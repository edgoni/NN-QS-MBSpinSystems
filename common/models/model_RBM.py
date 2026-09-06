import jax
import jax.numpy as jnp
import jaxlib
import flax
import flax.linen as nn
import optax
import netket as nk
from typing import Any, Optional
import numpy as np

from netket.jax import logsumexp_cplx
from netket.utils.group import PermutationGroup
from netket.utils import HashableArray


class DeepMLP(nn.Module):
  num_layers: int
  alpha: float
  @nn.compact
  def __call__(self, x):
    for i in range(self.num_layers):
      n_hidden = int(x.shape[-1] * self.alpha)
      x = nn.Dense(features=n_hidden,
                         dtype=complex,
                         use_bias=True,
                         kernel_init=nn.initializers.normal(stddev=0.1))(x)
      x = nk.nn.log_cosh(x)

    return jnp.sum(x, axis=-1)


class DeepRBM(nn.Module):
    num_layers: int = 2
    alpha: float = 1.0
    param_dtype: Any = jnp.complex128
    stable_cosh: bool = False

    @nn.compact
    def __call__(self, x):
        input_spins = x

        kernel_init = nn.initializers.normal(stddev=0.01)
        bias_init = nn.initializers.normal(stddev=0.1)

        for i in range(self.num_layers):
            n_hidden = int(self.alpha * x.shape[-1])
            x = nn.Dense(
                features=n_hidden,
                use_bias=True,
                param_dtype=self.param_dtype,
                kernel_init=kernel_init,
                bias_init=bias_init,
                name=f"layer_{i}"
            )(x)

            x = nn.LayerNorm(
                param_dtype=self.param_dtype,
                use_scale=False,
                use_bias=False
            )(x)

            if self.stable_cosh:
                x = jnp.log(1 + x ** 2 / 2)
            else:
                x = nk.nn.log_cosh(x)

        res = jnp.sum(x, axis=-1)

        v_bias = self.param(
            "visible_bias",
            bias_init,
            (input_spins.shape[-1],),
            self.param_dtype,
        )
        out_bias = jnp.dot(input_spins, v_bias)

        return res + out_bias


class SymmExpSumChunked(nn.Module):
    """Equivalente a `nk.nn.blocks.SymmExpSum`, pero evaluando la orbita del
    grupo con `jax.lax.map(..., batch_size=group_chunk_size)` en vez de
    `jax.vmap`.

    `SymmExpSum` proyecta a UNA sola irrep (la de `character_id`/`characters`),
    no a las |G| irreps -- pero construir ESE proyector, para cualquier irrep
    que se elija, exige evaluar la red en las |G| configuraciones trasladadas
    de cada sigma (ec. 38 del paper de Noormandipour et al: suma sobre
    elementos g del grupo, ponderada por el caracter de la irrep objetivo).
    Con `vmap`, esas |G| copias del batch existen todas a la vez en memoria;
    para FullSumState con SymmExpSum eso multiplica el batch de
    `expect_and_forces_fullsum` (que NetKet no chunkea, ver comentario en
    `make_vstate` de try_Norman.py) por |G|, reventando la memoria a partir de
    3x3 (2^18 x 9 filas). `lax.map` con `batch_size` hace el mismo calculo
    (mismo gradiente exacto, sin aproximar nada) pero procesando el eje del
    grupo en trozos secuenciales, acotando el pico de memoria sin tocar el
    coste de computo total.
    """

    module: nn.Module
    symm_group: PermutationGroup
    characters: Optional[HashableArray] = None
    character_id: Optional[int] = None
    group_chunk_size: Optional[int] = None
    remat: bool = True

    def setup(self):
        if self.characters is None:
            if self.character_id is None:
                self._chi = np.ones(len(np.asarray(self.symm_group)))
            else:
                self._chi = self.symm_group.character_table()[self.character_id]
        else:
            if self.character_id is None:
                self._chi = self.characters.wrapped
            else:
                raise AttributeError(
                    "Must not specify both `characters` and `character_id`"
                )

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x_symm = self.symm_group @ x
        n_symm = x_symm.shape[0]

        chunk = n_symm if self.group_chunk_size is None else int(self.group_chunk_size)
        chunk = max(1, min(chunk, n_symm))
        while n_symm % chunk != 0:
            chunk -= 1

        if chunk == n_symm:
            psi_symm = self.module(x_symm)
        else:
            n_chunks = n_symm // chunk
            x_chunks = x_symm.reshape((n_chunks, chunk) + x_symm.shape[1:])

            def _apply(module, carry, xs):
                return carry, module(xs)

            body = nn.remat(_apply) if self.remat else _apply

            scan_apply = nn.scan(
                body,
                variable_broadcast="params",
                split_rngs={"params": False},
                in_axes=0,
                out_axes=0,
            )
            _, psi_chunks = scan_apply(self.module, None, x_chunks)
            psi_symm = psi_chunks.reshape((n_symm,) + psi_chunks.shape[2:])

        characters = np.expand_dims(self._chi, tuple(range(1, x.ndim)))

        logsumexp_fun = (
            jax.scipy.special.logsumexp if np.all(characters >= 0) else logsumexp_cplx
        )

        psi = logsumexp_fun(psi_symm, axis=0, b=characters / len(self.symm_group))
        return psi


class DeepRBMSymmProj(nn.Module):
    """
    DeepRBM con proyección explícita al sector de simetría,
    equivalente a lo que hace el repositorio TQC con self.symmetry.mm(bx).
    
    La proyección se aplica DESPUÉS de calcular log|ψ(σ)|,
    construyendo la amplitud proyectada como:
        ψ_proj(σ) = Σ_g  χ*(g) * ψ(P_g σ)
    que es exactamente S @ ψ en espacio de configuraciones.
    """
    num_layers:  int = 2
    alpha:       float = 1.0
    param_dtype: Any = jnp.complex128
    symmetries:  Any = None
    characters:  Any = None

    @nn.compact
    def __call__(self, x):
        """
        x: configuración de espines, shape (..., N) con valores ±1
        """
        input_spins = x

        kernel_init = nn.initializers.normal(stddev=0.01)
        bias_init   = nn.initializers.normal(stddev=0.1)

        def _rbm_logpsi(spin_config):
            """Aplica las capas RBM a una configuración dada."""
            h = spin_config
            for i in range(self.num_layers):
                n_hidden = int(self.alpha * h.shape[-1])
                h = nn.Dense(
                    features=n_hidden,
                    use_bias=True,
                    param_dtype=self.param_dtype,
                    kernel_init=kernel_init,
                    bias_init=bias_init,
                    name=f"layer_{i}"
                )(h)
                h = nn.LayerNorm(
                    param_dtype=self.param_dtype,
                    use_scale=False,
                    use_bias=False,
                    name=f"ln_{i}"
                )(h)
                h = nk.nn.log_cosh(h)

            res = jnp.sum(h, axis=-1)

            v_bias = self.param(
                "visible_bias",
                bias_init,
                (spin_config.shape[-1],),
                self.param_dtype,
            )
            return res + jnp.dot(spin_config, v_bias)

        if self.symmetries is None or self.characters is None:
            return _rbm_logpsi(input_spins)

        perms  = jnp.array(self.symmetries)
        chars  = jnp.array(self.characters)

        x_perm = input_spins[..., perms]


        log_amps = jax.vmap(
            _rbm_logpsi,
            in_axes=0,
            out_axes=0
        )(x_perm.reshape(-1, input_spins.shape[-1]))

        batch_shape = input_spins.shape[:-1]
        n_g = perms.shape[0]
        log_amps = log_amps.reshape(*batch_shape, n_g)

        chars_conj = jnp.conj(chars)

        log_max   = jnp.max(jnp.real(log_amps), axis=-1, keepdims=True)
        amps_rel  = jnp.exp(log_amps - log_max)
        weighted  = jnp.sum(chars_conj * amps_rel, axis=-1)
        log_proj  = jnp.log(weighted) + log_max[..., 0]

        return log_proj

class MonomialSymmExpSum(nn.Module):
    """Like `SymmExpSumChunked`, but for a group whose elements act as a
    permutation *times a configuration-dependent phase*.

    `SymmExpSum` and `SymmExpSumChunked` assume every group element merely
    relabels a configuration, so the projected amplitude is
    `sum_g chi*(g) psi(g^-1 sigma)`. The C3 of the isotropic Kitaev point is
    not of that kind: it rotates the lattice *and* spin space together, and
    in the rotated spin frame (see `common/physics/isotropic_symmetry.py`) it
    acts as `R_g |sigma> = omega**(k_g n_-(sigma)) |P_g sigma>` -- a
    *monomial* operator, one connection per configuration but with a phase.

    Because `n_-` (the number of sites in the second local state) is
    invariant under site permutations, that phase depends only on the
    configuration and on the C3 power `k_g`, never on which translation. So
    it factors straight into the logsumexp weight as an effective character,
    and the cost stays O(|G|) per sample -- the whole reason this is usable
    at 3x3, where the same operator in the computational basis would connect
    each configuration to all 2^18.

    :param element_powers: `k_g` for each group element, shape (|G|,)
    """

    module: nn.Module
    symm_group: Any
    """(|G|, N) permutations, or a PermutationGroup."""
    characters: HashableArray
    """Characters of the target irrep, shape (|G|,)."""
    element_powers: HashableArray
    """Power of the spin rotation carried by each group element, shape (|G|,).

    For the C3 group (`root_order=3`) this is the C3 power in {0,1,2}; for
    the C2v group (`root_order=2`) it is the mirror grade in {0,1}."""
    root_order: int = 3
    """Order n of the root of unity: the phase is exp(2 pi i / n)**(power *
    n_minus). 3 for the C3 (default, unchanged), 2 for the xy mirror / C2v
    where the rotated spin factor is diag(1, -1)."""
    group_chunk_size: Optional[int] = None
    remat: bool = True

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        perms = jnp.asarray(np.asarray(self.symm_group))
        powers = jnp.asarray(np.asarray(self.element_powers))

        chars = jnp.conj(jnp.asarray(np.asarray(self.characters)))
        n_g = perms.shape[0]

        x_symm = jnp.moveaxis(x[..., perms], -2, 0)

        chunk = n_g if self.group_chunk_size is None else int(self.group_chunk_size)
        chunk = max(1, min(chunk, n_g))
        while n_g % chunk != 0:
            chunk -= 1

        if chunk == n_g:
            psi_symm = self.module(x_symm)
        else:
            n_chunks = n_g // chunk
            x_chunks = x_symm.reshape((n_chunks, chunk) + x_symm.shape[1:])

            def _apply(module, carry, xs):
                return carry, module(xs)

            body = nn.remat(_apply) if self.remat else _apply
            scan_apply = nn.scan(
                body,
                variable_broadcast="params",
                split_rngs={"params": False},
                in_axes=0,
                out_axes=0,
            )
            _, psi_chunks = scan_apply(self.module, None, x_chunks)
            psi_symm = psi_chunks.reshape((n_g,) + psi_chunks.shape[2:])

        n_minus = jnp.sum(x < 0, axis=-1)
        omega = jnp.exp(2j * jnp.pi / float(self.root_order))
        exponent = powers.reshape((-1,) + (1,) * (x.ndim - 1)) * n_minus[None, ...]
        phase = omega ** jnp.mod(exponent, self.root_order)

        weights = chars.reshape((-1,) + (1,) * (x.ndim - 1)) * phase / n_g
        return logsumexp_cplx(psi_symm, axis=0, b=weights)
