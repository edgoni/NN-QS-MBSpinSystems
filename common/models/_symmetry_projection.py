import jax
import jax.numpy as jnp
import numpy as np
from netket.jax import logsumexp_cplx


def project_log_amplitude(log_amplitude_fn, x, symmetries, characters):
    """Apply `log_amplitude_fn` to `x`, optionally projected onto an irrep.

    :param log_amplitude_fn: callable (..., N) spins -> (...,) log-amplitude
    :param x: spin configuration(s), shape (..., N) with values +/-1
    :param symmetries: None, or an array-like of shape (n_g, N) with one
        site permutation per row
    :param characters: None, or an array-like of shape (n_g,) with the
        (complex) characters of the target irrep
    """
    if symmetries is None or characters is None:
        return log_amplitude_fn(x)

    perms = jnp.array(symmetries)
    chars = jnp.array(characters)

    x_perm = x[..., perms]
    n_g = perms.shape[0]
    n_sites = x.shape[-1]
    batch_shape = x.shape[:-1]

    x_super_batch = x_perm.reshape(-1, n_sites)
    log_amps = log_amplitude_fn(x_super_batch).reshape(*batch_shape, n_g)

    chars_conj = jnp.conj(chars)
    log_max = jnp.max(jnp.real(log_amps), axis=-1, keepdims=True)
    amps_rel = jnp.exp(log_amps - log_max)
    weighted = jnp.sum(chars_conj * amps_rel, axis=-1)
    return jnp.log(weighted) + log_max[..., 0]


def project_log_amplitude_stable(log_amplitude_fn, x, symmetries, characters, group_chunk_size=None):
    """Numerically stable, memory-chunked alternative to
    `project_log_amplitude`, matching `nk.nn.blocks.SymmExpSum` /
    `common/models/model_RBM.py::SymmExpSumChunked`.

    :param log_amplitude_fn: callable (..., N) spins -> (...,) log-amplitude
    :param x: spin configuration(s), shape (..., N) with values +/-1
    :param symmetries: None, or an array-like of shape (n_g, N) with one
        site permutation per row
    :param characters: None, or an array-like of shape (n_g,) with the
        (complex) characters of the target irrep
    :param group_chunk_size: if given, `log_amplitude_fn` is evaluated over
        the group orbit in chunks of this many elements at a time (via
        `jax.lax.map`) instead of a single batch of size `n_g * batch`,
        bounding peak memory for large groups. None evaluates the whole
        orbit at once.
    """
    if symmetries is None or characters is None:
        return log_amplitude_fn(x)

    perms = jnp.array(symmetries)
    chars_conj = jnp.conj(jnp.array(characters))
    n_g = perms.shape[0]

    x_perm = jnp.moveaxis(x[..., perms], -2, 0)

    if group_chunk_size is None:
        log_amps = jax.vmap(log_amplitude_fn)(x_perm)
    else:
        log_amps = jax.lax.map(log_amplitude_fn, x_perm, batch_size=group_chunk_size)

    weights = chars_conj / n_g
    weights = weights.reshape((n_g,) + (1,) * (log_amps.ndim - 1))
    logsumexp_fun = jax.scipy.special.logsumexp if np.all(np.asarray(characters) >= 0) else logsumexp_cplx
    return logsumexp_fun(log_amps, axis=0, b=weights)
