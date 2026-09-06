from typing import Any

import jax.numpy as jnp
import flax.linen as nn
import netket as nk

from ._symmetry_projection import project_log_amplitude, project_log_amplitude_stable


class ProjectedRBM(nn.Module):
    """Deep restricted Boltzmann machine log-amplitude ansatz.

    When `symmetries`/`characters` are left as None this behaves as a plain
    deep RBM. Passing the (n_g, N) permutation table and (n_g,) irrep
    characters of a symmetry point (see
    `src.physics.symmetries.symmetry_projector_inputs`) instead projects the
    ansatz onto that symmetry sector.
    """

    num_layers: int = 2
    alpha: float = 1.0
    param_dtype: Any = jnp.complex128
    symmetries: Any = None
    characters: Any = None
    projector: str = "stable"
    """Which projection routine to use: "stable" (default, `logsumexp`-based,
    recommended) or "legacy" (original hand-rolled `log(sum(...))`, kept for
    backward compatibility / comparison). See `src.models._symmetry_projection`."""
    group_chunk_size: Any = None
    """Only used by the "stable" projector: evaluate the group orbit in
    chunks of this size instead of one big batch (bounds memory for large
    groups); None evaluates the whole orbit at once."""

    @nn.compact
    def __call__(self, x):
        kernel_init = nn.initializers.normal(stddev=0.01)
        bias_init = nn.initializers.normal(stddev=0.1)

        def log_amplitude(spins):
            h = spins
            for i in range(self.num_layers):
                n_hidden = int(self.alpha * h.shape[-1])
                h = nn.Dense(
                    features=n_hidden,
                    use_bias=True,
                    param_dtype=self.param_dtype,
                    kernel_init=kernel_init,
                    bias_init=bias_init,
                    name=f"layer_{i}",
                )(h)
                h = nn.LayerNorm(
                    param_dtype=self.param_dtype,
                    use_scale=False,
                    use_bias=False,
                    name=f"ln_{i}",
                )(h)
                h = nk.nn.log_cosh(h)

            res = jnp.sum(h, axis=-1)
            v_bias = self.param(
                "visible_bias", bias_init, (spins.shape[-1],), self.param_dtype
            )
            return res + jnp.dot(spins, v_bias)

        if self.projector == "legacy":
            return project_log_amplitude(log_amplitude, x, self.symmetries, self.characters)
        return project_log_amplitude_stable(
            log_amplitude, x, self.symmetries, self.characters, group_chunk_size=self.group_chunk_size
        )
