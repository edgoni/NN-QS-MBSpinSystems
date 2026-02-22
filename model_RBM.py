#model_RBM
import jax
import jax.numpy as jnp
import jaxlib
import flax
import flax.linen as nn
import optax
import netket as nk
from typing import Any


class DeepMLP(nn.Module):
  # features: int # número de features
  num_layers: int # número de capas
  alpha: float #
  @nn.compact
  def __call__(self, x):
    for i in range(self.num_layers):
      # ensure n_hidden is an integer
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

    @nn.compact
    def __call__(self, x):
        input_spins = x

        # inicializadores estables cómo en artículo
        kernel_init = nn.initializers.normal(stddev=0.01)
        # change bias_init from uniform to normal for complex dtype compatibility
        bias_init = nn.initializers.normal(stddev=0.1)

        for i in range(self.num_layers):
            # transformación lineal (W)
            n_hidden = int(self.alpha * x.shape[-1])
            x = nn.Dense(
                features=n_hidden,
                use_bias=True,
                param_dtype=self.param_dtype,
                kernel_init=kernel_init,
                bias_init=bias_init,
                name=f"layer_{i}"
            )(x)

            # usamos LayerNorm para mantener la varianza cerca de 1 antes del log_cosh
            x = nn.LayerNorm(
                param_dtype=self.param_dtype,
                use_scale=False, # Evitamos añadir más parámetros inestables al inicio
                use_bias=False
            )(x)

            # Activación No Lineal
            x = nk.nn.log_cosh(x) # Using nk.nn for consistency and to avoid deprecation warning

        # Colapso final a log-amplitud
        res = jnp.sum(x, axis=-1)

        # Visible Bias (Campo local directo sobre los espines originales)
        v_bias = self.param(
            "visible_bias",
            bias_init,
            (input_spins.shape[-1],),
            self.param_dtype,
        )
        out_bias = jnp.dot(input_spins, v_bias)

        return res + out_bias