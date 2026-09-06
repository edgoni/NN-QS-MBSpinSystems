# -*- coding: utf-8 -*-
from typing import Any

import jax
import jax.numpy as jnp
import flax.linen as nn


DOT = "dot"
FACTORED = "factored"


def _is_complex(dtype) -> bool:
    return jnp.issubdtype(jnp.dtype(dtype), jnp.complexfloating)


def _real_counterpart(dtype):
    """El dtype real asociado a `dtype` (float64 para complex128, etc.).

    El ARGUMENTO del softmax tiene que ser real aunque el resto de la red sea
    compleja: el softmax sobre complejos corre pero NO devuelve una
    distribucion -- medido, produce pesos con parte real negativa. Un mapa de
    atencion con pesos negativos no es un promedio ponderado sobre sitios, es
    otra cosa; y ademas `jnp.max` (que el softmax usa para estabilizar) ordena
    complejos lexicograficamente, sin relacion suave con la magnitud, lo que da
    gradientes caoticos.
    """
    return jnp.finfo(jnp.dtype(dtype)).dtype


class TransformerBlock(nn.Module):
    """Un bloque pre-LN: x -> x + MHA(LN(x)) -> x + MLP(LN(x)).

    `attention` selecciona el mapa de atencion:
        "dot"      -> A = softmax(Q K^T / sqrt(dk)), depende de la configuracion.
        "factored" -> A = softmax(P), con P un parametro (heads, N, N) aprendido
                      e independiente de la configuracion.

    **Con `param_dtype` complejo** el tronco entero (embedding, V, out_proj,
    LayerNorm, MLP) pasa a ser complejo, pero el MAPA DE ATENCION se mantiene
    real a proposito:

    - "dot": las puntuaciones se calculan con el producto interno hermitico
      `Re(Q . K*)`, siempre real -- el mismo patron que
      `legacy/SelfAtt/selfatt_models.ComplexSelfAttention`.
    - "factored": `A = softmax(Re P) * exp(i Im P)`, con `P` el parametro
      complejo. El MODULO sigue siendo una distribucion de verdad sobre sitios
      (softmax de un argumento real) y la FASE queda libre. Con `P` real se
      recupera `softmax(P)` exactamente.

    `LayerNorm` si acepta complejo tal cual: normaliza con la norma hermitica
    `E|x-mu|^2` (verificado), que es la definicion sensata. `gelu` tambien.

    Espera SIEMPRE (batch, N, d_model) exactamente 3D. `Transformer` aplana los
    ejes de batch antes de llamarlo, ver alli el porque.
    """

    heads: int
    dk: int
    d_model: int
    d_ff_mult: int = 2
    attention: str = DOT
    param_dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, x):
        b, n, _ = x.shape
        inner = self.heads * self.dk

        def dense(feats, name, **kw):
            return nn.Dense(
                features=feats, param_dtype=self.param_dtype, name=name, **kw
            )

        h = nn.LayerNorm(param_dtype=self.param_dtype, name="ln_att")(x)

        if self.attention == DOT:
            qkv = dense(3 * inner, "qkv", use_bias=False)(h)
            qkv = qkv.reshape(b, n, 3, self.heads, self.dk)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

            real_dtype = _real_counterpart(self.param_dtype)
            logits = jnp.einsum("...ihd,...jhd->...hij", q, jnp.conj(k))
            logits = jnp.real(logits) / jnp.sqrt(
                jnp.asarray(self.dk, dtype=real_dtype)
            )
            weights = jax.nn.softmax(logits, axis=-1).astype(self.param_dtype)
            att = jnp.einsum("...hij,...jhd->...ihd", weights, v)

        elif self.attention == FACTORED:
            v = dense(inner, "v", use_bias=False)(h)
            v = v.reshape(b, n, self.heads, self.dk)

            logits = self.param(
                "att_logits",
                nn.initializers.normal(stddev=0.02),
                (self.heads, n, n),
                self.param_dtype,
            )
            if _is_complex(self.param_dtype):
                weights = (jax.nn.softmax(jnp.real(logits), axis=-1)
                           * jnp.exp(1j * jnp.imag(logits))).astype(self.param_dtype)
            else:
                weights = jax.nn.softmax(logits, axis=-1)
            att = jnp.einsum("hij,...jhd->...ihd", weights, v)

        else:
            raise ValueError(
                "attention debe ser %r o %r, no %r" % (DOT, FACTORED, self.attention)
            )

        att = att.reshape(b, n, inner)
        att = dense(self.d_model, "out_proj")(att)

        x = x + att

        h = nn.LayerNorm(param_dtype=self.param_dtype, name="ln_mlp")(x)
        h = dense(self.d_ff_mult * self.d_model, "mlp_up")(h)
        h = nn.gelu(h)
        h = dense(self.d_model, "mlp_down")(h)

        return x + h


class Transformer(nn.Module):
    """Funcion de onda variacional Transformer.

    Argumentos:
        layers: numero de bloques transformer.
        heads: numero de cabezas de atencion (independientes entre si).
        dk: dimension de cada cabeza. El ancho interno de la atencion es
            heads*dk y se reproyecta a d_model, asi que heads y d_model no
            tienen que ser divisibles entre si.
        d_model: dimension del embedding por sitio.
        d_ff_mult: factor de expansion del MLP interno.
        attention: "dot" (estandar) o "factored" (mapa aprendido).
        param_dtype: dtype de los parametros. Real (`jnp.float64`, por
            defecto) o COMPLEJO (`jnp.complex128`).

            Con parametros reales el tronco es real y log(psi) se construye al
            final combinando DOS canales reales: log|psi| y theta salen de dos
            readouts independientes, cada uno una suma sobre sitios.

            Con parametros complejos el tronco entero es complejo y la cabeza
            emite UN solo canal, ya complejo -- amplitud y fase salen
            entrelazadas de los mismos parametros, como en `DeepRBM`. El mapa
            de atencion se mantiene real en los dos casos (ver
            `TransformerBlock`).

            Motivo: es la hipotesis de `docs/codigo/simetrias_de_espin_en_el_codigo.md`
            §7.3 -- que el tronco real limita la estructura de fase
            representable, y que eso explicaria por que el RBM (complejo
            nativo) aguanta el Hamiltoniano ROTADO de c3/c2v y el transformer
            no. Con esta opcion se puede medir en vez de conjeturar.

            AVISO: cambia el pytree de parametros (dtype, y la cabeza pasa de
            2 canales a 1), asi que un checkpoint real NO se puede cargar en
            un modelo complejo ni al reves.
        out_dtype: complejo para log(psi) = log|psi| + i*theta, o real para un
            ansatz de amplitudes positivas. Se ignora si `param_dtype` ya es
            complejo: ahi la salida es compleja por construccion.

    **Ejes de batch arbitrarios.** `TransformerBlock` desempaqueta `b, n, _ =
    x.shape`, asi que exige exactamente 3D. Pero los envoltorios de proyeccion
    de simetria (`SymmExpSumChunked`, `MonomialSymmExpSum` en model_RBM.py)
    llaman al modulo con la orbita del grupo apilada delante: (|G|, batch, N),
    o (chunk, batch, N) al trocear. Con la version original eso reventaba en el
    desempaquetado. `DeepRBM` no tiene el problema porque solo usa `nn.Dense` y
    reducciones sobre el ultimo eje, que difunden sobre cualquier numero de
    ejes por delante.

    La solucion es aplanar aqui todos los ejes de batch a uno y restaurar la
    forma al final. Es exacto (ningun paso mezcla muestras entre si) y deja
    `TransformerBlock` con su contrato 3D intacto.
    """

    layers: int = 2
    heads: int = 4
    dk: int = 8
    d_model: int = 32
    d_ff_mult: int = 2
    attention: str = DOT
    param_dtype: Any = jnp.float64
    out_dtype: Any = jnp.complex128

    @nn.compact
    def __call__(self, x):
        x = x.astype(self.param_dtype)
        n_sites = x.shape[-1]

        lead_shape = x.shape[:-1]
        x = x.reshape((-1, n_sites))

        h = x[..., None]
        h = nn.Dense(
            features=self.d_model, param_dtype=self.param_dtype, name="embed"
        )(h)

        pos = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (n_sites, self.d_model),
            self.param_dtype,
        )
        h = h + pos

        for i in range(self.layers):
            h = TransformerBlock(
                heads=self.heads,
                dk=self.dk,
                d_model=self.d_model,
                d_ff_mult=self.d_ff_mult,
                attention=self.attention,
                param_dtype=self.param_dtype,
                name="block_%d" % i,
            )(h)

        h = nn.LayerNorm(param_dtype=self.param_dtype, name="ln_final")(h)

        if _is_complex(self.param_dtype):
            out = nn.Dense(
                features=1,
                param_dtype=self.param_dtype,
                kernel_init=nn.initializers.normal(stddev=0.01),
                name="head",
            )(h)
            out = jnp.sum(out, axis=-2)
            out = out[..., 0]
        else:
            out = nn.Dense(
                features=2,
                param_dtype=self.param_dtype,
                kernel_init=nn.initializers.normal(stddev=0.01),
                name="head",
            )(h)
            out = jnp.sum(out, axis=-2)
            if jnp.issubdtype(jnp.dtype(self.out_dtype), jnp.complexfloating):
                out = out[..., 0] + 1j * out[..., 1]
            else:
                out = out[..., 0]
        out = out.astype(self.out_dtype)

        return out.reshape(lead_shape)


class FactoredSelfAttention(Transformer):
    """Transformer con atencion factorizada (mapa aprendido, no Q K^T).

    Identico a `Transformer` en todo lo demas, para que la comparacion entre
    ambos aisle el efecto del mecanismo de atencion.

    Coste: el mapa son heads*N*N parametros por bloque, pero se ahorra las
    proyecciones Q y K y el producto QK^T en cada paso, y el mapa no se
    recalcula por muestra.
    """

    attention: str = FACTORED
