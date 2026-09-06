import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..'))
sys.path.append(_HERE)
sys.path.append(_REPO)
sys.path.append(os.path.join(_REPO, 'common', 'models'))

import copy
import pickle
import time
import numpy as np
import jax.numpy as jnp
import flax
import optax
import netket as nk
import netket.experimental as nkx
import pandas as pd
import csv

from cycles_Kitaev import obtener_plaquetas_kitaev, build_Wp_operators
from utils import BestIterKeeper, make_extract_metrics, KitaevTransverse_H
from model_RBM import DeepRBM, SymmExpSumChunked, MonomialSymmExpSum
from transformer import Transformer, FactoredSelfAttention
from kitaev_implementation import declare_kitaev
from netket.utils import HashableArray
from common.physics.symmetries import get_kitaev_symmetries, get_projection_group
from common.physics.isotropic_symmetry import (
    c3_translation_group,
    c3_character_table,
    c3_irrep_weights,
    c2v_translation_group,
    c2v_character_table,
    c2v_irrep_weights,
    check_c2xy_applicable,
    rotated_kitaev_hamiltonian,
)

_GROUP_AXIS = {"c3": (1.0, 1.0, 1.0), "c2v": (1.0, 1.0, 0.0)}

_GROUP_ROOT_ORDER = {"c3": 3, "c2v": 2}

_PROJECTION_GROUPS = ("none", "translation", "space", "c3", "c2v")

_ANSATZE = ("rbm", "transformer", "factored")
from common.physics.exact_diag import (
    degenerate_manifold,
    manifold_irrep_weights,
    sectors_hosting_manifold,
    manifold_fidelity,
)

FULLSUM_MAX_SPINS = 20

OG_LR_SCHEDULE_KWARGS = dict(
    init_value=0.01,
    peak_value=0.05,
    warmup_steps=30,
    transition_steps=100,
    decay_rate=0.90,
)
OG_DIAG_SHIFT = 0.01
OG_N_SAMPLES = 2048

OG_LONG_LR_SCHEDULE_KWARGS = dict(OG_LR_SCHEDULE_KWARGS, transition_steps=400)

OG_HOT_LR_SCHEDULE_KWARGS = dict(
    OG_LONG_LR_SCHEDULE_KWARGS, peak_value=0.10, decay_rate=0.9118
)

_HPARAMS = ("og", "og_long", "og_hot", "staged")

_OG_FAMILY = {
    "og": OG_LR_SCHEDULE_KWARGS,
    "og_long": OG_LONG_LR_SCHEDULE_KWARGS,
    "og_hot": OG_HOT_LR_SCHEDULE_KWARGS,
}

_PRESET = "preset"


class BestManifoldFidelity:
    '''Best-iterate checkpoint que sigue la fidelidad frente al MANIFOLD
    degenerado completo del groundstate (F = sum_i |<g_i|psi>|^2, ver
    `manifold_fidelity` en src/physics/exact_diag.py), no el overlap contra
    un único autovector arbitrario.

    Esto importa porque ahora entrenamos un modelo por cada sector de
    simetría que hospeda el manifold degenerado (ver `train_projected`): cada
    corrida converge a un representante distinto del mismo manifold, y
    comparar contra un único autovector fijo (el que devolvió Lanczos, en una
    base arbitraria dentro del subespacio degenerado) penalizaría
    injustamente sectores "correctos" que no coinciden con esa base
    concreta. La fidelidad de manifold es invariante a esa elección de base.

    :param eigvecs: autovectores exactos (columnas), como devuelve lanczos_ed
    :param manifold_idx: índices de columnas de `eigvecs` que forman el
        subespacio degenerado del groundstate (ver `degenerate_manifold`)
    :param every: calcular la fidelidad solo cada `every` pasos
    :param print_every: imprimir la fidelidad ACTUAL (no solo la mejor
        histórica) cada `print_every` pasos, para poder ver en vivo si un
        plateau de energía es convergencia real o un colapso de SR
        transitorio (ver discusión en el log de entrenamiento). None
        desactiva el print.
    :param history: lista compartida (mutable) donde `update` añade
        `(step, fidelity, best_fidelity)` cada vez que evalúa. Antes de esto
        la EVOLUCIÓN de la fidelidad solo existía como texto impreso por
        consola -- si la sesión de Colab se desconecta, se pierde entera (fue
        justo lo que pasó con la corrida c3 del punto isótropo). Pasando la
        MISMA lista a las `BestManifoldFidelity` de las fases 0/1/2 (y de cada
        bloque de `plateau_patience`), la traza queda completa y en orden de
        step GLOBAL a través de toda la corrida, no solo de una fase. None
        (por defecto) crea una lista propia, no compartida -- úsese `history`
        explícito para acumular entre fases.
    '''

    def __init__(self, eigvecs, manifold_idx, every=1, print_every=20, history=None):
        self.eigvecs = eigvecs
        self.manifold_idx = manifold_idx
        self.every = every
        self.print_every = print_every
        self.best_fidelity = -1.0
        self.best_state = None
        self.last_fidelity = None
        self.history = history if history is not None else []

    def update(self, step, log_data, driver):
        if step % self.every != 0:
            return True
        psi = driver.state.to_array()
        psi = psi / np.linalg.norm(psi)
        fidelity = manifold_fidelity(self.eigvecs, self.manifold_idx, psi)
        self.last_fidelity = fidelity
        if fidelity > self.best_fidelity:
            self.best_fidelity = fidelity
            self.best_state = copy.copy(driver.state)
            self.best_state.parameters = flax.core.copy(driver.state.parameters)
        self.history.append((step, fidelity, self.best_fidelity))
        if self.print_every is not None and step % self.print_every == 0:
            print(f"  [fidelity] step {step}: F_manifold={fidelity:.4f}  (mejor hasta ahora={self.best_fidelity:.4f})")
        return True


class PeriodicCheckpoint:
    '''Guarda a disco, cada `every` pasos, el mejor estado conocido hasta
    ese momento (por fidelidad de manifold si la tenemos, si no por
    energía) y las métricas acumuladas -- sobrescribiendo los mismos
    ficheros que el guardado final. Así, si el proceso muere a mitad de
    entrenamiento (p.ej. Colab se desconecta a las 3 horas de una corrida
    de 4.5h), queda en disco el último checkpoint en vez de perderlo todo.

    :param vstate_path: ruta del .pkl de parámetros (se sobrescribe)
    :param metrics_path: ruta del .csv de métricas (se sobrescribe)
    :param metrics_history: dict compartido que `make_extract_metrics` va
        rellenando; se vuelca tal cual está en cada checkpoint
    :param keeper: `BestIterKeeper` (fallback si no hay `fidelity_keeper`)
    :param fidelity_keeper: `BestManifoldFidelity` opcional, preferido
    :param every: cada cuántos pasos guardar
    :param fidelity_trace: lista compartida `(step, fidelity, best_fidelity)`
        (el `history` de `BestManifoldFidelity`, ver ahí). Antes de esto la
        EVOLUCION de la fidelidad solo vivía en la consola -- se pierde
        entera si la sesión muere, que es justo lo que le paso a la corrida
        c3 del punto isótropo (su traza completa solo sobrevivió porque el
        usuario pegó el log a mano). Con esto queda en disco cada `every`
        pasos, igual que `metrics_history`. None desactiva el volcado (p.ej.
        si no hay diagonalización exacta y por tanto no hay fidelidad).
    :param fidelity_trace_path: ruta del .csv donde se vuelca `fidelity_trace`.
    '''

    def __init__(self, vstate_path, metrics_path, metrics_history, keeper,
                 fidelity_keeper=None, every=100, energy_vstate_path=None,
                 fidelity_trace=None, fidelity_trace_path=None):
        self.vstate_path = vstate_path
        self.energy_vstate_path = energy_vstate_path
        self.metrics_path = metrics_path
        self.metrics_history = metrics_history
        self.keeper = keeper
        self.fidelity_keeper = fidelity_keeper
        self.every = every
        self.fidelity_trace = fidelity_trace
        self.fidelity_trace_path = fidelity_trace_path

    def update(self, step, log_data, driver):
        if step == 0 or step % self.every != 0:
            return True
        best_state = None
        if self.fidelity_keeper is not None and self.fidelity_keeper.best_state is not None:
            best_state = self.fidelity_keeper.best_state
        elif self.keeper.best_state is not None:
            best_state = self.keeper.best_state
        if best_state is not None:
            with open(self.vstate_path, "wb") as f:
                pickle.dump(best_state.parameters, f)
        if self.energy_vstate_path is not None and self.keeper.best_state is not None:
            with open(self.energy_vstate_path, "wb") as f:
                pickle.dump(self.keeper.best_state.parameters, f)
        pd.DataFrame(self.metrics_history).to_csv(self.metrics_path, index=False)
        if self.fidelity_trace is not None and self.fidelity_trace_path is not None and self.fidelity_trace:
            pd.DataFrame(
                self.fidelity_trace, columns=["step", "fidelity", "best_fidelity"]
            ).to_csv(self.fidelity_trace_path, index=False)
        print(f"  [checkpoint] step {step}: guardado en {self.vstate_path}")
        return True


def make_vstate(hi, model, N, sampler=None, n_samples=2048, seed=0, chunk_size=4096, use_mcstate=False):
    '''Usa suma exacta sobre todo el Hilbert space si el sistema es pequeño
    (como hacía el paper -- ver Noormandipour et al 2022, sec. 5.1: su propio
    RBM entrenado con NetKet+Metropolis se queda muy por debajo en precisión
    frente a su version con suma exacta + proyección de simetría), o Monte
    Carlo si es demasiado grande o si `use_mcstate=True` fuerza esa rama
    explícitamente.

    IMPORTANTE sobre `chunk_size` en FullSumState: solo trocea `to_array()`/
    `expect()` (ver `netket/vqs/full_summ/state.py`). El cálculo del
    GRADIENTE de la energía (`expect_and_forces_fullsum` en
    `netket/vqs/full_summ/expect.py`) hace un único VJP sin chunkear sobre
    TODO el Hilbert space -- `chunk_size` no lo alcanza. Con
    `nk.nn.blocks.SymmExpSum` (que multiplica ese batch por el tamaño del
    grupo de simetría ANTES de reducir, vía `jax.vmap`) esto hacía que
    FullSumState dejara de ser viable en memoria a partir de lattices ~3x3
    (2^18 configuraciones x 9 traslaciones ~2.36M, intentaba reservar ~21 GiB
    de golpe). Usando en su lugar `SymmExpSumChunked` (ver model_RBM.py, que
    hace lo mismo pero con `jax.lax.map(..., batch_size=group_chunk_size)`
    en vez de `vmap`) el pico de memoria queda acotado por `group_chunk_size`
    en vez de por el tamaño del grupo completo, así que FullSumState vuelve a
    ser viable a 3x3 (N=18, sin el multiplicador de grupo el batch base son
    solo 2^18 filas). `use_mcstate=True` sigue disponible para forzar Monte
    Carlo en lattices más grandes donde ni siquiera el batch base cabe en
    memoria.
    '''
    if N <= FULLSUM_MAX_SPINS and not use_mcstate:
        return nk.vqs.FullSumState(hi, model=model, seed=seed, chunk_size=chunk_size)
    if sampler is None:
        raise ValueError("Se necesita un sampler para MCState (N > FULLSUM_MAX_SPINS o use_mcstate=True)")
    vstate = nk.vqs.MCState(sampler, model=model, n_samples=n_samples, seed=seed)
    if chunk_size is not None:
        cs = min(int(chunk_size), int(vstate.n_samples))
        while cs > 1 and vstate.n_samples % cs != 0:
            cs -= 1
        vstate.chunk_size = cs if cs > 1 else None
    return vstate


def _match_params_to_model(params, projected):
    """Adapta un pytree de parametros entre el ansatz DESNUDO y el PROYECTADO.

    Los envoltorios de proyeccion (`SymmExpSumChunked`, `MonomialSymmExpSum`)
    guardan el RBM como submodulo de Flax, asi que sus parametros viven un
    nivel mas abajo, bajo la clave 'module':

        desnudo   : {'layer_0': ..., 'layer_1': ..., 'visible_bias': ...}
        proyectado: {'module': {'layer_0': ..., ...}}

    Trasplantar pesos entre la fase sin proyectar y la proyectada -- o
    reanudar una corrida proyectada desde un checkpoint sin proyectar --
    necesita anadir o quitar ese nivel. Flax lanza si las estructuras no
    casan, asi que no es un fallo silencioso, pero el arreglo corresponde
    aqui y no en cada sitio que llame.

    Es idempotente: si el pytree ya tiene la forma que pide `projected`, se
    devuelve tal cual.
    """
    if params is None:
        return None
    keys = set(params.keys())
    wrapped = (keys == {"module"})
    if projected and not wrapped:
        return {"module": params}
    if not projected and wrapped:
        return params["module"]
    return params


def _load_exact_spectrum(path, jx, jy, jz, H, k_needed, tol=1e-6):
    """Carga `(evals, evecs)` de un `.npz` de diagonalizacion exacta en vez de
    volver a diagonalizar. Devuelve None si el cache no aplica (y dice por que).

    El fichero es el `energies_eigenvecs_dict_k40.npz` que ya usa
    `legacy/Retry_OG/training.py`: un dict `{jz: {...}}` bajo la clave
    `data_dict`, con 'energies' (k,) y 'eigenvectors' (2^N, k) por punto.

    Tres condiciones, y ninguna es opcional:

    1. **La parametrizacion tiene que coincidir.** El fichero esta indexado
       SOLO por jz, asumiendo `jx = jy = (1-jz)/2`. Con `couplings` explicitos
       que no sigan esa relacion (p.ej. el punto isotropico) la clave jz no
       identifica el Hamiltoniano y el cache no vale.
    2. **Tiene que haber al menos `k_needed` autovalores.** Si se pidieron mas
       de los que hay guardados, se rediagonaliza: truncar en silencio seria
       reintroducir justo el bug de manifold que `k_eigenvals` bajo provoca.
    3. **El H tiene que ser el mismo.** Se comprueba midiendo
       `<v0|H|v0>` contra la energia guardada -- un unico producto
       matriz-vector disperso, barato. Es lo que separa esto de un cache que
       devuelve el espectro equivocado en silencio: si el fichero se calculo
       con otros acoplos, otra red, u otra convencion de signo, salta aqui.

       Esta condicion es lo que hace la funcion valida para CUALQUIER frame,
       no solo el plano: `H` es lo que el llamante le pase, y con
       `projection_group` "c3"/"c2v" `train_projected` ya le pasa el H
       ROTADO. Un cache de autovectores PLANOS (`energies_eigenvecs_dict_k40.npz`
       tal cual) falla esta condicion con un H rotado -- `<v0|H_rotado|v0>`
       no coincide con la energia plana guardada -- y lanza `ValueError` en
       vez de proyectar sobre el sector equivocado en silencio. Un cache ya
       rotado a ese frame (`common/analysis/rotate_exact_spectrum.py`) la pasa.
    """
    jz_expected = (1.0 - float(jz)) / 2.0
    if not (abs(jx - jy) < 1e-12 and abs(jx - jz_expected) < 1e-12):
        print(f"  [exact] el cache asume jx=jy=(1-jz)/2; estos acoplos son "
              f"Jx={jx:.4f} Jy={jy:.4f} Jz={jz:.4f} -> se diagonaliza")
        return None

    try:
        raw = np.load(path, allow_pickle=True)["data_dict"].item()
    except Exception as exc:  # noqa: BLE001
        print(f"  [exact] no se pudo leer {path} ({exc}) -> se diagonaliza")
        return None

    key = None
    for cand in raw:
        if abs(float(cand) - float(jz)) < 1e-9:
            key = cand
            break
    if key is None:
        print(f"  [exact] jz={jz:.4f} no esta en {path} "
              f"(hay {sorted(float(c) for c in raw)}) -> se diagonaliza")
        return None

    entry = raw[key]
    evals = np.asarray(entry["energies"]).real
    evecs = np.asarray(entry["eigenvectors"])
    if evals.shape[0] < k_needed:
        print(f"  [exact] el cache trae {evals.shape[0]} autovalores pero se "
              f"pidieron {k_needed} -> se diagonaliza")
        return None

    order = np.argsort(evals)
    evals, evecs = evals[order], evecs[:, order]

    v0 = evecs[:, 0]
    v0 = v0 / np.linalg.norm(v0)
    e_check = float(np.real(np.vdot(v0, H.to_sparse() @ v0)))
    if abs(e_check - evals[0]) > max(tol, tol * abs(evals[0])):
        raise ValueError(
            f"El espectro de {path} NO corresponde a este Hamiltoniano en "
            f"jz={jz:.4f}: la energia guardada es {evals[0]:.8f} pero "
            f"<v0|H|v0> = {e_check:.8f} (dif {abs(e_check-evals[0]):.2e}). "
            f"Revisa que el fichero se calculo con estos acoplos y esta "
            f"convencion de signo."
        )
    print(f"  [exact] espectro cargado de {path}: {evals.shape[0]} autovalores, "
          f"E0={evals[0]:.6f} (verificado con <v0|H|v0>)")
    return evals, evecs


def _offset_schedule(schedule, offset):
    '''Envuelve un schedule de optax para que, evaluado en el step LOCAL de
    un driver nuevo (que siempre arranca su contador en 0), devuelva el
    mismo valor que tendría en el step GLOBAL `step + offset` del schedule
    original. Se usa para que la segunda fase del warmup de estabilidad
    (ver `train_projected`) continúe el mismo lr/diag_shift schedule de la
    primera fase en vez de reiniciarlo.
    '''
    return lambda step: schedule(step + offset)


class _PlateauController:
    '''Decide, mirando SOLO lo medido en el ULTIMO bloque de pasos, si hay
    que tocar el lr o el `diag_shift` de SR en la fase 2 (ver el docstring de
    `plateau_patience` en `train_projected`). Es estado Python puro: vive
    FUERA del jit, y el bucle de fase 2 traduce sus decisiones a
    `optax.constant_schedule` para el driver del bloque siguiente.

    Tres ideas, las tres por la misma razon -- distinguir CONVERGENCIA de
    RUIDO y de DESASTRE:

    1. La referencia contra la que se mide "ha mejorado?" es el mejor valor
       visto DESDE EL ULTIMO CAMBIO DE REGIMEN (bajada de lr o ajuste de
       `diag_shift`), no el mejor GLOBAL de la corrida. Con el mejor global
       (lo que guardan `BestManifoldFidelity`/`BestIterKeeper`, monotono por
       construccion) basta UN pico de ruido de Monte Carlo que la trayectoria
       no vuelva a tocar para congelar la referencia: a partir de ahi el lr
       baja cada `patience` pasos aunque el entrenamiento este mejorando de
       verdad desde el ultimo cambio. Los keepers siguen decidiendo QUE
       ESTADO se guarda a disco; esta clase solo decide si el regimen actual
       sigue dando de si.
    2. La INESTABILIDAD se detecta con la VARIANZA de la energia, no con la
       metrica de plateau. Es la leccion de la corrida c2v/Jz=0.6: entre los
       steps 530-570 la energia se fue de -5.72 a -5.36 con la varianza en
       0.86 (300x la mediana de su bloque, ~0.003), pero la fidelidad solo se
       mide cada `fidelity_every` pasos -- el bloque dejo las muestras
       [0.2730, 0.2610, 0.2295], una caida del 16%, por debajo del 25% de
       `collapse_frac`. El estallido cayo ENTRE dos medidas y el controlador,
       ciego, lo leyo como plateau y bajo la amortiguacion justo despues de
       un colapso: F se hundio de 0.23 a 0.07 y no volvio. La varianza viene
       de `metrics_history`, que se llena cada `eval_every` pasos (10 frente
       a 100), asi que ve lo que la fidelidad no ve, y un pico relativo a la
       MEDIANA DEL PROPIO BLOQUE separa limpiamente los casos (245x en el
       bloque que colapso; 1.4x y 2.2x en los que no).
    3. Un bloque inestable, o que EMPEORA de forma clara, no es un plateau y
       pide lo contrario: mas amortiguacion, no menos. Se mira antes que el
       plateau y no cuenta como "pasos sin mejorar".

    :param lr: learning rate de arranque (el que `hparams` daria a la fase 2).
    :param diag: `diag_shift` de arranque; se recorta a `diag_range` si
        `diag_adapt=True` (el aviso queda en `self.notes`).
    :param patience: pasos sin mejorar que disparan un ajuste. Es tambien el
        tamano del bloque, asi que `observe` se llama una vez por bloque.
    :param decay: factor que multiplica al lr en cada bajada.
    :param min_lr: suelo del lr.
    :param use_fidelity: True -> la metrica es `F_manifold` y mejora al SUBIR;
        False -> es la energia y mejora al BAJAR.
    :param diag_adapt: activa el control por evidencia del `diag_shift`.
    :param diag_lower: permite ADEMAS bajar el `diag_shift` en un plateau,
        alternando con el lr. False por defecto: es la mitad que estropeo la
        corrida c2v/Jz=0.6 (ver punto 2), asi que hay que pedirla a mano.
    :param diag_factor: factor multiplicativo del `diag_shift` (>1).
    :param diag_range: `(suelo, techo)` del `diag_shift` adaptativo.
    :param collapse_frac: caida relativa de la METRICA dentro de un bloque que
        cuenta como colapso/regresion (solo se aplica con `use_fidelity`: para
        la energia, |E| es enorme frente a las diferencias que importan y una
        fraccion de |E| no discrimina nada).
    :param var_spike: cuantas veces la MEDIANA de la varianza del bloque tiene
        que alcanzar su maximo para llamarlo colapso.
    :param metric_name: solo para los mensajes de log.
    '''

    def __init__(self, lr, diag, patience, decay, min_lr, use_fidelity,
                 diag_adapt=False, diag_lower=False, diag_factor=2.0,
                 diag_range=(1e-4, 1e-1), collapse_frac=0.25, var_spike=10.0,
                 metric_name="fidelidad"):
        self.lr = float(lr)
        self.diag = float(diag)
        self.patience = int(patience)
        self.decay = float(decay)
        self.min_lr = float(min_lr)
        self.use_fidelity = bool(use_fidelity)
        self.diag_adapt = bool(diag_adapt)
        self.diag_lower = bool(diag_lower) and self.diag_adapt
        self.diag_factor = float(diag_factor)
        self.diag_min, self.diag_max = float(diag_range[0]), float(diag_range[1])
        self.collapse_frac = float(collapse_frac)
        self.var_spike = float(var_spike)
        self.metric_name = metric_name

        self.worst = -float("inf") if self.use_fidelity else float("inf")
        self.window_best = self.worst
        self.steps_since_improve = 0
        self.next_knob = "diag" if self.diag_lower else "lr"
        self.notes = []
        if self.diag_adapt:
            clipped = min(max(self.diag, self.diag_min), self.diag_max)
            if clipped != self.diag:
                self.notes.append(
                    f"diag_shift de arranque {self.diag:.2e} fuera de "
                    f"diag_range=({self.diag_min:.1e}, {self.diag_max:.1e}) "
                    f"-> se recorta a {clipped:.2e}"
                )
                self.diag = clipped

    def _instability(self, energies, variances):
        '''Mira la traza DENSA de energia/varianza del bloque (una medida
        cada `eval_every` pasos, frente a cada `fidelity_every` de la
        metrica de plateau) y devuelve el motivo del colapso, o None.'''
        e = [float(x) for x in (energies or [])]
        v = [float(x) for x in (variances or [])]
        if any(not np.isfinite(x) for x in e + v):
            return "energia/varianza no finita en el bloque"
        if len(v) >= 4:
            med = float(np.median(v))
            top = max(v)
            if med > 0 and top > self.var_spike * med:
                return (f"la varianza pico a {top:.3g}, {top / med:.0f}x la mediana "
                        f"del bloque ({med:.3g})")
        return None

    def _summarise(self, window, energies=None, variances=None):
        '''(mejor del bloque, motivo de colapso) a partir de lo medido en el
        bloque. `None` como mejor significa que el bloque no dejo ni una
        medida utilizable de la metrica de plateau.'''
        reason = self._instability(energies, variances) if self.diag_adapt else None
        finite = [float(x) for x in window if np.isfinite(x)]
        if not finite:
            if window and reason is None:
                reason = f"todas las medidas de {self.metric_name} del bloque son no finitas"
            return None, reason
        best = max(finite) if self.use_fidelity else min(finite)
        if reason is None and self.diag_adapt and self.use_fidelity and len(finite) >= 2:
            if finite[-1] < best - self.collapse_frac * abs(best):
                reason = (f"el bloque acabo en {finite[-1]:.4f}, mas de un "
                          f"{self.collapse_frac:.0%} por debajo de su propio mejor "
                          f"({best:.4f})")
        return best, reason

    def observe(self, window, n_steps, global_best=None, energies=None, variances=None):
        '''Contabiliza un bloque de `n_steps` pasos: `window` son las medidas
        de la metrica de plateau (fidelidad, o energia sin ED) en orden
        temporal, y `energies`/`variances` la traza densa de `metrics_history`
        del mismo bloque, que es la que detecta las inestabilidades. Ajusta
        `self.lr`/`self.diag` si toca y devuelve las lineas de log.'''
        out = []
        chunk_best, collapse_reason = self._summarise(window, energies, variances)

        regressed = False
        if chunk_best is None:
            out.append(f"  [plateau] bloque de {n_steps} pasos sin ninguna medida "
                       f"utilizable de {self.metric_name}: no se cuenta como plateau "
                       f"(sube `plateau_patience` por encima de `fidelity_every`/"
                       f"`eval_every`)")
        else:
            ref = self.window_best
            ref_txt = "-" if ref == self.worst else f"{ref:.4f}"
            if self.use_fidelity:
                improved = chunk_best > ref + 1e-6
                regressed = (ref != self.worst
                             and chunk_best < ref - self.collapse_frac * abs(ref))
            else:
                improved = chunk_best < ref - 1e-8
            if improved:
                self.window_best = chunk_best
                self.steps_since_improve = 0
            else:
                self.steps_since_improve += n_steps
            gb_txt = "-" if global_best is None else f"{float(global_best):.4f}"
            estado = "MEJORA" if improved else ("EMPEORA" if regressed else "plano")
            out.append(f"  [plateau] bloque de {n_steps} pasos: mejor del bloque="
                       f"{chunk_best:.4f}  referencia={ref_txt}  mejor global={gb_txt}"
                       f"  {estado}{'  [COLAPSO]' if collapse_reason else ''}")
            if collapse_reason:
                out.append(f"  [plateau] inestabilidad: {collapse_reason}")

        changed = None
        if self.diag_adapt and (collapse_reason or regressed):
            motivo = "colapso" if collapse_reason else "regresion de la metrica"
            if self.diag < self.diag_max:
                new_diag = min(self.diag * self.diag_factor, self.diag_max)
                out.append(f"  [plateau] {motivo} -> mas amortiguacion: "
                           f"diag_shift {self.diag:.2e} -> {new_diag:.2e}")
                self.diag = new_diag
                changed = "diag"
            else:
                new_lr = max(self.lr * self.decay, self.min_lr)
                out.append(f"  [plateau] {motivo} con diag_shift ya en el techo "
                           f"({self.diag_max:.1e}) -> lr {self.lr:.2e} -> {new_lr:.2e}")
                self.lr = new_lr
                changed = "lr"
        elif chunk_best is not None and self.steps_since_improve >= self.patience:
            if self.diag_lower and self.next_knob == "diag" and self.diag > self.diag_min:
                new_diag = max(self.diag / self.diag_factor, self.diag_min)
                out.append(f"  [plateau] sin mejora de {self.metric_name} en "
                           f"{self.steps_since_improve} pasos (referencia="
                           f"{self.window_best:.4f}) -> menos amortiguacion: diag_shift "
                           f"{self.diag:.2e} -> {new_diag:.2e}")
                self.diag = new_diag
                self.next_knob = "lr"
                changed = "diag"
            else:
                new_lr = max(self.lr * self.decay, self.min_lr)
                out.append(f"  [plateau] sin mejora de {self.metric_name} en "
                           f"{self.steps_since_improve} pasos (referencia="
                           f"{self.window_best:.4f}) -> lr {self.lr:.2e} -> {new_lr:.2e}")
                self.lr = new_lr
                self.next_knob = "diag"
                changed = "lr"

        if changed is not None:
            self.window_best = self.worst
            self.steps_since_improve = 0
        return out


def train_projected(
    extent=(2, 2),
    jz_values=np.linspace(0.3, 0.4, 2),
    couplings=None,
    ansatz="rbm",
    num_layers=2,
    alpha=2.0,
    heads=4,
    dk=8,
    d_model=32,
    complex_trunk=None,
    n_iter=1500,
    k_eigenvals=6,
    exact_path=None,
    seed=0,
    out_prefix="projected_noinfid",
    use_mcstate=False,
    n_samples=_PRESET,
    checkpoint_every=100,
    group_chunk_size=3,
    stable_warmup_frac=0.2,
    eval_every=1,
    fidelity_every=1,
    clip_norm=_PRESET,
    remat=True,
    projection_group="space",
    resume_from=None,
    resume_at_step=0,
    lr_stages=None,
    diag_stages=None,
    sectors=None,
    unprojected_frac=0.0,
    optimizer="sgd",
    sampler_rules="local",
    hparams="og",
    plateau_patience=0,
    plateau_decay=0.5,
    plateau_min_lr=1e-4,
    plateau_diag_adapt=False,
    plateau_diag_lower=False,
    plateau_diag_factor=2.0,
    plateau_diag_range=(1e-4, 1e-1),
    plateau_collapse_frac=0.25,
    plateau_var_spike=10.0,
):
    '''Entrena, para cada Jz, un modelo por cada sector de simetría
    que hospeda el manifold degenerado del groundstate exacto (en vez de
    asumir a ciegas el irrep trivial), proyectando con la ec. 38 del paper.

    :param hparams: preset de hiperparametros de optimizacion.

        - "og" (por defecto): los de `legacy/Retry_OG/training.py`, la corrida
          de referencia que SI baja la energia con este mismo DeepRBM y este
          mismo H (ver `OG_LR_SCHEDULE_KWARGS` arriba, con los numeros
          medidos). `lr` = warmup exponencial continuo 0.01 -> 0.05,
          `diag_shift` = 1e-2 constante, sin `clip_by_global_norm`,
          `n_samples` = 2048.
        - "og_long": identico a "og" salvo que el lr decae 4 veces mas
          despacio (`transition_steps` 100 -> 400), de forma que
          `og_long(4*n) ~= og(n)` (con un 2.4% de desfase por el warmup, ver
          `OG_LONG_LR_SCHEDULE_KWARGS`). Para corridas de 4000 pasos o mas,
          donde "og" deja el lr 65 veces por debajo del pico y estrangula el
          entrenamiento antes de que sature -- medido en la corrida c3 del
          punto isotropico, ver `OG_LONG_LR_SCHEDULE_KWARGS`.
        - "og_hot": identico a "og_long" salvo que el pico del lr se dobla,
          0.05 -> 0.10 (ver `OG_HOT_LR_SCHEDULE_KWARGS`). SIN VALIDAR todavia:
          es la version mas agresiva a probar si con "og_long" la fidelidad
          sigue subiendo sin saturar al final de la corrida. Riesgo: un pico
          mas alto puede reventar el primer paso (la misma firma que goes
          delato "staged": energia SUBIENDO en los primeros steps).
        - "staged": los schedules escalonados que tenia este fichero, pensados
          para el ansatz PROYECTADO (`diag_shift` arranca en 1e-1 porque con
          `SymmExpSum` el QGT es casi singular) mas `clip_norm=5.0`. Se
          conservan para reproducir corridas anteriores.

        El preset solo fija los DEFAULTS: `lr_stages`, `diag_stages`,
        `clip_norm` y `n_samples` explicitos siguen ganando, asi que se puede
        partir de "og" y cambiar una sola cosa.

    :param plateau_patience: si es > 0, sustituye el schedule de lr de la
        FASE 2 (refinamiento proyectado) por uno controlado por PLATEAU de
        fidelidad: el lr se queda CONSTANTE mientras `F_manifold` siga
        mejorando, y solo baja (multiplicado por `plateau_decay`) si pasan
        `plateau_patience` pasos sin una mejora. 0 (por defecto) desactiva
        esto y deja el schedule de `hparams` tal cual, step a step, sin mirar
        la fidelidad -- el comportamiento de siempre.

        QUE CUENTA COMO "MEJORA": el mejor `F_manifold` visto DESDE EL ULTIMO
        CAMBIO DE REGIMEN -- la ultima bajada de lr, o el ultimo ajuste de
        `diag_shift` si `plateau_diag_adapt=True` --, NO el mejor GLOBAL de la
        corrida. La version anterior comparaba contra el mejor global
        (`fidelity_keeper.best_fidelity`, monotono por construccion) y eso se
        rompe justo en el caso que importa: si un pico de RUIDO de Monte Carlo
        fija un `best_fidelity` que la trayectoria no vuelve a tocar, la
        referencia queda congelada para siempre y el lr baja cada
        `plateau_patience` pasos aunque el entrenamiento este mejorando de
        verdad desde el ultimo cambio -- se estrangula el lr por un unico
        outlier, que es justo lo contrario de lo que este control busca. Con
        la referencia por VENTANA, cada regimen de lr/diag_shift se juzga por
        lo que consigue EL; el mejor global sigue mandando solo donde siempre
        mando: en que estado se guarda a disco.

        Motivo: con "og_hot" en la corrida c3 del punto isotropico, el
        cociente (mejora de fidelidad)/(lr) SUBIA de forma monotona a lo
        largo de la corrida (0.94 -> 2.88 entre step 1100 y 3700) -- la firma
        de un lr que decae mas rapido de lo que el optimizador necesita, no
        de uno que haya saturado. El schedule de "og"/"og_long"/"og_hot" baja
        el lr por CALENDARIO (numero de step), sin mirar si hace falta.
        `plateau_patience` lo baja por EVIDENCIA en su lugar.

        Requiere trocear `driver.run()` en bloques de `plateau_patience`
        pasos, cada uno con su propio driver (mismo patron que ya usa este
        fichero entre fase 0/1/2: nueva instancia de `nk.driver.VMC_SR`, pesos
        trasplantados). Es necesario y no cosmetico: los schedules de optax
        corren DENTRO del jit de cada paso de NetKet, así que no pueden leer
        estado Python mutable (como "cuantos pasos sin mejorar") a mitad de un
        `driver.run()` -- quedaria congelado al valor de la primera traza. Un
        driver nuevo por bloque fuerza una traza nueva, que si ve el valor de
        lr actualizado.

        Coste: un driver nuevo por bloque es una recompilacion XLA, asi que
        `plateau_patience` pequeno (p.ej. 20) es mucho mas lento por paso que
        uno grande (p.ej. 300). No poner por debajo de unos ~100-200 pasos
        salvo que el coste de recompilar sea aceptable.

        Sin diagonalizacion exacta (N > FULLSUM_MAX_SPINS) no hay
        `F_manifold` que mirar: se usa la energia (`keeper.best_energy`,
        mejora si baja) como sustituto, y se avisa por print de cual metrica
        se esta usando.

        Por defecto no toca el `diag_shift` (sigue el schedule de `hparams`,
        por step); `plateau_diag_adapt=True` lo pone tambien bajo control por
        evidencia. Las fases 0/1 (sin proyectar / warmup polinomico) no se
        tocan nunca, solo la fase 2. El valor de arranque (de lr y de
        `diag_shift`) es el que `hparams` le habria dado a la fase 2 en su
        primer step; a partir de ahi manda la fidelidad, no el calendario.

        AVISO: el contador de "pasos sin mejorar" y la referencia de ventana
        arrancan vacios al empezar la fase 2 (o al reanudar con
        `resume_from`), no recuerdan cuanto lleva estancada la fidelidad de
        una corrida anterior -- `BestManifoldFidelity` solo guarda el mejor
        valor visto, no CUANDO se vio. La primera ventana de la fase 2 mejora
        siempre (parte de referencia vacia), asi que el primer ajuste posible
        cae, como pronto, tras 2*`plateau_patience` pasos.
    :param plateau_decay: factor multiplicativo aplicado al lr cuando se
        cumple `plateau_patience` sin mejora. 0.5 por defecto (baja a la
        mitad). Ignorado si `plateau_patience=0`.
    :param plateau_min_lr: suelo del lr bajo `plateau_patience`, para que no
        decaiga indefinidamente hacia 0. 1e-4 por defecto. Ignorado si
        `plateau_patience=0`.
    :param plateau_diag_adapt: si es True (y `plateau_patience > 0`), el
        `diag_shift` de SR deja de seguir el calendario en la fase 2 y sube
        por evidencia cuando el entrenamiento se desestabiliza:

        - COLAPSO o REGRESION -> MAS amortiguacion. Colapso = la varianza de
          la energia pica a `plateau_var_spike` veces la mediana de su propio
          bloque, o aparece un no-finito, o (solo con fidelidad) el bloque
          acaba `plateau_collapse_frac` por debajo de su propio mejor.
          Regresion = el mejor del bloque se queda `plateau_collapse_frac` por
          debajo de la referencia. En los dos casos el paso natural S^-1 F se
          esta yendo por las direcciones mal condicionadas del QGT: se
          MULTIPLICA `diag_shift` por `plateau_diag_factor` (hasta el techo de
          `plateau_diag_range`) y el bloque NO cuenta como plateau, porque no
          es convergencia. Con el `diag_shift` ya en el techo, lo unico que
          queda para cortarlo es acortar el paso, asi que ahi si baja el lr.
        - PLATEAU -> baja el lr, como sin `plateau_diag_adapt`. Bajar tambien
          el `diag_shift` en un plateau existe, pero hay que pedirlo aparte
          con `plateau_diag_lower=True`; ver ahi por que no viene de serie.

        La deteccion de inestabilidad usa la traza DENSA de
        `metrics_history` (una medida cada `eval_every` pasos), no la metrica
        de plateau (cada `fidelity_every`). Es lo que fallo en la corrida
        c2v/Jz=0.6: el estallido de los steps 530-570 (E de -5.72 a -5.36,
        varianza 0.86 frente a ~0.003) cayo entre dos medidas de fidelidad y
        el bloque parecio un simple plateau.

        Cada ajuste, de lr o de `diag_shift`, reinicia la referencia de
        ventana y el contador de pasos sin mejora. No aumenta el coste: el
        bucle ya construia un driver nuevo por bloque, y el `diag_shift`
        entra en el como un `optax.constant_schedule` mas.
    :param plateau_diag_lower: permite ademas BAJAR el `diag_shift` en un
        plateau, alternando con el lr (primer plateau divide `diag_shift`,
        el siguiente baja el lr, y asi). False por defecto, y el default no es
        conservadurismo gratuito: en la corrida c2v/Jz=0.6 esta rama disparo
        justo despues de un colapso que el detector no vio, bajo `diag_shift`
        de 1e-2 a 5e-3 y hundio la fidelidad de 0.23 a 0.07 sin recuperacion.
        La idea de fondo sigue en pie -- un plateau puede venir de SR
        SOBRE-amortiguado, donde un `diag_shift` grande convierte S^-1 F en
        SGD y mata las direcciones de curvatura pequena --, pero solo tiene
        sentido probarla sobre una trayectoria ESTABLE, asi que se pide a
        mano. Requiere `plateau_diag_adapt=True`.
    :param plateau_diag_factor: factor multiplicativo del `diag_shift` bajo
        `plateau_diag_adapt`: se multiplica por el al detectar un colapso o
        una regresion y se divide por el en un plateau (esto ultimo solo con
        `plateau_diag_lower=True`). 2.0 por defecto.
    :param plateau_diag_range: `(suelo, techo)` del `diag_shift` adaptativo,
        (1e-4, 1e-1) por defecto. El techo es el `diag_shift` con el que
        arranca la fase proyectada del preset no-"og" (con `SymmExpSum` el QGT
        es casi singular y hace falta esa amortiguacion al principio); el
        suelo, el minimo que aguantaron las corridas de prueba antes de que SR
        colapsara. Ignorado si `plateau_diag_adapt=False`.
    :param plateau_collapse_frac: caida RELATIVA de la fidelidad que cuenta
        como colapso (acabar el bloque un 25%, por defecto, por debajo del
        mejor de ese mismo bloque) o como regresion (quedarse ese 25% por
        debajo de la referencia de ventana). Solo se aplica con fidelidad: en
        la energia, |E| es enorme frente a las diferencias que importan -- el
        estallido de la corrida c2v/Jz=0.6 fue de -5.72 a -5.36, un 6% -- y
        una fraccion de |E| no discrimina nada; ahi manda `plateau_var_spike`.
        Ignorado si `plateau_diag_adapt=False`.
    :param plateau_var_spike: cuantas veces la MEDIANA de la varianza del
        bloque tiene que alcanzar su maximo para llamarlo colapso. 10.0 por
        defecto, con margen de sobra: en la corrida c2v/Jz=0.6 el bloque que
        colapso dio 245x y los que no, 1.4x y 2.2x. Relativo a la mediana del
        PROPIO bloque, asi que no depende de la escala del Hamiltoniano ni de
        `n_samples`. Ignorado si `plateau_diag_adapt=False`.

    :param optimizer: "sgd" (por defecto) o "adam", el optax que recibe el
        driver DESPUES del precondicionamiento de SR.

        **El default cambio de "adam" a "sgd", y no es cosmetico.** `VMC_SR` ya
        devuelve el gradiente natural S^-1 F; encadenarle Adam encima le aplica
        su normalizacion por RMS y destruye la escala que SR acaba de
        construir. El docstring de `Infidelity_SR` de NetKet lo dice
        literalmente: el optimizador "should be an instance of optax.sgd. Other
        optimizers ... will not make mathematical sense". `run_vmc.py` ya tenia
        "sgd" por defecto por esto mismo, y `docs/DEVLOG.md` (2026-08-06)
        registra que se midio; `try_Norman.py` se habia quedado con Adam.

        "adam" sigue disponible para reproducir corridas antiguas.

    :param sampler_rules: "local" (por defecto) o "local+exchange".

        **El default cambio de "local+exchange" a "local".** `ExchangeRule`
        intercambia espines entre sitios vecinos, lo que CONSERVA la
        magnetizacion -- pero la magnetizacion no es una carga conservada del
        Kitaev, asi que ese 10% de propuestas estaba confinado a un subespacio
        de magnetizacion fija sin motivo fisico. La corrida de referencia que
        alcanza la energia exacta a 3x3 usa `LocalRule` sola.

        Solo afecta a `use_mcstate=True`; con `FullSumState` no hay muestreo.

    :param ansatz: que red variacional usar.

        - "rbm" (por defecto): `DeepRBM` de model_RBM.py. `num_layers` y
          `alpha` son sus hiperparametros; `heads`/`dk`/`d_model` se ignoran.
        - "transformer": `Transformer` de transformer.py, atencion estandar
          A = softmax(QK^T/sqrt(dk)), que depende de la configuracion.
        - "factored": `FactoredSelfAttention`, atencion FACTORIZADA -- el mapa
          A = softmax(P) es un parametro aprendido (heads, N, N), independiente
          de la configuracion. Captura que pares de sitios interactuan (la
          geometria de la red) pero no modula esa geometria segun el estado.

        Los dos transformers comparten toda la arquitectura salvo el mapa de
        atencion, para que cualquier diferencia de resultados sea atribuible a
        ese unico cambio. Sus hiperparametros son `num_layers`, `heads`, `dk` y
        `d_model`; `alpha` se ignora.

    :param complex_trunk: solo para "transformer"/"factored". None (por
        defecto) = TRONCO COMPLEJO, lo MISMO que el RBM: `param_dtype=
        jnp.complex128` en toda la red y la cabeza emitiendo UN canal ya
        complejo, con amplitud y fase entrelazadas en los mismos parametros,
        como en `DeepRBM`. False vuelve al tronco REAL de antes: log(psi) se
        arma al final combinando dos canales reales, o sea log|psi| y theta son
        dos readouts independientes, cada uno una suma sobre sitios. True hace
        exactamente lo mismo que None en los transformers; se distinguen solo
        en que True con ansatz="rbm" imprime el aviso de que ahi no cambia nada.

        El mapa de atencion se mantiene REAL en los dos casos, a proposito: el
        softmax sobre complejos corre pero devuelve pesos con parte real
        negativa (medido), que no son una distribucion. Con "dot" las
        puntuaciones pasan a `Re(Q.K*)`; con "factored" los `att_logits` se
        declaran en float64. Ver `common/models/transformer.py`.

        Para que sirve: es la hipotesis de
        `docs/codigo/simetrias_de_espin_en_el_codigo.md` §7.3 -- que el tronco
        real es lo que limita al transformer sobre el Hamiltoniano ROTADO de
        c3/c2v, donde el RBM (complejo nativo) si llega (F=0.85 medido). Con
        esto se mide en vez de conjeturar.

        AVISO: cambia el pytree (dtype, y la cabeza pasa de 2 canales a 1), asi
        que un checkpoint de tronco real NO carga en uno complejo ni al reves.

        AVISO: solo el RBM tiene `stable_cosh`, asi que con un transformer el
        pre-entreno polinomico de `stable_warmup_frac` no aplica y se desactiva
        con un aviso impreso, en vez de ignorarse en silencio.

        Con el default los tres ansatze llevan ya parametros complex128, pero
        el pytree sigue siendo distinto (el RBM tiene `layer_i`; el transformer
        `embed`/`pos_embed`/`block_i`/`head`), asi que un checkpoint de uno NO
        es trasplantable al otro: `resume_from` solo vale entre corridas del
        mismo ansatz, y dentro de los transformers solo entre corridas con el
        mismo `complex_trunk`.

    :param unprojected_frac: fraccion inicial de `n_iter` que se entrena con el
        ansatz SIN proyectar, antes de trasplantar los pesos al proyectado.

        Motivo: cada paso del ansatz proyectado evalua la red en |G|
        configuraciones por muestra (27 con "c3", 36 con "c2v"), asi que
        cuesta ~|G| veces mas que un paso sin proyectar. Bajar la energia es
        lo que mas pasos consume, y no necesita la proyeccion para nada: se
        puede hacer barato primero y proyectar despues, ya cerca del
        fundamental.

        El reparto del presupuesto queda:

            unprojected_steps = n_iter * unprojected_frac
            projected_budget  = n_iter - unprojected_steps
            warmup_steps      = projected_budget * stable_warmup_frac

        de forma que cada fraccion significa lo que dice y `stable_warmup_frac`
        sigue midiendo sobre la parte proyectada, que es donde el paper reporta
        el estancamiento numerico que motiva el pre-entreno polinomico.

        AVISO historico: `Supervised_Infid_min/run_vmc.py --unprojected-phase1` documenta
        que esta estrategia fallaba, porque un estado de fase 1 que salga
        simetrico bajo el grupo es ANIQUILADO por el proyector de cualquier
        irrep no trivial (sum_g chi(g) = 0 sobre un estado invariante). Aquello
        fue con proyeccion sobre `translation`. Para poder ver ese modo de
        fallo si reaparece, al empezar cada fase con pesos trasplantados se
        imprime la energia ANTES de entrenar: si la proyeccion hubiera
        aniquilado el estado, esa energia sale disparatada o NaN en vez de
        parecida a la que traia.

        La fase sin proyectar usa la MISMA activacion que la fase que le sigue
        (`stable_cosh=True` si hay warmup polinomico despues, `False` si no),
        para que el trasplante cambie UNA sola cosa -- la proyeccion -- en vez
        de dos. Si no, se entrenaria con log_cosh real para retroceder despues
        al polinomio, y el diagnostico de energia dejaria de ser legible: no se
        sabria si el salto lo causo proyectar o cambiar la no linealidad.

    :param sectors: lista de indices de irrep a entrenar, en vez de TODOS los
        que hospedan el manifold. Sin esto, un punto con manifold grande puede
        lanzar muchos entrenamientos sin avisar: a 3x3 con Jz>=0.48 el manifold
        es 18-dimensional y `sectors_hosting_manifold` puede devolver varios
        irreps, uno por entrenamiento completo. Pedir un sector que NO hospeda
        el manifold es un error, no un no-op silencioso -- entrenarlo
        perseguiria un subespacio de norma de proyeccion casi nula.

        Equivalente al `--sectors` de `Supervised_Infid_min/run_vmc.py`.

    :param resume_from: ruta a un `.pkl` de parametros (o un dict de
        parametros ya cargado) con el que arrancar en vez de una
        inicializacion aleatoria. Con un dict `{k_sector: ruta}` se puede dar
        uno distinto por sector. El pytree es el mismo con `stable_cosh`
        True o False, asi que un checkpoint de la fase polinomica se
        trasplanta sin mas a la fase de `log_cosh` real.

        OJO con CUAL checkpoint se carga. Cuando hay diagonalizacion exacta
        (N <= FULLSUM_MAX_SPINS) `{out_prefix}_vstate_{tag}.pkl` guarda el
        mejor por FIDELIDAD, no por energia, y las dos divergen: si la
        fidelidad hace pico pronto y se estanca, ese fichero se queda
        congelado en un estado energeticamente peor que el actual. Desde
        ahora se guarda tambien `{out_prefix}_vstate_{tag}_bestE.pkl` con el
        mejor por energia; elige segun lo que quieras continuar.

        Y `{out_prefix}_vstate_{tag}_final.pkl` con el estado en que ACABO la
        corrida. Los otros dos son selecciones, y las dos pueden quedar peor
        en energia exacta que la media de la cola convergida: medido en
        c3/Jz=1/3, la traza cruza el umbral de la cota (E=-4.7192 contra
        E0+gap=-4.6915) pero `bestF` y `bestE` se quedan los dos en -4.671.
        Para REANUDAR sigue siendo razonable partir de un "mejor"; para
        MEDIR el resultado de la corrida, este es el fichero.

    :param resume_at_step: cuantos pasos se dieron ya. Desplaza tanto la
        numeracion de steps en el CSV de metricas como la EVALUACION de los
        schedules, y los boundaries se calculan sobre `resume_at_step +
        n_iter` (el total). Sin esto, reanudar reinicia el schedule en su
        primera etapa -- que con los valores por defecto significa volver a
        `diag_shift=1e-1` y `lr=3e-2`, la etapa mas amortiguada, justo la que
        se queria dejar atras.

    :param lr_stages: `(valores, fracciones)` para sustituir el schedule de
        learning rate, p.ej. `([1e-2, 3e-3, 1e-3], [0.4, 0.75])`: len(valores)
        == len(fracciones) + 1, y las fracciones son de `resume_at_step +
        n_iter`. None usa el de siempre.
    :param diag_stages: idem para el `diag_shift` de SR.

    :param projection_group: "space" (por defecto), "none", "translation",
        "c3" o "c2v".

        "none" NO proyecta: entrena el ansatz desnudo minimizando energia, sin
        restringir a ningun sector. Sigue diagonalizando para reportar
        `E_exacta`, el gap y la fidelidad de manifold, asi que es la linea base
        con la que comparar cualquier proyeccion. El sector se registra como
        -1, igual que el convenio de `Supervised_Infid_min/run_vmc.py --projection none`.

        Es tambien el modo a usar cuando la proyeccion resulta contraproducente.
        Medido a 3x3, Jz=0.1: tanto la corrida sin proyectar como la proyectada
        a "c2v" colapsaron a un estado producto (E = -N_x * Jx exacto, varianza
        muestral 0 y el muestreador congelado), asi que proyectar NO es una
        defensa contra ese colapso -- son problemas ortogonales.

        "c2v" es el grupo del caso Jx == Jy, o sea de TODO el barrido de
        produccion (`jx = jy = (1-jz)/2`), no de un punto suelto como la C3.
        Combina el espejo de red que fija enlaces z e intercambia x<->y (mas
        una rotacion de espin de pi alrededor de (1,1,0)) con el C2 de red
        que preserva colores y no necesita rotacion de espin ninguna. A 3x3
        son 36 elementos, frente a los 18 de "space" y los 27 de "c3".

        OJO: mas grande no es mas util automaticamente. Medido, sus 9 irreps
        tienen dims [1,1,1,1,2,2,2,2,4] -- solo CUATRO unidimensionales,
        frente a nueve de once en "c3". Un irrep de dimension d>1 proyecta
        sobre un subespacio isotipico d-dimensional, no sobre un estado; para
        minimizar energia sigue siendo valido, pero no nombra un estado unico.

        Requiere Jx == Jy (y h == 0, que es siempre el caso aqui):
        `check_c2xy_applicable` lo comprueba antes de entrenar. Como "c3",
        entrena en la base de espin rotada -- pero con el eje (1,1,0), no
        (1,1,1) -- porque solo ahi el operador combinado es monomial.

        "c3" proyecta sobre traslaciones combinadas con la rotacion de 120
        grados del punto ISOTROPICO: 27 elementos a 3x3, frente a los 18 de
        "space" (que son traslaciones x un C2). Solo es una simetria si
        Jx == Jy == Jz, y `train_projected` lo comprueba antes de usarlo.

        Esa C3 es una rotacion del panal combinada SIMULTANEAMENTE con una
        rotacion de 120 grados en el espacio de espin alrededor de (1,1,1)
        (sigma^x -> sigma^y -> sigma^z). Ninguna de las dos mitades por
        separado conmuta con H (ver tests/test_isotropic_symmetry.py). En la
        base computacional el operador combinado es DENSO -- conecta cada
        configuracion con las 2^N -- asi que con "c3" se entrena en la base de
        espin rotada, donde pasa a ser monomial (permutacion x fase, UNA
        conexion por configuracion) y `MonomialSymmExpSum` puede evaluarlo en
        O(|G|) por muestra.

        AVISO: en esa base el Hamiltoniano es `rotated_kitaev_hamiltonian`,
        unitariamente equivalente (mismo espectro, misma fisica) pero con cada
        enlace expandido de 1 a 9 terminos Pauli. Los `Wp` de plaqueta que se
        reportan al final NO estan rotados, asi que con "c3" no son
        comparables con los de las otras corridas; ver `rotate_state_to_frame`
        para llevar estados de una base a la otra.

        Para "space" y "translation" se usa
        `src.physics.symmetries.get_projection_group` -- el mismo selector que
        usa `Supervised_Infid_min/run_vmc.py --group`, para que legacy y refactorizado no
        proyecten sobre grupos distintos sin querer. A 3x3: translation tiene
        |G|=9 con los 9 irreps 1D (sectores de momento genuinos); space tiene
        |G|=18 con dims [1,1,2,2,2,2].

        La objeción registrada contra "space" (irreps 2D -> el proyector de
        caracteres cae sobre un subespacio isotípico 2D, no sobre un estado)
        es una objeción al TARGET de la minimización de infidelidad, que sí
        necesita un estado único bien definido. Aquí NO hay fase de
        infidelidad: se minimiza energía, y restringir el ansatz a un
        subespacio isotípico 2D es perfectamente válido variacionalmente. Por
        eso el default es "space", que es el grupo mayor y por tanto la
        restricción de simetría más fuerte. `identify_irreps`
        (src/physics/exact_diag.py) ya incluye el factor d_mu/|G| del
        proyector, así que los pesos por sector salen bien también para los
        irreps 2D.
    :param couplings: lista de triplas (Jx, Jy, Jz) explícitas. Si se da,
        IGNORA `jz_values` y su parametrización `jx = jy = (1-jz)/2`. Hace
        falta para el punto isótropo Jx=Jy=Jz=1: con `jz_values=[1.0]` esa
        parametrización da Jx=Jy=0, Jz=1, que es el límite de dímeros
        desacoplados (exactamente resoluble), justo lo contrario del punto
        isótropo. Nota: H(1,1,1) = 3*H(1/3,1/3,1/3), o sea mismos
        autovectores (misma fidelidad/overlap) y energías x3.
    :param exact_path: ruta a un `.npz` con la diagonalizacion exacta ya
        hecha (el `energies_eigenvecs_dict_k40.npz` que usa
        `legacy/Retry_OG/training.py`, tambien en `data/raw/`). Si se da, se
        carga de ahi en vez de llamar a `lanczos_ed`, que es el coste fijo que
        se paga en CADA corrida y en cada punto de `couplings` -- incluso al
        reanudar desde un checkpoint.

        Ventaja secundaria y nada menor: el fichero trae k=40 autovalores,
        asi que resuelve de paso el problema de `k_eigenvals` demasiado bajo.
        Con el default `k_eigenvals=6` y un manifold 18-dimensional (Jz de 0.5
        a 0.9) Lanczos devuelve 6 de los 18 estados degenerados y la
        descomposicion en irreps se hace sobre una rebanada arbitraria del
        manifold -> `sectors_hosting_manifold` puede dejar fuera sectores que
        SI hospedan. Cargando el cache se usan los 40, y la deteccion sale
        bien.

        Con `projection_group` "c3" o "c2v" el Hamiltoniano esta ROTADO
        (`rotated_kitaev_hamiltonian`), y el fichero de arriba trae
        autovectores del H PLANO -- mismo espectro, distinta base, asi que
        NO sirve tal cual: se detecta (el chequeo `<v0|H|v0>` de
        `_load_exact_spectrum` no cuadra) y se diagonaliza en su lugar, con
        `ValueError` si de verdad se le fuerza a aceptarlo. Lo que SI sirve es
        un cache ya rotado a ese frame, generado con
        `common/analysis/rotate_exact_spectrum.py --group c3` (o `c2v`) a partir del
        `.npz` plano -- no rediagonaliza nada, solo rota los autovectores
        (`H_rotado = W^dagger H W` es una conjugacion unitaria: mismo
        autovector salvo `W^dagger`, misma energia). Pasale ESE fichero aqui.

        Tampoco aplica si los `couplings` no siguen `jx = jy = (1-jz)/2`, que
        es la parametrizacion con la que esta indexado el fichero (plano o
        rotado da igual, la relacion es sobre los acoplos fisicos).

        Se verifica que el espectro cargado corresponde de verdad a este H
        midiendo `<v0|H|v0>` contra la energia guardada; si no cuadra, es
        ValueError y no un resultado silenciosamente equivocado.

    :param eval_every: cada cuántos pasos se evalúan los callbacks caros de
        energía (`make_extract_metrics` y `BestIterKeeper`, que llaman a
        `expect(H)`). Con FullSumState cada `expect` recorre las 2^N
        configuraciones x |G| traslaciones, así que a 1 paso de cada uno
        estos callbacks cuestan del orden del propio paso de entrenamiento.
        Subirlo es lo que hace viable una corrida de 10k épocas.
    :param fidelity_every: ídem para `BestManifoldFidelity`, que además hace
        un `to_array()` completo por evaluación.
    :param clip_norm: cota de `clip_by_global_norm` sobre el gradiente
        natural, ANTES de Adam. Es el único knob sensible a la escala global
        de H (Adam es casi invariante a reescalar el gradiente, el clip no):
        si se multiplica H por un factor c, hay que multiplicar `clip_norm`
        por c para conservar el comportamiento ya ajustado.
    :param remat: rematerializar (gradient checkpointing) el módulo dentro
        del scan sobre el grupo, ver `SymmExpSumChunked` en model_RBM.py. Sin
        esto `group_chunk_size` NO acota la memoria del gradiente y
        FullSumState a 3x3 pide ~26 GB. Con remat el pico escala con
        `group_chunk_size`, a cambio de ~2x de cómputo en el backward. El
        gradiente sigue siendo exacto (verificado bit a bit).
    :param group_chunk_size: cuántos elementos del grupo de simetría evalúa
        `SymmExpSumChunked` a la vez (ver model_RBM.py). Acota el pico de
        memoria de FullSumState+simetría; no cambia el resultado (gradiente
        exacto), solo el pico de RAM/VRAM. Más pequeño = menos memoria, más
        pasos secuenciales (más lento).
    :param stable_warmup_frac: fracción inicial de `n_iter` que se entrena
        con `DeepRBM(stable_cosh=True)` (log(1+x^2/2) en vez de log_cosh,
        ec. 39 del paper) antes de trasplantar los parámetros a un modelo
        con el log_cosh real para refinar el resto de pasos. Mitiga el
        estancamiento numérico que el propio paper reporta (sec. 5.1) para
        RBMs entrenados con NetKet, causado por valores extremos de
        cosh/sinh en la fase final de entrenamiento. 0 desactiva el warmup
        (una sola fase, como antes).
    '''
    if projection_group not in _PROJECTION_GROUPS:
        raise ValueError(
            f"projection_group={projection_group!r} no reconocido; "
            f"admitidos: {_PROJECTION_GROUPS}"
        )
    if ansatz not in _ANSATZE:
        raise ValueError(f"ansatz={ansatz!r} no reconocido; admitidos: {_ANSATZE}")
    if complex_trunk and ansatz == "rbm":
        print("[aviso] `DeepRBM` ya es complejo nativo (param_dtype=complex128): "
              "complex_trunk=True no cambia nada con ansatz='rbm'")
    if complex_trunk is None:
        complex_trunk = True
    if optimizer not in ("sgd", "adam"):
        raise ValueError(f"optimizer={optimizer!r} no reconocido; admitidos: sgd, adam")
    if sampler_rules not in ("local", "local+exchange"):
        raise ValueError(
            f"sampler_rules={sampler_rules!r} no reconocido; "
            f"admitidos: local, local+exchange"
        )
    if hparams not in _HPARAMS:
        raise ValueError(f"hparams={hparams!r} no reconocido; admitidos: {_HPARAMS}")
    if plateau_patience < 0:
        raise ValueError(f"plateau_patience={plateau_patience!r} debe ser >= 0")
    if not (0.0 < plateau_decay < 1.0):
        raise ValueError(f"plateau_decay={plateau_decay!r} debe estar en (0, 1)")
    if plateau_diag_adapt and plateau_patience <= 0:
        print("[aviso] plateau_diag_adapt=True no hace nada con plateau_patience=0: "
              "el diag_shift adaptativo vive en el mismo bucle por bloques que el lr")
    if plateau_diag_lower and not plateau_diag_adapt:
        print("[aviso] plateau_diag_lower=True no hace nada sin plateau_diag_adapt=True")
    if plateau_var_spike <= 1.0:
        raise ValueError(
            f"plateau_var_spike={plateau_var_spike!r} debe ser > 1: es un pico de la "
            f"varianza RELATIVO a la mediana del propio bloque"
        )
    if plateau_diag_factor <= 1.0:
        raise ValueError(
            f"plateau_diag_factor={plateau_diag_factor!r} debe ser > 1: se MULTIPLICA "
            f"por el al colapsar (mas amortiguacion) y se DIVIDE por el en un plateau"
        )
    if len(tuple(plateau_diag_range)) != 2 or not (0.0 < plateau_diag_range[0] < plateau_diag_range[1]):
        raise ValueError(
            f"plateau_diag_range={plateau_diag_range!r} debe ser (suelo, techo) con 0 < suelo < techo"
        )
    if not (0.0 < plateau_collapse_frac < 1.0):
        raise ValueError(
            f"plateau_collapse_frac={plateau_collapse_frac!r} debe estar en (0, 1)"
        )

    if clip_norm is _PRESET:
        clip_norm = None if hparams in _OG_FAMILY else 5.0
    if n_samples is _PRESET:
        n_samples = OG_N_SAMPLES if hparams in _OG_FAMILY else 4096

    kitaev_graph, hi, Wp_list, Wp_total, N = declare_kitaev(extent=list(extent), pbc=True)
    direcciones = kitaev_graph.edge_colors
    bonds = kitaev_graph.edges()

    if ansatz != "rbm" and stable_warmup_frac > 0:
        print(f"[aviso] ansatz={ansatz!r} no tiene `stable_cosh`, asi que el "
              f"pre-entreno polinomico no aplica: stable_warmup_frac="
              f"{stable_warmup_frac} -> 0")
        stable_warmup_frac = 0.0

    project = (projection_group != "none")
    if not project and unprojected_frac:
        print(f"[aviso] projection_group='none' ya entrena sin proyectar: "
              f"unprojected_frac={unprojected_frac} -> 0 (no hay nada a lo que "
              f"trasplantar despues)")
        unprojected_frac = 0.0

    use_c3 = (projection_group == "c3")
    use_c2v = (projection_group == "c2v")
    monomial = use_c3 or use_c2v
    spin_axis = _GROUP_AXIS.get(projection_group)
    root_order = _GROUP_ROOT_ORDER.get(projection_group)
    if not project:
        group = character_table = element_powers = None
        print("Sin proyeccion: se entrena el ansatz desnudo (sector = -1)")
    elif use_c3:
        group, element_powers = c3_translation_group(kitaev_graph)
        _, character_table = c3_character_table(kitaev_graph, group)
    elif use_c2v:
        group, element_powers = c2v_translation_group(kitaev_graph)
        _, character_table = c2v_character_table(kitaev_graph, group)
    else:
        symmetries = get_kitaev_symmetries(kitaev_graph, hi)
        group, character_table = get_projection_group(symmetries, projection_group)
        element_powers = None
    if project:
        print(f"Proyectando sobre el grupo '{projection_group}': |G|={len(np.asarray(group))}, "
              f"{character_table.shape[0]} irreps, "
              f"dims={[int(round(d)) for d in np.real(character_table[:, 0])]}")

    if lr_stages is not None:
        _lr_desc = f"escalones {lr_stages}"
    elif hparams in _OG_FAMILY:
        _kw = _OG_FAMILY[hparams]
        _lr_desc = (f"warmup_exp({_kw['init_value']}->{_kw['peak_value']}, "
                    f"x{_kw['decay_rate']}/{_kw['transition_steps']})")
    else:
        _lr_desc = "escalones [3e-2, 1e-2, 3e-3, 1e-3]"
    if diag_stages is not None:
        _diag_desc = f"escalones {diag_stages}"
    elif hparams in _OG_FAMILY:
        _diag_desc = f"{OG_DIAG_SHIFT} (constante)"
    else:
        _diag_desc = "escalones [1e-1, 1e-2, 3e-3, 1e-3]"
    if plateau_patience > 0:
        _plateau_desc = (
            f"patience={plateau_patience} decay={plateau_decay} min_lr={plateau_min_lr:.1e} "
            f"(fase 2 por fidelidad de VENTANA -- la mejor desde el ultimo cambio de "
            f"regimen, no la mejor global -- en vez de por calendario)"
        )
        if plateau_diag_adapt:
            _plateau_desc += (
                f" + SR adaptativo(factor={plateau_diag_factor} "
                f"rango=({plateau_diag_range[0]:.1e}, {plateau_diag_range[1]:.1e}) "
                f"colapso={plateau_collapse_frac:.0%} o varianza x{plateau_var_spike:g}"
                f"{'; baja diag en plateau' if plateau_diag_lower else '; solo SUBE diag'})"
            )
    else:
        _plateau_desc = "desactivado (lr por calendario)"
    print(
        f"Hiperparametros: preset={hparams!r}  optimizer={optimizer}  "
        f"clip_norm={clip_norm}  n_samples={n_samples}  "
        f"lr={_lr_desc}  diag_shift={_diag_desc}  plateau={_plateau_desc}"
    )

    sampler = None
    if N > FULLSUM_MAX_SPINS or use_mcstate:
        if sampler_rules == "local":
            sampler = nk.sampler.MetropolisSampler(hi, nk.sampler.rules.LocalRule())
        else:
            rule1 = nk.sampler.rules.LocalRule()
            rule2 = nk.sampler.rules.ExchangeRule(graph=kitaev_graph)
            sampler = nk.sampler.MetropolisSampler(
                hi, nk.sampler.rules.MultipleRules([rule1, rule2], [0.9, 0.1])
            )
        print(f"Sampler: {sampler_rules}   Optimizador tras SR: {optimizer}")

    if couplings is not None:
        coupling_list = [tuple(float(c) for c in trio) for trio in couplings]
    else:
        coupling_list = [((1 - float(jz)) / 2, (1 - float(jz)) / 2, float(jz))
                         for jz in jz_values]

    results = []
    for jx, jy, jz in coupling_list:
        coupling_tag = f"jx{jx:.2f}_jy{jy:.2f}_jz{jz:.2f}_{projection_group}"
        if use_c3:
            if not (abs(jx - jy) < 1e-12 and abs(jy - jz) < 1e-12):
                raise ValueError(
                    f"projection_group='c3' solo es una simetria en el punto "
                    f"isotropico, y estos acoplos no lo son: "
                    f"Jx={jx}, Jy={jy}, Jz={jz}"
                )
            H = rotated_kitaev_hamiltonian(direcciones, bonds, jx, jy, jz, hi,
                                           axis=spin_axis)
        elif use_c2v:
            check_c2xy_applicable(jx, jy, jz, h=0.0)
            H = rotated_kitaev_hamiltonian(direcciones, bonds, jx, jy, jz, hi,
                                           axis=spin_axis)
        else:
            H = KitaevTransverse_H(direcciones, bonds, Jx=jx, Jy=jy, Jz=jz, h=0, hi=hi)

        evecs, manifold_idx, hosting = None, None, [0]
        if N <= FULLSUM_MAX_SPINS:
            k = min(k_eigenvals, 2 ** N - 1)
            cached = None
            if exact_path is not None:
                cached = _load_exact_spectrum(exact_path, jx, jy, jz, H, k)
            if cached is not None:
                evals, evecs = cached
            else:
                evals, evecs = nk.exact.lanczos_ed(H, k=k, compute_eigenvectors=True)
                order = np.argsort(evals.real)
                evals, evecs = evals[order], evecs[:, order]
            manifold_idx = degenerate_manifold(evals.real)
            if not project:
                mani_weights = {}
                hosting = [-1]
            elif monomial:
                weights_fn = c3_irrep_weights if use_c3 else c2v_irrep_weights
                mani_weights = {i: 0.0 for i in range(character_table.shape[0])}
                for idx in manifold_idx:
                    for k, w in weights_fn(
                        evecs[:, idx], group, element_powers, character_table
                    ).items():
                        mani_weights[k] += w
            else:
                mani_weights = manifold_irrep_weights(evecs, manifold_idx, hi, group, character_table)
            if project:
                hosting = sectors_hosting_manifold(mani_weights)
            print(f"Jx={jx:.3f} Jy={jy:.3f} Jz={jz:.3f}: E_exacta={evals.real[0]:.6f} "
                  f"(E/N={evals.real[0]/N:.6f}), manifold degenerado (tam={len(manifold_idx)}), "
                  f"sectores dominantes ({projection_group}) = {hosting}")
            if len(manifold_idx) < len(evals):
                gap = float(evals.real[len(manifold_idx)] - evals.real[0])
                print(f"  gap={gap:.6f} -> overlap garantizado solo si "
                      f"E < {evals.real[0] + gap:.6f} "
                      f"(precision relativa {gap/abs(evals.real[0]):.2e})")
        else:
            print(f"Jx={jx:.3f} Jy={jy:.3f} Jz={jz:.3f}: N={N} > {FULLSUM_MAX_SPINS}, "
                  f"no se puede diagonalizar; se asume el sector trivial (0) sin verificar.")

        def _run_phase(k_sector, stable_cosh, n_iter_phase, step_offset,
                        diag_schedule, lr_schedule, clip_norm,
                        init_params, keeper_state, fidelity_keeper_state,
                        metrics_history, vstate_path, metrics_path,
                        energy_vstate_path=None, projected=True,
                        fidelity_trace=None, fidelity_trace_path=None):
            '''Corre UNA fase de entrenamiento (todo o parte de `n_iter`) para
            un sector `k_sector`, con `DeepRBM(stable_cosh=stable_cosh)`.
            `init_params` (opcional) se trasplanta al vstate recién creado
            antes de entrenar -- mismo pytree de parámetros entre
            stable_cosh=True/False, así que el trasplante es directo.
            `keeper_state`/`fidelity_keeper_state` (opcionales) siembran el
            "mejor hasta ahora" de una fase anterior para no perder ese punto
            al empezar una fase nueva con keepers recién creados.
            '''
            if ansatz == "rbm":
                model_bare = DeepRBM(
                    num_layers=num_layers, alpha=alpha, param_dtype=jnp.complex128,
                    stable_cosh=stable_cosh,
                )
            else:
                cls = Transformer if ansatz == "transformer" else FactoredSelfAttention
                model_bare = cls(
                    layers=num_layers, heads=heads, dk=dk, d_model=d_model,
                    param_dtype=jnp.complex128 if complex_trunk else jnp.float64,
                    out_dtype=jnp.complex128,
                )
            is_projected = projected and project

            if not is_projected:
                model = model_bare
            elif monomial:
                model = MonomialSymmExpSum(
                    module=model_bare,
                    symm_group=HashableArray(np.asarray(group)),
                    characters=HashableArray(np.asarray(character_table[int(k_sector)])),
                    element_powers=HashableArray(np.asarray(element_powers)),
                    root_order=root_order,
                    group_chunk_size=group_chunk_size, remat=remat,
                )
            else:
                model = SymmExpSumChunked(
                    module=model_bare, symm_group=group,
                    characters=HashableArray(np.asarray(character_table[int(k_sector)])),
                    group_chunk_size=group_chunk_size, remat=remat,
                )

            vstate = make_vstate(
                hi, model, N, sampler=sampler, seed=seed,
                use_mcstate=use_mcstate, n_samples=n_samples,
            )
            if init_params is not None:
                vstate.parameters = flax.core.copy(
                    _match_params_to_model(init_params, is_projected)
                )
                label = "proyectada" if is_projected else "sin proyectar"
                try:
                    e_transfer = float(np.real(vstate.expect(H).mean))
                    print(f"  [transfer] fase {label}: E al arrancar = {e_transfer:.6f}")
                except Exception as exc:  # noqa: BLE001 - solo es un diagnostico
                    print(f"  [transfer] fase {label}: no se pudo medir la energia ({exc})")

            base_opt = (
                nk.optimizer.Sgd(learning_rate=lr_schedule)
                if optimizer == "sgd"
                else nk.optimizer.Adam(learning_rate=lr_schedule)
            )
            opt = (
                base_opt if clip_norm is None
                else optax.chain(optax.clip_by_global_norm(clip_norm), base_opt)
            )

            if isinstance(vstate, nk.vqs.MCState):
                driver = nk.driver.VMC_SR(
                    H, opt, diag_shift=diag_schedule, variational_state=vstate,
                )
            else:
                sr = nk.optimizer.SR(
                    qgt=nk.optimizer.qgt.QGTOnTheFly,
                    diag_shift=diag_schedule,
                    holomorphic=False,
                )
                driver = nk.driver.VMC(H, opt, variational_state=vstate, preconditioner=sr)

            keeper = BestIterKeeper(H, N, 1e-8, stop_variance=False)
            if keeper_state is not None:
                keeper.best_energy, keeper.best_state, keeper.vscore = keeper_state

            fidelity_keeper = (BestManifoldFidelity(evecs, manifold_idx, every=fidelity_every,
                                                     history=fidelity_trace)
                               if evecs is not None else None)
            if fidelity_keeper is not None and fidelity_keeper_state is not None:
                fidelity_keeper.best_fidelity, fidelity_keeper.best_state = fidelity_keeper_state

            base_extract = make_extract_metrics(metrics_history, H)

            def _extract(step, log_data, driver):
                if step % eval_every != 0:
                    return True
                return base_extract(step + step_offset, log_data, driver)

            def _keeper_update(step, log_data, driver):
                if step % eval_every != 0:
                    return True
                return keeper.update(step, log_data, driver)

            callback_fn = [_keeper_update, _extract]
            if fidelity_keeper is not None:
                def _fidelity_update(step, log_data, driver):
                    return fidelity_keeper.update(step + step_offset, log_data, driver)
                callback_fn.append(_fidelity_update)

            checkpoint = PeriodicCheckpoint(
                vstate_path, metrics_path, metrics_history, keeper,
                fidelity_keeper=fidelity_keeper, every=checkpoint_every,
                energy_vstate_path=energy_vstate_path,
                fidelity_trace=fidelity_trace, fidelity_trace_path=fidelity_trace_path,
            )

            def _checkpoint_update(step, log_data, driver):
                return checkpoint.update(step + step_offset, log_data, driver)

            callback_fn.append(_checkpoint_update)

            if n_iter_phase > 0:
                driver.run(n_iter=n_iter_phase, callback=callback_fn, show_progress=False)

            return vstate, keeper, fidelity_keeper

        if sectors is not None:
            missing = [k for k in sectors if k not in hosting]
            if missing:
                raise ValueError(
                    f"Jz={jz:.3f}: los sectores {missing} NO hospedan este "
                    f"manifold (hosting={hosting}). Entrenarlos perseguiria un "
                    f"subespacio de norma casi nula, asi que se para en vez de "
                    f"gastar la corrida."
                )
            sectors_to_train = list(sectors)
            print(f"  restringido a los sectores {sectors_to_train} de {hosting}")
        else:
            sectors_to_train = list(hosting)

        for k_sector in sectors_to_train:
            total_iter = int(resume_at_step) + int(n_iter)

            def _stages(values, fracs):
                return optax.join_schedules(
                    schedules=[optax.constant_schedule(float(v)) for v in values],
                    boundaries=[int(total_iter * f) for f in fracs],
                )

            if diag_stages is not None:
                diag_schedule = _stages(*diag_stages)
            elif hparams in _OG_FAMILY:
                diag_schedule = optax.constant_schedule(float(OG_DIAG_SHIFT))
            else:
                diag_schedule = _stages([1e-1, 1e-2, 3e-3, 1e-3], [0.15, 0.45, 0.75])

            if lr_stages is not None:
                lr_schedule = _stages(*lr_stages)
            elif hparams in _OG_FAMILY:
                lr_schedule = optax.warmup_exponential_decay_schedule(
                    **_OG_FAMILY[hparams]
                )
            else:
                lr_schedule = _stages([3e-2, 1e-2, 3e-3, 1e-3], [0.4, 0.75, 0.9])

            if resume_at_step:
                diag_schedule = _offset_schedule(diag_schedule, int(resume_at_step))
                lr_schedule = _offset_schedule(lr_schedule, int(resume_at_step))
            tag = f"{coupling_tag}_{ansatz}_k{k_sector}"
            vstate_path = f"{out_prefix}_vstate_{tag}.pkl"
            energy_vstate_path = f"{out_prefix}_vstate_{tag}_bestE.pkl"
            metrics_path = f"{out_prefix}_metrics_{tag}.csv"
            metrics_history = {'step': [], 'energy': [], 'energy_error': [], 'variance': []}
            fidelity_trace = []
            fidelity_trace_path = f"{out_prefix}_fidelitytrace_{tag}.csv"

            unprojected_steps = (
                int(n_iter * unprojected_frac) if unprojected_frac > 0 else 0
            )
            projected_budget = n_iter - unprojected_steps
            warmup_steps = (
                int(projected_budget * stable_warmup_frac)
                if stable_warmup_frac > 0 else 0
            )

            t0 = time.time()

            init_params, keeper_state, fidelity_keeper_state = None, None, None

            if resume_from is not None:
                source = resume_from
                if isinstance(resume_from, dict) and int(k_sector) in resume_from:
                    source = resume_from[int(k_sector)]
                if isinstance(source, (str, os.PathLike)):
                    with open(source, "rb") as f:
                        init_params = pickle.load(f)
                    print(f"  [resume] sector {k_sector}: parametros cargados de {source}")
                else:
                    init_params = source
                    print(f"  [resume] sector {k_sector}: parametros pasados en memoria")
                if resume_at_step:
                    print(f"  [resume] schedules evaluados desde el step {resume_at_step} "
                          f"(diag={float(diag_schedule(0)):.2e}, lr={float(lr_schedule(0)):.2e})")
            if unprojected_steps > 0:
                phase0_stable_cosh = warmup_steps > 0
                print(f"  [fase 0] {unprojected_steps} pasos SIN proyectar "
                      f"(|G|={len(np.asarray(group))}, asi que ~{len(np.asarray(group))}x "
                      f"mas baratos por paso; stable_cosh={phase0_stable_cosh})")
                free_vstate, free_keeper, free_fid_keeper = _run_phase(
                    k_sector, stable_cosh=phase0_stable_cosh, n_iter_phase=unprojected_steps,
                    step_offset=int(resume_at_step),
                    diag_schedule=diag_schedule, lr_schedule=lr_schedule, clip_norm=clip_norm,
                    init_params=init_params, keeper_state=None, fidelity_keeper_state=None,
                    metrics_history=metrics_history, vstate_path=vstate_path, metrics_path=metrics_path,
                    energy_vstate_path=energy_vstate_path, projected=False,
                    fidelity_trace=fidelity_trace, fidelity_trace_path=fidelity_trace_path,
                )
                init_params = flax.core.copy(free_vstate.parameters)

            if warmup_steps > 0:
                diag_schedule_warm = (
                    _offset_schedule(diag_schedule, unprojected_steps)
                    if unprojected_steps > 0 else diag_schedule
                )
                lr_schedule_warm = (
                    _offset_schedule(lr_schedule, unprojected_steps)
                    if unprojected_steps > 0 else lr_schedule
                )
                warm_vstate, warm_keeper, warm_fid_keeper = _run_phase(
                    k_sector, stable_cosh=True, n_iter_phase=warmup_steps,
                    step_offset=int(resume_at_step) + unprojected_steps,
                    diag_schedule=diag_schedule_warm, lr_schedule=lr_schedule_warm,
                    clip_norm=clip_norm,
                    init_params=init_params, keeper_state=None, fidelity_keeper_state=None,
                    metrics_history=metrics_history, vstate_path=vstate_path, metrics_path=metrics_path,
                    energy_vstate_path=energy_vstate_path,
                    fidelity_trace=fidelity_trace, fidelity_trace_path=fidelity_trace_path,
                )
                init_params = flax.core.copy(warm_vstate.parameters)
                keeper_state = (warm_keeper.best_energy, warm_keeper.best_state, warm_keeper.vscore)
                if warm_fid_keeper is not None:
                    fidelity_keeper_state = (warm_fid_keeper.best_fidelity, warm_fid_keeper.best_state)

            done_steps = unprojected_steps + warmup_steps
            diag_schedule_p2 = _offset_schedule(diag_schedule, done_steps) if done_steps > 0 else diag_schedule
            lr_schedule_p2 = _offset_schedule(lr_schedule, done_steps) if done_steps > 0 else lr_schedule
            phase2_iter = projected_budget - warmup_steps

            if plateau_patience > 0 and phase2_iter > 0:
                use_fidelity = evecs is not None
                metric_name = "fidelidad" if use_fidelity else "energia (sin ED: no hay F_manifold)"
                ctrl = _PlateauController(
                    lr=float(lr_schedule_p2(0)), diag=float(diag_schedule_p2(0)),
                    patience=plateau_patience, decay=plateau_decay,
                    min_lr=plateau_min_lr, use_fidelity=use_fidelity,
                    diag_adapt=plateau_diag_adapt, diag_lower=plateau_diag_lower,
                    diag_factor=plateau_diag_factor, diag_range=plateau_diag_range,
                    collapse_frac=plateau_collapse_frac, var_spike=plateau_var_spike,
                    metric_name=metric_name,
                )
                for note in ctrl.notes:
                    print(f"  [plateau] {note}")
                print(f"  [plateau] fase 2 controlada por {metric_name}: "
                      f"lr inicial={ctrl.lr:.2e}, patience={plateau_patience}, "
                      f"decay={plateau_decay}, suelo={plateau_min_lr:.1e} "
                      f"(referencia = mejor desde el ultimo cambio de regimen, "
                      f"no el mejor global)")
                if plateau_diag_adapt:
                    print(f"  [plateau] SR adaptativo: diag_shift inicial={ctrl.diag:.2e}, "
                          f"factor={plateau_diag_factor}, rango=({ctrl.diag_min:.1e}, "
                          f"{ctrl.diag_max:.1e}); SUBE si la varianza pica a "
                          f"x{plateau_var_spike:g} la mediana del bloque o la fidelidad cae "
                          f">{plateau_collapse_frac:.0%}"
                          + (", y BAJA en los plateaus (alternando con el lr)"
                             if plateau_diag_lower else ", y no baja nunca "
                             "(plateau_diag_lower=False)"))
                remaining = phase2_iter
                local_off = 0
                while remaining > 0:
                    this_chunk = min(plateau_patience, remaining)
                    if plateau_diag_adapt:
                        diag_chunk = optax.constant_schedule(ctrl.diag)
                    else:
                        diag_chunk = (_offset_schedule(diag_schedule_p2, local_off)
                                      if local_off > 0 else diag_schedule_p2)
                    fid_mark = len(fidelity_trace)
                    e_mark = len(metrics_history['energy'])
                    vstate, keeper, fidelity_keeper = _run_phase(
                        k_sector, stable_cosh=False, n_iter_phase=this_chunk,
                        step_offset=int(resume_at_step) + done_steps + local_off,
                        diag_schedule=diag_chunk,
                        lr_schedule=optax.constant_schedule(ctrl.lr),
                        clip_norm=clip_norm,
                        init_params=init_params, keeper_state=keeper_state,
                        fidelity_keeper_state=fidelity_keeper_state,
                        metrics_history=metrics_history, vstate_path=vstate_path,
                        metrics_path=metrics_path, energy_vstate_path=energy_vstate_path,
                        fidelity_trace=fidelity_trace, fidelity_trace_path=fidelity_trace_path,
                    )
                    init_params = flax.core.copy(vstate.parameters)
                    keeper_state = (keeper.best_energy, keeper.best_state, keeper.vscore)
                    if use_fidelity and fidelity_keeper is not None:
                        fidelity_keeper_state = (fidelity_keeper.best_fidelity,
                                                 fidelity_keeper.best_state)
                        window = [f for (_, f, _) in fidelity_trace[fid_mark:]]
                        global_best = float(fidelity_keeper.best_fidelity)
                    else:
                        window = list(metrics_history['energy'][e_mark:])
                        global_best = float(keeper.best_energy)
                    for line in ctrl.observe(
                        window, this_chunk, global_best=global_best,
                        energies=metrics_history['energy'][e_mark:],
                        variances=metrics_history['variance'][e_mark:],
                    ):
                        print(line)
                    remaining -= this_chunk
                    local_off += this_chunk
            else:
                vstate, keeper, fidelity_keeper = _run_phase(
                    k_sector, stable_cosh=False, n_iter_phase=phase2_iter,
                    step_offset=int(resume_at_step) + done_steps,
                    diag_schedule=diag_schedule_p2, lr_schedule=lr_schedule_p2, clip_norm=clip_norm,
                    init_params=init_params, keeper_state=keeper_state, fidelity_keeper_state=fidelity_keeper_state,
                    metrics_history=metrics_history, vstate_path=vstate_path, metrics_path=metrics_path,
                    energy_vstate_path=energy_vstate_path,
                    fidelity_trace=fidelity_trace, fidelity_trace_path=fidelity_trace_path,
                )
            dt = time.time() - t0

            if fidelity_keeper is not None and fidelity_keeper.best_state is not None:
                best = fidelity_keeper.best_state
                fidelity = fidelity_keeper.best_fidelity
            else:
                best = keeper.best_state
                fidelity = float("nan")

            E_best = float(np.real(best.expect(H).mean))
            wp_vals = [float(np.real(best.expect(Wp).mean)) for Wp in Wp_list] if Wp_list else []

            print(f"Jx={jx:.3f} Jy={jy:.3f} Jz={jz:.3f} sector={k_sector}  "
                  f"E={E_best:.6f}  E/N={E_best/N:.6f}  "
                  f"Wp_mean={np.mean(wp_vals) if wp_vals else float('nan'):.4f}  "
                  f"fidelity_manifold={fidelity:.6f}  ({dt:.1f}s)")

            results.append({
                'Jx': jx, 'Jy': jy, 'Jz': jz, 'group': projection_group,
                'sector': k_sector, 'N': N,
                'energy': E_best, 'energy_per_site': E_best / N,
                'wp_mean': np.mean(wp_vals) if wp_vals else np.nan,
                'fidelity_manifold': fidelity, 'time_s': dt,
            })

            pd.DataFrame(metrics_history).to_csv(f"{out_prefix}_metrics_{tag}.csv", index=False)
            if fidelity_trace:
                pd.DataFrame(
                    fidelity_trace, columns=["step", "fidelity", "best_fidelity"]
                ).to_csv(fidelity_trace_path, index=False)
            with open(f"{out_prefix}_vstate_{tag}.pkl", "wb") as f:
                pickle.dump(best.parameters, f)
            if keeper.best_state is not None:
                with open(energy_vstate_path, "wb") as f:
                    pickle.dump(keeper.best_state.parameters, f)
            with open(f"{out_prefix}_vstate_{tag}_final.pkl", "wb") as f:
                pickle.dump(vstate.parameters, f)

    df = pd.DataFrame(results)
    df.to_csv(f"{out_prefix}_summary.csv", index=False)
    return df


if __name__ == "__main__":

    OUT_DIR = os.environ.get("NQS_OUT_DIR", ".")
    os.makedirs(OUT_DIR, exist_ok=True)

    PROJECTION = os.environ.get("NQS_PROJECTION", "space")

    JZ_VALUES = tuple(
        float(x) for x in os.environ.get("NQS_JZ", "0.1,0.6,0.8").split(",")
    )
    couplings = [((1 - jz) / 2, (1 - jz) / 2, jz) for jz in JZ_VALUES]

    df = train_projected(
        extent=(3, 3),
        couplings=couplings,
        projection_group=PROJECTION,
        sectors=None,
        ansatz="rbm",

        hparams="og",

        num_layers=1,
        alpha=2.0,

        n_iter=4000,
        unprojected_frac=0.5,

        stable_warmup_frac=0.0,

        use_mcstate=True,
        group_chunk_size=None,
        remat=False,

        eval_every=10,
        fidelity_every=100,
        checkpoint_every=250,

        out_prefix=os.path.join(OUT_DIR, f"proj_{PROJECTION}_og"),
    )
    print(df)
