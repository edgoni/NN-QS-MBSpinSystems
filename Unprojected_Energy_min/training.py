import os
import sys

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common", "models")
)

import netket as nk
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import pandas as pd
import optax
import numpy.typing as npt
from typing import Optional
import pathlib
import copy
import flax
import flax.linen as nn
import time
from jax.nn.initializers import uniform, normal
import netket.experimental as nkx
from netket.operator.spin import sigmaz, sigmax, sigmay
import csv
import os
from typing import Any
import pickle
from transformer import Transformer, FactoredSelfAttention


from utils import BestIterKeeper, make_extract_metrics, KitaevTransverse_H, all_plaquettes
from transformer import Transformer, FactoredSelfAttention
from model_RBM import DeepRBM


MODEL = os.environ.get('NQS_MODEL', 'factored')

D_MODEL = 4
DK = 6

RBM_ALPHA = float(os.environ.get('NQS_RBM_ALPHA', '2.0'))

MODEL_TAGS = {
    'transformer': 'Transformer',
    'factored': 'FactoredAtt',
    'rbm': 'RBM',
}
if MODEL not in MODEL_TAGS:
    raise ValueError(
        f"NQS_MODEL={MODEL!r} no valido. Opciones: {sorted(MODEL_TAGS)}"
    )
TAG = MODEL_TAGS[MODEL]


def build_model(layers, heads):
    """Construye el ansatz seleccionado con el numero de capas/cabezas dado.

    'heads' se ignora para 'rbm' (DeepRBM no tiene cabezas de atencion); se
    mantiene en la firma para que el bucle de barrido sea uniforme para
    todos los modelos.
    """
    if MODEL == 'rbm':
        return DeepRBM(num_layers=layers, alpha=RBM_ALPHA)
    cls = Transformer if MODEL == 'transformer' else FactoredSelfAttention
    return cls(layers=layers, heads=heads, dk=DK, d_model=D_MODEL)


print(f"### Ansatz: {TAG} ({MODEL}) ###")


kitaev_graph= nk.graph.KitaevHoneycomb(extent=[3,3], pbc = True)
N = kitaev_graph.n_nodes
adj_list = kitaev_graph.adjacency_list()
direcciones = kitaev_graph.edge_colors
bonds = kitaev_graph.edges()

hi = nk.hilbert.Spin(s=1 / 2, N=kitaev_graph.n_nodes)
kitaev_graph.hi=hi

renyi = nkx.observable.Renyi2EntanglementEntropy(
    hi, np.arange(0, N / 2 + 1, dtype=int)
)
mags = sum([(-1) ** i * sigmaz(hi, i) / N for i in range(N)])
magnet = sum([sigmaz(hi, i) / N for i in range(N)])

Wp_ops = all_plaquettes(hi, kitaev_graph)
Wp_op = sum(Wp_ops) / len(Wp_ops)


rule1 = nk.sampler.rules.LocalRule()
rule2 = nk.sampler.rules.GaussianRule()
rules = [rule1]
sampler = nk.sampler.MetropolisSampler(
    hi, nk.sampler.rules.MultipleRules([rule1],[1.0])
)

epochs = 2500

weights_init = normal(stddev=0.01)
bias_init = normal(stddev=0.1)
vstate_init =  []

_jz_env = os.environ.get('NQS_JZ', '').strip()
jz_values = (np.array([float(v) for v in _jz_env.split(',') if v.strip()])
             if _jz_env else np.linspace(0.1, 0.4, 4))
energies_exact = []
matrics_history = {}


EXACT_PATH = 'energies_eigenvecs_dict_k40.npz'
DEGEN_TOL = 1e-8


def load_exact_manifolds(path=EXACT_PATH, tol=DEGEN_TOL):
    """Devuelve {jz: (E0, V)} con V = columnas del subespacio fundamental."""
    raw = np.load(path, allow_pickle=True)['data_dict'].item()
    out = {}
    for jz_key, entry in raw.items():
        e = np.asarray(entry['energies'])
        v = np.asarray(entry['eigenvectors'])
        order = np.argsort(e)
        e, v = e[order], v[:, order]
        manifold = np.abs(e - e[0]) < tol
        out[round(float(jz_key), 2)] = (float(e[0]), np.ascontiguousarray(v[:, manifold]))
    return out


EXACT_CACHE = os.environ.get('NQS_EXACT_CACHE', 'exact_manifolds_cache.npz')


def load_exact_cache(path=EXACT_CACHE):
    """{jz: (E0, V)} desde el cache pequeno; {} si no existe."""
    if not os.path.isfile(path):
        return {}
    out = {}
    with np.load(path) as raw:
        for key in raw.files:
            if key.startswith('v_'):
                tag = key[2:]
                out[round(float(tag), 2)] = (float(raw['e0_' + tag]),
                                             np.ascontiguousarray(raw[key]))
    return out


def save_exact_cache(manifolds, jz_list, path=EXACT_CACHE):
    """Guarda en el cache solo los jz pedidos."""
    data = {}
    for jz in jz_list:
        if jz in manifolds:
            e0, V = manifolds[jz]
            data['e0_%.2f' % jz] = np.asarray(e0)
            data['v_%.2f' % jz] = V
    if data:
        np.savez(path, **data)
        print(f"   cache escrito en {path} ({len(data)//2} jz)")


_jz_needed = [round(float(j), 2) for j in jz_values]
exact_manifolds = load_exact_cache()
if exact_manifolds:
    print(f"Cache de referencia exacta: {sorted(exact_manifolds)}")
_missing = [j for j in _jz_needed if j not in exact_manifolds]
if _missing:
    print(f"   jz sin cache {_missing} -> cargando {EXACT_PATH} (~1.8 GB)")
    exact_manifolds.update(load_exact_manifolds())
    save_exact_cache(exact_manifolds, _jz_needed)

print("Referencia exacta cargada:")
for _jz in sorted(exact_manifolds):
    _e0, _V = exact_manifolds[_jz]
    print(f"   jz={_jz:.2f}  E0={_e0:+.6f}  degeneracion={_V.shape[1]}")


FLUX_SIGN_TOL = 1e-3
_Wp_sparse = [W.to_sparse().astype(complex) for W in Wp_ops]


def flux_sector_dim(V_exact, sign, tol=0.5):
    """Dimension del sector Wp=sign (en las 9 plaquetas) dentro de V_exact.

    Los Wp conmutan entre si y con H, luego dejan invariante el subespacio
    fundamental; restringidos a el (M_k = V^dag Wp V, con V ortonormal) son
    matrices (degen x degen) hermiticas de autovalores +-1 que conmutan. El
    proyector al sector comun es el producto de (I + sign*M_k)/2 -- producto de
    proyectores que conmutan, luego proyector -- y su rango (numero de valores
    singulares ~1) es la dimension buscada.
    """
    d = V_exact.shape[1]
    if d == 0:
        return 0
    P = np.eye(d, dtype=complex)
    for W in _Wp_sparse:
        M = V_exact.conj().T @ (W @ V_exact)
        P = P @ ((np.eye(d, dtype=complex) + sign * M) / 2.0)
    return int(np.sum(np.linalg.svd(P, compute_uv=False) > tol))


def flux_target_sign(V_exact, tol=FLUX_SIGN_TOL):
    """Signo del sector de flujo del subespacio fundamental, o None si mezcla.

    degeneracion==1 se resuelve por el camino barato (<Wp> en las 9 plaquetas);
    para degeneracion>1 se exige que el sector Wp=s cubra el subespacio ENTERO,
    es decir dim(sector) == degeneracion.
    """
    d = V_exact.shape[1]
    if d == 0:
        return None
    if d == 1:
        psi = V_exact[:, 0]
        vals = [float(np.real(psi.conj() @ (W @ psi))) for W in _Wp_sparse]
        if max(vals) - min(vals) < tol and abs(abs(vals[0]) - 1.0) < tol:
            return float(np.sign(vals[0]))
        return None
    for sign in (+1.0, -1.0):
        if flux_sector_dim(V_exact, sign) == d:
            return sign
    return None


flux_signs = {jz: flux_target_sign(V) for jz, (_, V) in exact_manifolds.items()}
print("Signo de proyeccion de flujo por jz (None = sin proyectar, sector mixto):")
for _jz in sorted(flux_signs):
    _degen = exact_manifolds[_jz][1].shape[1]
    print(f"   jz={_jz:.2f}  degen={_degen:<3d} signo={flux_signs[_jz]}")


def parse_flux_force(spec):
    """'' -> (None, {}); '+1' -> (+1.0, {}); '0.4:+1,0.1:-1' -> (None, {...})."""
    spec = spec.strip()
    if not spec:
        return None, {}
    if ':' not in spec:
        sign = float(spec)
        if sign not in (1.0, -1.0):
            raise ValueError(f"NQS_FLUX_FORCE={spec!r}: el signo debe ser +1 o -1")
        return sign, {}
    per_jz = {}
    for item in spec.split(','):
        if not item.strip():
            continue
        jz_s, sign_s = item.split(':')
        sign = float(sign_s)
        if sign not in (1.0, -1.0):
            raise ValueError(f"NQS_FLUX_FORCE={spec!r}: el signo debe ser +1 o -1")
        per_jz[round(float(jz_s), 2)] = sign
    return None, per_jz


FLUX_FORCE_ALL, FLUX_FORCE_JZ = parse_flux_force(os.environ.get('NQS_FLUX_FORCE', ''))

if FLUX_FORCE_ALL is not None or FLUX_FORCE_JZ:
    print("Signo de flujo FORZADO a mano (NQS_FLUX_FORCE):")
    for _jz_raw in jz_values:
        _jz = round(float(_jz_raw), 2)
        _forced = FLUX_FORCE_JZ.get(_jz, FLUX_FORCE_ALL)
        if _forced is None:
            continue
        if _jz not in exact_manifolds:
            raise KeyError(f"NQS_FLUX_FORCE: jz={_jz:.2f} no esta en {EXACT_PATH}")
        _auto = flux_signs[_jz]
        _dim = flux_sector_dim(exact_manifolds[_jz][1], _forced)
        _degen = exact_manifolds[_jz][1].shape[1]
        flux_signs[_jz] = _forced
        _nota = 'coincide con el automatico' if _auto == _forced else (
            f'automatico={_auto}')
        print(f"   jz={_jz:.2f}  signo={_forced:+.0f}  ({_nota}); "
              f"dim(sector Wp={_forced:+.0f}) = {_dim} de {_degen}")
        if _dim == 0:
            print(f"      AVISO: el subespacio fundamental de jz={_jz:.2f} no "
                  f"contiene ninguna componente con Wp={_forced:+.0f} en las 9 "
                  f"plaquetas. Forzar ese signo empuja fuera del fundamental: "
                  f"el overlap maximo alcanzable es 0.")


OVERLAP_EVERY = int(os.environ.get('NQS_OVERLAP_EVERY', '50'))

OBS_STATE = os.environ.get('NQS_OBS_STATE', 'last')
if OBS_STATE not in ('last', 'best'):
    raise ValueError(f"NQS_OBS_STATE={OBS_STATE!r} no valido. Opciones: ['last', 'best']")


PLATEAU_WINDOW = int(os.environ.get('NQS_PLATEAU_WINDOW', '40'))
PLATEAU_MIN_STEP = int(os.environ.get('NQS_PLATEAU_MIN_STEP', '150'))
PLATEAU_E_TOL = float(os.environ.get('NQS_PLATEAU_E_TOL', '5e-4'))
PLATEAU_VSCORE_TOL = float(os.environ.get('NQS_PLATEAU_VSCORE_TOL', '1e-4'))


class PlateauStopper:
    """Callback: para el entrenamiento si energia y V-score llevan `window`
    pasos SEGUIDOS estables, en vez de agotar siempre los `epochs` fijos.

    Usa log_data[driver._loss_name] (ya calculado por el propio paso de SR,
    sin coste extra), igual que BestIterKeeper.
    """

    def __init__(self, N, window=PLATEAU_WINDOW, min_step=PLATEAU_MIN_STEP,
                 e_tol=PLATEAU_E_TOL, vscore_tol=PLATEAU_VSCORE_TOL):
        self.N = N
        self.window = window
        self.min_step = min_step
        self.e_tol = e_tol
        self.vscore_tol = vscore_tol
        self.energies = []
        self.vscores = []
        self.stopped_at = None

    def update(self, step, log_data, driver):
        mean = float(np.real(getattr(log_data[driver._loss_name], "mean")))
        var = float(np.real(getattr(log_data[driver._loss_name], "variance")))
        vscore = self.N * var / mean**2 if mean != 0 else np.inf
        self.energies.append(mean / self.N)
        self.vscores.append(vscore)

        if step < self.min_step or len(self.energies) < self.window:
            return True

        e_win = self.energies[-self.window:]
        v_win = self.vscores[-self.window:]
        spread = max(e_win) - min(e_win)
        if spread < self.e_tol and max(v_win) < self.vscore_tol:
            self.stopped_at = step
            print(
                f"  Plateau detectado en step {step}: spread E/N={spread:.2e} "
                f"(< {self.e_tol:.0e}), max V-score={max(v_win):.2e} "
                f"(< {self.vscore_tol:.0e}) sostenido {self.window} pasos -> "
                f"parando entrenamiento"
            )
            return False
        return True


COLLAPSE_WINDOW = int(os.environ.get('NQS_COLLAPSE_WINDOW', '200'))
COLLAPSE_VSCORE = float(os.environ.get('NQS_COLLAPSE_VSCORE', '1e-12'))


class CollapseStopper:
    """Callback: para el entrenamiento si la mediana del V-score en una
    ventana cae por debajo de `vscore_tol` (funcion de onda colapsada sobre
    una unica configuracion, o autoestado exacto)."""

    def __init__(self, N, window=COLLAPSE_WINDOW, vscore_tol=COLLAPSE_VSCORE):
        self.N = N
        self.window = window
        self.vscore_tol = vscore_tol
        self.vscores = []
        self.stopped_at = None

    def update(self, step, log_data, driver):
        if self.window <= 0:
            return True
        mean = float(np.real(getattr(log_data[driver._loss_name], "mean")))
        var = float(np.real(getattr(log_data[driver._loss_name], "variance")))
        self.vscores.append(self.N * var / mean**2 if mean != 0 else np.inf)
        if len(self.vscores) < self.window:
            return True
        med = float(np.median(self.vscores[-self.window:]))
        if med < self.vscore_tol:
            self.stopped_at = step
            print(
                f"  COLAPSO detectado en step {step}: mediana del V-score en "
                f"{self.window} pasos = {med:.2e} (< {self.vscore_tol:.0e}), "
                f"E/N = {mean/self.N:+.6f} -> parando entrenamiento"
            )
            return False
        return True


class OverlapTracker:
    """Callback de NetKet: cada `every` pasos mide el overlap y guarda el mejor.

    Guarda tambien el historial (step, overlap, E_var) para poder ver la
    trayectoria y no solo el extremo.
    """

    def __init__(self, V_exact, H_sparse, every):
        self.V = V_exact
        self.Hs = H_sparse
        self.every = every
        self.best_overlap = -np.inf
        self.best_state = None
        self.best_step = -1
        self.history = []

    def update(self, step, log_data, driver):
        if self.every <= 0 or (step % self.every) != 0:
            return True
        psi = np.asarray(driver.state.to_array())
        psi = psi / np.linalg.norm(psi)
        ov = float(np.sum(np.abs(self.V.conj().T @ psi) ** 2))
        e_var = float(np.real(np.vdot(psi, self.Hs @ psi)))
        self.history.append((step, ov, e_var))
        if ov > self.best_overlap:
            self.best_overlap = ov
            self.best_step = step
            st = copy.copy(driver.state)
            st.parameters = flax.core.copy(driver.state.parameters)
            self.best_state = st
        return True


lr = 0.1
ramp_iter = 50
lrmax = 0.05
epsilon = 1e-7

lr_schedule_try = optax.warmup_exponential_decay_schedule(
    init_value=0.01,
    peak_value=0.05,
    warmup_steps=30,
    transition_steps=100,
    decay_rate=0.90
)

lr = lr_schedule_try
lr_name = f'{lr}'
if lr == lr_schedule_try:
    lr_name = 'sched'


OPTIMIZER = os.environ.get('NQS_OPT', 'sr')
SR_DIAG_SHIFT = 0.01

if OPTIMIZER not in ('sr', 'adagrad'):
    raise ValueError(f"NQS_OPT={OPTIMIZER!r} no valido. Opciones: ['sr', 'adagrad']")


PROJECT_FLUX = os.environ.get('NQS_PROJECT_FLUX', '1') == '1'
FLUX_LAMBDA = float(os.environ.get('NQS_FLUX_LAMBDA', '1.0'))


def build_driver(H, vstate):
    """Devuelve el driver VMC con el optimizador seleccionado.

    Para 'sr' se usa nk.driver.VMC_SR y no VMC(preconditioner=SR): con
    n_samples <= n_params (2048 muestras frente a ~17k parametros del
    Transformer) la formulacion QGT estandar es ineficiente e inestable, y
    NetKet recomienda explicitamente la formulacion kernel/minSR. VMC_SR elige
    la implementacion optima automaticamente y da el mismo resultado.
    """
    if OPTIMIZER == 'sr':
        return nk.driver.VMC_SR(
            H,
            optax.sgd(learning_rate=lr),
            diag_shift=SR_DIAG_SHIFT,
            variational_state=vstate,
        )
    return nk.driver.VMC(
        H, nk.optimizer.AdaGrad(learning_rate=lr, epscut=1e-7), variational_state=vstate
    )

TRANSFER = os.environ.get('NQS_TRANSFER', '0') == '1'
TRANSFER_FROM = os.environ.get('NQS_TRANSFER_FROM', 'last')
if TRANSFER_FROM not in ('last', 'best'):
    raise ValueError(
        f"NQS_TRANSFER_FROM={TRANSFER_FROM!r} no valido. Opciones: ['last', 'best']")
if TRANSFER:
    _jz_sorted = list(jz_values) == sorted(jz_values)
    print(f"### Transfer learning ENTRE jz: ON (pesos de '{TRANSFER_FROM}') ###")
    if not _jz_sorted:
        print("   AVISO: jz no viene ordenado y la transferencia sigue ese mismo "
              "orden; para una continuacion adiabatica ordenalo (NQS_JZ=0.1,0.2,...)")
else:
    print("### Transfer learning ENTRE jz: OFF (init fresca en cada jz) ###")


RESULTS_ROOT = os.environ.get('NQS_RESULTS_DIR', 'results')
RESULTS_DIR = os.path.join(RESULTS_ROOT, TAG)
os.makedirs(RESULTS_DIR, exist_ok=True)

energies_rbm = np.zeros((len(jz_values), 2))
np.copyto(energies_rbm[:, 0], jz_values)

maxlayers = 3
maxheads = 4

if MODEL == 'rbm':
    LAYERS_VALUES = range(1, 5)
    HEADS_VALUES = [1]
else:
    LAYERS_VALUES = range(1, maxlayers + 1)
    HEADS_VALUES = range(2, maxheads + 1)

_layers_env = os.environ.get('NQS_LAYERS', '').strip()
if _layers_env:
    LAYERS_VALUES = [int(v) for v in _layers_env.split(',') if v.strip()]
_heads_env = os.environ.get('NQS_HEADS', '').strip()
if _heads_env:
    HEADS_VALUES = [int(v) for v in _heads_env.split(',') if v.strip()]

print(f"### Barrido: layers={list(LAYERS_VALUES)} heads={list(HEADS_VALUES)} "
      f"jz={[round(float(j), 2) for j in jz_values]} ###")


for layers in LAYERS_VALUES:
    for heads in HEADS_VALUES:

        transfer_params = None

        for i, jz in enumerate(jz_values):
            path_metrics = os.path.join(RESULTS_DIR, f'{TAG}_metrics{layers}_head{heads}_{jz:.2f}_{lr_name}.csv')
            filename = os.path.join(RESULTS_DIR, f"{TAG}{layers}_head{heads}_{jz:.2f}_{lr_name}.mpack")
            vstate_path = os.path.join(RESULTS_DIR, f"vstate_{TAG}{layers}_head{heads}_{jz:.2f}_{lr_name}.pkl")
            filename_last = os.path.join(RESULTS_DIR, f"{TAG}{layers}_head{heads}_{jz:.2f}_{lr_name}_last.mpack")
            vstate_path_last = os.path.join(RESULTS_DIR, f"vstate_{TAG}{layers}_head{heads}_{jz:.2f}_{lr_name}_last.pkl")
            filename_bestov = os.path.join(RESULTS_DIR, f"{TAG}{layers}_head{heads}_{jz:.2f}_{lr_name}_bestov.mpack")
            vstate_path_bestov = os.path.join(RESULTS_DIR, f"vstate_{TAG}{layers}_head{heads}_{jz:.2f}_{lr_name}_bestov.pkl")
            overlap_path = os.path.join(RESULTS_DIR, f'{TAG}_overlaptrace{layers}_head{heads}_{jz:.2f}_{lr_name}.csv')
            obs_path = os.path.join(RESULTS_DIR, f'obs_{TAG}_layers{layers}_head{heads}_{lr_name}.csv')

            print(f"\n--- Entrenando para Jz = {jz:.2f} ---")

            RBM = build_model(layers, heads)
            vstate = nk.vqs.MCState(sampler, model=RBM, n_samples=2048)

            if transfer_params is not None:
                vstate.parameters = transfer_params
                print(f"   [transfer] arrancando de los pesos '{TRANSFER_FROM}' "
                      f"del jz anterior")

            jx = jy = (1 - jz) / 2
            H = KitaevTransverse_H(direcciones, bonds, Jx=jx, Jy=jy, Jz=jz, h=0, hi=hi)

            flux_sign = flux_signs.get(round(float(jz), 2)) if PROJECT_FLUX else None
            H_train = H - flux_sign * FLUX_LAMBDA * sum(Wp_ops) if flux_sign is not None else H
            if flux_sign is not None:
                print(f"   [flux] jz={jz:.2f}: proyectando hacia Wp={flux_sign:+.0f} "
                      f"(lambda={FLUX_LAMBDA})")

            driver = build_driver(H_train, vstate)

            keeper = BestIterKeeper(H, N, 1e-8)

            log = nk.logging.RuntimeLog()
            metrics_history = {'step': [], 'energy': [], 'energy_error': [], 'loss': [], 'variance': [], 'vscore': []}

            e0_exact, V_exact = exact_manifolds[round(float(jz), 2)]
            H_sparse = H.to_sparse()
            tracker = OverlapTracker(V_exact, H_sparse, OVERLAP_EVERY)

            plateau = PlateauStopper(N)
            collapse = CollapseStopper(N)
            callback_fn = [keeper.update, make_extract_metrics(metrics_history, H, N),
                           plateau.update, collapse.update]
            if OVERLAP_EVERY > 0:
                callback_fn.append(tracker.update)

            driver.run(n_iter=epochs, out=log, callback=callback_fn, show_progress=True)

            last_params = flax.core.copy(vstate.parameters)
            best = keeper.best_state

            def evaluate_state(state):
                """(E_var exacta, overlap con el subespacio fundamental) del estado."""
                psi = np.asarray(state.to_array())
                psi = psi / np.linalg.norm(psi)
                ov = float(np.sum(np.abs(V_exact.conj().T @ psi) ** 2))
                ev = float(np.real(np.vdot(psi, H_sparse @ psi)))
                return ev, ov

            e_var_last, overlap_last = evaluate_state(vstate)
            e_var_best, overlap_best = evaluate_state(best)

            if OBS_STATE == 'last':
                obs_state = copy.copy(vstate)
                obs_state.parameters = flax.core.copy(last_params)
            else:
                obs_state = best

            if tracker.best_state is not None:
                e_var_bestov = tracker.history[
                    [h[0] for h in tracker.history].index(tracker.best_step)][2]
                overlap_bestov = tracker.best_overlap
                step_bestov = tracker.best_step
            else:
                e_var_bestov, overlap_bestov, step_bestov = float('nan'), float('nan'), -1

            if not (overlap_bestov >= overlap_last):
                e_var_bestov, overlap_bestov = e_var_last, overlap_last
                step_bestov = epochs - 1
                bestov_params = last_params
            elif tracker.best_state is not None:
                bestov_params = tracker.best_state.parameters
            else:
                bestov_params = None

            to_save = [(filename, vstate_path, best.parameters),
                       (filename_last, vstate_path_last, last_params)]
            if bestov_params is not None:
                to_save.append((filename_bestov, vstate_path_bestov, bestov_params))

            for mpack_path, pkl_path, params in to_save:
                with open(mpack_path, "wb") as f:
                    f.write(flax.serialization.to_bytes(params))
                with open(pkl_path, "wb") as f:
                    pickle.dump(params, f)

            if tracker.history:
                pd.DataFrame(tracker.history,
                             columns=['step', 'overlap', 'E_var']).to_csv(
                    overlap_path, index=False)

            if TRANSFER:
                if collapse.stopped_at is not None:
                    transfer_params = None
                    print(f"   [transfer] esta corrida colapso (step "
                          f"{collapse.stopped_at}): NO se propagan sus pesos, el "
                          f"jz siguiente arranca de init fresca")
                else:
                    transfer_params = flax.core.copy(
                        last_params if TRANSFER_FROM == 'last' else best.parameters)

            header = ['model', 'opt', 'layers', 'heads', 'n_params', 'Jz', 'Energy',
                      'E_var_best', 'overlap_best', 'E_var_last', 'overlap_last',
                      'E_var_bestov', 'overlap_bestov', 'step_bestov',
                      'E0_exact', 'degen', 'flux_sign', 'obs_state', 'transfer',
                      'S', 'm', 'ms', 'fluct', 'fluct_s', 'Wp']
            wp_val = np.real(obs_state.expect(Wp_op).mean)
            obs = [renyi, magnet, mags, magnet @ magnet, mags @ mags]

            results = [TAG, OPTIMIZER, layers, heads, vstate.n_parameters, jz,
                       keeper.best_energy/N,
                       e_var_best/N, overlap_best,
                       e_var_last/N, overlap_last,
                       e_var_bestov/N, overlap_bestov, step_bestov,
                       e0_exact/N, V_exact.shape[1],
                       '' if flux_sign is None else f'{flux_sign:+.0f}',
                       OBS_STATE, TRANSFER_FROM if TRANSFER else ''] \
                      + [np.real(obs_state.expect(o).mean) for o in obs] + [wp_val]

            file_exists = os.path.isfile(obs_path)
            if file_exists:
                with open(obs_path, newline='') as f:
                    old_header = next(csv.reader(f, delimiter='\t'), None)
                if old_header != header:
                    backup, n = obs_path + '.bak', 1
                    while os.path.exists(backup):
                        backup = f'{obs_path}.bak{n}'
                        n += 1
                    os.rename(obs_path, backup)
                    print(f"   [obs] cabecera antigua en {obs_path} -> movida a "
                          f"{backup}; se empieza un CSV nuevo con la cabecera actual")
                    file_exists = False
            with open(obs_path, 'a', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                if not file_exists:
                    writer.writerow(header)
                writer.writerow(results)

            df_metrics = pd.DataFrame(metrics_history)

            df_metrics.to_csv(path_metrics, index=False)

