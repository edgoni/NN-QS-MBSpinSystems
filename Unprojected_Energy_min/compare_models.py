# -*- coding: utf-8 -*-
import copy
import os
import time

import numpy as np
import flax
import optax
import netket as nk

import sys
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common", "models")
)

from transformer import Transformer, FactoredSelfAttention
from utils import KitaevTransverse_H


EXTENT = [int(v) for v in os.environ.get('CMP_EXTENT', '2,2').split(',')]
ITERS = int(os.environ.get('CMP_ITERS', '300'))
JZ = float(os.environ.get('CMP_JZ', '0.6'))

D_MODEL = 32
DK = 8
LAYERS = 2
HEADS = 4
N_SAMPLES = 1024
DEGEN_TOL = 1e-8


def build(name):
    if name == 'Transformer':
        return Transformer(layers=LAYERS, heads=HEADS, dk=DK, d_model=D_MODEL)
    if name == 'FactoredAtt':
        return FactoredSelfAttention(layers=LAYERS, heads=HEADS, dk=DK, d_model=D_MODEL)
    raise ValueError(name)


def build_driver(opt_name, H, vstate):
    if opt_name == 'SR':
        return nk.driver.VMC(
            H, optax.sgd(0.02), variational_state=vstate,
            preconditioner=nk.optimizer.SR(diag_shift=0.01, holomorphic=False),
        )
    return nk.driver.VMC(
        H, nk.optimizer.AdaGrad(learning_rate=0.02, epscut=1e-7), variational_state=vstate
    )


def main():
    graph = nk.graph.KitaevHoneycomb(extent=EXTENT, pbc=True)
    n_sites = graph.n_nodes
    hi = nk.hilbert.Spin(s=1 / 2, N=n_sites)

    jx = jy = (1 - JZ) / 2
    H = KitaevTransverse_H(graph.edge_colors, graph.edges(), Jx=jx, Jy=jy, Jz=JZ, h=0, hi=hi)

    H_dense = H.to_dense()
    evals, evecs = np.linalg.eigh(H_dense)
    e_exact = evals[0]
    manifold = np.abs(evals - e_exact) < DEGEN_TOL
    V_exact = evecs[:, manifold]

    print(f"Red Kitaev extent={EXTENT} -> N={n_sites} sitios, Jz={JZ}, {ITERS} iteraciones")
    print(f"E_exacta/N = {e_exact / n_sites:+.6f}   degeneracion = {V_exact.shape[1]}"
          f"   gap = {evals[V_exact.shape[1]] - e_exact:.4f}\n")

    sampler = nk.sampler.MetropolisSampler(
        hi, nk.sampler.rules.MultipleRules([nk.sampler.rules.LocalRule()], [1.0])
    )
    H_sparse = H.to_sparse()

    rows = []
    for name in ['Transformer', 'FactoredAtt']:
        for opt_name in ['AdaGrad', 'SR']:
            vstate = nk.vqs.MCState(sampler, model=build(name), n_samples=N_SAMPLES, seed=0)
            driver = build_driver(opt_name, H, vstate)

            best = {'energy': np.inf, 'state': None}

            def track(step, log_data, drv, _best=best):
                e = float(np.real(drv.state.expect(H).mean))
                if e < _best['energy']:
                    _best['energy'] = e
                    st = copy.copy(drv.state)
                    st.parameters = flax.core.copy(drv.state.parameters)
                    _best['state'] = st
                return True

            t0 = time.time()
            driver.run(n_iter=ITERS, callback=track, show_progress=False)
            elapsed = time.time() - t0

            psi = np.asarray(best['state'].to_array())
            psi = psi / np.linalg.norm(psi)
            overlap = float(np.sum(np.abs(V_exact.conj().T @ psi) ** 2))
            e_var = float(np.real(np.vdot(psi, H_sparse @ psi)))
            err = abs((e_var - e_exact) / e_exact)

            rows.append((name, opt_name, vstate.n_parameters, best['energy'] / n_sites,
                         e_var / n_sites, err, overlap, elapsed))

    print(f"{'modelo':<13}{'opt':<9}{'params':>7}{'E_mc/N':>11}{'E_var/N':>11}"
          f"{'err rel':>10}{'overlap':>9}{'seg':>8}")
    print('-' * 68)
    for name, opt_name, npar, e_mc, e_var, err, ov, secs in rows:
        print(f"{name:<13}{opt_name:<9}{npar:>7}{e_mc:>11.6f}{e_var:>11.6f}"
              f"{err:>10.2e}{ov:>9.4f}{secs:>8.1f}")
    print('-' * 68)
    print(f"{'exacto':<13}{'':<9}{'-':>7}{'-':>11}{e_exact / n_sites:>11.6f}"
          f"{0.0:>10.2e}{1.0:>9.4f}")
    print("\nE_mc  = minimo de la estimacion Monte Carlo (SESGADO hacia abajo)")
    print("E_var = <psi|H|psi> exacto del mejor estado (variacional, >= E_exacta)")


if __name__ == '__main__':
    main()
