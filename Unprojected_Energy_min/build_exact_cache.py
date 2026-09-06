# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
import netket as nk
from scipy.sparse.linalg import eigsh

from utils import KitaevTransverse_H

DEGEN_TOL = 1e-8
CACHE_PATH = os.environ.get('NQS_EXACT_CACHE', 'exact_manifolds_cache.npz')
EXTENT = [3, 3]


def ground_manifold(hi, graph, jz, k=8, k_max=64):
    """(E0, V) con V = base ortonormal del subespacio fundamental.

    Sube k hasta que el subespacio degenerado quepa CON hueco por encima: si
    los k autovalores pedidos son todos del fundamental, no hay forma de saber
    si el subespacio sigue mas alla.
    """
    jx = jy = (1 - jz) / 2
    H = KitaevTransverse_H(graph.edge_colors, graph.edges(),
                           Jx=jx, Jy=jy, Jz=jz, h=0, hi=hi)
    Hs = H.to_sparse().astype(complex)
    while True:
        vals, vecs = eigsh(Hs, k=k, which='SA', tol=1e-11)
        order = np.argsort(vals)
        vals, vecs = vals[order], vecs[:, order]
        man = np.abs(vals - vals[0]) < DEGEN_TOL
        d = int(man.sum())
        if d < k or k >= k_max:
            break
        k = min(2 * k, k_max)
        print(f"   degeneracion >= {d}, repitiendo con k={k}")
    if d == k:
        raise RuntimeError(
            f"jz={jz}: el subespacio fundamental no cabe en k={k_max}")
    return float(vals[0]), np.ascontiguousarray(vecs[:, man]), float(vals[d] - vals[0])


def main(jz_list):
    graph = nk.graph.KitaevHoneycomb(extent=EXTENT, pbc=True)
    hi = nk.hilbert.Spin(s=1 / 2, N=graph.n_nodes)
    N = graph.n_nodes
    print(f"Kitaev extent={EXTENT} -> N={N} sitios, dim={2**N}")

    data = {}
    if os.path.isfile(CACHE_PATH):
        with np.load(CACHE_PATH) as raw:
            data = {k: raw[k] for k in raw.files}
        print(f"Cache existente: {sorted(k[2:] for k in data if k.startswith('v_'))}")

    for jz in jz_list:
        tag = '%.2f' % round(float(jz), 2)
        print(f"\njz={tag} ...", flush=True)
        e0, V, gap = ground_manifold(hi, graph, float(jz))
        data['e0_' + tag] = np.asarray(e0)
        data['v_' + tag] = V
        print(f"   E0/N={e0/N:+.10f}  degeneracion={V.shape[1]}  gap={gap:.3e}"
              f"  -> {V.nbytes/1e6:.1f} MB")

    np.savez(CACHE_PATH, **data)
    size = os.path.getsize(CACHE_PATH) / 1e6
    print(f"\nEscrito {CACHE_PATH} ({size:.1f} MB, "
          f"jz = {sorted(k[2:] for k in data if k.startswith('v_'))})")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main([float(a) for a in sys.argv[1:]])
