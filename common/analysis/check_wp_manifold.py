#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import numpy as np
import netket as nk

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from common.physics.hamiltonian import build_kitaev_lattice, KitaevTransverse_H
from common.physics.symmetries import get_kitaev_symmetries, get_projection_group
from common.physics.exact_diag import (
    load_exact_results,
    degenerate_manifold,
    manifold_energy_gaps,
    detect_manifold_tail,
    manifold_irrep_weights,
    sectors_hosting_manifold,
    pick_source_for_sector,
)
from common.physics.observables import get_kitaev_plaquettes, build_wilson_loops


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jz", type=float, required=True)
    p.add_argument("--extent", type=int, nargs=2, default=[3, 3])
    p.add_argument("--group", choices=["translation", "space"], default="space",
                    help="Debe coincidir con el --group que uses en run_vmc.py.")
    p.add_argument("--tol", type=float, default=1e-6, help="Tolerancia de degenerate_manifold.")
    p.add_argument("--exact-data", type=str, default=None,
                    help="npz de generate_exact_data.py; si se da, reutiliza sus autovectores.")
    p.add_argument("--recompute", action="store_true",
                    help="Ignora --exact-data y corre Lanczos desde cero.")
    p.add_argument("--k-eigenvals", type=int, default=30, help="Solo si --recompute.")
    return p.parse_args()


def main():
    args = parse_args()

    graph, hi = build_kitaev_lattice(extent=args.extent, pbc=True)
    symmetries = get_kitaev_symmetries(graph, hi)
    sg, ct = get_projection_group(symmetries, args.group)

    if args.exact_data and not args.recompute:
        exact_results = load_exact_results(args.exact_data)
        jz_key = min(exact_results, key=lambda k: abs(k - args.jz))
        result = exact_results[jz_key]
        energies = np.asarray(result["energies"]).real
        eigvecs = result["eigenvectors"]
        print(f"Cargado Jz={jz_key} de {args.exact_data} ({eigvecs.shape[1]} autovectores)")
    else:
        jx = jy = (1 - args.jz) / 2
        H = KitaevTransverse_H(graph.edge_colors, graph.edges(), Jx=jx, Jy=jy, Jz=args.jz, h=0, hi=hi)
        print(f"Corriendo Lanczos k={args.k_eigenvals} para Jz={args.jz} (puede tardar varios minutos)...")
        evals, eigvecs = nk.exact.lanczos_ed(H, k=args.k_eigenvals, compute_eigenvectors=True)
        order = np.argsort(evals.real)
        energies = evals.real[order]
        eigvecs = eigvecs[:, order]
        jz_key = args.jz

    manifold_idx = degenerate_manifold(energies, tol=args.tol)
    gaps = manifold_energy_gaps(energies, manifold_idx)
    tail = detect_manifold_tail(gaps)
    print(f"\nE0 = {energies[0]:.6f}")
    print(f"Manifold degenerado (tol={args.tol}): {len(manifold_idx)} estados -> idx {manifold_idx}")
    print("gaps respecto a E0:", ["%.2e" % g for g in gaps])
    print("detect_manifold_tail:", tail, " (None = nucleo homogeneo, sin cola sospechosa)")

    weights = manifold_irrep_weights(eigvecs, manifold_idx, hi, sg, ct)
    hosting = sectors_hosting_manifold(weights)
    print(f"\nmanifold_irrep_weights (group={args.group}):", {k: round(v, 3) for k, v in weights.items()})
    print("hosting_sectors:", hosting)
    if not hosting:
        print("[!] hosting_sectors vacio, no hay targets g_k que comprobar con este --group/--tol.")

    plaquettes, ops = get_kitaev_plaquettes(graph)
    wilson_loops = build_wilson_loops(hi, plaquettes, ops)
    Wp_sparse = [w.to_sparse() for w in wilson_loops]

    def wp_vals(psi):
        return [float(np.real(np.vdot(psi, W @ psi))) for W in Wp_sparse]

    print(f"\n=== Wp_i de los {len(manifold_idx)} autovectores CRUDOS del manifold ===")
    raw_avgs = []
    for idx in manifold_idx:
        wv = wp_vals(eigvecs[:, idx])
        avg = float(np.mean(wv))
        raw_avgs.append(avg)
        print(f"  idx {idx:2d}: avg={avg:+.6f}  Wp_i={[round(w, 2) for w in wv]}")

    print(f"\n=== Wp_i de los targets g_k (uno por hosting sector, via pick_source_for_sector) ===")
    target_avgs = []
    g_ks = {}
    for k in hosting:
        g_k, norm = pick_source_for_sector(eigvecs, manifold_idx, hi, sg, ct, k)
        g_ks[k] = (g_k, norm)
        wv = wp_vals(g_k)
        avg = float(np.mean(wv))
        target_avgs.append(avg)
        print(f"  k={k} (norma proy={norm:.3f}): avg={avg:+.6f}  Wp_i={[round(w, 3) for w in wv]}")

    sum_avg = None
    if hosting:
        print(f"\n=== Suma coherente ponderada de los g_k de todos los hosting sectors ===")
        g_sum = np.zeros(eigvecs.shape[0], dtype=complex)
        for k in hosting:
            g_k, norm = g_ks[k]
            g_sum = g_sum + g_k * norm
        g_sum = g_sum / np.linalg.norm(g_sum)
        sum_avg = float(np.mean(wp_vals(g_sum)))
        print(f"  avg={sum_avg:+.6f}")

    all_avgs = raw_avgs + target_avgs + ([sum_avg] if sum_avg is not None else [])
    spread = max(all_avgs) - min(all_avgs)
    print(f"\n=== Veredicto ===")
    print(f"  {len(all_avgs)} vectores probados. avg(Wp) va de {min(all_avgs):.6f} a {max(all_avgs):.6f}"
          f"  (dispersion={spread:.2e})")
    if spread < 1e-6:
        print(f"  -> INVARIANTE: Wp_total = {np.mean(all_avgs):.6f} en TODO el subespacio degenerado.")
        print(f"     Ningun entrenamiento ni combinacion de sectores puede superar este techo")
        print(f"     sin salir de este nivel de energia.")
    else:
        print(f"  -> NO es invariante (dispersion={spread:.2e} > 1e-6): el 'manifold' mezcla mas de")
        print(f"     un autoespacio de Wp_total, o --tol={args.tol} esta metiendo estados espurios")
        print(f"     (revisa detect_manifold_tail de arriba).")


if __name__ == "__main__":
    main()
