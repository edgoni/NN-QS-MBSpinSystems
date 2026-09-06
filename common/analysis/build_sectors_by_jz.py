#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from common.physics.hamiltonian import build_kitaev_lattice
from common.physics.symmetries import get_kitaev_symmetries, get_projection_group
from common.physics.exact_diag import (
    load_exact_results,
    degenerate_manifold,
    manifold_energy_gaps,
    detect_manifold_tail,
    manifold_irrep_weights,
    sectors_hosting_manifold,
)
from common.utils.schema import SECTORS_BY_JZ_COLUMNS, validate_row

DEFAULT_TOL_GRID = np.logspace(-12, -2, 21)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exact-data", type=str, default="data/raw/energies_eigenvecs_dict_k40.npz")
    parser.add_argument("--extent", type=int, nargs=2, default=[3, 3])
    parser.add_argument("--no-pbc", action="store_true")
    parser.add_argument(
        "--group", choices=["translation", "space"], default="space",
        help="Symmetry group whose irreps label the sectors. Must match the "
             "--group used by run_vmc.py, or the sector indices in runs.csv "
             "and the ones here name different things.",
    )
    parser.add_argument(
        "--tol", type=float, default=1e-6,
        help="Degeneracy tolerance for the reported manifold (the sweep for "
             "S3 is recorded separately in manifold_dim_vs_tol).",
    )
    parser.add_argument("--n-levels", type=int, default=30, help="Levels kept in spectrum_head (S2).")
    parser.add_argument("--out", type=str, default="data/results/sectors_by_jz.csv")
    return parser.parse_args()


def manifold_dim_vs_tol(energies, tol_grid=DEFAULT_TOL_GRID) -> dict:
    """`{tol: dim(manifold)}` over `tol_grid` -- the data behind S3."""
    return {float(t): len(degenerate_manifold(energies, tol=t)) for t in tol_grid}


def gap_above_manifold(energies, manifold_idx) -> float:
    """Energy gap separating the manifold from the first state above it.

    NaN when the stored spectrum stops at the top of the manifold, i.e. the
    Lanczos run did not compute enough eigenpairs to see the gap. That is a
    real limitation of the input file, not a zero gap, so it must not be
    reported as one.
    """
    n = len(manifold_idx)
    if n >= len(energies):
        return float("nan")
    return float(energies[n] - energies[0])


def build_rows(exact_results, hi, sg, ct, *, extent, N, group, tol, n_levels):
    rows = []
    for jz in sorted(exact_results):
        entry = exact_results[jz]
        energies = np.asarray(entry["energies"]).real
        eigvecs = entry["eigenvectors"]

        order = np.argsort(energies)
        energies = energies[order]
        eigvecs = eigvecs[:, order]

        manifold_idx = degenerate_manifold(energies, tol=tol)
        gaps = manifold_energy_gaps(energies, manifold_idx)
        tail = detect_manifold_tail(gaps)
        weights = manifold_irrep_weights(eigvecs, manifold_idx, hi, sg, ct)
        hosting = sectors_hosting_manifold(weights)

        row = {
            "Jz": float(jz),
            "extent_x": extent[0],
            "extent_y": extent[1],
            "N": N,
            "group": group,
            "tol": tol,
            "E0": float(energies[0]),
            "n_eigenvalues": int(len(energies)),
            "manifold_dim": len(manifold_idx),
            "gap_manifold": gap_above_manifold(energies, manifold_idx),
            "hosting_sectors": json.dumps(list(hosting)),
            "manifold_irrep_weights": json.dumps({str(k): float(v) for k, v in weights.items()}),
            "manifold_gaps": json.dumps([float(g) for g in gaps]),
            "manifold_tail_warning": tail is not None,
            "manifold_dim_vs_tol": json.dumps(manifold_dim_vs_tol(energies)),
            "spectrum_head": json.dumps([float(e) for e in energies[:n_levels]]),
            "wilson_labels": None,
            "hsym_eigenvalues": None,
            "topological_degeneracy_ok": None,
        }
        validate_row(row, SECTORS_BY_JZ_COLUMNS, context=f"Jz={jz}")
        rows.append(row)

        warn = "  [WARN] cola de casi-degenerados" if tail is not None else ""
        print(
            f"Jz={jz:.2f}  E0={energies[0]:+.6f}  dim={len(manifold_idx):2d}  "
            f"gap={row['gap_manifold']:.2e}  hosting={hosting}{warn}"
        )
    return rows


def main():
    args = parse_args()

    graph, hi = build_kitaev_lattice(extent=args.extent, pbc=not args.no_pbc)
    symmetries = get_kitaev_symmetries(graph, hi)
    sg, ct = get_projection_group(symmetries, args.group)

    try:
        exact_results = load_exact_results(args.exact_data)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{args.exact_data} not found. Run `python common/analysis/generate_exact_data.py` first."
        ) from exc

    print(
        f"{len(exact_results)} Jz points, group={args.group} (n_g={len(sg)}, "
        f"{ct.shape[0]} irreps), N={graph.n_nodes}"
    )
    rows = build_rows(
        exact_results, hi, sg, ct,
        extent=args.extent, N=graph.n_nodes, group=args.group,
        tol=args.tol, n_levels=args.n_levels,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(SECTORS_BY_JZ_COLUMNS)).to_csv(out, index=False)
    print(f"\n[OK] {len(rows)} filas -> {out}")


if __name__ == "__main__":
    main()
