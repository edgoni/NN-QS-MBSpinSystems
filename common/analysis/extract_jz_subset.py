#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from common.physics.hamiltonian import build_kitaev_lattice
from common.physics.observables import get_kitaev_plaquettes, build_wilson_loops
from common.physics.exact_diag import (
    load_exact_results,
    degenerate_manifold,
    diagonalize_wp_in_manifold_robust,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--exact-data", type=str, default="data/raw/energies_eigenvecs_dict_k40.npz"
    )
    parser.add_argument("--jz", type=float, nargs="+", required=True)
    parser.add_argument("--extent", type=int, nargs=2, default=[3, 3])
    parser.add_argument("--no-pbc", action="store_true")
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument(
        "--max-eigenvectors", type=int, default=None,
        help="Keep only this many lowest eigenpairs per Jz. Rejected if it "
             "would truncate the ground level.",
    )
    parser.add_argument(
        "--skip-check", action="store_true",
        help="Write without verifying the level is complete. Only for points "
             "you do not intend to build phase-2 targets from.",
    )
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def check_level_is_complete(energies, eigvecs, wp_sparse, tol):
    """(ok, message) for the same completeness test the training path applies."""
    manifold_idx = degenerate_manifold(energies, tol=tol)
    if len(manifold_idx) >= len(energies):
        return False, (
            f"the manifold fills all {len(energies)} kept eigenpairs, so its "
            f"dimension is where the file stops, not where the degeneracy ends"
        )
    result = diagonalize_wp_in_manifold_robust(eigvecs, manifold_idx, wp_sparse)
    if not result["pure"]:
        return False, (
            f"the joint W_p diagonalization is impure "
            f"({result['max_impurity']:.1e}); the level is incomplete"
        )
    return True, f"manifold dim {len(manifold_idx)} of {len(energies)} kept, W_p pure"


def main():
    args = parse_args()

    graph, hi = build_kitaev_lattice(extent=args.extent, pbc=not args.no_pbc)
    plaquettes, ops = get_kitaev_plaquettes(graph)
    wp_sparse = [w.to_sparse() for w in build_wilson_loops(hi, plaquettes, ops)]

    print(f"Loading {args.exact_data} (this is the slow part) ...", flush=True)
    archive = load_exact_results(args.exact_data)
    print(f"  {len(archive)} Jz points available: {sorted(archive)}", flush=True)

    subset = {}
    for jz in args.jz:
        key = min(archive, key=lambda k: abs(float(k) - jz))
        if abs(float(key) - jz) > 1e-6:
            raise SystemExit(
                f"Jz={jz} is not in the archive (nearest is {key}). "
                f"Available: {sorted(archive)}"
            )
        entry = dict(archive[key])

        energies = np.asarray(entry["energies"]).real
        eigvecs = np.asarray(entry["eigenvectors"])
        order = np.argsort(energies)
        energies, eigvecs = energies[order], eigvecs[:, order]

        if args.max_eigenvectors is not None:
            energies = energies[: args.max_eigenvectors]
            eigvecs = eigvecs[:, : args.max_eigenvectors]

        entry["energies"] = energies
        entry["eigenvectors"] = eigvecs

        if args.skip_check:
            print(f"Jz={key}: {eigvecs.shape[1]} eigenpairs (unchecked)")
        else:
            ok, message = check_level_is_complete(energies, eigvecs, wp_sparse, args.tol)
            if not ok:
                raise SystemExit(
                    f"Jz={key}: refusing to write -- {message}.\n"
                    f"Raise --max-eigenvectors (or drop it) so the ground level "
                    f"fits strictly inside the kept spectrum."
                )
            print(f"Jz={key}: {eigvecs.shape[1]} eigenpairs kept -- {message}")

        subset[key] = entry

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, data_dict=subset)
    size_mb = out.stat().st_size / 1e6
    print(f"\n[OK] {len(subset)} Jz point(s) -> {out}  ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
