#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import netket as nk
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from common.physics.hamiltonian import build_kitaev_lattice, KitaevTransverse_H
from common.physics.symmetries import get_kitaev_symmetries, get_projection_group
from common.physics.exact_diag import (
    load_exact_results,
    degenerate_manifold,
    manifold_irrep_weights,
    sectors_hosting_manifold,
    diagonalize_wp_in_manifold_robust,
    plaquette_permutations,
    vortex_pattern_summary,
)
from common.physics.observables import get_kitaev_plaquettes, build_wilson_loops
from common.utils.schema import VORTEX_SECTORS_COLUMNS, validate_row

COARSE_GRID = np.round(np.linspace(0.0, 1.0, 11), 4)

FINE_WINDOWS = ((0.40, 0.5001, 0.02), (0.70, 0.8001, 0.02))

DEFAULT_JZ_MIN = 0.74

DEFAULT_JZ_MAX = 0.99


def default_jz_grid(jz_min=None, jz_max=None):
    """Coarse scan plus the two refined windows, deduplicated and sorted.

    Rounded to 4 decimals *before* deduplicating: `np.arange(0.4, 0.5, 0.02)`
    produces 0.44000000000000006, which is a different float from the 0.44 a
    later window would produce and would otherwise be diagonalized twice.

    :param jz_min: keep only points >= this, or None for no lower bound
    :param jz_max: keep only points <= this, or None for no upper bound
    """
    values = list(COARSE_GRID)
    for start, stop, step in FINE_WINDOWS:
        values.extend(np.arange(start, stop, step))
    grid = sorted({round(float(v), 4) for v in values})
    if jz_min is not None:
        grid = [v for v in grid if v >= jz_min - 1e-9]
    if jz_max is not None:
        grid = [v for v in grid if v <= jz_max + 1e-9]
    return grid


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--extent", type=int, nargs=2, default=[3, 3])
    parser.add_argument("--no-pbc", action="store_true")
    parser.add_argument(
        "--group", choices=["translation", "space"], default="space",
        help="Symmetry group whose irreps fill `hosting_sectors`. Must match "
             "the --group used by run_vmc.py and build_sectors_by_jz.py.",
    )
    parser.add_argument(
        "--jz", type=float, nargs="+", default=None,
        help="Explicit Jz values. Overrides --jz-min/--jz-max entirely. "
             "Default: the coarse 0.1 scan refined to 0.02 across 0.4-0.5 and "
             "0.7-0.8, restricted to [--jz-min, --jz-max].",
    )
    parser.add_argument(
        "--jz-min", type=float, default=DEFAULT_JZ_MIN,
        help=f"Lower end of the default grid (default {DEFAULT_JZ_MIN}, which "
             f"resumes the run already covering 0.0-0.72). Pass 0.0 for the "
             f"whole scan. Rows this run does not recompute are kept.",
    )
    parser.add_argument(
        "--jz-max", type=float, default=DEFAULT_JZ_MAX,
        help=f"Upper end of the default grid (default {DEFAULT_JZ_MAX}, i.e. "
             f"just short of the exactly-solvable Jz=1 limit, whose degeneracy "
             f"Lanczos does not resolve at any affordable k).",
    )
    parser.add_argument("--tol", type=float, default=1e-6, help="Degeneracy tolerance.")
    parser.add_argument(
        "--exact-data", type=str, default="data/raw/energies_eigenvecs_dict_k40.npz",
        help="npz from generate_exact_data.py. Points it already contains are "
             "reused instead of re-running Lanczos; the rest are computed.",
    )
    parser.add_argument(
        "--cache-match-tol", type=float, default=1e-6,
        help="How close a cached Jz must be to count as the requested one.",
    )
    parser.add_argument(
        "--k-eigenvals", type=int, default=30,
        help="Eigenpairs per Lanczos run. Escalated automatically (see --k-max) "
             "when the manifold fills the computed spectrum, since then the "
             "reported dimension is where the solver stopped, not where the "
             "degeneracy ends.",
    )
    parser.add_argument("--k-max", type=int, default=48, help="Ceiling for that escalation.")
    parser.add_argument(
        "--wp-seeds", type=int, nargs="+", default=[0, 1, 2],
        help="Seeds tried for the random W_p combination.",
    )
    parser.add_argument(
        "--no-symmetry-quotient", action="store_true",
        help="Count flux patterns literally, without merging ones related by "
             "a lattice symmetry. Off by default: without the quotient, the "
             "nine translations of one vortex pair read as nine sectors.",
    )
    parser.add_argument("--out", type=str, default="data/results/vortex_sectors_by_jz.csv")
    return parser.parse_args()


def lanczos_with_resolved_manifold(H, k, k_max, tol, wp_sparse, wp_seeds):
    """Lanczos, escalating `k` until the manifold is a *complete* energy level.

    Two ways a manifold can come back truncated, both of which corrupt every
    quantity downstream:

    1. It fills the computed spectrum, so its dimension is where the solver
       stopped rather than where the degeneracy ends, and `gap_manifold` is
       the gap to nothing.
    2. It does not fill the spectrum but is still missing members, because
       Lanczos resolved a highly degenerate level unevenly and the stragglers
       landed outside `tol`.

    Case 2 is invisible to a dimension check, but not to the physics: the W_p
    commute with H, so a *complete* level is an invariant subspace of every
    W_p and diagonalizing them jointly inside it must give eigenvalues of
    exactly +-1. A partial level is not invariant, and the joint
    diagonalization comes back impure. So impurity is used here as the
    convergence criterion it actually is. (Observed on the 2x2 lattice at
    Jz=1: k=8 returns 6 of the 16 degenerate states and an impurity of 0.8,
    while dense ED gives all 16 and impurity 1e-15.)

    :return: (energies, eigvecs, k_used) with `energies` ascending.
    """
    while True:
        evals, eigvecs = nk.exact.lanczos_ed(H, k=k, compute_eigenvectors=True)
        order = np.argsort(evals.real)
        energies, eigvecs = evals.real[order], eigvecs[:, order]

        manifold_idx = degenerate_manifold(energies, tol=tol)
        fills_spectrum = len(manifold_idx) == len(energies)
        impurity = diagonalize_wp_in_manifold_robust(
            eigvecs, manifold_idx, wp_sparse, seeds=tuple(wp_seeds)
        )["max_impurity"]
        truncated = fills_spectrum or impurity >= 1e-6

        if not truncated:
            return energies, eigvecs, k

        if k >= k_max:
            reason = (
                "the manifold fills every computed eigenpair"
                if fills_spectrum
                else f"the W_p basis is impure ({impurity:.1e})"
            )
            print(
                f"    [WARN] at the k-max ceiling k={k}, {reason}. The level is "
                f"probably incomplete: manifold_dim and gap_manifold are lower "
                f"bounds and the flux census for this point is unreliable."
            )
            return energies, eigvecs, k

        k = min(k * 2, k_max)
        reason = "manifold filled the spectrum" if fills_spectrum else f"impure W_p basis ({impurity:.1e})"
        print(f"    {reason}, retrying with k={k}", flush=True)


def wp_expectation(psi, wp_sparse):
    """<W_p_i> of one state, as a plain float array."""
    return np.array([float(np.real(np.vdot(psi, W @ psi))) for W in wp_sparse])


def analyze_point(
    energies, eigvecs, *, hi, sg, ct, wp_sparse, plaquette_perms, tol, wp_seeds,
):
    """All per-Jz quantities, given a spectrum and the operators."""
    manifold_idx = degenerate_manifold(energies, tol=tol)

    gap = (
        float(energies[len(manifold_idx)] - energies[0])
        if len(manifold_idx) < len(energies)
        else float("nan")
    )

    wp_result = diagonalize_wp_in_manifold_robust(
        eigvecs, manifold_idx, wp_sparse, seeds=tuple(wp_seeds)
    )
    summary = vortex_pattern_summary(wp_result["sign_patterns"], plaquette_perms)

    raw_avgs = [float(np.mean(wp_expectation(eigvecs[:, i], wp_sparse))) for i in manifold_idx]

    weights = manifold_irrep_weights(eigvecs, manifold_idx, hi, sg, ct)

    return {
        "manifold_idx": manifold_idx,
        "gap": gap,
        "wp_result": wp_result,
        "summary": summary,
        "avg_Wp": float(np.mean(raw_avgs)),
        "avg_Wp_spread": float(max(raw_avgs) - min(raw_avgs)),
        "hosting": sectors_hosting_manifold(weights),
    }


def build_row(jz, analysis, *, energies, extent, N, n_plaquettes, group, tol, source):
    summary = analysis["summary"]
    wp = analysis["wp_result"]
    row = {
        "Jz": float(jz),
        "extent_x": extent[0],
        "extent_y": extent[1],
        "N": N,
        "n_plaquettes": n_plaquettes,
        "group": group,
        "tol": tol,
        "E0": float(energies[0]),
        "n_eigenvalues": int(len(energies)),
        "manifold_dim": len(analysis["manifold_idx"]),
        "gap_manifold": analysis["gap"],
        "n_minus": json.dumps(summary["n_minus"]),
        "vortex_plaquette_idx": json.dumps(summary["vortex_plaquette_idx"]),
        "n_distinct_patterns": summary["n_distinct_patterns"],
        "pattern_multiplicity": json.dumps(summary["multiplicity"]),
        "all_same_n_minus": summary["all_same_n_minus"],
        "representatives": json.dumps(summary["representatives"]),
        "avg_Wp": analysis["avg_Wp"],
        "avg_Wp_spread": analysis["avg_Wp_spread"],
        "hosting_sectors": json.dumps(list(analysis["hosting"])),
        "wp_pure": bool(wp["pure"]),
        "wp_max_impurity": float(wp["max_impurity"]),
        "wp_seed": int(wp.get("seed", -1)),
        "quotiented_by_symmetry": summary["quotiented_by_symmetry"],
        "source": source,
    }
    validate_row(row, VORTEX_SECTORS_COLUMNS, context=f"Jz={jz}")
    return row


def report(jz, row, analysis):
    summary = analysis["summary"]
    flags = []
    if not row["wp_pure"]:
        flags.append(f"[WARN] impure W_p basis ({row['wp_max_impurity']:.1e}) -- try other --wp-seeds")
    if row["avg_Wp_spread"] > 1e-6:
        flags.append(f"[WARN] avg(W_p) varies by {row['avg_Wp_spread']:.1e} across the manifold")
    if row["n_distinct_patterns"] > 1:
        if summary["all_same_n_minus"]:
            flags.append(
                f"[OPEN, mild] {row['n_distinct_patterns']} symmetry-inequivalent "
                f"placements of the SAME vortex number N_-={summary['n_minus'][0]} "
                f"(multiplicity={summary['multiplicity']}, placements="
                f"{summary['vortex_plaquette_idx']}). The target's flux label is "
                f"well defined; only the placement class is a choice."
            )
        else:
            flags.append(
                f"[OPEN] the level mixes vortex NUMBERS "
                f"(n_minus={summary['n_minus']}, multiplicity={summary['multiplicity']}) "
                f"-- there is no single flux label here; do NOT pick one arbitrarily"
            )
    print(
        f"Jz={jz:.2f}  E0={row['E0']:+.6f}  dim={row['manifold_dim']:2d}  "
        f"gap={row['gap_manifold']:.2e}  avg(Wp)={row['avg_Wp']:+.6f}  "
        f"N_minus={summary['n_minus']}  vortices@{summary['vortex_plaquette_idx']}  "
        f"hosting={row['hosting_sectors']}"
    )
    for flag in flags:
        print(f"    {flag}")


ROW_KEY = ("Jz", "extent_x", "extent_y", "group", "tol")


def merge_with_existing(rows, out_path: Path) -> pd.DataFrame:
    """Combine this run's rows with whatever `out_path` already holds.

    The scan runs for hours and is normally done in pieces (`--jz-min` /
    `--jz-max`), so writing only the current pieces would silently delete
    every point computed by an earlier invocation -- the expensive half of the
    table, and no way to tell it had happened except by reading the file.

    New rows win on a key collision, so re-running a Jz to fix it does what it
    looks like.
    """
    fresh = pd.DataFrame(rows, columns=list(VORTEX_SECTORS_COLUMNS))
    if not out_path.is_file():
        return fresh.sort_values("Jz").reset_index(drop=True)

    previous = pd.read_csv(out_path).reindex(columns=list(VORTEX_SECTORS_COLUMNS))
    for frame in (previous, fresh):
        frame["Jz"] = frame["Jz"].astype(float).round(4)
        frame["tol"] = frame["tol"].astype(float)

    combined = pd.concat([previous, fresh], ignore_index=True)
    combined = combined.drop_duplicates(subset=list(ROW_KEY), keep="last")
    return combined.sort_values("Jz").reset_index(drop=True)


def main():
    args = parse_args()

    graph, hi = build_kitaev_lattice(extent=args.extent, pbc=not args.no_pbc)
    N = graph.n_nodes
    symmetries = get_kitaev_symmetries(graph, hi)
    sg, ct = get_projection_group(symmetries, args.group)

    plaquettes, plaquette_ops = get_kitaev_plaquettes(graph)
    wp_sparse = [w.to_sparse() for w in build_wilson_loops(hi, plaquettes, plaquette_ops)]
    plaquette_perms = (
        None if args.no_symmetry_quotient else plaquette_permutations(plaquettes, sg)
    )

    jz_values = (
        args.jz
        if args.jz is not None
        else default_jz_grid(args.jz_min, args.jz_max)
    )
    jz_values = sorted({round(float(v), 4) for v in jz_values})
    if not jz_values:
        raise SystemExit(
            f"No Jz points in [{args.jz_min}, {args.jz_max}]. Widen the range "
            f"or pass --jz explicitly."
        )

    cache = {}
    if args.exact_data:
        print(f"Loading cached eigenvectors from {args.exact_data} ...", flush=True)
        cache = load_exact_results(args.exact_data)
        print(f"  {len(cache)} cached Jz points: {sorted(cache)}", flush=True)

    print(
        f"\n{len(jz_values)} Jz points, extent={args.extent} (N={N}, "
        f"{len(plaquettes)} plaquettes), group={args.group} "
        f"(n_g={len(sg)}, {ct.shape[0]} irreps), "
        f"symmetry quotient={'off' if plaquette_perms is None else f'{len(plaquette_perms)} elements'}\n",
        flush=True,
    )

    rows = []
    for jz in jz_values:
        cached_key = next(
            (k for k in cache if abs(float(k) - jz) < args.cache_match_tol), None
        )
        energies = eigvecs = source = None

        if cached_key is not None:
            entry = cache[cached_key]
            cached_energies = np.asarray(entry["energies"]).real
            cached_eigvecs = entry["eigenvectors"]
            order = np.argsort(cached_energies)
            cached_energies, cached_eigvecs = cached_energies[order], cached_eigvecs[:, order]

            cached_manifold = degenerate_manifold(cached_energies, tol=args.tol)
            impurity = diagonalize_wp_in_manifold_robust(
                cached_eigvecs, cached_manifold, wp_sparse, seeds=tuple(args.wp_seeds)
            )["max_impurity"]
            complete = (
                len(cached_manifold) < len(cached_energies) and impurity < 1e-6
            )
            if complete:
                energies, eigvecs = cached_energies, cached_eigvecs
                source = f"cache:{Path(args.exact_data).name}"
            else:
                print(
                    f"Jz={jz:.2f}: cached level looks truncated "
                    f"(dim={len(cached_manifold)}/{len(cached_energies)}, "
                    f"impurity={impurity:.1e}); recomputing.",
                    flush=True,
                )

        if energies is None:
            jx = jy = (1.0 - jz) / 2.0
            H = KitaevTransverse_H(
                graph.edge_colors, graph.edges(), Jx=jx, Jy=jy, Jz=jz, h=0, hi=hi
            )
            print(f"Jz={jz:.2f}: running Lanczos (k={args.k_eigenvals}) ...", flush=True)
            energies, eigvecs, _ = lanczos_with_resolved_manifold(
                H, args.k_eigenvals, args.k_max, args.tol, wp_sparse, args.wp_seeds
            )
            source = "lanczos"

        analysis = analyze_point(
            energies, eigvecs, hi=hi, sg=sg, ct=ct, wp_sparse=wp_sparse,
            plaquette_perms=plaquette_perms, tol=args.tol, wp_seeds=args.wp_seeds,
        )
        row = build_row(
            jz, analysis, energies=energies, extent=args.extent, N=N,
            n_plaquettes=len(plaquettes), group=args.group, tol=args.tol,
            source=source,
        )
        rows.append(row)
        report(jz, row, analysis)

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        table = merge_with_existing(rows, out)
        table.to_csv(out, index=False)

    print(f"\n[OK] {len(rows)} row(s) computed, {len(table)} in {args.out}")
    _print_transitions(table.to_dict("records"))


def _print_transitions(rows):
    """Bracket each change of vortex number, which is the point of the scan.

    Compares the *multiset* of vortex counts, so a level that keeps the same
    number of vortices while rearranging them does not register as a
    transition -- only a change in vortex content does.
    """
    print("\n=== vortex-number transitions ===")
    found = False
    for prev, curr in zip(rows, rows[1:]):
        if sorted(json.loads(prev["n_minus"])) != sorted(json.loads(curr["n_minus"])):
            found = True
            print(
                f"  {prev['n_minus']} -> {curr['n_minus']} somewhere in "
                f"({prev['Jz']:.2f}, {curr['Jz']:.2f}]  "
                f"(width {curr['Jz'] - prev['Jz']:.2f})"
            )
    if not found:
        print("  none: the vortex content is constant over the scanned grid.")


if __name__ == "__main__":
    main()
