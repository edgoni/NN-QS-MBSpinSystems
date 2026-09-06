#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from common.physics.hamiltonian import build_kitaev_lattice, KitaevTransverse_H
from common.physics.observables import (
    get_kitaev_plaquettes,
    build_wilson_loops,
    build_magnetization_observables,
)
from common.physics.symmetries import get_kitaev_symmetries, get_projection_group
from common.physics.isotropic_symmetry import c2xy_translation_group
from common.physics.exact_diag import (
    permutation_matrices,
    load_exact_results,
    degenerate_manifold,
    manifold_irrep_weights,
    sectors_hosting_manifold,
    plaquette_permutations,
    vortex_resolved_manifold,
    pick_target_for_sector,
)

OBSERVABLE_NAMES = ("E", "Wp0", "flux_total", "m", "ms", "m2", "ms2")


def orthonormal_image(mat, tol=0.5):
    """Orthonormal basis of the column space of `mat`, plus its rank.

    Used on `P_k @ class_basis`. Because the class span is invariant under
    the group, `P_k` restricted to it is an orthogonal projector, so its
    singular values are exactly 0 or 1: `tol=0.5` is the midpoint of that
    gap, not a fitted knob.

    A small tolerance is actively WRONG here and not conservatively so. The
    class basis comes from Lanczos vectors carrying a residual, so `P_k V`
    has a tail of singular values around 1e-6..1e-3 that are pure noise
    directions. Cutting at 1e-8 kept them: at Jz=0.7 the five blocks came
    out with ranks 3+6+6+6+6 = 27 inside an 18-dimensional class, and at
    Jz=0.9 four of the five blocks came out at the full rank 18. Those
    phantom directions then widened every `block_range` -- which is exactly
    the number the "is one sector enough?" question is read off. `gap()`
    below reports the two singular values straddling the cut so the 0/1
    structure is visible rather than assumed.
    """
    u, s, _ = np.linalg.svd(mat, full_matrices=False)
    keep = s > tol
    return u[:, keep], int(keep.sum()), s


def gap(singular_values, tol=0.5):
    """(smallest kept, largest discarded) singular value around `tol`.

    A healthy projector block shows ~1.0 and ~1e-6; anything near `tol` on
    either side means the rank is not well defined and the block ranges
    below it are not either.
    """
    kept = singular_values[singular_values > tol]
    dropped = singular_values[singular_values <= tol]
    return (
        float(kept.min()) if kept.size else float("nan"),
        float(dropped.max()) if dropped.size else 0.0,
    )


def project_block(basis, perm_mats, characters, d_mu):
    """P_k applied to every column of `basis` (dim, m) at once."""
    out = np.zeros_like(basis)
    for mat, chi in zip(perm_mats, characters):
        out = out + np.conj(chi) * (mat @ basis)
    return out * (d_mu / len(perm_mats))


def block_range(op_sparse, basis):
    """(min, max, mean) of <psi|O|psi> over all normalized psi in the span.

    The extrema of a Rayleigh quotient over a subspace are the extreme
    eigenvalues of the compression B^dag O B, so this is exact, not a sample.
    """
    if basis.shape[1] == 0:
        return float("nan"), float("nan"), float("nan")
    compressed = basis.conj().T @ (op_sparse @ basis)
    compressed = 0.5 * (compressed + compressed.conj().T)
    evals = np.linalg.eigvalsh(compressed)
    return float(evals[0]), float(evals[-1]), float(np.mean(evals))


def expect(op_sparse, psi):
    return float(np.real(np.vdot(psi, op_sparse @ psi)))


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--extent", type=int, nargs=2, default=[3, 3])
    p.add_argument("--no-pbc", action="store_true")
    p.add_argument("--exact-data", type=str,
                   default="data/raw/energies_eigenvecs_dict_k40.npz")
    p.add_argument("--jz", type=float, nargs="+", default=None,
                   help="Jz values to analyze (default: every one in the .npz).")
    p.add_argument("--group", choices=["translation", "space"], default="space",
                   help="Same meaning as in run_vmc.py; 'space' is that script's default.")
    p.add_argument("--vortex-class", type=int, default=0,
                   help="Which vortex class to build targets in (run_vmc.py default: 0).")
    p.add_argument("--tol", type=float, default=1e-6,
                   help="Spread below this counts as 'sector-independent' in the verdict.")
    p.add_argument("--rank-tol", type=float, default=0.5,
                   help="SVD cut for the rank of an isotypic block. The exact "
                        "singular values of a projector are 0 or 1, so 0.5 is "
                        "the midpoint; lowering it admits Lanczos-residual "
                        "noise directions that inflate every block range.")
    p.add_argument("--out", type=str, default="data/results/sector_observables.csv")
    return p.parse_args()


def main():
    args = parse_args()
    graph, hi = build_kitaev_lattice(extent=args.extent, pbc=not args.no_pbc)
    N = graph.n_nodes
    symmetries = get_kitaev_symmetries(graph, hi)
    sector_sg, sector_ct = get_projection_group(symmetries, args.group)
    perm_mats = permutation_matrices(hi, sector_sg)

    plaquettes, plaquette_ops = get_kitaev_plaquettes(graph)
    wilson_loops = build_wilson_loops(hi, plaquettes, plaquette_ops)
    wilson_sparse = [w.to_sparse() for w in wilson_loops]
    flux_total = wilson_sparse[0]
    for w in wilson_sparse[1:]:
        flux_total = flux_total + w
    staggered_mag, uniform_mag = build_magnetization_observables(hi, N)
    m_sp, ms_sp = uniform_mag.to_sparse(), staggered_mag.to_sparse()

    plaquette_perms = plaquette_permutations(plaquettes, sector_sg) + plaquette_permutations(
        plaquettes, c2xy_translation_group(graph)
    )

    exact_results = load_exact_results(args.exact_data)
    jz_values = sorted(exact_results) if args.jz is None else [
        min(exact_results, key=lambda kk: abs(kk - v)) for v in args.jz
    ]

    rows = []
    skipped = []
    for jz in jz_values:
        result = exact_results[jz]
        eigvecs = result["eigenvectors"]
        manifold_idx = degenerate_manifold(np.asarray(result["energies"]).real)
        weights = manifold_irrep_weights(eigvecs, manifold_idx, hi, sector_sg, sector_ct)
        hosting = sectors_hosting_manifold(weights)

        jx = jy = (1 - jz) / 2
        H = KitaevTransverse_H(graph.edge_colors, graph.edges(), Jx=jx, Jy=jy,
                               Jz=jz, h=0, hi=hi)
        H_sp = H.to_sparse()

        resolved = vortex_resolved_manifold(
            eigvecs, manifold_idx, wilson_sparse, plaquette_perms
        )
        classes = resolved["classes"]
        chosen = classes[args.vortex_class]
        class_basis, class_rank, _ = orthonormal_image(chosen["vectors"], args.rank_tol)

        print(
            f"\n=== Jz={jz:.2f} | manifold={len(manifold_idx)} | hosting={hosting} | "
            f"vortex classes={len(classes)} N_-={[c['n_minus'] for c in classes]} | "
            f"class {args.vortex_class}: N_-={chosen['n_minus']}, dim={class_rank}, "
            f"{chosen['n_placements']} placement(s) ==="
        )

        if not resolved["pure"]:
            print(
                f"  [SKIP] the W_p basis of this manifold is impure "
                f"({resolved['diagonalization']['max_impurity']:.1e}): the level is "
                f"truncated at k={eigvecs.shape[1]} stored eigenpairs. Nothing "
                f"measured here would be trustworthy; rerun the ED with a larger k."
            )
            skipped.append((jz, "truncated manifold (impure W_p basis)"))
            del H, H_sp
            continue

        observables = {
            "E": H_sp,
            "Wp0": wilson_sparse[0],
            "flux_total": flux_total,
            "m": m_sp,
            "ms": ms_sp,
            "m2": m_sp @ m_sp,
            "ms2": ms_sp @ ms_sp,
        }

        ref = {name: block_range(op, class_basis) for name, op in observables.items()}
        rows.append({
            "Jz": jz, "group": args.group, "sector": "class", "rank": class_rank,
            "proj_norm": np.nan, "n_minus": chosen["n_minus"],
            **{f"{n}_target": np.nan for n in observables},
            **{f"{n}_{stat}": ref[n][i]
               for n in observables for i, stat in enumerate(("min", "max", "mean"))},
        })

        blocks_k = {}
        for k_all in range(sector_ct.shape[0]):
            d_all = float(np.real(sector_ct[k_all, 0]))
            projected = project_block(class_basis, perm_mats, sector_ct[k_all], d_all)
            blocks_k[k_all] = orthonormal_image(projected, args.rank_tol)
        rank_budget = sum(r for _, r, _ in blocks_k.values())
        if rank_budget != class_rank:
            print(
                f"  [!] block ranks over all {sector_ct.shape[0]} irreps sum to "
                f"{rank_budget} but the class is {class_rank}-dimensional. "
                f"sum_k P_k = 1 on an invariant subspace, so they must agree: "
                f"either --rank-tol is cutting through a real singular value, "
                f"or the class span is not closed under the group."
            )

        for k in hosting:
            d_mu = float(np.real(sector_ct[k, 0]))
            basis_k, rank_k, sv_k = blocks_k[k]
            kept_min, dropped_max = gap(sv_k, args.rank_tol)

            try:
                target = pick_target_for_sector(
                    eigvecs, manifold_idx, hi, sector_sg, sector_ct, k,
                    wilson_loops=wilson_sparse, plaquette_perms=plaquette_perms,
                    class_index=args.vortex_class, verbose=False,
                )
            except ValueError as exc:
                print(f"  k={k:<2d} [SKIP] {exc}")
                skipped.append((jz, f"k={k}: {str(exc).splitlines()[0]}"))
                continue
            g_k = target["vector"]

            wp_all = [expect(w, g_k) for w in wilson_sparse]

            row = {
                "Jz": jz, "group": args.group, "sector": k, "rank": rank_k,
                "proj_norm": target["norm"], "n_minus": target["n_minus"],
                "sv_kept_min": kept_min, "sv_dropped_max": dropped_max,
                "wp_uniform": bool(np.ptp(wp_all) < 1e-8),
                "wp_spread_over_plaquettes": float(np.ptp(wp_all)),
            }
            for name, op in observables.items():
                lo, hi_, mean = block_range(op, basis_k)
                row[f"{name}_target"] = expect(op, g_k)
                row[f"{name}_min"], row[f"{name}_max"], row[f"{name}_mean"] = lo, hi_, mean
            rows.append(row)

            print(
                f"  k={k:<2d} d={round(d_mu)} rank={rank_k:<2d} |P_k psi|={target['norm']:.3f}  "
                f"E={row['E_target']:+.9f}  Wp0={row['Wp0_target']:+.6f}  "
                f"sumWp={row['flux_total_target']:+.6f}  m={row['m_target']:+.2e}  "
                f"ms={row['ms_target']:+.2e}  m2={row['m2_target']:.6f}  "
                f"ms2={row['ms2_target']:.6f}"
            )
            print(
                f"       <W_p> per plaquette: "
                f"[{', '.join(f'{v:+.3f}' for v in wp_all)}]"
                + ("  (uniform)" if row["wp_uniform"] else
                   f"  (spread {row['wp_spread_over_plaquettes']:.3f} over p)")
            )
            if rank_k > 1:
                print(
                    f"       block rank {rank_k} (sv gap {dropped_max:.1e} | "
                    f"{kept_min:.3f}): <m2> in [{row['m2_min']:.6f}, "
                    f"{row['m2_max']:.6f}], <ms2> in [{row['ms2_min']:.6f}, "
                    f"{row['ms2_max']:.6f}] inside this same sector"
                )

        del H, H_sp

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")

    print("\n=== spread of the target observables ACROSS sectors ===")
    targets = df[df["sector"] != "class"]
    for jz, grp in targets.groupby("Jz"):
        spreads = {
            name: float(np.ptp(grp[f"{name}_target"].to_numpy(dtype=float)))
            for name in OBSERVABLE_NAMES
        }
        disagree = [n for n, v in spreads.items() if v >= args.tol]
        verdict = "IDENTICAL" if not disagree else f"DIFFERENT in {','.join(disagree)}"
        parts = "  ".join(f"{n}:{v:.2e}" for n, v in spreads.items())
        print(f"  Jz={jz:.2f} n_sectors={len(grp)}  {parts}   -> {verdict}")

    print("\n=== is the sector label enough to FIX the observable? ===")
    for jz, grp in targets.groupby("Jz"):
        for name in OBSERVABLE_NAMES:
            widths = (grp[f"{name}_max"] - grp[f"{name}_min"]).to_numpy(dtype=float)
            worst = float(np.nanmax(widths))
            if worst >= args.tol:
                k_worst = grp["sector"].to_numpy()[int(np.nanargmax(widths))]
                print(
                    f"  Jz={jz:.2f} {name}: undetermined inside a sector "
                    f"(widest block range {worst:.3e} at k={k_worst}) -- the "
                    f"target's value there is an artifact of which vector "
                    f"pick_source_for_sector happened to pick."
                )

    if skipped:
        print("\n=== not analyzed ===")
        for jz, why in skipped:
            print(f"  Jz={jz:.2f}: {why}")


if __name__ == "__main__":
    main()
