#!/usr/bin/env python
import argparse
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
    _orthonormal_manifold_basis,
)
from common.physics.isotropic_symmetry import (
    apply_site_permutation,
    c2v_translation_group,
    c2v_character_table,
    c3_translation_group,
    c3_character_table,
    rotate_state_to_frame,
)

GROUP_FRAME = {
    "c2v": {"axis": (1.0, 1.0, 0.0), "root_order": 2},
    "c3": {"axis": (1.0, 1.0, 1.0), "root_order": 3},
}

CEILING_COLUMNS = [
    "Jz", "Jx", "Jy", "extent_x", "extent_y", "N",
    "group", "group_order", "irrep", "irrep_dim",
    "manifold_dim", "weight", "rank", "F_max",
    "E0", "gap", "threshold_E", "n_eigenvalues", "source",
]


def group_elements(graph, hi, group_name):
    """`(perms, powers, character_table)` for a named projection group.

    `powers` is None for the permutational groups, and the per-element spin
    power for the monomial ones.
    """
    if group_name == "c2v":
        perms, powers = c2v_translation_group(graph)
        _, table = c2v_character_table(graph, perms)
        return np.asarray(perms), np.asarray(powers), table
    if group_name == "c3":
        perms, powers = c3_translation_group(graph)
        _, table = c3_character_table(graph, perms)
        return np.asarray(perms), np.asarray(powers), table

    symmetries = get_kitaev_symmetries(graph, hi)
    sg, table = get_projection_group(symmetries, group_name)
    return np.asarray(sg), None, table


def configuration_phases(n_sites, root_order):
    """`phases[k][basis index] = omega**(k * n_minus)`, precomputed per power.

    `n_minus` counts sites in the second local state, which is what the spin
    factor is diagonal in once the frame rotation has been applied.
    """
    dim = 2 ** n_sites
    bits = (np.arange(dim)[:, None] >> np.arange(n_sites)[::-1]) & 1
    n_minus = bits.sum(axis=1)
    omega = np.exp(2j * np.pi / root_order)
    return [omega ** ((k * n_minus) % root_order) for k in range(root_order)]


def group_gram_matrices(basis, perms, powers, phases):
    """`A_g = B^dagger R_g B` for every group element, shape (|G|, m, m).

    One permutation (and one elementwise multiply, when the element carries a
    phase) per group element per manifold vector -- the whole cost of this
    script.
    """
    n_g, m = len(perms), basis.shape[1]
    grams = np.empty((n_g, m, m), dtype=complex)
    for g, perm in enumerate(perms):
        rotated = np.empty_like(basis)
        for j in range(m):
            column = basis[:, j]
            if powers is not None:
                column = phases[int(powers[g])] * column
            rotated[:, j] = apply_site_permutation(column, perm)
        grams[g] = basis.conj().T @ rotated
    return grams


def sector_ceilings(grams, character_table, rank_tol=1e-8):
    """Weight, rank and ceiling of every irrep, from the `A_g` matrices."""
    n_g = grams.shape[0]
    rows = []
    for mu in range(character_table.shape[0]):
        dim_mu = float(np.real(character_table[mu, 0]))
        projected = (dim_mu / n_g) * np.einsum(
            "g,gij->ij", np.conj(character_table[mu, :]), grams
        )
        projected = 0.5 * (projected + projected.conj().T)
        eigenvalues = np.linalg.eigvalsh(projected)
        rows.append({
            "irrep": mu,
            "irrep_dim": int(round(dim_mu)),
            "weight": float(np.real(np.trace(projected))),
            "rank": int((eigenvalues > rank_tol).sum()),
            "F_max": float(min(1.0, max(0.0, eigenvalues.max()))),
        })
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--extent", type=int, nargs=2, default=[3, 3])
    parser.add_argument("--no-pbc", action="store_true")
    parser.add_argument(
        "--groups", nargs="+", default=["translation", "space", "c2v"],
        help="Projection groups to tabulate. 'c3' is only a symmetry at the "
             "isotropic point and is skipped at every other Jz.",
    )
    parser.add_argument(
        "--exact-data", type=str,
        default="data/raw/energies_eigenvecs_dict_k40.npz",
    )
    parser.add_argument(
        "--out", type=str, default="data/results/sector_ceiling_by_jz.csv"
    )
    parser.add_argument(
        "--tol", type=float, default=1e-6,
        help="Degeneracy tolerance for `degenerate_manifold`.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    graph, hi = build_kitaev_lattice(extent=args.extent, pbc=not args.no_pbc)
    n_sites = graph.n_nodes
    lx, ly = args.extent

    exact = load_exact_results(args.exact_data)
    print(f"{args.exact_data}: {len(exact)} puntos de Jz, N={n_sites}", flush=True)

    prepared = {}
    for name in args.groups:
        perms, powers, table = group_elements(graph, hi, name)
        frame = GROUP_FRAME.get(name)
        phases = (
            configuration_phases(n_sites, frame["root_order"])
            if frame is not None else None
        )
        prepared[name] = (perms, powers, table, frame, phases)
        dims = np.round(np.real(table[:, 0])).astype(int)
        print(f"  grupo '{name}': |G|={len(perms)}  irreps={table.shape[0]}  "
              f"dims={dims.tolist()}", flush=True)

    rows = []
    for jz in sorted(exact):
        result = exact[jz]
        energies = np.asarray(result["energies"]).real
        eigvecs = np.asarray(result["eigenvectors"])

        order = np.argsort(energies)
        energies, eigvecs = energies[order], eigvecs[:, order]

        manifold_idx = degenerate_manifold(energies, tol=args.tol)
        m = len(manifold_idx)
        if manifold_idx != list(range(m)):
            raise RuntimeError(
                f"Jz={jz}: tras ordenar, el manifold sigue sin ser un prefijo "
                f"contiguo ({manifold_idx[:5]}...). Eso solo puede pasar si "
                f"`degenerate_manifold` cambio de semantica; el gap calculado "
                f"abajo seria erroneo."
            )
        gap = (
            float(energies[m] - energies[0]) if m < len(energies) else float("nan")
        )
        jx = jy = (1.0 - float(jz)) / 2.0
        isotropic = abs(jx - jz) < 1e-9

        basis_computational = _orthonormal_manifold_basis(eigvecs, manifold_idx)
        print(f"\nJz={jz:.4f} (Jx=Jy={jx:.4f}): E0={energies[0]:.6f} "
              f"manifold={m} gap={gap:.3e}", flush=True)
        if m == len(energies):
            print("  [!] el manifold llena el espectro almacenado: esta "
                  "truncado y estas cifras son cotas inferiores", flush=True)

        for name, (perms, powers, table, frame, phases) in prepared.items():
            if name == "c3" and not isotropic:
                print(f"  grupo '{name}': omitido, solo es simetria en el "
                      f"punto isotropico (Jz=1/3)", flush=True)
                continue

            if frame is None:
                basis = basis_computational
            else:
                basis = np.stack(
                    [rotate_state_to_frame(basis_computational[:, j], axis=frame["axis"])
                     for j in range(basis_computational.shape[1])],
                    axis=1,
                )

            grams = group_gram_matrices(basis, perms, powers, phases)
            for entry in sector_ceilings(grams, table):
                rows.append({
                    "Jz": float(jz), "Jx": jx, "Jy": jy,
                    "extent_x": lx, "extent_y": ly, "N": n_sites,
                    "group": name, "group_order": len(perms),
                    "manifold_dim": m,
                    "E0": float(energies[0]), "gap": gap,
                    "threshold_E": float(energies[0]) + gap,
                    "n_eigenvalues": len(energies),
                    "source": Path(args.exact_data).name,
                    **entry,
                })
            best = max(
                (r for r in rows if r["Jz"] == float(jz) and r["group"] == name),
                key=lambda r: r["F_max"],
            )
            hosting = [
                r["irrep"] for r in rows
                if r["Jz"] == float(jz) and r["group"] == name and r["F_max"] > 0.5
            ]
            print(f"  grupo '{name}': mejor irrep {best['irrep']} "
                  f"(d={best['irrep_dim']}) F_max={best['F_max']:.6f}; "
                  f"sectores con F_max>0.5: {hosting}", flush=True)

    frame = pd.DataFrame(rows, columns=CEILING_COLUMNS)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"\n{len(frame)} filas -> {out}", flush=True)


if __name__ == "__main__":
    main()
