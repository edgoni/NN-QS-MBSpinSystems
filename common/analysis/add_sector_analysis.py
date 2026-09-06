#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from common.physics.hamiltonian import build_kitaev_lattice
from common.physics.symmetries import get_kitaev_symmetries, get_projection_group
from common.physics.exact_diag import include_sector_analysis


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-path", type=str, default="data/raw/energies_eigenvecs_dict_k30.npz",
                         help="Path to the existing energies/eigenvectors .npz file")
    parser.add_argument("--extent", type=int, nargs=2, default=[3, 3],
                         help="Lattice extent used to originally generate the .npz file")
    parser.add_argument("--save-path", type=str, default=None,
                         help="Where to save the enriched dict (defaults to overwriting --npz-path)")
    parser.add_argument("--group", type=str, choices=["translation", "space"], default="translation",
                         help="Symmetry group to label sectors with; see "
                              "src.physics.symmetries.get_projection_group (default: translation, "
                              "recommended -- all irreps 1-dimensional; 'space' has 2D irreps at 3x3+ "
                              "and is kept only for comparison experiments)")
    return parser.parse_args()


def main():
    args = parse_args()

    graph, hilbert = build_kitaev_lattice(extent=tuple(args.extent), pbc=True)
    symmetries = get_kitaev_symmetries(graph, hilbert)
    sg, character_table = get_projection_group(symmetries, args.group)

    npz = include_sector_analysis(
        npz_path=args.npz_path,
        hi=hilbert,
        sg=sg,
        character_table=character_table,
        save_path=args.save_path,
    )

    for jz_key, entry in sorted(npz.items()):
        print(
            f"Jz = {jz_key:.4f} | degenerate_idx = {entry['degenerate_idx']} "
            f"| hosting_sectors = {entry['hosting_sectors']} "
            f"| manifold_irrep_weights = {entry['manifold_irrep_weights']}"
        )


if __name__ == "__main__":
    main()
