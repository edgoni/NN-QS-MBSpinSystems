#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from common.physics.exact_diag import run_exact_diagonalization


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extent", type=int, nargs=2, default=[3, 3])
    parser.add_argument("--jz-steps", type=int, default=11)
    parser.add_argument("--k-eigenvals", type=int, default=1)
    parser.add_argument("--save-path", type=str, default="data/raw/energies_eigenvecs_dict.npz")
    parser.add_argument(
        "--group", choices=["translation", "space"], default="translation",
        help="Symmetry group whose irreps label the sectors. Must match what "
             "`run_vmc.py --group` uses. See src.physics.symmetries.get_projection_group.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_exact_diagonalization(
        extent=args.extent,
        jz_steps=args.jz_steps,
        k_eigenvals=args.k_eigenvals,
        save_path=args.save_path,
        save_debug_json=True,
        sector_group=args.group,
    )


if __name__ == "__main__":
    main()
