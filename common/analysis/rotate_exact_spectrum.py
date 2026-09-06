#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from common.physics.exact_diag import load_exact_results
from common.physics.isotropic_symmetry import rotate_state_to_frame

_AXIS = {"c3": (1.0, 1.0, 1.0), "c2v": (1.0, 1.0, 0.0)}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--group", choices=sorted(_AXIS), required=True,
                    help="grupo cuyo frame se usa para rotar")
    p.add_argument("--jz", type=float, nargs="*", default=None,
                    help="Jz a rotar (deben existir en --in-path). "
                         "Por defecto, TODOS los del cache de entrada.")
    p.add_argument("--in-path", default="data/raw/energies_eigenvecs_dict_k40.npz",
                    help="cache de entrada, frame COMPUTACIONAL")
    p.add_argument("--out-path", default=None,
                    help="por defecto, deriva el nombre del --in-path y --group")
    return p.parse_args()


def main():
    args = parse_args()
    axis = _AXIS[args.group]

    in_path = Path(args.in_path)
    out_path = Path(args.out_path) if args.out_path else (
        in_path.with_name(f"{in_path.stem}_{args.group}_rotated.npz")
    )

    raw = load_exact_results(str(in_path))
    available = sorted(float(k) for k in raw)

    if args.jz is None:
        targets = available
    else:
        targets = []
        for jz in args.jz:
            match = min(available, key=lambda a: abs(a - jz))
            if abs(match - jz) > 1e-6:
                print(f"[aviso] jz={jz} no esta en {in_path} "
                      f"(disponibles: {available}) -> se omite")
                continue
            targets.append(match)

    out = {}
    for jz in targets:
        key = next(k for k in raw if abs(float(k) - jz) < 1e-9)
        entry = raw[key]
        energies = np.asarray(entry["energies"])
        evecs = np.asarray(entry["eigenvectors"])

        rotated = np.stack(
            [rotate_state_to_frame(evecs[:, j], axis=axis)
             for j in range(evecs.shape[1])],
            axis=1,
        )
        out[float(jz)] = {
            "energies": energies,
            "eigenvectors": rotated,
            "E0": float(entry.get("E0", energies[0])),
        }
        print(f"Jz={jz:.6f}: {evecs.shape[1]} autovectores rotados a "
              f"frame '{args.group}' (axis={axis}), E0={energies[0]:.6f}")

    if not out:
        print("Nada que guardar (ningun --jz encontrado en el cache de entrada).")
        return

    np.savez(out_path, data_dict=out)
    print(f"\nGuardado: {out_path}  ({len(out)} puntos de Jz)")


if __name__ == "__main__":
    main()
