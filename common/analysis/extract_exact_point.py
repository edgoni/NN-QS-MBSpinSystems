#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from common.physics.exact_diag import load_exact_results, degenerate_manifold  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jz", type=float, nargs="+", required=True)
    ap.add_argument("--exact-data", default="data/raw/energies_eigenvecs_dict_k40.npz")
    ap.add_argument("--out-dir", default="data/raw")
    ap.add_argument("--keep", type=int, default=None,
                    help="Recortar a los `keep` autovalores mas bajos. Por "
                         "defecto se guardan todos los que haya.")
    args = ap.parse_args()

    print(f"cargando {args.exact_data} (1.8 GB, tarda)...", flush=True)
    grid = load_exact_results(args.exact_data)

    for jz_req in args.jz:
        jz = min(grid, key=lambda z: abs(z - jz_req))
        if abs(jz - jz_req) > 1e-6:
            print(f"[!] jz={jz_req} no esta en la rejilla; el mas cercano es "
                  f"{jz}. Saltado.")
            continue
        entry = dict(grid[jz])
        energies = np.asarray(entry["energies"]).real
        vecs = np.asarray(entry["eigenvectors"])
        if args.keep is not None and args.keep < len(energies):
            order = np.argsort(energies)[:args.keep]
            entry["energies"] = energies[order]
            entry["eigenvectors"] = vecs[:, order]
            energies, vecs = entry["energies"], entry["eigenvectors"]

        idx = degenerate_manifold(np.sort(energies))
        gap = (float(np.sort(energies)[len(idx)] - np.sort(energies)[0])
               if len(idx) < len(energies) else float("nan"))
        out = Path(args.out_dir) / f"ed_3x3_jz{jz:.2f}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, data_dict={jz: entry})

        mb = out.stat().st_size / 1024 ** 2
        print(f"\n== Jz={jz:.2f}  (jx=jy={(1-jz)/2:.4f}) -> {out}  [{mb:.0f} MB]")
        print(f"   E0={np.min(energies):.10f}  k guardados={len(energies)}  "
              f"manifold={len(idx)}  gap={gap:.4e}")
        if "hosting_sectors" in entry:
            print(f"   hosting_sectors = {entry['hosting_sectors']}")
        if "manifold_irrep_weights" in entry:
            w = {k: round(float(v), 4) for k, v in entry["manifold_irrep_weights"].items()}
            print(f"   peso del manifold por irrep = {w}")
            uni = [k for k, v in w.items() if abs(v - 1.0) < 1e-2]
            print(f"   irreps que lo hospedan con dimension 1 (el proyector "
                  f"nombra un ESTADO): {uni if uni else 'ninguno'}")


if __name__ == "__main__":
    main()
