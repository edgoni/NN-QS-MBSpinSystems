#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load_measure_observables():
    """`measure_observables.py` como modulo.

    Via importlib porque el bloque no es un paquete (no tiene `__init__.py`),
    asi que un `import measure_observables` normal no lo encuentra desde aqui.
    """
    path = Path(__file__).resolve().parent / "measure_observables.py"
    if not path.is_file():
        raise SystemExit(f"No encuentro {path}. Se ha movido el repo?")
    spec = importlib.util.spec_from_file_location("_measure_observables", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BLOCKS = [
    ("IDENTIDAD", ["tag", "ansatz", "layers", "extent_x", "extent_y", "N",
                   "Jz", "sector", "group", "seed", "iteration", "n_params",
                   "dtype", "phase"]),
    ("FLUJOS DE PLAQUETA", ["W_mean", "W_sum", "W_min", "W_max", "W_spread",
                            "W_uniform", "flux_pattern", "n_minus",
                            "n_plaquettes"]),
    ("ENERGIA", ["E", "E_per_site", "Var_E", "vscore"]),
    ("MAGNETIZACION", ["Mx", "My", "Mz", "M_total", "m", "ms", "fluct",
                       "fluct_s"]),
    ("CORRELACIONES", ["Szz", "corr_xx_bond", "corr_yy_bond", "corr_zz_bond"]),
    ("COHERENCIA", ["Cl1", "Cl1_normalized"]),
    ("COMPARACION CON ED", ["E_ED", "delta_eps", "manifold_dim", "overlap_gs",
                            "fidelity_manifold", "fidelity_sector",
                            "infidelity", "target_n_minus"]),
]


def _fmt(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if isinstance(value, bool):
        return "si" if value else "no"
    if isinstance(value, float):
        if value != 0 and (abs(value) < 1e-4 or abs(value) >= 1e6):
            return f"{value:.6e}"
        return f"{value:.10f}".rstrip("0").rstrip(".")
    return str(value)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("checkpoint", help="Ruta a un .pkl o .mpack.")
    p.add_argument("--csv", type=str, default=None,
                   help="Vuelca la fila completa a este CSV (lo sobrescribe).")
    p.add_argument("--json", type=str, default=None,
                   help="Vuelca todo a este JSON.")
    p.add_argument("--ansatz", choices=["rbm", "factored", "transformer", "vit"],
                   default=None, help="Por defecto, el del nombre del fichero.")
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--alpha", type=float, default=2.0,
                   help="Densidad de ocultas de la RBM (defecto 2).")
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--dk", type=int, default=8)
    p.add_argument("--extent", type=int, nargs=2, default=None,
                   help="Por defecto, el del nombre, o 3 3.")
    p.add_argument("--no-pbc", action="store_true")
    p.add_argument("--jz", type=float, default=None,
                   help="Por defecto, el del nombre del fichero.")
    p.add_argument("--sector", type=int, default=None,
                   help="Irrep de la proyeccion; -1 = sin proyectar. Un valor "
                        "equivocado NO da error, mide otro estado.")
    p.add_argument("--vortex-class", type=int, default=None)
    p.add_argument("--group", choices=["translation", "space"], default="space")
    p.add_argument("--vortex-group", choices=["space+mirror", "space"],
                   default="space+mirror")
    p.add_argument("--projector", choices=["stable", "legacy"], default="stable")
    p.add_argument("--group-chunk-size", type=int, default=None)
    p.add_argument("--exact-data", type=str,
                   default="data/raw/energies_eigenvecs_dict_k40.npz",
                   help="Espectro de ED para el bloque de comparacion. Si no "
                        "existe o es de otra red, ese bloque sale vacio y el "
                        "resto de medidas sigue siendo valido.")
    p.add_argument("--max-dim", type=int, default=300000,
                   help="Tope del espacio de Hilbert; estas medidas son "
                        "exactas sobre el vector denso.")
    p.add_argument("--no-target", action="store_true",
                   help="Salta F_sector/infidelidad (es la parte cara).")
    return p.parse_args()


def main():
    args = parse_args()
    mo = _load_measure_observables()
    import jax
    import netket as nk

    path = Path(args.checkpoint)
    if not path.is_file():
        raise SystemExit(f"No existe: {path}")

    info = mo.parse_tag(str(path))
    spec = mo.resolve_spec(info, args)
    lat = mo.Lattice(spec.extent, not args.no_pbc, args.max_dim,
                     args.vortex_group)

    perms = characters = None
    if spec.sector >= 0:
        perms, characters = mo.symmetry_projector_inputs(
            lat.symmetries, spec.sector, group=args.group
        )
    model = mo.build_model(spec, lat.graph, symmetries=perms,
                           characters=characters)
    vstate = nk.vqs.MCState(
        nk.sampler.MetropolisLocal(lat.hi), model, n_samples=16, seed=0
    )
    params = mo.load_params(str(path))
    mo.check_params_match(params, vstate.parameters, context=path.name)
    vstate.parameters = params

    psi = np.asarray(vstate.to_array())
    norm = np.linalg.norm(psi)
    if not np.isfinite(norm) or norm == 0:
        raise SystemExit(f"El estado no es normalizable (norma={norm}).")
    psi = psi / norm

    exact = None
    try:
        exact_all = mo.load_exact_results(args.exact_data)
    except FileNotFoundError:
        print(f"[!] {args.exact_data} no existe: sin comparacion con ED.\n")
    else:
        jz_key = min(exact_all, key=lambda k: abs(k - spec.Jz))
        candidate = exact_all[jz_key]
        ed_dim = int(np.asarray(candidate["eigenvectors"]).shape[0])
        if ed_dim != lat.dim:
            print(f"[!] El espectro de ED es de dimension {ed_dim} y esta red "
                  f"tiene {lat.dim}: sin comparacion con ED.\n")
        else:
            exact = candidate

    jx = jy = (1 - spec.Jz) / 2
    H = mo.KitaevTransverse_H(lat.graph.edge_colors, lat.graph.edges(),
                              Jx=jx, Jy=jy, Jz=spec.Jz, h=0,
                              hi=lat.hi).to_sparse()

    row = {
        "tag": info["tag"], "phase": info["phase"], "ansatz": spec.ansatz,
        "layers": spec.layers, "extent_x": spec.extent[0],
        "extent_y": spec.extent[1], "N": lat.N, "Jz": spec.Jz,
        "sector": spec.sector, "group": args.group, "seed": info.get("seed"),
        "iteration": info.get("iteration"),
        "n_params": int(nk.jax.tree_size(vstate.parameters)),
        "dtype": "/".join(sorted({
            np.dtype(leaf.dtype).name
            for leaf in jax.tree.leaves(vstate.parameters)
        })),
        "file": str(path),
    }
    row.update(mo.measure(psi, spec, lat, H, exact))

    row["fidelity_sector"] = row["infidelity"] = row["target_n_minus"] = None
    if exact is not None and spec.sector >= 0 and not args.no_target:
        try:
            target = mo.pick_target_for_sector(
                exact["eigenvectors"],
                mo.degenerate_manifold(np.asarray(exact["energies"]).real),
                lat.hi, *mo._projection_group(lat, args.group), spec.sector,
                wilson_loops=lat.wilson,
                plaquette_perms=lat.plaquette_perms,
                class_index=spec.vortex_class, verbose=False,
            )
            fidelity = float(np.abs(np.vdot(target["vector"], psi)) ** 2)
            row["fidelity_sector"] = fidelity
            row["infidelity"] = 1.0 - fidelity
            row["target_n_minus"] = target["n_minus"]
        except (ValueError, IndexError) as exc:
            print(f"[!] sin target de fase 2: {str(exc).splitlines()[0]}\n")

    print("=" * 72)
    print(f"  {path.name}")
    print("=" * 72)

    n_plaq = int(row.get("n_plaquettes") or 0)
    for title, cols in BLOCKS:
        if title == "FLUJOS DE PLAQUETA":
            cols = [f"W_{i}" for i in range(n_plaq)] + cols
        present = [c for c in cols if c in row]
        if not present:
            continue
        print(f"\n-- {title} " + "-" * max(0, 68 - len(title)))
        width = max(len(c) for c in present)
        for c in present:
            print(f"   {c:<{width}}  {_fmt(row[c])}")

    claimed = {c for _, cols in BLOCKS for c in cols}
    claimed |= {f"W_{i}" for i in range(n_plaq)} | {"W_all", "file"}
    rest = [c for c in row if c not in claimed]
    if rest:
        print("\n-- OTRAS " + "-" * 63)
        width = max(len(c) for c in rest)
        for c in rest:
            print(f"   {c:<{width}}  {_fmt(row[c])}")

    if args.csv:
        import pandas as pd
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(out, index=False)
        print(f"\n-> {out}")
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
        print(f"\n-> {out}")


if __name__ == "__main__":
    main()
