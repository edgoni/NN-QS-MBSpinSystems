#!/usr/bin/env python
import argparse
import hashlib
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import importlib.util

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "common" / "models"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import jax.numpy as jnp                                            # noqa: E402
import netket as nk                                                # noqa: E402
from netket.utils import HashableArray                             # noqa: E402

from model_RBM import DeepRBM, SymmExpSumChunked, MonomialSymmExpSum  # noqa: E402
from transformer import FactoredSelfAttention                      # noqa: E402

from common.physics.hamiltonian import KitaevTransverse_H             # noqa: E402
from common.physics.symmetries import get_kitaev_symmetries, get_projection_group  # noqa: E402
from common.physics.isotropic_symmetry import (                       # noqa: E402
    c3_translation_group, c3_character_table,
    c2v_translation_group, c2v_character_table,
    rotate_state_to_frame,
)
from common.physics.exact_diag import (                               # noqa: E402
    load_exact_results, degenerate_manifold, manifold_fidelity,
)
from common.utils.io import append_observables_csv                    # noqa: E402

import importlib.util                                              # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "mo", ROOT / "Supervised_Infid_min" / "measure_observables.py")
mo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mo)

GROUP_AXIS = {"c3": (1.0, 1.0, 1.0), "c2v": (1.0, 1.0, 0.0)}
ROOT_ORDER = {"c3": 3, "c2v": 2}

TAG_RE = re.compile(
    r"jx(?P<jx>[\d.]+)_jy(?P<jy>[\d.]+)_jz(?P<jz>[\d.]+)_"
    r"(?P<group>space|c3|c2v|translation|none)_(?P<ansatz>rbm|factored)_k(?P<k>\d+)")
TRY_RE = re.compile(r"(?i)^\d*try[ _]?\d*$")


from common.utils.naming import short_variants  # noqa: E402


def run_variant(path):
    """Run directory a checkpoint belongs to.

    Mirrors `plot_try_norman_curves.run_variant`: the filename suffix names
    the physics, not the experiment, and two directories can hold two attempts
    at the same point (`standard_..._jz06_..._rbm_k0` and
    `new_standard_..._jz06_..._rbm_k0`). Without this they collapsed onto one
    key and `setdefault` kept whichever the sort reached first.
    """
    run_dir = path.parent.parent if TRY_RE.match(path.parent.name) else path.parent
    return run_dir.name


def collect(base):
    """{(ansatz, group, jz, k, variant): {stage: {"bestF"|"bestE": path}}}.

    Same reasoning as `plot_try_norman_curves.collect`: the filename SUFFIX is
    the only trustworthy label (the directory names and `out_prefix` are
    recycled by hand), byte-identical `Copia de ...` duplicates are dropped by
    hash, and a `try*/` subdirectory is a later re-optimization.

    `_bestE(1).pkl` is classified by its hash-mates, not its stem: that file is
    a copy of the best-ENERGY checkpoint even though the trailing `(1)` stops
    it from ending in `_bestE`.
    """
    runs = defaultdict(lambda: defaultdict(dict))
    seen = defaultdict(set)
    for path in sorted(base.rglob("*.pkl")):
        tag = TAG_RE.search(path.name)
        if not tag:
            continue
        g = tag.groupdict()
        key = (g["ansatz"], g["group"], float(g["jz"]), int(g["k"]),
               run_variant(path))
        stage = path.parent.name if TRY_RE.match(path.parent.name) else ""
        kind = "bestE" if re.search(r"_bestE(\(\d+\))?$", path.stem) else "bestF"
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest in seen[(key, stage)]:
            continue
        seen[(key, stage)].add(digest)
        runs[key][stage].setdefault(kind, path)
    return runs


def build_group(graph, hi, group_name):
    """(perms, character_table, element_powers, monomial) for a projection."""
    if group_name == "c3":
        group, powers = c3_translation_group(graph)
        _, table = c3_character_table(graph, group)
        return group, table, powers, True
    if group_name == "c2v":
        group, powers = c2v_translation_group(graph)
        _, table = c2v_character_table(graph, group)
        return group, table, powers, True
    symmetries = get_kitaev_symmetries(graph, hi)
    group, table = get_projection_group(symmetries, group_name)
    return group, table, None, False


def build_model(ansatz, group, table, powers, monomial, k_sector, hp, group_chunk):
    """Rebuild the trained module, wrapper included.

    Mirrors `try_Norman.train_projected`'s builder (`:1476-1535`) rather than
    importing it, because that function trains; only the module is needed.
    Hyperparameters come from the checkpoint's own shapes, so a mismatch is
    caught by `mo.check_params_match` instead of silently measuring a
    different wavefunction.
    """
    if ansatz == "rbm":
        bare = DeepRBM(num_layers=hp.layers, alpha=hp.alpha,
                       param_dtype=jnp.complex128, stable_cosh=hp.stable_cosh)
    else:
        bare = FactoredSelfAttention(
            layers=hp.layers, heads=hp.heads, dk=hp.dk, d_model=hp.d_model,
            param_dtype=jnp.complex128 if hp.complex_trunk else jnp.float64,
            out_dtype=jnp.complex128,
        )
    if monomial:
        return MonomialSymmExpSum(
            module=bare,
            symm_group=HashableArray(np.asarray(group)),
            characters=HashableArray(np.asarray(table[int(k_sector)])),
            element_powers=HashableArray(np.asarray(powers)),
            root_order=hp.root_order, group_chunk_size=group_chunk, remat=False,
        )
    return SymmExpSumChunked(
        module=bare, symm_group=group,
        characters=HashableArray(np.asarray(table[int(k_sector)])),
        group_chunk_size=group_chunk,
    )


def infer_hparams(params, ansatz, group_name):
    """Read the architecture off the checkpoint instead of guessing it.

    Nothing in the tree records the ansatz's hyperparameters, but the shapes
    determine them: the RBM's `layer_0/kernel` is (N, alpha*N), and the
    transformer's `embed/kernel` is (1, d_model), `v/kernel` (d_model,
    heads*dk) and `block_0/att_logits` (heads, N, N). `head/kernel`'s second
    axis is 2 for the real two-channel trunk and 1 for a complex one.
    """
    tree = params["module"] if set(params) == {"module"} else params
    hp = SimpleNamespace(stable_cosh=False, root_order=ROOT_ORDER.get(group_name, 3),
                         alpha=2.0, layers=1, heads=2, dk=6, d_model=4,
                         complex_trunk=False)
    if ansatz == "rbm":
        n_visible, n_hidden = np.shape(tree["layer_0"]["kernel"])
        hp.alpha = n_hidden / n_visible
        hp.layers = sum(1 for k in tree if k.startswith("layer_"))
    else:
        hp.layers = sum(1 for k in tree if k.startswith("block_"))
        hp.d_model = int(np.shape(tree["embed"]["kernel"])[1])
        hp.heads = int(np.shape(tree["block_0"]["att_logits"])[0])
        hp.dk = int(np.shape(tree["block_0"]["v"]["kernel"])[1]) // hp.heads
        hp.complex_trunk = int(np.shape(tree["head"]["kernel"])[1]) == 1
    return hp


def load_exact(jz, grid_path, isotropic_path):
    """The exact spectrum for one point, from whichever file holds it.

    The grid file runs jx=jy=(1-jz)/2 over jz = 0.0 .. 1.0, so the isotropic
    point jx=jy=jz=1/3 -- which is what the c3 runs use -- is simply not in
    it: the nearest tabulated jz=0.3 is jx=jy=0.35, a different Hamiltonian.
    """
    if abs(jz - 1 / 3) < 0.02:
        path = Path(isotropic_path)
        if not path.is_file():
            print(f"   [aviso] falta {path}: sin columnas de ED en el punto isotropo.")
            return None
        data = np.load(path)
        energies = np.asarray(data["energies"]).real
        return {"E0": float(np.min(energies)), "energies": energies,
                "eigenvectors": np.asarray(data["eigenvectors"])}
    grid = load_exact_results(grid_path)
    entry = grid[min(grid, key=lambda z: abs(z - jz))]
    energies = np.asarray(entry["energies"]).real
    return {"E0": float(np.min(energies)), "energies": energies,
            "eigenvectors": np.asarray(entry["eigenvectors"])}


def trace_reference(base, key, stage):
    """(E_cola, err_MC) of the metrics CSV for one stage, or (None, None).

    The converged tail mean, not the last value: `energy` is a Monte-Carlo
    estimate, so a single eval can sit below the variational bound (at Jz=0.10
    the `space` run's minimum is -5.4590 against E_ED = -5.4449).
    """
    ansatz, group, jz, k, variant = key
    pattern = f"jx*_jz{jz:.2f}_{group}_{ansatz}_k{k}.csv"
    best = None
    for path in base.rglob("*metrics*" + pattern.split("jx*")[-1]):
        st = path.parent.name if TRY_RE.match(path.parent.name) else ""
        if st != stage or run_variant(path) != variant:
            continue
        if not TAG_RE.search(path.name):
            continue
        if best is None or path.stat().st_size > best.stat().st_size:
            best = path
    if best is None:
        return None, None
    df = pd.read_csv(best)
    tail = df.iloc[len(df) // 2:]
    return float(tail["energy"].mean()), float(tail["energy_error"].mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="data/Projected_Energy_min")
    ap.add_argument("--out", default="data/results/obs_try_norman.csv")
    ap.add_argument("--exact-data", default="data/raw/energies_eigenvecs_dict_k40.npz")
    ap.add_argument("--isotropic-data", default="data/raw/ed_3x3_isotropic_k40.npz")
    ap.add_argument("--stable-cosh", action="store_true",
                    help="Reconstruir el RBM con log(1+x^2/2) en vez de log_cosh.")
    ap.add_argument("--group-chunk", type=int, default=2,
                    help="Elementos del grupo evaluados a la vez en to_array().")
    ap.add_argument("--chunk-size", type=int, default=4096,
                    help="Configuraciones por trozo en to_array().")
    ap.add_argument("--max-dim", type=int, default=2 ** 22)
    args = ap.parse_args()

    base = Path(args.base)
    runs = collect(base)
    if not runs:
        raise SystemExit(f"Ningun checkpoint bajo {base}")
    lat = mo.Lattice([3, 3], True, args.max_dim, "space+mirror")
    exact_cache, rows = {}, []

    by_physics = {}
    for k in runs:
        by_physics.setdefault(k[:4], set()).add(k[4])
    variant_label = {ph: short_variants(vs)
                     for ph, vs in by_physics.items()}

    for key in sorted(runs):
        ansatz, group_name, jz, k_sector, variant = key
        corrida = variant_label[key[:4]][variant]
        jx = jy = jz if abs(jz - 1 / 3) < 0.02 else (1 - jz) / 2
        print(f"\n===== {ansatz} · {group_name} · jz={jz:.2f} · k{k_sector}"
              f"{' · ' + corrida if corrida else ''}"
              f" (jx=jy={jx:.4f}) =====")
        if jz not in exact_cache:
            exact_cache[jz] = load_exact(jz, args.exact_data, args.isotropic_data)
        exact = exact_cache[jz]
        group, table, powers, monomial = build_group(lat.graph, lat.hi, group_name)
        H_sparse = KitaevTransverse_H(lat.graph.edge_colors, lat.graph.edges(),
                                      Jx=jx, Jy=jy, Jz=jz, h=0, hi=lat.hi).to_sparse()

        for stage in sorted(runs[key], key=lambda s: (s != "", s)):
            e_trace, err_trace = trace_reference(base, key, stage)
            for kind, path in sorted(runs[key][stage].items()):
                params = pickle.loads(path.read_bytes())
                hp = infer_hparams(params, ansatz, group_name)
                hp.stable_cosh = args.stable_cosh
                model = build_model(ansatz, group, table, powers, monomial,
                                    k_sector, hp, args.group_chunk)
                vs = nk.vqs.MCState(nk.sampler.MetropolisLocal(lat.hi), model,
                                    n_samples=16, seed=0,
                                    chunk_size=args.chunk_size)
                mo.check_params_match(params, vs.parameters, context=path.name)
                vs.parameters = params

                psi = np.array(vs.to_array())
                if monomial:
                    psi = rotate_state_to_frame(psi, inverse=True,
                                                axis=GROUP_AXIS[group_name])
                norm = np.linalg.norm(psi)
                if not np.isfinite(norm) or norm == 0:
                    print(f"   [SKIP] {kind} etapa{stage or '1'}: norma={norm}")
                    continue
                psi = psi / norm

                spec = SimpleNamespace(Jz=jz, sector=k_sector, extent=[3, 3])
                row = {
                    "file": str(path), "ansatz": ansatz, "group": group_name,
                    "corrida": corrida or "-",
                    "Jz": jz, "Jx": jx, "Jy": jy, "sector": k_sector,
                    "etapa": stage or "1a", "checkpoint": kind,
                    "rotado": bool(monomial), "N": lat.N,
                    "n_params": int(nk.jax.tree_size(vs.parameters)),
                    "alpha": hp.alpha, "layers": hp.layers,
                    "heads": hp.heads, "dk": hp.dk, "d_model": hp.d_model,
                }
                row.update(mo.measure(psi, spec, lat, H_sparse, exact))
                if exact is not None:
                    idx = degenerate_manifold(exact["energies"])
                    row["manifold_dim"] = len(idx)
                    row["fidelity_manifold"] = manifold_fidelity(
                        exact["eigenvectors"], idx, psi)
                    rest = np.sort(exact["energies"])[len(idx)] if len(idx) < len(
                        exact["energies"]) else None
                    gap = None if rest is None else float(rest - exact["E0"])
                    row["gap_ED"] = gap
                    row["cota_infid"] = (None if not gap else
                                         (row["E"] - exact["E0"]) / gap)
                row["E_traza"] = e_trace
                row["err_MC_traza"] = err_trace
                row["dE_vs_traza"] = None if e_trace is None else row["E"] - e_trace
                rows.append(row)
                print(f"   etapa{stage or '1':<5s} {kind:5s} "
                      f"E={row['E']:+.6f} (traza {'' if e_trace is None else f'{e_trace:+.6f}'}"
                      f"{'' if err_trace is None else f' ±{err_trace:.4f}'})"
                      f"  F_man={row['fidelity_manifold']}"
                      f"  W_mean={row['W_mean']:+.4f}  N_-={row['n_minus']}")

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = mo.existing_header(out)
    header += [c for c in df.columns if c not in set(header)]
    for record in df.to_dict("records"):
        append_observables_csv(out, header, [record.get(c) for c in header])
    print(f"\n{len(rows)} fila(s) -> {out}")


if __name__ == "__main__":
    main()
