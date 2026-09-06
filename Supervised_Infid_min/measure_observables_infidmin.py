import csv
import importlib.util
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

_p = Path(__file__).resolve().parent / "measure_observables.py"
_spec = importlib.util.spec_from_file_location("mo", _p)
mo = importlib.util.module_from_spec(_spec)
sys.modules["mo"] = mo
_spec.loader.exec_module(mo)

import netket as nk  # noqa: E402

INF = ROOT / "data" / "InfidMin"
EXACT = ROOT / "data" / "raw" / "energies_eigenvecs_dict_k40.npz"
GROUP, VORTEX_GROUP = "space", "space+mirror"
DEST = INF / "observables_todos_los_sectores.csv"

TAG = re.compile(
    r"^(?:(?P<ph1>ph1)_)?(?P<ansatz>[a-z]+)_(?P<lx>\d+)x(?P<ly>\d+)_L(?P<L>\d+)"
    r"_jz(?P<jz>[\d.]+)_(?:k(?P<k>\d+)(?:_v(?P<v>\d+))?|knone)"
    r"_s(?P<s>\d+)_(?P<it>\d+)$")

found = {}
for d in ("Infid_gapped", "Infid_k0_gapless", "Infid_proy_gapless", "Restokno0"):
    for ext in ("pkl", "mpack"):
        for p in sorted((INF / d).glob(f"*.{ext}")):
            stem = p.name.rsplit(".", 1)[0]
            if stem.startswith("vstate_"):
                stem = stem[len("vstate_"):]
            if TAG.match(stem):
                found[stem] = p

FIELDS = ["Jz", "k", "vclass", "phase", "jobid", "carpeta", "tag",
          "manifold_dim", "E", "E_ED", "delta_eps", "Var_E", "vscore",
          "overlap_gs", "fidelity_manifold", "fidelity_sector", "infidelity",
          "W_mean", "n_minus", "estado", "file"]

prev = []
if DEST.exists():
    with open(DEST, newline="") as fh:
        for r in csv.DictReader(fh):
            r.setdefault("jobid", TAG.match(r["tag"]).group("it")
                         if TAG.match(r["tag"]) else "")
            prev.append({k: r.get(k, "") for k in FIELDS})
done = {r["tag"] for r in prev}
todo = {t: p for t, p in found.items() if t not in done}
print(f"{len(found)} corridas reales, {len(done)} ya medidas, {len(todo)} por medir",
      flush=True)

if todo:
    print("cargando espectro exacto...", flush=True)
    exact_all = mo.load_exact_results(str(EXACT))
    lat = mo.Lattice([3, 3], True, 2 ** 22, VORTEX_GROUP)
    H_cache, ex_cache = {}, {}

    def H_for(jz):
        if jz not in H_cache:
            jx = jy = (1 - jz) / 2
            H_cache[jz] = mo.KitaevTransverse_H(
                lat.graph.edge_colors, lat.graph.edges(),
                Jx=jx, Jy=jy, Jz=jz, h=0, hi=lat.hi).to_sparse()
        return H_cache[jz]

    def exact_for(jz):
        if jz not in ex_cache:
            k = min(exact_all, key=lambda x: abs(x - jz))
            c = exact_all[k]
            ex_cache[jz] = c if int(np.asarray(c["eigenvectors"]).shape[0]) == lat.dim else None
        return ex_cache[jz]

    t0 = time.time()
    for i, (tag, path) in enumerate(sorted(todo.items()), 1):
        g = TAG.match(tag).groupdict()
        jz = float(g["jz"])
        rec = dict.fromkeys(FIELDS)
        rec.update(Jz=jz, k=int(g["k"]), vclass=g["v"],
                   phase="ph1" if g["ph1"] else "ph2", jobid=g["it"],
                   carpeta=path.parent.name, tag=tag,
                   file=str(path.relative_to(ROOT)), estado="")
        try:
            info = mo.parse_tag(str(path))
            args = SimpleNamespace(
                ansatz=None, layers=None, alpha=2.0, heads=2, dk=8,
                extent=None, jz=None, sector=None, vortex_class=None,
                projector="stable", group_chunk_size=None)
            spec = mo.resolve_spec(info, args)
            perms = chars = None
            if spec.sector >= 0:
                perms, chars = mo.symmetry_projector_inputs(
                    lat.symmetries, spec.sector, group=GROUP)
            model = mo.build_model(spec, lat.graph, symmetries=perms,
                                   characters=chars)
            vs = nk.vqs.MCState(nk.sampler.MetropolisLocal(lat.hi), model,
                                n_samples=16, seed=0)
            params = mo.load_params(str(path))
            mo.check_params_match(params, vs.parameters, context=path.name)
            vs.parameters = params
            psi = np.asarray(vs.to_array())
            norm = np.linalg.norm(psi)
            if not np.isfinite(norm) or norm == 0:
                rec["estado"] = f"norma={norm}"
            else:
                psi = psi / norm
                ex = exact_for(jz)
                row = mo.measure(psi, spec, lat, H_for(jz), ex)
                for c in ("E", "Var_E", "vscore", "W_mean", "n_minus", "E_ED",
                          "delta_eps", "manifold_dim", "overlap_gs",
                          "fidelity_manifold"):
                    if c in row:
                        rec[c] = row[c]
                if ex is not None and spec.sector >= 0:
                    try:
                        tgt = mo.pick_target_for_sector(
                            ex["eigenvectors"],
                            mo.degenerate_manifold(np.asarray(ex["energies"]).real),
                            lat.hi, *mo._projection_group(lat, GROUP), spec.sector,
                            wilson_loops=lat.wilson,
                            plaquette_perms=lat.plaquette_perms,
                            class_index=spec.vortex_class, verbose=False)
                        fid = float(np.abs(np.vdot(tgt["vector"], psi)) ** 2)
                        rec["fidelity_sector"] = fid
                        rec["infidelity"] = 1.0 - fid
                    except (ValueError, IndexError) as exc:
                        rec["estado"] = f"sin target: {str(exc).splitlines()[0][:40]}"
                rec["estado"] = rec["estado"] or "ok"
        except Exception as exc:  # noqa: BLE001
            rec["estado"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:60]}"
        prev.append(rec)
        if i % 4 == 0 or i == len(todo):
            print(f"[{i}/{len(todo)}] {time.time()-t0:.0f}s", flush=True)

prev.sort(key=lambda r: (float(r["Jz"]), int(r["k"]), str(r["vclass"]),
                         r["phase"], str(r["jobid"])))
with open(DEST, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(prev)
print(f"\nescrito {DEST}: {len(prev)} filas")
from collections import Counter
for st, n in Counter(r["estado"] for r in prev).most_common():
    print(f"   {n:3d}x {st[:60]}")
