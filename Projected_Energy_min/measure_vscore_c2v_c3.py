import csv
import importlib.util
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

_src = Path(__file__).resolve().parent / "measure_try_norman.py"
if not _src.is_file():
    raise SystemExit(f"no encuentro {_src}")
spec = importlib.util.spec_from_file_location("mor", _src)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

import netket as nk  # noqa: E402

N_SITES = 18
base = ROOT / "data" / "ENERGY_min_proj_c2v_c3"
runs = M.collect(base)
lat = M.mo.Lattice([3, 3], True, 2 ** 22, "space+mirror")

out = []
camp = list(csv.DictReader(open(base / "campaign_summary.csv", newline="")))
t0 = time.time()

for i, r in enumerate(camp, 1):
    ansatz, group_name = r["ansatz"], r["group"]
    jz, k = float(r["Jz"]), int(r["sector"])
    corrida, etapa = r["corrida"], r["etapa_elegida"]
    rec = {"ansatz": ansatz, "group": group_name, "sector": k, "Jz": r["Jz"],
           "corrida": corrida, "etapa_elegida": etapa,
           "E_exacta": None, "Var_E": None, "vscore": None,
           "E_campaign": float(r["E"]), "dE_check": None, "estado": ""}

    cand = [(key, st) for key in runs for st in runs[key]
            if key[0] == ansatz and key[1] == group_name
            and abs(key[2] - jz) < 0.02 and key[3] == k
            and st == etapa and "bestF" in runs[key][st]
            and (corrida == "-" or key[4].startswith(corrida))]
    if corrida == "-":
        cand = [c for c in cand if not c[0][4].startswith("new_")] or cand
    if len(cand) != 1:
        rec["estado"] = f"{len(cand)} candidatos"
        out.append(rec)
        print(f"[{i}/8] {rec['estado']}", flush=True)
        continue

    key, st = cand[0]
    path = runs[key][st]["bestF"]
    try:
        jx = jy = jz if abs(jz - 1 / 3) < 0.02 else (1 - jz) / 2
        group, table, powers, monomial = M.build_group(lat.graph, lat.hi, group_name)
        H = M.KitaevTransverse_H(lat.graph.edge_colors, lat.graph.edges(),
                                 Jx=jx, Jy=jy, Jz=jz, h=0, hi=lat.hi).to_sparse()
        params = pickle.loads(path.read_bytes())
        hp = M.infer_hparams(params, ansatz, group_name)
        hp.stable_cosh = False
        model = M.build_model(ansatz, group, table, powers, monomial, k, hp, 2)
        vs = nk.vqs.MCState(nk.sampler.MetropolisLocal(lat.hi), model,
                            n_samples=16, seed=0, chunk_size=4096)
        M.mo.check_params_match(params, vs.parameters, context=path.name)
        vs.parameters = params
        psi = np.array(vs.to_array())
        if monomial:
            psi = M.rotate_state_to_frame(psi, inverse=True,
                                          axis=M.GROUP_AXIS[group_name])
        norm = np.linalg.norm(psi)
        if not np.isfinite(norm) or norm == 0:
            rec["estado"] = f"norma={norm}"
        else:
            psi = psi / norm
            h_psi = H @ psi
            E = float(np.real(np.vdot(psi, h_psi)))
            var = float(np.real(np.vdot(h_psi, h_psi))) - E ** 2
            rec.update(E_exacta=E, Var_E=var,
                       vscore=(N_SITES * var / E ** 2) if E else None,
                       dE_check=abs(E - rec["E_campaign"]), estado="ok")
    except Exception as exc:  # noqa: BLE001
        rec["estado"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:70]}"
    out.append(rec)
    print(f"[{i}/8] {ansatz} {group_name} k{k} jz={jz} {etapa}: {rec['estado']} "
          f"E={rec['E_exacta']} dE={rec['dE_check']}  {time.time()-t0:.0f}s", flush=True)

dest = base / "vscore_exacto_bestF.csv"
with open(dest, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0]))
    w.writeheader()
    for x in out:
        w.writerow(x)
ok = [x for x in out if x["estado"] == "ok"]
print(f"\nescrito {dest}  ({len(ok)}/{len(out)} medidos)")
if ok:
    print("validacion E_exacta vs E_campaign: max|dE| = %.3e"
          % max(x["dE_check"] for x in ok))
