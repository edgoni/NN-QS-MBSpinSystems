import csv
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "common" / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import flax
import netket as nk

from model_RBM import DeepRBM
from transformer import FactoredSelfAttention
from utils import KitaevTransverse_H

RES = ROOT / "data"
N_SITES = 18
D_MODEL, DK, RBM_ALPHA = 4, 6, 2.0

graph = nk.graph.KitaevHoneycomb(extent=[3, 3], pbc=True)
hi = nk.hilbert.Spin(s=1 / 2, N=graph.n_nodes)
graph.hi = hi
sampler = nk.sampler.MetropolisLocal(hi)

H_cache = {}


def H_for(jz):
    if jz not in H_cache:
        jx = jy = (1 - jz) / 2
        H_cache[jz] = KitaevTransverse_H(graph.edge_colors, graph.edges(),
                                         Jx=jx, Jy=jy, Jz=jz, h=0,
                                         hi=hi).to_sparse()
    return H_cache[jz]


def build(model, layers, heads):
    if model == "RBM":
        return DeepRBM(num_layers=layers, alpha=RBM_ALPHA)
    return FactoredSelfAttention(layers=layers, heads=heads, dk=DK, d_model=D_MODEL)


resumen = list(csv.DictReader(open(RES / "energyNoProj_resumen.csv", newline="")))
out_rows = []
t0 = time.time()

for i, r in enumerate(resumen, 1):
    model, layers, heads = r["model"], int(r["layers"]), int(r["heads"])
    jz = float(r["Jz"])
    ck = RES / "Energy_min_no_proj" / model / f"{model}{layers}_head{heads}_{jz:.2f}_sched.mpack"
    rec = {"model": model, "layers": layers, "heads": heads, "Jz": jz,
           "E_exacta": None, "Var_E": None, "vscore": None,
           "E_resumen": float(r["E_var_best"]) * N_SITES, "dE_check": None,
           "estado": ""}
    if not ck.exists():
        rec["estado"] = "sin checkpoint"
        out_rows.append(rec)
        continue
    try:
        vs = nk.vqs.MCState(sampler, build(model, layers, heads),
                            n_samples=16, seed=0, chunk_size=8192)
        vs.parameters = flax.serialization.from_bytes(vs.parameters,
                                                      ck.read_bytes())
        psi = np.asarray(vs.to_array())
        norm = np.linalg.norm(psi)
        if not np.isfinite(norm) or norm == 0:
            rec["estado"] = f"norma={norm}"
            out_rows.append(rec)
            continue
        psi = psi / norm
        h_psi = H_for(jz) @ psi
        E = float(np.real(np.vdot(psi, h_psi)))
        var = float(np.real(np.vdot(h_psi, h_psi))) - E ** 2
        rec["E_exacta"] = E
        rec["Var_E"] = var
        rec["vscore"] = N_SITES * var / E ** 2 if E else None
        rec["dE_check"] = abs(E - rec["E_resumen"])
        rec["estado"] = "ok"
    except Exception as exc:  # noqa: BLE001
        rec["estado"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:70]}"
    out_rows.append(rec)
    if i % 10 == 0 or i == len(resumen):
        ok = sum(1 for x in out_rows if x["estado"] == "ok")
        print(f"[{i}/{len(resumen)}] ok={ok}  {time.time()-t0:.0f}s", flush=True)

dest = RES / "Energy_min_no_proj" / "vscore_exacto_best.csv"
with open(dest, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
    w.writeheader()
    for x in out_rows:
        w.writerow(x)

ok = [x for x in out_rows if x["estado"] == "ok"]
print(f"\nescrito {dest}  ({len(ok)}/{len(out_rows)} medidos)")
if ok:
    d = [x["dE_check"] for x in ok]
    print(f"validacion E_exacta vs E_var_best*18: max|dE| = {max(d):.3e}")
    peor = max(ok, key=lambda x: x["dE_check"])
    print(f"  peor: {peor['model']} L{peor['layers']} h{peor['heads']} "
          f"Jz={peor['Jz']}  {peor['E_exacta']:.9f} vs {peor['E_resumen']:.9f}")
for x in out_rows:
    if x["estado"] not in ("ok",):
        print(f"  [!] {x['model']} L{x['layers']} h{x['heads']} Jz={x['Jz']}: {x['estado']}")
