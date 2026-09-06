import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "common" / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import flax
import netket as nk

from model_RBM import DeepRBM
from transformer import FactoredSelfAttention
from utils import KitaevTransverse_H
from common.physics.exact_diag import (load_exact_results, degenerate_manifold,
                                    manifold_fidelity)

RES = ROOT / "data"
N_SITES, D_MODEL, DK, RBM_ALPHA = 18, 4, 6, 2.0
EXACT = ROOT / "data" / "raw" / "energies_eigenvecs_dict_k40.npz"

graph = nk.graph.KitaevHoneycomb(extent=[3, 3], pbc=True)
hi = nk.hilbert.Spin(s=1 / 2, N=graph.n_nodes)
sampler = nk.sampler.MetropolisLocal(hi)

print("cargando espectro exacto...", flush=True)
exact_all = load_exact_results(str(EXACT))
print(f"  {len(exact_all)} valores de Jz", flush=True)

H_cache, man_cache = {}, {}


def H_for(jz):
    if jz not in H_cache:
        jx = jy = (1 - jz) / 2
        H_cache[jz] = KitaevTransverse_H(graph.edge_colors, graph.edges(),
                                         Jx=jx, Jy=jy, Jz=jz, h=0,
                                         hi=hi).to_sparse()
    return H_cache[jz]


def manifold_for(jz, tol):
    key = (jz, tol)
    if key not in man_cache:
        k = min(exact_all, key=lambda x: abs(x - jz))
        ex = exact_all[k]
        idx = degenerate_manifold(np.asarray(ex["energies"]).real, tol=tol)
        man_cache[key] = (np.asarray(ex["eigenvectors"]), idx)
    return man_cache[key]


def build(model, layers, heads):
    if model == "RBM":
        return DeepRBM(num_layers=layers, alpha=RBM_ALPHA)
    return FactoredSelfAttention(layers=layers, heads=heads, dk=DK, d_model=D_MODEL)


def psi_of(model, layers, heads, path):
    vs = nk.vqs.MCState(sampler, build(model, layers, heads),
                        n_samples=16, seed=0, chunk_size=8192)
    vs.parameters = flax.serialization.from_bytes(vs.parameters, path.read_bytes())
    psi = np.asarray(vs.to_array())
    n = np.linalg.norm(psi)
    return None if (not np.isfinite(n) or n == 0) else psi / n


resumen = list(csv.DictReader(open(RES / "energyNoProj_resumen.csv", newline="")))

print("\nvalidando tolerancia del manifold contra overlap_best:", flush=True)
for tol in (1e-6, 1e-8):
    errs = []
    for r in resumen[:12]:
        model, L, h, jz = r["model"], int(r["layers"]), int(r["heads"]), float(r["Jz"])
        p = RES / "Energy_min_no_proj" / model / f"{model}{L}_head{h}_{jz:.2f}_sched.mpack"
        if not p.exists():
            continue
        psi = psi_of(model, L, h, p)
        if psi is None:
            continue
        vecs, idx = manifold_for(jz, tol)
        F = float(manifold_fidelity(vecs, idx, psi))
        errs.append(abs(F - float(r["overlap_best"])))
    print(f"  tol={tol:.0e}: n={len(errs)} max|dF|={max(errs):.3e}", flush=True)

TOL = 1e-8

print(f"\nmidiendo estados finales (tol={TOL:.0e})...", flush=True)
out, t0 = [], time.time()
for i, r in enumerate(resumen, 1):
    model, L, h = r["model"], int(r["layers"]), int(r["heads"])
    jz = float(r["Jz"])
    p = RES / "Energy_min_no_proj" / model / f"{model}{L}_head{h}_{jz:.2f}_sched_last.mpack"
    rec = {"model": model, "layers": L, "heads": h, "Jz": jz,
           "E_last": None, "Var_E_last": None, "vscore_last_exacto": None,
           "F_manifold_last": None,
           "E_last_resumen": (float(r["E_var_last"]) * N_SITES
                              if r["E_var_last"].strip() else None),
           "dE_check": None, "estado": ""}
    if not p.exists():
        rec["estado"] = "sin checkpoint"
    else:
        try:
            psi = psi_of(model, L, h, p)
            if psi is None:
                rec["estado"] = "norma no finita"
            else:
                h_psi = H_for(jz) @ psi
                E = float(np.real(np.vdot(psi, h_psi)))
                var = float(np.real(np.vdot(h_psi, h_psi))) - E ** 2
                vecs, idx = manifold_for(jz, TOL)
                rec.update(E_last=E, Var_E_last=var,
                           vscore_last_exacto=(N_SITES * var / E ** 2) if E else None,
                           F_manifold_last=float(manifold_fidelity(vecs, idx, psi)),
                           estado="ok")
                if rec["E_last_resumen"] is not None:
                    rec["dE_check"] = abs(E - rec["E_last_resumen"])
        except Exception as exc:  # noqa: BLE001
            rec["estado"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:60]}"
    out.append(rec)
    if i % 20 == 0 or i == len(resumen):
        print(f"[{i}/{len(resumen)}] ok={sum(1 for x in out if x['estado']=='ok')}"
              f"  {time.time()-t0:.0f}s", flush=True)

dest = RES / "Energy_min_no_proj" / "vscore_exacto_last.csv"
with open(dest, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)

ok = [x for x in out if x["estado"] == "ok"]
print(f"\nescrito {dest} ({len(ok)}/{len(out)})")
d = [x["dE_check"] for x in ok if x["dE_check"] is not None]
if d:
    print(f"validacion E_last vs E_var_last*18: n={len(d)} mediana={sorted(d)[len(d)//2]:.2e} max={max(d):.2e}")
for x in out:
    if x["estado"] != "ok":
        print(f"  [!] {x['model']} L{x['layers']} h{x['heads']} Jz={x['Jz']}: {x['estado']}")
