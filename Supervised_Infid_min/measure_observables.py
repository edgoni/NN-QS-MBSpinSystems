#!/usr/bin/env python
import argparse
import glob
import json
import pickle
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import flax
import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
import pandas as pd
from netket.operator.spin import sigmax, sigmay, sigmaz

sys.path.append(str(Path(__file__).resolve().parent.parent))

from common.models.rbm import ProjectedRBM
from common.physics.hamiltonian import build_kitaev_lattice, KitaevTransverse_H
from common.physics.observables import (
    get_kitaev_plaquettes,
    build_wilson_loops,
    build_magnetization_observables,
)
from common.physics.symmetries import get_kitaev_symmetries, symmetry_projector_inputs
from common.physics.isotropic_symmetry import c2xy_translation_group
from common.physics.exact_diag import (
    load_exact_results,
    degenerate_manifold,
    manifold_fidelity,
    plaquette_permutations,
    pick_target_for_sector,
)
from common.utils.io import append_observables_csv

TAG_RE = re.compile(
    r"^(?P<ansatz>[a-z]+)_(?P<lx>\d+)x(?P<ly>\d+)_L(?P<layers>\d+)"
    r"_jz(?P<jz>[\d.]+)_(?:k(?P<sector>\d+)(?:_v(?P<vclass>\d+))?|knone)"
    r"_s(?P<seed>\d+)_(?P<iteration>\d+)$"
)

LEGACY_TAG_RE = re.compile(
    r"^(?P<ansatz>[a-z]+)_L(?P<layers>\d+)_jz(?P<jz>[\d.]+)"
    r"_k(?P<sector>\d+)_(?P<iteration>\d+)$"
)


def parse_tag(path):
    """Recover what the filename says about a checkpoint.

    Returns a dict of the fields it could read (never raises): anything
    missing is expected to arrive by flag. `phase` distinguishes the two
    checkpoints `run_vmc.py` saves per run -- `vstate_ph1_*` is the purely
    variational state, `vstate_*` the supervised one -- because their rows
    are not comparable and the CSV has to say which is which.
    """
    stem = Path(path).stem
    info = {"file": str(path), "tag": stem}

    for prefix, phase in (("vstate_ph1_", "ph1"), ("vstate_", "ph2"), ("ph1_", "ph1")):
        if stem.startswith(prefix):
            info["phase"] = phase
            stem = stem[len(prefix):]
            break
    else:
        info["phase"] = "ph2" if Path(path).suffix.lower() == ".mpack" else "unknown"
    info["tag"] = stem

    match = TAG_RE.match(stem) or LEGACY_TAG_RE.match(stem)
    if not match:
        return info

    fields = match.groupdict()
    info["ansatz"] = fields["ansatz"]
    info["layers"] = int(fields["layers"])
    info["Jz"] = float(fields["jz"])
    info["iteration"] = int(fields["iteration"])
    info["sector"] = int(fields["sector"]) if fields.get("sector") else -1
    if fields.get("vclass") is not None:
        info["vortex_class"] = int(fields["vclass"])
    if fields.get("seed") is not None:
        info["seed"] = int(fields["seed"])
    if fields.get("lx") is not None:
        info["extent"] = [int(fields["lx"]), int(fields["ly"])]
    return info


def load_params(path):
    """Read a checkpoint's parameter tree, from either format.

    `save_checkpoint` writes the *same* `state.parameters` twice: a `.mpack`
    through `flax.serialization.to_bytes` and a `.pkl` through
    `pickle.dump`. Verified leaf by leaf on
    `rbm_L1_jz0.50_k0_1`: identical arrays. So either is equally good to
    measure, and reading both matters because they are not always both
    there -- four `.mpack` in `data/checkpoints` (`RBM1_*_sched_1`,
    `Test_Inicial_Jz*`) have no pickle at all.

    `msgpack_restore` rather than `from_bytes`: the latter needs a target
    tree to restore into and raises its own error on a mismatch, which would
    pre-empt `check_params_match` and its much more specific message. This
    returns a bare nested dict of numpy arrays, exactly like the pickle path.

    Both formats then go through the same unwrapping: the checkpoints
    `try_Norman.py` left in the repo root nest everything under a single
    `module` key, because that projection wrapper was a Flax module holding
    the ansatz as a submodule. Today's `ProjectedRBM` is the ansatz itself,
    so its tree has `layer_0`/`visible_bias` at the top.
    """
    suffix = Path(path).suffix.lower()
    with open(path, "rb") as f:
        raw = f.read()
    if suffix == ".mpack":
        params = flax.serialization.msgpack_restore(raw)
    elif suffix == ".pkl":
        params = pickle.loads(raw)
    else:
        raise SystemExit(
            f"{path}: extensión {suffix!r} desconocida; se esperaba .pkl o .mpack."
        )
    if isinstance(params, dict) and set(params) == {"module"}:
        print("   [nota] tree envuelto en 'module' (checkpoint legacy): desenvuelto")
        return params["module"]
    return params


def tree_shapes(tree):
    """{path: shape} of every leaf, for comparing two param trees."""
    return {
        jax.tree_util.keystr(path): tuple(np.shape(leaf))
        for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]
    }


def tree_dtypes(tree):
    """{path: dtype name} of every leaf."""
    return {
        jax.tree_util.keystr(path): np.dtype(leaf.dtype).name
        for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]
    }


def check_params_match(loaded, reference, context):
    """Fail loudly when the rebuilt model is not the one that was trained.

    A pickled checkpoint records no architecture, so the only thing standing
    between a wrong `--alpha`/`--heads` and a silently wrong measurement is
    this comparison. It cannot catch a wrong `--sector`, though: the
    symmetry projection adds no parameters, so a checkpoint trained in one
    irrep loads without complaint into a model projected onto another and
    measures a different wavefunction. That is why `--sector` is read from
    the tag rather than given a default.

    Only *shapes* are fatal, because only shapes encode the architecture. A
    dtype difference is a precision difference: JAX's x64 mode is off until
    something enables it, so the very same pickle unpickles as complex64 or
    complex128 depending on import order, and `param_dtype` legitimately
    differs between ansatze (c128 for the RBM, c64 for the transformer). It
    is reported, and NetKet casts on assignment.
    """
    got, want = tree_shapes(loaded), tree_shapes(reference)
    if got != want:
        only_loaded = sorted(set(got) - set(want))
        only_model = sorted(set(want) - set(got))
        differing = sorted(k for k in set(got) & set(want) if got[k] != want[k])
        lines = [f"{context}: el checkpoint no encaja con el modelo reconstruido."]
        for key in differing:
            lines.append(f"  {key}: checkpoint {got[key]} vs modelo {want[key]}")
        for key in only_loaded:
            lines.append(f"  {key}: solo en el checkpoint {got[key]}")
        for key in only_model:
            lines.append(f"  {key}: solo en el modelo {want[key]}")
        lines.append(
            "  Ajusta --ansatz/--layers/--alpha/--heads/--dk/--extent para que "
            "coincidan con la corrida que lo generó."
        )
        raise SystemExit("\n".join(lines))

    got_dt, want_dt = tree_dtypes(loaded), tree_dtypes(reference)
    changed = sorted(k for k in got_dt if got_dt[k] != want_dt[k])
    if changed:
        print(
            f"   [nota] dtype distinto en {len(changed)} hoja(s), p.ej. "
            f"{changed[0]}: checkpoint {got_dt[changed[0]]} -> modelo "
            f"{want_dt[changed[0]]}. Se castea al del modelo."
        )


def build_model(spec, graph, symmetries=None, characters=None):
    """Instantiate the ansatz named by `spec.ansatz`.

    Deliberately a local copy of `run_vmc.py`'s builders rather than an
    import of that script: this tool must keep reading checkpoints that were
    written months ago, so it should not track a training script's current
    defaults. `check_params_match` is what catches a drift between the two.
    """
    common = dict(
        symmetries=symmetries,
        characters=characters,
        projector=spec.projector,
        group_chunk_size=spec.group_chunk_size,
    )
    if spec.ansatz == "rbm":
        return ProjectedRBM(
            num_layers=spec.layers, alpha=spec.alpha,
            param_dtype=jnp.complex128, **common,
        )
    raise SystemExit(
        f"--ansatz {spec.ansatz!r} desconocido. Solo queda 'rbm': ver "
        f"legacy/amputado/ para los ansatze de atencion."
    )


def expect(op_sparse, psi):
    return complex(np.vdot(psi, op_sparse @ psi))


def magnetization_components(psi, single_ops, N):
    """Mx, My, Mz per site and their vector norm -- the legacy definition
    (`legacy/observables_comparar.py`), kept identical so old CSVs and new
    ones can go on the same axes."""
    values = {}
    total_sq = 0.0
    for direction in ("x", "y", "z"):
        val = float(np.real(sum(np.vdot(psi, op @ psi) for op in single_ops[direction]))) / N
        values[direction] = val
        total_sq += val ** 2
    values["total"] = float(np.sqrt(total_sq))
    return values


def s_corr_zz(psi, total_z_sparse, N):
    """`sum_{i != j} <sigma^z_i sigma^z_j>`, the legacy `Szz`.

    Computed as `<(sum_i sigma^z_i)^2> - N` rather than by the O(N^2) double
    loop the legacy script ran. That is not an approximation: the Pauli
    matrices square to the identity, so `(sum_i s_i)^2 = N + sum_{i!=j}
    s_i s_j` exactly. The double loop was 306 sparse mat-vecs per checkpoint
    on 3x3; this is one.
    """
    return float(np.real(expect(total_z_sparse @ total_z_sparse, psi))) - N


def kitaev_bond_correlators(psi, graph, hi):
    """Mean `<sigma^g_i sigma^g_j>` over the bonds of each colour g.

    The signature of the Kitaev exact solution: on a g-bond only the g-g
    correlator survives, every other two-spin correlator vanishing
    identically at any distance. So these three numbers are the physical
    content of the state's short-range order, and `-J_g * <s^g s^g>` summed
    over bonds is exactly the energy.
    """
    sigmas = {0: (sigmax, "x"), 1: (sigmay, "y"), 2: (sigmaz, "z")}
    sums = {"x": 0.0, "y": 0.0, "z": 0.0}
    counts = {"x": 0, "y": 0, "z": 0}
    for color, (u, v) in zip(graph.edge_colors, graph.edges()):
        sigma, name = sigmas[int(color)]
        op = (sigma(hi, u) @ sigma(hi, v)).to_sparse()
        sums[name] += float(np.real(expect(op, psi)))
        counts[name] += 1
    return {
        f"corr_{name}{name}_bond": (sums[name] / counts[name] if counts[name] else float("nan"))
        for name in ("x", "y", "z")
    }


def quantum_coherence_l1(psi):
    """`C_l1 = sum_{a != b} |rho_ab|` for a pure state, i.e.
    `(sum_a |psi_a|)^2 - sum_a |psi_a|^2` -- the legacy definition.

    Basis dependent (computational basis) and unbounded above by the Hilbert
    dimension, so raw values are not comparable across lattice sizes; the
    caller also records `Cl1_normalized = C_l1 / (dim - 1)`, whose maximum is
    1 for the uniform superposition whatever the size.
    """
    abs_psi = np.abs(psi)
    return float(np.sum(abs_psi) ** 2 - np.sum(abs_psi ** 2))


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("checkpoints", nargs="+",
                   help="Rutas .pkl o .mpack (acepta comodines entre "
                        "comillas). save_checkpoint escribe los mismos "
                        "pesos en ambos formatos; da igual cuál pases.")
    p.add_argument("--out", type=str,
                   default="data/results/observables_from_checkpoints.csv")
    p.add_argument("--ansatz", choices=["rbm"],
                   default=None, help="Por defecto, el del nombre del fichero.")
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--alpha", type=float, default=2.0, help="Densidad de ocultas de la RBM.")
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--dk", type=int, default=8)
    p.add_argument("--extent", type=int, nargs=2, default=None,
                   help="Por defecto, el del nombre del fichero, o 3 3.")
    p.add_argument("--no-pbc", action="store_true")
    p.add_argument("--jz", type=float, default=None,
                   help="Por defecto, el del nombre del fichero.")
    p.add_argument("--sector", type=int, default=None,
                   help="Irrep sobre el que se proyectó el ansatz; -1 = sin "
                        "proyectar. Por defecto, el del nombre del fichero. "
                        "NO se puede validar contra el checkpoint (la "
                        "proyección no añade parámetros), así que un valor "
                        "equivocado mide un estado distinto sin avisar.")
    p.add_argument("--vortex-class", type=int, default=None,
                   help="Clase de vórtices del target de fase 2 (defecto: la "
                        "del nombre, o 0).")
    p.add_argument("--group", choices=["translation", "space"], default="space")
    p.add_argument("--projector", choices=["stable", "legacy"], default="stable")
    p.add_argument("--group-chunk-size", type=int, default=None)
    p.add_argument("--exact-data", type=str,
                   default="data/raw/energies_eigenvecs_dict_k40.npz")
    p.add_argument("--vortex-group", choices=["space+mirror", "space"],
                   default="space+mirror",
                   help="Cómo se agrupan las colocaciones de vórtices en "
                        "clases, que es lo que indexa --vortex-class. "
                        "'space+mirror' es lo que hace run_vmc.py hoy. Usa "
                        "'space' para checkpoints entrenados antes de plegar "
                        "el espejo xy: en 3x3 con Jz en [0.48, 0.90] aquello "
                        "daba 2 clases de dimensión 9 donde hoy hay 1 de 18, "
                        "así que un target de --vortex-class 1 sólo se "
                        "reconstruye con esta opción.")
    p.add_argument("--no-target", action="store_true",
                   help="No construir el target de fase 2 (salta "
                        "fidelity_sector/infidelity, que es la parte cara).")
    p.add_argument("--max-dim", type=int, default=2 ** 22,
                   help="Rechaza redes cuyo espacio de Hilbert no quepa denso.")
    return p.parse_args()


def resolve_spec(info, args):
    """Merge filename, flags and defaults into one description of the run.

    Precedence is flag > filename > default, so a checkpoint whose name lies
    (renamed by hand, produced by a legacy script) is still measurable, and
    the common case needs no flags at all.
    """
    def pick(flag_value, key, default=None):
        if flag_value is not None:
            return flag_value
        return info.get(key, default)

    ansatz = pick(args.ansatz, "ansatz")
    if ansatz is None:
        raise SystemExit(
            f"{info['file']}: no se puede deducir el ansatz del nombre "
            f"('{info['tag']}'). Pásalo con --ansatz."
        )
    jz = pick(args.jz, "Jz")
    if jz is None:
        raise SystemExit(
            f"{info['file']}: no se puede deducir Jz del nombre "
            f"('{info['tag']}'). Pásalo con --jz."
        )
    return SimpleNamespace(
        ansatz=ansatz,
        layers=pick(args.layers, "layers", 1),
        alpha=args.alpha,
        heads=args.heads,
        dk=args.dk,
        Jz=float(jz),
        sector=pick(args.sector, "sector", -1),
        vortex_class=pick(args.vortex_class, "vortex_class", 0),
        extent=list(pick(args.extent, "extent", [3, 3])),
        projector=args.projector,
        group_chunk_size=args.group_chunk_size,
    )


class Lattice:
    """Everything that depends only on (extent, pbc), built once and reused.

    Rebuilding the sparse single-site operators per checkpoint dominated the
    legacy script's runtime on a directory of pickles; they only depend on
    the Hilbert space, so a sweep over checkpoints of one lattice builds them
    once.
    """

    def __init__(self, extent, pbc, max_dim, vortex_group="space+mirror"):
        self.graph, self.hi = build_kitaev_lattice(extent=extent, pbc=pbc)
        self.N = self.graph.n_nodes
        self.dim = self.hi.n_states
        if self.dim > max_dim:
            raise SystemExit(
                f"El espacio de Hilbert de {extent} tiene {self.dim} estados, "
                f"por encima de --max-dim ({max_dim}). Estas medidas son "
                f"exactas sobre el vector denso, así que no caben; súbelo "
                f"sólo si tienes memoria de sobra."
            )
        self.symmetries = get_kitaev_symmetries(self.graph, self.hi)
        self.plaquettes, plaq_ops = get_kitaev_plaquettes(self.graph)
        self.wilson = [w.to_sparse() for w in build_wilson_loops(self.hi, self.plaquettes, plaq_ops)]
        staggered, uniform = build_magnetization_observables(self.hi, self.N)
        self.m_uniform = uniform.to_sparse()
        self.m_staggered = staggered.to_sparse()
        self.single_ops = {
            "x": [sigmax(self.hi, i).to_sparse() for i in range(self.N)],
            "y": [sigmay(self.hi, i).to_sparse() for i in range(self.N)],
            "z": [sigmaz(self.hi, i).to_sparse() for i in range(self.N)],
        }
        total_z = self.single_ops["z"][0]
        for op in self.single_ops["z"][1:]:
            total_z = total_z + op
        self.total_z = total_z
        self.plaquette_perms = plaquette_permutations(
            self.plaquettes, self.symmetries.automorphisms
        )
        if vortex_group == "space+mirror":
            self.plaquette_perms = self.plaquette_perms + plaquette_permutations(
                self.plaquettes, c2xy_translation_group(self.graph)
            )


def measure(psi, spec, lat, H_sparse, exact):
    """Every observable of one state, as a flat dict ready to be a CSV row."""
    row = {}

    wp = [float(np.real(expect(w, psi))) for w in lat.wilson]
    for i, value in enumerate(wp):
        row[f"W_{i}"] = value
    row["W_all"] = json.dumps([round(v, 8) for v in wp])
    row["W_mean"] = float(np.mean(wp))
    row["W_sum"] = float(np.sum(wp))
    row["W_min"], row["W_max"] = float(np.min(wp)), float(np.max(wp))
    row["W_spread"] = float(np.ptp(wp))
    row["W_uniform"] = bool(np.ptp(wp) < 1e-6)
    row["flux_pattern"] = json.dumps([int(np.sign(v)) if abs(v) > 1e-6 else 0 for v in wp])
    row["n_minus"] = int(sum(1 for v in wp if v < -1e-6))
    row["n_plaquettes"] = len(wp)

    h_psi = H_sparse @ psi
    energy = float(np.real(np.vdot(psi, h_psi)))
    variance = float(np.real(np.vdot(h_psi, h_psi))) - energy ** 2
    row["E"] = energy
    row["E_per_site"] = energy / lat.N
    row["Var_E"] = variance
    row["vscore"] = lat.N * variance / energy ** 2 if energy else float("inf")

    mag = magnetization_components(psi, lat.single_ops, lat.N)
    row["Mx"], row["My"], row["Mz"] = mag["x"], mag["y"], mag["z"]
    row["M_total"] = mag["total"]
    row["m"] = float(np.real(expect(lat.m_uniform, psi)))
    row["ms"] = float(np.real(expect(lat.m_staggered, psi)))
    row["fluct"] = float(np.real(expect(lat.m_uniform @ lat.m_uniform, psi)))
    row["fluct_s"] = float(np.real(expect(lat.m_staggered @ lat.m_staggered, psi)))

    row["Szz"] = s_corr_zz(psi, lat.total_z, lat.N)
    row.update(kitaev_bond_correlators(psi, lat.graph, lat.hi))

    cl1 = quantum_coherence_l1(psi)
    row["Cl1"] = cl1
    row["Cl1_normalized"] = cl1 / (lat.dim - 1)

    if exact is None:
        row["E_ED"] = row["delta_eps"] = None
        row["overlap_gs"] = row["fidelity_manifold"] = None
        row["manifold_dim"] = None
    else:
        e_ed = float(exact["E0"])
        row["E_ED"] = e_ed
        row["delta_eps"] = abs(energy - e_ed) / abs(e_ed) if e_ed else float("nan")
        eigvecs = exact["eigenvectors"]
        manifold_idx = degenerate_manifold(np.asarray(exact["energies"]).real)
        row["manifold_dim"] = len(manifold_idx)
        row["overlap_gs"] = float(np.abs(np.vdot(eigvecs[:, 0], psi)) ** 2)
        row["fidelity_manifold"] = manifold_fidelity(eigvecs, manifold_idx, psi)
    return row


def main():
    args = parse_args()

    paths = []
    for pattern in args.checkpoints:
        matched = sorted(glob.glob(pattern))
        if matched:
            paths.extend(matched)
        elif Path(pattern).is_file():
            paths.append(pattern)
        else:
            print(f"[!] sin coincidencias: {pattern}")
    if not paths:
        raise SystemExit("Ningún checkpoint que medir.")

    try:
        exact_results = load_exact_results(args.exact_data)
    except FileNotFoundError:
        print(f"[!] {args.exact_data} no existe: sin columnas de comparación con ED.")
        exact_results = None

    lattices = {}
    rows = []
    for path in paths:
        info = parse_tag(path)
        spec = resolve_spec(info, args)
        print(f"\n== {Path(path).name} ==")
        print(
            f"   ansatz={spec.ansatz} L={spec.layers} extent={spec.extent} "
            f"Jz={spec.Jz:.2f} sector={spec.sector} fase={info['phase']}"
        )

        key = (tuple(spec.extent), not args.no_pbc, args.vortex_group)
        if key not in lattices:
            lattices[key] = Lattice(
                spec.extent, not args.no_pbc, args.max_dim, args.vortex_group
            )
        lat = lattices[key]

        perms = characters = None
        if spec.sector >= 0:
            perms, characters = symmetry_projector_inputs(
                lat.symmetries, spec.sector, group=args.group
            )
        model = build_model(spec, lat.graph, symmetries=perms, characters=characters)

        params = load_params(path)
        vstate = nk.vqs.MCState(
            nk.sampler.MetropolisLocal(lat.hi), model, n_samples=16, seed=0
        )
        check_params_match(params, vstate.parameters, context=Path(path).name)
        vstate.parameters = params

        psi = np.asarray(vstate.to_array())
        norm = np.linalg.norm(psi)
        if not np.isfinite(norm) or norm == 0:
            print(f"   [SKIP] el estado no es normalizable (norma={norm}).")
            continue
        psi = psi / norm

        jx = jy = (1 - spec.Jz) / 2
        H_sparse = KitaevTransverse_H(
            lat.graph.edge_colors, lat.graph.edges(),
            Jx=jx, Jy=jy, Jz=spec.Jz, h=0, hi=lat.hi,
        ).to_sparse()

        exact = None
        if exact_results is not None:
            jz_key = min(exact_results, key=lambda k: abs(k - spec.Jz))
            candidate = exact_results[jz_key]
            ed_dim = int(np.asarray(candidate["eigenvectors"]).shape[0])
            if ed_dim != lat.dim:
                print(
                    f"   [aviso] {args.exact_data} es de dimension {ed_dim} y "
                    f"esta red tiene {lat.dim} ({spec.extent}): sin columnas "
                    f"de ED. Pasa el .npz de esta red con --exact-data."
                )
            else:
                exact = candidate

        row = {
            "file": str(path), "tag": info["tag"], "phase": info["phase"],
            "ansatz": spec.ansatz, "layers": spec.layers,
            "extent_x": spec.extent[0], "extent_y": spec.extent[1], "N": lat.N,
            "Jz": spec.Jz, "sector": spec.sector, "group": args.group,
            "seed": info.get("seed"), "iteration": info.get("iteration"),
            "n_params": int(nk.jax.tree_size(vstate.parameters)),
            "dtype": "/".join(sorted({
                np.dtype(leaf.dtype).name for leaf in jax.tree.leaves(vstate.parameters)
            })),
        }
        row.update(measure(psi, spec, lat, H_sparse, exact))

        row["fidelity_sector"] = row["infidelity"] = None
        row["target_n_minus"] = None
        if exact is not None and spec.sector >= 0 and not args.no_target:
            try:
                target = pick_target_for_sector(
                    exact["eigenvectors"],
                    degenerate_manifold(np.asarray(exact["energies"]).real),
                    lat.hi, *_projection_group(lat, args.group), spec.sector,
                    wilson_loops=lat.wilson,
                    plaquette_perms=lat.plaquette_perms,
                    class_index=spec.vortex_class, verbose=False,
                )
                fidelity = float(np.abs(np.vdot(target["vector"], psi)) ** 2)
                row["fidelity_sector"] = fidelity
                row["infidelity"] = 1.0 - fidelity
                row["target_n_minus"] = target["n_minus"]
            except (ValueError, IndexError) as exc:
                print(f"   [aviso] sin target de fase 2: {str(exc).splitlines()[0]}")
                if "class_index" in str(exc) and args.vortex_group == "space+mirror":
                    print(
                        "           Si el checkpoint es anterior al plegado del "
                        "espejo xy, prueba --vortex-group space."
                    )

        rows.append(row)
        print(
            f"   E={row['E']:.6f} (ED {row['E_ED']})  W_mean={row['W_mean']:+.4f}"
            f"  N_-={row['n_minus']}  M_total={row['M_total']:.4f}"
        )
        print(
            f"   Szz={row['Szz']:.4f}  Cl1={row['Cl1']:.3e}  "
            f"corr_xx/yy/zz={row['corr_xx_bond']:+.3f}/"
            f"{row['corr_yy_bond']:+.3f}/{row['corr_zz_bond']:+.3f}"
        )
        overlap = row["overlap_gs"]
        fidelity = row["fidelity_sector"]
        print(
            f"   overlap_gs={overlap if overlap is None else round(overlap, 6)}  "
            f"F_manifold={row['fidelity_manifold']}  "
            f"F_sector={fidelity if fidelity is None else round(fidelity, 6)}"
        )
        del psi, H_sparse

    if not rows:
        raise SystemExit("Ninguna medida realizada.")

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = existing_header(out)
    header += [c for c in df.columns if c not in set(header)]
    for record in df.to_dict("records"):
        append_observables_csv(out, header, [record.get(c) for c in header])
    print(f"\n{len(rows)} fila(s) -> {out}")


def existing_header(path):
    """Column names already in `path`, or [] if it does not exist yet."""
    path = Path(path)
    if not path.is_file():
        return []
    with open(path, newline="") as f:
        import csv
        return next(csv.reader(f), [])


def _projection_group(lat, group):
    """(sg, character_table) for `pick_target_for_sector`, matching --group."""
    from common.physics.symmetries import get_projection_group
    return get_projection_group(lat.symmetries, group)


if __name__ == "__main__":
    main()
