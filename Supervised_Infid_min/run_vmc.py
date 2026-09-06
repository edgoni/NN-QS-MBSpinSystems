#!/usr/bin/env python
import argparse
import gc
import itertools
import json
import subprocess
import sys
from pathlib import Path

import netket as nk
import numpy as np
import jax.numpy as jnp
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from common.physics.hamiltonian import build_kitaev_lattice, KitaevTransverse_H
from common.physics.observables import get_kitaev_plaquettes, build_wilson_loops, build_magnetization_observables
from common.physics.symmetries import (
    get_kitaev_symmetries,
    get_projection_group,
    symmetry_projector_inputs,
)
from common.physics.isotropic_symmetry import c2xy_translation_group
from common.physics.exact_diag import (
    load_exact_results,
    degenerate_manifold,
    manifold_irrep_weights,
    sectors_hosting_manifold,
    pick_target_for_sector,
    plaquette_permutations,
    vortex_resolved_manifold,
    manifold_fidelity,
)
from common.models.rbm import ProjectedRBM
from common.training.drivers import build_sampler, run_ground_state, run_infidelity_projection
from common.utils.io import (
    save_checkpoint,
    append_observables_csv,
    load_target_state_from_vector,
    load_target_state_exact,
)
from common.utils.schema import RUN_COLUMNS, empty_reserved, first_step_below, row_to_list


def params_dtype(state) -> str:
    """Short name of the dtype the variational parameters actually carry.

    Read off the state instead of hardcoded. Historicamente hacia falta
    porque los ansatze no coincidian: `build_rbm` pasa
    `param_dtype=jnp.complex128` explicito mientras que el transformer
    heredaba `complex64`, y S11 es justo la figura c64-vs-c128. Tras amputar
    los ansatze de atencion solo queda la RBM, pero se sigue leyendo del
    estado: es la fuente de verdad y no cuesta nada.
    """
    import jax
    leaves = jax.tree.leaves(state.parameters)
    if not leaves:
        return "unknown"
    names = {np.dtype(leaf.dtype).name for leaf in leaves}
    short = {"complex64": "c64", "complex128": "c128", "float32": "f32", "float64": "f64"}
    return "/".join(sorted(short.get(n, n) for n in names))

import optax


def git_commit() -> str:
    """Short git SHA of the working tree, or 'unknown' outside a repo.

    Recorded on every results row: table T3 of the guideline is the
    hyperparameter table, and a run you cannot map back to the code that
    produced it is not reproducible no matter how many hyperparameters
    you wrote down.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def step_metrics_frame(metrics_history: dict) -> pd.DataFrame:
    """Per-step columns of a `metrics_history`, dropping the run-level
    entries (`best_infidelity`, `wall_clock`, `collapsed`, ...) that the
    drivers stash in the same dict. Those belong in the run summary row,
    not repeated on every line of the training curve.

    Selecting on `isinstance(v, list)` alone is not enough: phase 2 also
    stashes `target_flux_pattern`, a list with one entry per *plaquette*.
    Treating that as a training curve made `pd.DataFrame` raise "All arrays
    must be of the same length" whenever the plaquette count differed from
    the step count -- i.e. on essentially every phase-2 run, after the
    results row had already been written. Matching the length of `step` is
    what makes "per-step" mean per-step.
    """
    n_steps = len(metrics_history.get("step", []))
    return pd.DataFrame({
        k: v for k, v in metrics_history.items()
        if isinstance(v, list) and len(v) == n_steps
    })


def build_rbm(layers, args, graph, symmetries=None, characters=None):
    return ProjectedRBM(
        num_layers=layers,
        alpha=args.alpha,
        param_dtype=jnp.complex128,
        symmetries=symmetries,
        characters=characters,
        projector=args.projector,
        group_chunk_size=args.group_chunk_size,
    )


ANSATZ_BUILDERS = {
    "rbm": build_rbm,
}

LAYERLESS_ANSATZE = frozenset()

PHASE1_LR_INIT = 0.01
PHASE1_LR_PEAK = 0.05


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ansatz", choices=[*ANSATZ_BUILDERS], default="rbm",
        help="Solo 'rbm' (ProjectedRBM). 'transformer', 'vit' y 'factored' se "
             "amputaron en el refactor por bloques: no respaldan ninguna corrida "
             "de data/Supervised_Infid_min. Ver legacy/amputado/ y "
             "docs/código/expresividad_de_factored_y_vit.md",
    )
    parser.add_argument("--extent", type=int, nargs=2, default=[3, 3])
    parser.add_argument("--no-pbc", action="store_true")
    parser.add_argument("--max-layers", type=int, default=4)
    parser.add_argument(
        "--layers", type=int, default=None,
        help="Train this single layer count instead of sweeping "
             "1..--max-layers. Use it when the sweep axis is Jz alone.",
    )
    parser.add_argument("--epochs1", type=int, default=100)
    parser.add_argument(
        "--epochs2", type=int, default=20,
        help="Maximum phase-2 steps. Unless --no-plateau-stop is passed this "
             "is a safety ceiling, not a target: the run ends when the best "
             "infidelity stops improving.",
    )
    parser.add_argument("--jz-min", type=float, default=0.3)
    parser.add_argument("--jz-max", type=float, default=0.4)
    parser.add_argument("--jz-steps", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=2.0, help="RBM hidden-unit density.")
    parser.add_argument("--heads", type=int, default=2, help="Transformer attention heads.")
    parser.add_argument("--dk", type=int, default=8, help="Transformer per-head dimension.")
    parser.add_argument("--n-samples", type=int, default=2048)
    parser.add_argument("--diag-shift", type=float, default=0.01, help="SR regularization for phase 1.")
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seeds parameter init and sampler chains, and is recorded on the "
             "results row. S12 (V-score spread) and S6 (relative-phase spread) "
             "are seed sweeps, so this has to be an explicit, logged knob.",
    )
    parser.add_argument(
        "--vscore-threshold", type=float, default=1e-2,
        help="V-score bar for `n_iter_convergencia`, the step count F7b turns "
             "into wall-clock-to-threshold bars. Recorded on the row so the "
             "column is interpretable without knowing this flag's value.",
    )
    parser.add_argument(
        "--sectors", type=int, nargs="+", default=None,
        help="Train only these irrep indices instead of every sector hosting "
             "the manifold. k=0 converges reliably and is the slow half of a "
             "scan, so restricting to the sector under test (e.g. --sectors 2) "
             "is the usual way to iterate. Requesting a k the manifold does "
             "not host is an error, not a silent no-op.",
    )
    parser.add_argument(
        "--vortex-class", type=int, nargs="+", default=None,
        help="Which vortex sector(s) to build the target from, by index into "
             "the classes `pick_target_for_sector` reports (deterministic "
             "order). Defaults to 0. This matters whenever a level holds "
             "several sectors with the SAME vortex number: on 3x3 at "
             "Jz>=0.48 there are two, they are mutually ORTHOGONAL, and each "
             "hosts every irrep -- so a target from the wrong one cannot be "
             "reached by any state in the right one. Pass e.g. "
             "`--vortex-class 0 1` to train against both and compare.",
    )
    parser.add_argument(
        "--all-vortex-classes", action="store_true",
        help="Train against every vortex class present at each Jz. Overrides "
             "--vortex-class.",
    )
    parser.add_argument(
        "--target-n-minus", type=int, default=None,
        help="Restrict the target to vortex sectors carrying this many "
             "vortices. Combined with --vortex-class, the index then counts "
             "only within those.",
    )
    parser.add_argument("--iteration", type=int, default=1, help="Run tag, kept in output filenames.")
    parser.add_argument("--exact-data", type=str, default="data/raw/energies_eigenvecs_dict_k40.npz")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument(
        "--group", choices=["translation", "space"], default="space",
        help="Symmetry group to project onto. 'space' (default) reproduces the "
             "older behavior. 'translation' has only 1-dimensional irreps, so "
             "each sector is a well-defined momentum sector, but has not shown "
             "a clear win in 3x3/Jz=0.6 validation so far -- pass explicitly to "
             "opt in. See src.physics.symmetries.get_projection_group.",
    )
    parser.add_argument(
        "--projector", choices=["stable", "legacy"], default="stable",
        help="Symmetry-projection routine: 'stable' (logsumexp-based) or "
             "'legacy' (original hand-rolled log-of-sum).",
    )
    parser.add_argument(
        "--group-chunk-size", type=int, default=None,
        help="Evaluate the symmetry-group orbit in chunks of this size "
             "(bounds peak memory for large groups). Only used by the "
             "'stable' projector.",
    )
    parser.add_argument(
        "--optimizer", choices=["sgd", "adam"], default="sgd",
        help="Optimizer for both phases. 'sgd' is correct on SR/NGD-preconditioned "
             "gradients (NetKet's own recommendation); 'adam' reproduces the older behavior.",
    )
    parser.add_argument(
        "--clip-norm", type=float, default=5.0,
        help="Global gradient-norm clip; pass a negative value to disable.",
    )
    parser.add_argument(
        "--no-collapse-guard", action="store_true",
        help="Disable the phase-2 dead-gradient / divergence early stop.",
    )
    parser.add_argument(
        "--no-plateau-stop", action="store_true",
        help="Disable the phase-2 plateau early stop, making --epochs2 the "
             "actual stopping criterion again instead of a ceiling.",
    )
    parser.add_argument(
        "--plateau-patience", type=int, default=750,
        help="Window (steps) the phase-2 plateau is measured over.",
    )
    parser.add_argument(
        "--plateau-min-improvement", type=float, default=0.05,
        help="Relative improvement of the best infidelity required within "
             "--plateau-patience steps to keep training (0.05 = 5%%).",
    )
    parser.add_argument(
        "--plateau-min-steps", type=int, default=500,
        help="Warmup steps before the plateau check arms, so the noisy "
             "opening transient of a projected ansatz cannot trigger it.",
    )
    parser.add_argument(
        "--exact", action="store_true",
        help="Use exact FullSumState instead of Monte-Carlo sampling for both "
             "phases and for the infidelity target (only feasible for small N).",
    )
    parser.add_argument(
        "--projection", choices=["irrep", "none"], default="irrep",
        help="'irrep' (default) projects the ansatz onto a symmetry sector "
             "and trains one run per hosting sector. 'none' trains a plain, "
             "unprojected ansatz: the sector loop collapses to a single run "
             "recorded with sector=-1. Incompatible with --sectors and the "
             "--vortex-class flags, which only mean something per sector.",
    )
    parser.add_argument(
        "--phases", choices=["both", "vmc"], default="both",
        help="'both' (default) runs the variational phase and then the "
             "supervised infidelity phase against the ED target. 'vmc' runs "
             "only the variational energy minimization -- no target is built "
             "and every *_ph2 column is left empty. That is the guideline's "
             "phase B, the one that stands without ED access.",
    )
    parser.add_argument(
        "--unprojected-phase1", action="store_true",
        help="Reproduce the original behavior of training phase 1 WITHOUT the "
             "symmetry projection and transferring those weights into the "
             "projected phase-2 ansatz. Not recommended: a group-symmetric "
             "phase-1 state is annihilated by any non-trivial irrep projector "
             "(sum_g chi(g) = 0), which is what made non-trivial sectors fail.",
    )
    return validate_args(parser.parse_args())


def validate_args(args):
    """Reject flag combinations that cannot mean what they say.

    Each of these used to be a silent no-op or a wasted run, which on a
    cluster is discovered hours later from an empty results row.
    """
    if args.projection == "none":
        conflicting = [
            name
            for name, value in (
                ("--sectors", args.sectors),
                ("--vortex-class", args.vortex_class),
                ("--all-vortex-classes", args.all_vortex_classes or None),
                ("--target-n-minus", args.target_n_minus),
            )
            if value is not None
        ]
        if conflicting:
            raise SystemExit(
                f"--projection none is incompatible with {', '.join(conflicting)}: "
                "without a projection there is no sector to select, and no target "
                "is built for a vortex class to label. Drop the flag, or use "
                "--projection irrep."
            )
        if args.phases == "both":
            raise SystemExit(
                "--projection none requires --phases vmc: the phase-2 target is "
                "the manifold projected onto an irrep, so there is none to "
                "minimize the infidelity against when nothing is projected."
            )

    layer_values = layer_sweep(args)
    if args.ansatz in LAYERLESS_ANSATZE and len(layer_values) > 1:
        raise SystemExit(
            f"--ansatz {args.ansatz} has no layer axis (it applies a single "
            f"circulant-mixing block), so training {len(layer_values)} layer "
            f"counts would train the identical model {len(layer_values)} times "
            "-- exactly the silent no-op of the legacy sweep. Pass --layers 1."
        )
    return args


def layer_sweep(args):
    """Layer counts to train: the single `--layers` value if given, else the
    original `1..--max-layers` range."""
    if args.layers is not None:
        return [args.layers]
    return list(range(1, args.max_layers + 1))


def report_sample_budget(model, hi, n_samples, label):
    """Print the parameter count against `n_samples`, and warn when SR is
    about to run under-sampled.

    NetKet raises `InsufficientSamplesForSRWarning` once the parameters
    outnumber the samples, and that regime is not academic here: the first
    3x3 transformer run destabilized from step ~125 with npar=3745 against
    2048 samples (see `legacy/amputado/slurm_transformer_kitaev.py`). Printing the
    ratio up front turns "the run went strange halfway" into a line you can
    read before spending the cluster time. Counted on the unprojected model
    because the symmetry projection adds no parameters.
    """
    x = hi.random_state(nk.jax.PRNGKey(0), (2,))
    npar = int(nk.jax.tree_size(model.init(nk.jax.PRNGKey(0), x)))
    ratio = n_samples / npar if npar else float("inf")
    print(f"[{label}] npar={npar}  n_samples={n_samples}  ratio={ratio:.2f}x")
    if npar >= n_samples:
        print(
            f"[!] n_samples ({n_samples}) <= npar ({npar}): SR runs in the "
            f"regime NetKet flags as unstable. Raise --n-samples above {npar}, "
            f"or shrink the ansatz (--heads/--dk lower the width; d_model = "
            f"heads*dk for the attention ansatze)."
        )
    return npar


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    results_dir = output_dir / "results"

    projected = args.projection == "irrep"
    run_phase2 = args.phases == "both"

    ansatze = [args.ansatz]

    jz_values = sorted({round(float(v), 6) for v in np.linspace(args.jz_min, args.jz_max, args.jz_steps)})
    if len(jz_values) < args.jz_steps:
        print(
            f"[!] --jz-min/--jz-max/--jz-steps pedian {args.jz_steps} puntos "
            f"pero colapsan a {len(jz_values)} valor(es) unico(s): {jz_values}. "
            f"Se ejecuta cada Jz una sola vez."
        )

    graph, hi = build_kitaev_lattice(extent=args.extent, pbc=not args.no_pbc)
    N = graph.n_nodes
    symmetries = get_kitaev_symmetries(graph, hi)

    plaquettes, plaquette_ops = get_kitaev_plaquettes(graph)
    wilson_loops = build_wilson_loops(hi, plaquettes, plaquette_ops)
    wilson_sparse = [w.to_sparse() for w in wilson_loops]
    staggered_mag, uniform_mag = build_magnetization_observables(hi, N)

    sampler = build_sampler(hi, graph)
    lr_schedule_phase1 = optax.warmup_exponential_decay_schedule(
        init_value=PHASE1_LR_INIT, peak_value=PHASE1_LR_PEAK,
        warmup_steps=30, transition_steps=100, decay_rate=0.95,
    )

    try:
        exact_results = load_exact_results(args.exact_data)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{args.exact_data} not found. Run `python common/analysis/generate_exact_data.py` first."
        ) from exc

    runs_csv = results_dir / "runs.csv"
    commit = git_commit()

    sector_sg, sector_ct = get_projection_group(symmetries, args.group)
    plaquette_perms = plaquette_permutations(plaquettes, sector_sg) + plaquette_permutations(
        plaquettes, c2xy_translation_group(graph)
    )
    clip_norm = None if args.clip_norm is not None and args.clip_norm < 0 else args.clip_norm

    for ansatz in ansatze:
        build_model = ANSATZ_BUILDERS[ansatz]
        for layers in layer_sweep(args):
            report_sample_budget(
                build_model(layers, args, graph), hi, args.n_samples,
                f"{ansatz} L{layers}",
            )
            for jz in jz_values:
                jz_key = min(exact_results, key=lambda k: abs(k - jz))
                result = exact_results[jz_key]
                eigvecs = result["eigenvectors"]
                manifold_idx = degenerate_manifold(np.asarray(result["energies"]).real)

                if projected:
                    weights = manifold_irrep_weights(
                        eigvecs, manifold_idx, hi, sector_sg, sector_ct
                    )
                    hosting = sectors_hosting_manifold(weights)
                    print(
                        f"Jz={jz:.2f}: group={args.group} manifold={len(manifold_idx)} "
                        f"hosting_sectors={hosting} weights="
                        f"{ {kk: round(vv, 3) for kk, vv in weights.items() if abs(vv) > 1e-3} }"
                    )

                    if args.sectors is not None:
                        missing = [k for k in args.sectors if k not in hosting]
                        if missing:
                            raise SystemExit(
                                f"Jz={jz:.2f}: sector(s) {missing} do not host this "
                                f"manifold (hosting={hosting}). Training them would "
                                f"chase a target of near-zero projection norm, so "
                                f"this stops rather than wasting the run."
                            )
                        sectors = list(args.sectors)
                        print(f"Jz={jz:.2f}: restricted to sectors {sectors}")
                    else:
                        sectors = list(hosting)

                    resolved = vortex_resolved_manifold(
                        eigvecs, manifold_idx, wilson_sparse, plaquette_perms
                    )
                    classes = resolved["classes"]
                    if args.target_n_minus is not None:
                        classes = [c for c in classes if c["n_minus"] == args.target_n_minus]
                        if not classes:
                            raise SystemExit(
                                f"Jz={jz:.2f}: no vortex sector with N_-="
                                f"{args.target_n_minus} (available "
                                f"{[c['n_minus'] for c in resolved['classes']]})."
                            )
                    if args.all_vortex_classes:
                        vortex_classes = list(range(len(classes)))
                    elif args.vortex_class is not None:
                        vortex_classes = list(args.vortex_class)
                        out_of_range = [c for c in vortex_classes if not 0 <= c < len(classes)]
                        if out_of_range:
                            raise SystemExit(
                                f"Jz={jz:.2f}: --vortex-class {out_of_range} out of range; "
                                f"this level has {len(classes)} vortex sector(s) "
                                f"(N_-={[c['n_minus'] for c in classes]})."
                            )
                    else:
                        vortex_classes = [0]
                    print(
                        f"Jz={jz:.2f}: {len(resolved['classes'])} vortex sector(s) "
                        f"N_-={[c['n_minus'] for c in resolved['classes']]}, "
                        f"training class(es) {vortex_classes}"
                    )
                else:
                    sectors, vortex_classes = [-1], [None]
                    print(
                        f"Jz={jz:.2f}: unprojected ansatz, manifold="
                        f"{len(manifold_idx)} (sector/vortex analysis skipped)"
                    )

                jx = jy = (1 - jz) / 2
                H = KitaevTransverse_H(graph.edge_colors, graph.edges(), Jx=jx, Jy=jy, Jz=jz, h=0, hi=hi)

                for k, vclass in itertools.product(sectors, vortex_classes):
                    target = None
                    perms = characters = None
                    vs_target = None
                    if projected:
                        target = pick_target_for_sector(
                            eigvecs, manifold_idx, hi, sector_sg, sector_ct, k,
                            wilson_loops=wilson_sparse,
                            plaquette_perms=plaquette_perms,
                            class_index=vclass,
                            n_minus=args.target_n_minus,
                        )
                        g_k, norm = target["vector"], target["norm"]
                        print(
                            f"Jz={jz:.2f} sector k={k} vortex class {vclass}: "
                            f"norma de la proyección = {norm:.4f}"
                        )

                        perms, characters = symmetry_projector_inputs(
                            symmetries, k, group=args.group
                        )
                        if run_phase2:
                            if args.exact:
                                vs_target = load_target_state_exact(hi, g_k)
                            else:
                                vs_target = load_target_state_from_vector(
                                    hi, sampler, g_k, args.n_samples
                                )

                    phase1_projected = projected and not args.unprojected_phase1
                    phase1_label = "projected" if phase1_projected else "unprojected"
                    print(f"--- Phase 1: {phase1_label}, Jz={jz}, sector k={k} ---")
                    model_phase1 = (
                        build_model(layers, args, graph, symmetries=perms, characters=characters)
                        if phase1_projected
                        else build_model(layers, args, graph)
                    )
                    best_phase1, metrics_phase1 = run_ground_state(
                        H, sampler, model_phase1, lr_schedule_phase1, args.epochs1, N, args.n_samples,
                        diag_shift=args.diag_shift,
                        optimizer=args.optimizer, clip_norm=clip_norm, exact=args.exact,
                        seed=args.seed,
                    )

                    model_phase2 = best_phase2 = None
                    metrics_phase2 = {}
                    if run_phase2:
                        print(f"--- Phase 2: projected to irrep {k}, infidelity ---")
                        model_phase2 = build_model(
                            layers, args, graph, symmetries=perms, characters=characters
                        )
                        best_phase2, metrics_phase2 = run_infidelity_projection(
                            H, vs_target, sampler, model_phase2, best_phase1, args.epochs2, N,
                            wilson_loops=wilson_loops, n_samples=args.n_samples,
                            target_flux_pattern=target["pattern"],
                            target_n_minus=target["n_minus"],
                            optimizer=args.optimizer, clip_norm=clip_norm,
                            collapse_guard=not args.no_collapse_guard,
                            plateau_stop=not args.no_plateau_stop,
                            plateau_patience=args.plateau_patience,
                            plateau_min_rel_improvement=args.plateau_min_improvement,
                            plateau_min_steps=args.plateau_min_steps,
                            exact=args.exact,
                            seed=args.seed,
                        )

                    measured = best_phase2 if run_phase2 else best_phase1

                    lx, ly = args.extent
                    sector_tag = f"k{k}_v{vclass}" if projected else "knone"
                    tag = (
                        f"{ansatz}_{lx}x{ly}_L{layers}_jz{jz:.2f}"
                        f"_{sector_tag}_s{args.seed}_{args.iteration}"
                    )
                    if run_phase2:
                        save_checkpoint(
                            best_phase2,
                            checkpoints_dir / f"{tag}.mpack",
                            checkpoints_dir / f"vstate_{tag}.pkl",
                        )
                    save_checkpoint(
                        best_phase1,
                        checkpoints_dir / f"ph1_{tag}.mpack",
                        checkpoints_dir / f"vstate_ph1_{tag}.pkl",
                    )

                    psi_ph1 = best_phase1.to_array()
                    fidelity_manifold_ph1 = manifold_fidelity(eigvecs, manifold_idx, psi_ph1)
                    fidelity_sector_ph1 = (
                        float(np.abs(np.vdot(g_k, psi_ph1)) ** 2) if projected else None
                    )
                    energy_total_ph1 = float(np.real(best_phase1.expect(H).mean))

                    if run_phase2:
                        psi_nqs = best_phase2.to_array()
                        fidelity_manifold = manifold_fidelity(eigvecs, manifold_idx, psi_nqs)
                        fidelity_sector = float(np.abs(np.vdot(g_k, psi_nqs)) ** 2)
                        energy_stats = best_phase2.expect(H)
                        energy_total = float(np.real(energy_stats.mean))
                        energy_error = float(np.real(energy_stats.error_of_mean))
                    else:
                        fidelity_manifold = fidelity_sector = None
                        energy_total = energy_error = None

                    wp_val = float(np.real(measured.expect(wilson_loops[0]).mean))
                    m_val, ms_val, fluct_val, fluct_s_val = (
                        float(np.real(measured.expect(o).mean))
                        for o in (uniform_mag, staggered_mag, uniform_mag @ uniform_mag, staggered_mag @ staggered_mag)
                    )

                    e_ed = float(result["E0"])
                    energy_measured = energy_total if run_phase2 else energy_total_ph1
                    delta_eps = abs(energy_measured - e_ed) / abs(e_ed) if e_ed else float("nan")
                    row = {
                        "run_id": f"{tag}@{commit}",
                        "tag": tag,
                        "ansatz": ansatz,
                        "extent_x": lx,
                        "extent_y": ly,
                        "N": N,
                        "layers": layers,
                        "Jz": jz,
                        "sector": k,
                        "seed": args.seed,
                        "iteration": args.iteration,
                        "git_commit": commit,
                        "group": args.group,
                        "projector": args.projector,
                        "optimizer": args.optimizer,
                        "n_samples": args.n_samples,
                        "diag_shift": args.diag_shift,
                        "lr_phase1_peak": PHASE1_LR_PEAK,
                        "epochs1": args.epochs1,
                        "epochs2": args.epochs2,
                        "vscore_threshold": args.vscore_threshold,
                        "alpha": args.alpha,
                        "heads": args.heads,
                        "dk": args.dk,
                        "dtype": params_dtype(measured),
                        "sampler": "exact" if args.exact else "metropolis",
                        "clip_norm": clip_norm,
                        "phase1_projected": phase1_projected,
                        "projection": args.projection,
                        "energy_total_ph1": energy_total_ph1,
                        "energy_per_site_ph1": energy_total_ph1 / N,
                        "vscore_ph1": metrics_phase1.get("best_vscore", float("nan")),
                        "fidelity_sector_ph1": fidelity_sector_ph1,
                        "fidelity_manifold_ph1": fidelity_manifold_ph1,
                        "n_steps_ph1": len(metrics_phase1["step"]),
                        "n_iter_convergencia_ph1": first_step_below(
                            metrics_phase1["vscore"], metrics_phase1["step"],
                            args.vscore_threshold,
                        ),
                        "energy_total_ph2": energy_total,
                        "energy_per_site_ph2": energy_total / N if run_phase2 else None,
                        "energy_error_ph2": energy_error,
                        "vscore_ph2": metrics_phase2.get("best_vscore", float("nan")),
                        "infidelity": (
                            metrics_phase2["infidelity"][-1]
                            if metrics_phase2.get("infidelity") else float("nan")
                        ),
                        "best_infidelity": metrics_phase2.get("best_infidelity", float("nan")),
                        "fidelity_sector": fidelity_sector,
                        "fidelity_manifold": fidelity_manifold,
                        "collapsed": metrics_phase2.get("collapsed", False) if run_phase2 else None,
                        "collapse_reason": metrics_phase2.get("collapse_reason"),
                        "stop_reason": metrics_phase2.get("stop_reason"),
                        "n_steps_ph2": len(metrics_phase2["step"]) if run_phase2 else None,
                        "n_iter_convergencia_ph2": first_step_below(
                            metrics_phase2["vscore"], metrics_phase2["step"],
                            args.vscore_threshold,
                        ) if run_phase2 else None,
                        "E_ED": e_ed,
                        "delta_eps": delta_eps,
                        "proj_norm": norm if projected else None,
                        "target_n_minus": target["n_minus"] if projected else None,
                        "target_flux_pattern": json.dumps(target["pattern"]) if projected else None,
                        "target_vortex_plaquette_idx": (
                            json.dumps(target["vortex_plaquette_idx"]) if projected else None
                        ),
                        "target_n_equivalent_placements": (
                            target["n_equivalent_placements"] if projected else None
                        ),
                        "target_vortex_class": target["class_index"] if projected else None,
                        "target_n_vortex_classes": target["n_classes"] if projected else None,
                        "target_flux_ambiguous": target["ambiguous"] if projected else None,
                        "target_n_distinct_patterns": (
                            target["n_distinct_patterns"] if projected else None
                        ),
                        "m": m_val,
                        "ms": ms_val,
                        "fluct": fluct_val,
                        "fluct_s": fluct_s_val,
                        "Wp": wp_val,
                        "n_params": int(nk.jax.tree_size(measured.parameters)),
                        "wall_clock_ph1": metrics_phase1.get("wall_clock", float("nan")),
                        "wall_clock_ph2": metrics_phase2.get("wall_clock", float("nan")),
                        "time_per_step_ph1": metrics_phase1.get("time_per_step", float("nan")),
                        "time_per_step_ph2": metrics_phase2.get("time_per_step", float("nan")),
                        **empty_reserved(),
                    }
                    append_observables_csv(
                        runs_csv, list(RUN_COLUMNS),
                        row_to_list(row, RUN_COLUMNS, context=f"run {tag}"),
                    )

                    step_metrics_frame(metrics_phase1).to_csv(
                        results_dir / f"metrics_ph1_{tag}.csv", index=False
                    )
                    if run_phase2:
                        step_metrics_frame(metrics_phase2).to_csv(
                            results_dir / f"metrics_ph2_{tag}.csv", index=False
                        )

                    del model_phase1, model_phase2, best_phase1, best_phase2
                    del vs_target, measured
                    gc.collect()

                del H
                gc.collect()


if __name__ == "__main__":
    main()
