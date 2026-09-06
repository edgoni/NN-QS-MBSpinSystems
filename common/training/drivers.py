import time
from typing import Optional, Sequence

import netket as nk
import netket.experimental as nkx
import numpy as np
import optax

from .callbacks import (
    BestEnergyCheckpoint,
    BestOverlapCheckpoint,
    InfidelityCollapseGuard,
    InfidelityPlateauStopper,
    StepEnergyStats,
    build_observables_logger,
)

N_SAMPLES_DEFAULT = 2048


def build_sampler(hi, graph, rule_weights: Sequence[float] = (0.9, 0.1)):
    """Metropolis sampler mixing single-spin flips and neighbor exchanges."""
    rule1 = nk.sampler.rules.LocalRule()
    rule2 = nk.sampler.rules.ExchangeRule(graph=graph)
    return nk.sampler.MetropolisSampler(
        hi, nk.sampler.rules.MultipleRules([rule1, rule2], list(rule_weights))
    )


def build_optimizer(learning_rate, optimizer: str = "sgd", clip_norm: Optional[float] = 5.0):
    """Build the optax optimizer used by both training phases.

    :param optimizer: "sgd" (default) or "adam" (original default, kept
        available for backward compatibility / comparison). `Sgd` is
        recommended whenever gradients are SR/NGD-preconditioned: NetKet's
        own `Infidelity_SR` docstring states the optimizer "should be an
        instance of optax.sgd. Other optimizers ... will not make
        mathematical sense", and `docs/DEVLOG.md` (2026-08-06) measured this
        swap taking a comparable run's error from 27.9% to 11.1% on this
        project.
    :param clip_norm: if not None, chains `optax.clip_by_global_norm(clip_norm)`
        before the optimizer. A symmetry-projected sum vanishes exactly for
        any non-trivial irrep on a group-symmetric input (`sum_g chi(g)=0`
        for mu != trivial), which can produce a very large local gradient
        for a sampled configuration; clipping bounds a single such step
        without lowering the global learning rate (see
        `Projected_Energy_min/try_Norman.py`, which added this for the same reason).
        None disables clipping (original behavior).
    """
    if optimizer == "sgd":
        base = nk.optimizer.Sgd(learning_rate=learning_rate)
    elif optimizer == "adam":
        base = nk.optimizer.Adam(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer {optimizer!r}, expected 'sgd' or 'adam'")

    if clip_norm is None:
        return base
    return optax.chain(optax.clip_by_global_norm(clip_norm), base)


def build_vstate(
    hi, sampler, model, n_samples=N_SAMPLES_DEFAULT, exact: bool = False,
    seed: Optional[int] = None,
):
    """Build the variational state for a training phase.

    :param exact: if True, build a `nk.vqs.FullSumState` (enumerates the
        whole Hilbert space, exact gradients, no Monte-Carlo noise) instead
        of a sampled `nk.vqs.MCState`. Only affordable for small systems
        (2**N amplitudes), but it removes sampling noise entirely -- which
        is what `docs/proyeccion_simetria_rbm_kitaev.md` used for its 2x2
        benchmarks.
    :param seed: seeds parameter initialization (and, for an `MCState`, the
        sampler chains). None keeps NetKet's non-reproducible default. The
        seed is recorded per run because S12 (V-score spread over ~5 seeds)
        and S6 (relative-phase spread between independent trainings of the
        same irrep) are both seed sweeps -- without it, "run it again with a
        different seed" is not a reproducible instruction.
    """
    if exact:
        return nk.vqs.FullSumState(hilbert=hi, model=model, seed=seed)
    return nk.vqs.MCState(
        sampler, model=model, n_samples=n_samples, seed=seed, sampler_seed=seed
    )


def setup_vmc_driver(
    H, sampler, model, learning_rate, n_samples=N_SAMPLES_DEFAULT, diag_shift=0.01,
    optimizer: str = "sgd", clip_norm: Optional[float] = 5.0, exact: bool = False,
    seed: Optional[int] = None,
):
    """Build a variational state + SR-preconditioned `nk.driver.VMC` pair."""
    vstate = build_vstate(
        H.hilbert, sampler, model, n_samples=n_samples, exact=exact, seed=seed
    )
    sr = nk.optimizer.SR(diag_shift=diag_shift, holomorphic=False)
    opt = build_optimizer(learning_rate, optimizer=optimizer, clip_norm=clip_norm)
    driver = nk.driver.VMC(H, opt, variational_state=vstate, preconditioner=sr)
    return driver, vstate


def default_infidelity_lr_schedule(epochs):
    return optax.join_schedules(
        schedules=[
            optax.constant_schedule(2e-2),
            optax.constant_schedule(1e-2),
            optax.constant_schedule(5e-3),
            optax.constant_schedule(1e-3),
        ],
        boundaries=[int(epochs * 0.2), int(epochs * 0.5), int(epochs * 0.7)],
    )


def default_infidelity_diag_schedule(epochs: Optional[int] = None):
    """Piecewise-constant diag_shift schedule for Phase 2.

    :param epochs: if given, boundaries are scaled to this many total steps
        (same fractions as `default_infidelity_lr_schedule`: 0.2/0.5/0.7) so
        a short `epochs2` run actually traverses the schedule instead of
        getting stuck at the first (largest, least-regularizing) diag_shift
        value for its whole duration. If None, keeps the original absolute
        step boundaries `[200, 800, 2000]` (kept for backward compatibility
        / comparison -- this is what a long enough run, e.g. `epochs2>2000`,
        traverses regardless of which mode is used).
    """
    if epochs is None:
        boundaries = [200, 800, 2000]
    else:
        boundaries = [int(epochs * 0.2), int(epochs * 0.5), int(epochs * 0.7)]
    return optax.join_schedules(
        schedules=[optax.constant_schedule(x) for x in (0.1, 0.01, 1e-3, 1e-4)],
        boundaries=boundaries,
    )


def run_ground_state(
    H, sampler, model, learning_rate, epochs, N, n_samples=N_SAMPLES_DEFAULT, diag_shift=0.01,
    optimizer: str = "sgd", clip_norm: Optional[float] = 5.0, exact: bool = False,
    seed: Optional[int] = None,
):
    """Phase 1: plain VMC/SR energy minimization.

    :param exact: use an exact `FullSumState` instead of sampling; see `build_vstate`
    :return: (best_state, metrics_history). `metrics_history` also carries
        the scalar summaries of the run (`best_energy`, `best_vscore`,
        `wall_clock`, `time_per_step`) alongside the per-step lists, so the
        caller does not need the checkpoint object to build its results row.
    """
    driver, _ = setup_vmc_driver(
        H, sampler, model, learning_rate, n_samples, diag_shift,
        optimizer=optimizer, clip_norm=clip_norm, exact=exact, seed=seed,
    )
    energy_stats = StepEnergyStats(H, from_driver_loss=True)
    checkpoint = BestEnergyCheckpoint(H, N, baseline=1e-8, energy_stats=energy_stats)
    metrics_history = {
        "step": [],
        "energy": [],
        "energy_error": [],
        "energy_variance": [],
        "vscore": [],
        "loss_variance": [],
    }

    t0 = time.perf_counter()
    driver.run(
        n_iter=epochs,
        out=nk.logging.RuntimeLog(),
        callback=[
            checkpoint.update,
            build_observables_logger(metrics_history, H, N, energy_stats=energy_stats),
        ],
        show_progress=True,
    )
    _record_timing(metrics_history, t0)
    metrics_history["best_energy"] = float(checkpoint.best_energy)
    metrics_history["best_vscore"] = float(checkpoint.vscore)
    return checkpoint.best_state, metrics_history


def _record_timing(metrics_history: dict, t0: float) -> None:
    """Stamp wall-clock cost onto a finished run's metrics.

    `time_per_step` divides by the number of steps actually recorded, not
    by the requested `epochs`, so an early stop (the V-score baseline in
    phase 1, `InfidelityCollapseGuard` in phase 2) reports the real
    per-step cost instead of an artificially small one. T4 of the guideline
    wants seconds, not "fast".
    """
    wall_clock = time.perf_counter() - t0
    n_steps = len(metrics_history["step"])
    metrics_history["wall_clock"] = wall_clock
    metrics_history["time_per_step"] = wall_clock / n_steps if n_steps else float("nan")


class TargetFluxMismatch(ValueError):
    """The phase-2 target does not carry the flux pattern it was labeled with."""


def _verify_target_flux(
    target_state, wilson_loops, target_flux_pattern=None, target_n_minus=None, tol=1e-6
):
    """Check the target really is in the vortex sector the caller claims.

    Cheap (one pass over the target's amplitudes, once per run) and worth it:
    a target built from the wrong flux sector trains perfectly happily and
    produces an infidelity curve that looks like a hard optimization problem
    rather than the wrong-target problem it is.

    What is checkable depends on the target, and getting this wrong rejects
    perfectly good targets. `H` commutes with both the plaquette operators and
    the translations, but translations *permute* plaquettes, so `W_p_i` and
    `T` do not commute with each other. A target projected onto a momentum
    irrep therefore cannot also be a per-plaquette `W_p` eigenstate unless its
    vortex pattern is translation-invariant -- when the sector has several
    symmetry-equivalent placements, the momentum projection superposes exactly
    those placements and every `<W_p_i>` comes out at 0.

    What always survives is the *total* flux `sum_i W_p_i`, which commutes
    with translations and equals `n_plaq - 2*N_-` on the whole sector. So:

    * `target_n_minus` -- checked always, via the total flux. This is the
      label `pick_target_for_sector` can promise for any target.
    * `target_flux_pattern` -- the stronger per-plaquette check, run only when
      the caller supplies a pattern, which `pick_target_for_sector` does only
      for a sector with a single placement.

    Silently returns when there is nothing to check, or no Wilson loops to
    measure it with.
    """
    if not wilson_loops or (target_flux_pattern is None and target_n_minus is None):
        return

    psi = np.asarray(target_state.to_array())
    psi = psi / np.linalg.norm(psi)
    wp_sparse = [w if hasattr(w, "nnz") else w.to_sparse() for w in wilson_loops]
    measured = np.array([float(np.real(np.vdot(psi, W @ psi))) for W in wp_sparse])
    n_plaquettes = len(measured)

    if target_flux_pattern is not None:
        expected = np.asarray(target_flux_pattern, dtype=float)
        if len(expected) != n_plaquettes:
            raise TargetFluxMismatch(
                f"target_flux_pattern has {len(expected)} entries but there are "
                f"{n_plaquettes} plaquettes."
            )
        deviation = np.abs(measured - expected)
        if np.max(deviation) > tol:
            worst = int(np.argmax(deviation))
            raise TargetFluxMismatch(
                f"target is not in the declared flux sector: plaquette {worst} has "
                f"<W_p> = {measured[worst]:+.6f}, expected {expected[worst]:+.0f} "
                f"(max deviation {np.max(deviation):.2e} over {n_plaquettes} "
                f"plaquettes). Measured pattern: {[round(float(v), 3) for v in measured]}."
            )
        if target_n_minus is None:
            target_n_minus = int(np.sum(expected < 0))

    if target_n_minus is not None:
        expected_total = n_plaquettes - 2 * int(target_n_minus)
        measured_total = float(np.sum(measured))
        if abs(measured_total - expected_total) > tol * max(1, n_plaquettes):
            raise TargetFluxMismatch(
                f"target carries the wrong vortex number: <sum W_p> = "
                f"{measured_total:+.6f}, expected {expected_total:+d} for "
                f"N_minus={target_n_minus} over {n_plaquettes} plaquettes. "
                f"Measured <W_p_i> = {[round(float(v), 3) for v in measured]}."
            )
        print(
            f"[phase 2] target vortex sector verified: N_minus={target_n_minus}, "
            f"<sum W_p>={measured_total:+.6f}/{n_plaquettes}"
        )


def run_infidelity_projection(
    H,
    target_state,
    sampler,
    model,
    init_state,
    epochs,
    N,
    wilson_loops: Optional[Sequence] = None,
    lr_schedule=None,
    diag_schedule=None,
    n_samples=N_SAMPLES_DEFAULT,
    optimizer: str = "sgd",
    clip_norm: Optional[float] = 5.0,
    collapse_guard: bool = True,
    plateau_stop: bool = True,
    plateau_patience: int = 750,
    plateau_min_rel_improvement: float = 0.05,
    plateau_min_steps: int = 500,
    exact: bool = False,
    seed: Optional[int] = None,
    target_flux_pattern: Optional[Sequence[int]] = None,
    target_n_minus: Optional[int] = None,
):
    """Phase 2: transfer-learn `init_state`'s weights into `model` (typically
    the symmetry-projected version of the phase-1 ansatz) and minimize the
    infidelity against `target_state`.

    The target is supplied by the caller, and *which* state that is carries a
    physics choice this driver used to be silent about. A target has two
    independent labels: the spatial irrep it was projected onto, and the flux
    (vortex) pattern of the energy level it came from. Nothing here assumes
    the level is vortex-free -- on the 3x3 torus it is not for Jz >= 0.5 --
    but the label has to be *recorded*, or a run's infidelity cannot be read
    against the state it was actually chasing. Pass `target_flux_pattern`
    (from `src.physics.exact_diag.pick_target_for_sector`) and it is checked
    against the target and reported back in `metrics_history`.

    :param init_state: the phase-1 `MCState` (or its best checkpoint) whose
        parameters seed this phase's variational state
    :param target_n_minus: how many vortices the target is supposed to carry.
        Checked before training starts via the total flux `sum_i W_p_i`, the
        one flux label a momentum-projected target can always carry (see
        `_verify_target_flux`), and echoed into `metrics_history` so it lands
        in the results row.
    :param target_flux_pattern: the stronger, per-plaquette +-1 pattern, when
        the target's vortex sector has a single symmetry-equivalent placement
        and therefore does have a definite one. None skips that check; use
        `target_n_minus` on its own otherwise.
    :param epochs: *maximum* number of steps. With `plateau_stop` (the
        default) this is a safety ceiling, not a target: the run normally
        ends when the best infidelity stops improving. Only `plateau_stop=False`
        makes it the actual stopping criterion again.
    :param wilson_loops: optional plaquette operators to track per step
    :param optimizer: "sgd" (default) or "adam"; see `build_optimizer`
    :param clip_norm: gradient-norm clip, or None to disable; see `build_optimizer`
    :param collapse_guard: stop early when the infidelity estimator hits its
        dead-gradient fixed point (`I == 0.5` at zero variance) or diverges;
        see `src.training.callbacks.InfidelityCollapseGuard`. Set False to
        keep the original behavior of always running the full `epochs`.
    :param plateau_stop: stop early once the best infidelity has stopped
        improving; see `src.training.callbacks.InfidelityPlateauStopper`.
        Orthogonal to `collapse_guard`: that one catches runs that broke,
        this one catches runs that finished.
    :param plateau_patience: window in steps the plateau is measured over
    :param plateau_min_rel_improvement: relative improvement of the best
        infidelity required within that window to keep going
    :param plateau_min_steps: warmup steps before the plateau check arms
    :param exact: use an exact `FullSumState` instead of sampling; see
        `build_vstate`. `target_state` must then also be a `FullSumState`
        (see `src.utils.io.load_target_state_exact`), because NetKet's
        sampled infidelity estimator reads `target_state.samples`, which a
        `FullSumState` does not have. Note that in exact mode the infidelity
        is computed in closed form and `error_of_mean` is identically 0, so
        the collapse guard's zero-variance discriminator no longer
        discriminates -- there its check reduces to "I stayed within `eps`
        of 0.5 for `patience` steps", still a valid stall signal.
    :return: (best_state, metrics_history)
    """
    _verify_target_flux(
        target_state, wilson_loops, target_flux_pattern, target_n_minus
    )

    vstate = build_vstate(
        H.hilbert, sampler, model, n_samples=n_samples, exact=exact, seed=seed
    )
    vstate.variables = init_state.variables

    lr_schedule = lr_schedule if lr_schedule is not None else default_infidelity_lr_schedule(epochs)
    diag_schedule = diag_schedule if diag_schedule is not None else default_infidelity_diag_schedule(epochs)
    opt = build_optimizer(lr_schedule, optimizer=optimizer, clip_norm=clip_norm)

    driver = nkx.driver.Infidelity_SR(
        target_state=target_state,
        optimizer=opt,
        diag_shift=diag_schedule,
        variational_state=vstate,
    )

    energy_stats = StepEnergyStats(H)
    checkpoint = BestOverlapCheckpoint(
        H, N, baseline=1e-8, stop_variance=True, energy_stats=energy_stats
    )
    metrics_history = {
        "step": [],
        "infidelity": [],
        "infidelity_error": [],
        "energy": [],
        "energy_error": [],
        "energy_variance": [],
        "vscore": [],
        "loss_variance": [],
        "wp_mean": [],
    }
    infidelity_op = nkx.observable.InfidelityOperator(target_state=target_state, cv_coeff=-0.5)

    def infidelity_callback(step, log_data, drv):
        infid = drv.estimate(infidelity_op)
        log_data["Infidelity"] = infid
        print(f"Step {step}  I = {infid.mean:.6f} +/- {infid.error_of_mean:.2e}")
        return True

    guard = InfidelityCollapseGuard() if collapse_guard else None
    callbacks = [infidelity_callback]
    if guard is not None:
        callbacks.append(guard.update)
    callbacks += [
        checkpoint.update,
        build_observables_logger(
            metrics_history, H, N, wilson_loops=wilson_loops, energy_stats=energy_stats
        ),
    ]

    stopper = (
        InfidelityPlateauStopper(
            checkpoint,
            patience=plateau_patience,
            min_rel_improvement=plateau_min_rel_improvement,
            min_steps=plateau_min_steps,
        )
        if plateau_stop
        else None
    )
    if stopper is not None:
        callbacks.append(stopper.update)

    t0 = time.perf_counter()
    driver.run(
        n_iter=epochs,
        out=nk.logging.RuntimeLog(),
        callback=callbacks,
        show_progress=True,
    )
    _record_timing(metrics_history, t0)

    best_state = checkpoint.best_state
    if best_state is None:
        print(
            "[run_infidelity_projection] no checkpoint recorded "
            f"({guard.trigger_reason if guard is not None else 'unknown reason'}); "
            "returning the final variational state."
        )
        best_state = driver.state

    metrics_history["best_infidelity"] = checkpoint.best_infid
    metrics_history["best_energy"] = float(checkpoint.best_energy)
    metrics_history["best_vscore"] = float(checkpoint.vscore)
    metrics_history["collapsed"] = bool(guard.triggered) if guard is not None else False
    metrics_history["collapse_reason"] = guard.trigger_reason if guard is not None else None
    metrics_history["plateaued"] = bool(stopper.triggered) if stopper is not None else False
    metrics_history["plateau_reason"] = stopper.trigger_reason if stopper is not None else None
    if guard is not None and guard.triggered:
        stop_reason = "collapsed"
    elif stopper is not None and stopper.triggered:
        stop_reason = "plateau"
    else:
        stop_reason = "max_epochs"
    metrics_history["stop_reason"] = stop_reason
    metrics_history["target_flux_pattern"] = (
        list(target_flux_pattern) if target_flux_pattern is not None else None
    )
    metrics_history["target_n_minus"] = target_n_minus
    return best_state, metrics_history
