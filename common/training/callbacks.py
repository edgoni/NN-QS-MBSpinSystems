import copy
import pathlib
from collections import deque
from typing import Optional

import flax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt


def vscore_from_energy_stats(stats, N: int) -> float:
    """V-score of an energy estimate, `V = N * Var(E) / <E>^2`.

    This takes the `Stats` object returned by `vstate.expect(H)` rather than
    the driver's own loss statistics, because the latter are only the energy
    in phase 1. In phase 2 the driver is `Infidelity_SR`, whose
    `_loss_name` is "Infidelity", so `log_data[driver._loss_name].variance`
    there is Var(I) and `N*Var(I)/<I>^2` is not a V-score of anything. Every
    V-score in this project goes through this function so the two phases
    cannot silently mean different quantities.

    Note the `E_inf = 0` convention (guideline §8): the definition is
    `N*Var(E)/(<E> - E_inf)^2` and we take `E_inf = 0`. That has to be
    declared alongside any V-score we report, or the numbers are not
    comparable with the literature.
    """
    mean = float(np.real(stats.mean))
    variance = float(np.real(stats.variance))
    if mean == 0.0:
        return float("inf")
    return N * variance / mean**2


class StepEnergyStats:
    """One energy estimate per training step, shared by every callback.

    Both the checkpoint and the metrics logger need `<H>` on the current
    iterate, and each used to call `vstate.expect(H)` itself. That is not
    cached by NetKet -- the samples are reused while the parameters hold,
    but the local energies are recomputed every call -- so the two extra
    calls measured **1160-1415 ms per step** on 3x3 with 2048 samples,
    against a ~2.2 s step. Routing both through one instance of this class
    removes the duplication.

    :param from_driver_loss: read `log_data[driver._loss_name]` instead of
        calling `expect`. Only valid when the driver's loss *is* the energy,
        i.e. phase 1 (`nk.driver.VMC`). It is then free, because the driver
        has already computed it, and verified bit-identical to
        `driver.state.expect(H)` at callback time (max deviation 1.6e-15
        over every step of both a `FullSumState` and an `MCState` run), so
        nothing recorded changes and the same iterate is checkpointed.

        It must stay False in phase 2, where the driver is `Infidelity_SR`
        and `log_data[driver._loss_name]` is Var(I), not the energy -- the
        exact confusion `vscore_from_energy_stats` exists to prevent. There
        this class still earns its place by memoizing the one `expect` call.

    Memoized on `step` alone, which is sound because each phase calls
    `driver.run` exactly once: a second run would restart the step counter
    and hand back the stale first step.
    """

    def __init__(self, Hamiltonian: npt.ArrayLike, from_driver_loss: bool = False):
        self.Hamiltonian = Hamiltonian
        self.from_driver_loss = from_driver_loss
        self._step = None
        self._stats = None

    def __call__(self, step, log_data, driver):
        if self._stats is None or self._step != step:
            if self.from_driver_loss:
                self._stats = log_data[driver._loss_name]
            else:
                self._stats = driver.state.expect(self.Hamiltonian)
            self._step = step
        return self._stats


def _resolve_energy_stats(energy_stats, Hamiltonian, step, log_data, driver):
    """`energy_stats(step, log_data, driver)` if one was supplied, else the
    unshared `driver.state.expect(H)` the callbacks have always used."""
    if energy_stats is None:
        return driver.state.expect(Hamiltonian)
    return energy_stats(step, log_data, driver)


class BestEnergyCheckpoint:
    """Keeps a copy of the variational state at its lowest-energy iterate.

    :param Hamiltonian: the Hamiltonian operator to evaluate at each step
    :param N: number of spins, used to normalize the V-score
    :param baseline: V-score threshold below which training may stop early
    :param filename: optional path to persist the best state to on update
    :param stop_variance: if True, never stop early on the V-score baseline
        (useful when this callback's early-stopping is not the driver's
        actual termination condition, e.g. during infidelity training)
    :param energy_stats: optional `StepEnergyStats` shared with the metrics
        logger, so `<H>` is evaluated once per step instead of once per
        callback. None keeps the original unshared `expect` call.
    """

    def __init__(
        self,
        Hamiltonian: npt.ArrayLike,
        N: int,
        baseline: float,
        filename: Optional[pathlib.Path] = None,
        stop_variance: bool = False,
        energy_stats: Optional["StepEnergyStats"] = None,
    ):
        self.Hamiltonian = Hamiltonian
        self.N = N
        self.baseline = baseline
        self.filename = filename
        self.stop_variance = stop_variance
        self.energy_stats = energy_stats
        self.vscore = np.inf
        self.best_energy = np.inf
        self.best_state = None

    def update(self, step, log_data, driver):
        """NetKet training callback; see NetKet's callback API for details."""
        energy_stats = _resolve_energy_stats(
            self.energy_stats, self.Hamiltonian, step, log_data, driver
        )
        energy_step = float(np.real(energy_stats.mean))
        vscore_step = vscore_from_energy_stats(energy_stats, self.N)

        if self.best_energy > energy_step:
            self.best_energy = energy_step
            self.best_state = copy.copy(driver.state)
            self.best_state.parameters = flax.core.copy(driver.state.parameters)
            self.vscore = vscore_step
            if self.filename is not None:
                with open(self.filename, "wb") as f:
                    f.write(flax.serialization.to_bytes(driver.state))

        return True if self.stop_variance else self.vscore > self.baseline


class BestOverlapCheckpoint:
    """Keeps a copy of the variational state at its lowest-infidelity
    iterate. Same constructor/usage as `BestEnergyCheckpoint`, but tracks
    the "Infidelity" entry of `log_data` instead of the energy."""

    def __init__(
        self,
        Hamiltonian: npt.ArrayLike,
        N: int,
        baseline: float,
        filename: Optional[pathlib.Path] = None,
        stop_variance: bool = False,
        energy_stats: Optional["StepEnergyStats"] = None,
    ):
        self.Hamiltonian = Hamiltonian
        self.N = N
        self.baseline = baseline
        self.filename = filename
        self.stop_variance = stop_variance
        self.energy_stats = energy_stats
        self.vscore = np.inf
        self.best_energy = np.inf
        self.best_state = None
        self.best_infid = np.inf

    def update(self, step, log_data, driver):
        """NetKet training callback; see NetKet's callback API for details."""
        energy_stats = _resolve_energy_stats(
            self.energy_stats, self.Hamiltonian, step, log_data, driver
        )
        energy_step = float(np.real(energy_stats.mean))
        vscore_step = vscore_from_energy_stats(energy_stats, self.N)

        infidelity = np.inf
        if log_data.get("Infidelity") is not None:
            infidelity = float(jnp.real(log_data["Infidelity"].mean))

        if self.best_infid > infidelity:
            self.best_infid = infidelity
            self.best_energy = energy_step
            self.best_state = copy.copy(driver.state)
            self.best_state.parameters = flax.core.copy(driver.state.parameters)
            self.vscore = vscore_step
            if self.filename is not None:
                with open(self.filename, "wb") as f:
                    f.write(flax.serialization.to_bytes(driver.state))

        return True if self.stop_variance else self.vscore > self.baseline


class InfidelityCollapseGuard:
    """Detects the `Infidelity_SR` dead-gradient fixed point and stops the
    driver early instead of burning the rest of `epochs2` on it.

    With the optimal control-variate coefficient (`cv_coeff=-0.5`), the local
    infidelity estimator is `I_loc = 0.5 - Re[R] + 0.5|R|^2` where
    `R = (Phi(sigma)/psi(sigma)) * mean_{sigma_t}[psi(sigma_t)/Phi(sigma_t)]`
    (see `netket._src.observable.infidelity.expect`). `I == 0.5` with zero
    variance is therefore not "no progress yet" -- it is the signature of
    `R == 0` everywhere sampled, which also zeroes the (non-control-variate)
    local energies the SR/NGD update in `Infidelity_SR.compute_loss_and_update`
    is built from. The optimizer is then running on an identically-zero
    gradient and cannot recover on its own. This typically follows a
    `log(~0)` blow-up at the very start of a projected-ansatz run seeded
    from a symmetry-invariant state (any non-trivial irrep has
    `sum_g chi(g) = 0`, so a group-symmetric seed's projected amplitude is a
    near-perfect cancellation).

    Defaults are calibrated against a real collapsed run (3x3, Jz=0.6,
    sectors k=0,2,3,4): in the dead phase `I` hovers within ~2e-3 of 0.5
    while `error_of_mean` falls to 1e-4..1e-11, whereas the healthy opening
    steps of the same run sit at 0.494/0.455/0.538 with errors of
    2.9e-3..3.3e-2. `error_tol` is therefore the discriminating threshold --
    a genuinely-still-sampling `I` near 0.5 has error >~3e-3, a dead one has
    error orders of magnitude smaller. Replaying that log, these defaults
    stop the run at step 27 instead of burning all 2500 steps.

    :param eps: `|I - 0.5|` must be below this to count as "collapsed"
    :param error_tol: `error_of_mean` must be below this to count as
        "collapsed" (rules out a legitimately noisy `I` that merely happens
        to average near 0.5 for a step or two)
    :param patience: number of consecutive collapsed steps required before
        stopping (avoids a false positive on a transient plateau)
    :param max_infidelity: `I` above this counts as diverging -- the
        `log(~0)` blow-up signature itself
    :param divergence_patience: consecutive diverging steps required before
        stopping (a single large transient on step 0 is tolerated; a
        sustained blow-up is not). Non-finite `I` always stops immediately.
    """

    def __init__(
        self,
        eps: float = 2e-3,
        error_tol: float = 1e-3,
        patience: int = 25,
        max_infidelity: float = 10.0,
        divergence_patience: int = 3,
    ):
        self.eps = eps
        self.error_tol = error_tol
        self.patience = patience
        self.max_infidelity = max_infidelity
        self.divergence_patience = divergence_patience
        self._collapsed_streak = 0
        self._diverged_streak = 0
        self.triggered = False
        self.trigger_reason = None

    def _stop(self, reason):
        self.triggered = True
        self.trigger_reason = reason
        print(f"[InfidelityCollapseGuard] {reason} -- stopping.")
        return False

    def update(self, step, log_data, driver):
        """NetKet training callback; see NetKet's callback API for details."""
        infid = log_data.get("Infidelity")
        if infid is None:
            return True

        mean = float(jnp.real(infid.mean))
        error = float(jnp.real(infid.error_of_mean))

        if not np.isfinite(mean):
            return self._stop(f"non-finite infidelity at step {step}: I={mean}")

        if mean > self.max_infidelity:
            self._diverged_streak += 1
            if self._diverged_streak >= self.divergence_patience:
                return self._stop(
                    f"diverging infidelity at step {step}: I={mean:.6g} above "
                    f"{self.max_infidelity} for {self._diverged_streak} consecutive steps "
                    "(projected-amplitude cancellation blow-up)"
                )
        else:
            self._diverged_streak = 0

        if abs(mean - 0.5) < self.eps and error < self.error_tol:
            self._collapsed_streak += 1
        else:
            self._collapsed_streak = 0

        if self._collapsed_streak >= self.patience:
            return self._stop(
                f"dead-gradient collapse detected at step {step}: I stuck at "
                f"{mean:.6f} +/- {error:.2e} for {self._collapsed_streak} consecutive steps "
                "(R==0 fixed point, see class docstring)"
            )

        return True


class InfidelityPlateauStopper:
    """Stops the phase-2 driver once the *best* infidelity stops improving,
    so `epochs2` acts as a safety ceiling rather than as the target.

    Empirically a fixed `epochs2` is wrong in both directions at once: on
    real cluster logs (3x3, Jz=0.8, `epochs2=2500`) some sectors flatten by
    step 500-750 and spend the remaining ~1800 steps re-measuring the same
    `I`, while others are still descending, slowly and noisily, at step 2500
    and would use more. This callback ends the first kind early; the ceiling
    still bounds the second.

    Same pattern as `InfidelityCollapseGuard` -- a legacy NetKet callback
    that returns False to stop -- but it belongs on the *other* side of the
    checkpoint. The guard fires on a step whose state is bad (diverged or
    dead-gradient) and must therefore run before `checkpoint.update` so that
    step never becomes the "best" iterate; a legacy callback returning False
    raises `StopRun` immediately, skipping the rest of the list. This
    stopper's triggering step is perfectly valid -- the only complaint is
    that it is no better than its predecessors -- so it goes *after*
    `checkpoint.update` and the observables logger, letting the step be
    checkpointed and recorded before the run ends.

    The "best I so far" it watches is read straight off the checkpoint
    rather than tracked a second time here. Two independent minima of the
    same quantity is exactly the kind of duplication that drifts apart the
    day one of them starts filtering steps the other one doesn't, and the
    returned state is the checkpoint's, so the checkpoint's notion of "best"
    is the one that matters.

    Defaults come from the same Jz=0.8 logs: sectors that converge are flat
    well before step 750; sectors genuinely stuck show no measurable
    improvement across windows of 1500-2000 steps; sectors still making
    slow progress improve their best `I` by ~40-50% over ~1500-step windows,
    i.e. ~20-25% per 750, comfortably above a 5% bar. Treat both as
    hyperparameters to tune, not as constants -- they encode one lattice at
    one Jz.

    :param checkpoint: the `BestOverlapCheckpoint` of this same phase-2 run.
        It must appear *before* this callback in the driver's callback list,
        so `best_infid` is already up to date for the current step when
        `update` reads it.
    :param patience: window, in steps, over which improvement is measured
    :param min_rel_improvement: relative improvement of `best_infid` the
        window must show to not count as a plateau (0.05 = 5%)
    :param min_steps: warmup steps before the check arms at all. A projected
        ansatz seeded from a symmetry-invariant state can open at `I > 10`
        (`sum_g chi(g) = 0` makes the projected amplitude a near-perfect
        cancellation) and take 500-750 steps to settle; stopping inside that
        transient would kill runs that had not started yet.
    """

    def __init__(
        self,
        checkpoint: BestOverlapCheckpoint,
        patience: int = 750,
        min_rel_improvement: float = 0.05,
        min_steps: int = 500,
    ):
        self.checkpoint = checkpoint
        self.patience = patience
        self.min_rel_improvement = min_rel_improvement
        self.min_steps = min_steps
        self._window = deque()
        self.triggered = False
        self.trigger_reason = None

    def _stop(self, reason):
        self.triggered = True
        self.trigger_reason = reason
        print(f"[InfidelityPlateauStopper] {reason} -- stopping.")
        return False

    def update(self, step, log_data, driver):
        """NetKet training callback; see NetKet's callback API for details."""
        best = float(self.checkpoint.best_infid)
        self._window.append((step, best))

        while len(self._window) >= 2 and self._window[1][0] <= step - self.patience:
            self._window.popleft()

        if step < self.min_steps:
            return True

        ref_step, ref_best = self._window[0]
        if ref_step > step - self.patience:
            return True

        if not np.isfinite(ref_best) or not np.isfinite(best):
            return True

        denom = abs(ref_best)
        rel = 0.0 if denom == 0.0 else (ref_best - best) / denom
        if rel >= self.min_rel_improvement:
            return True

        return self._stop(
            f"plateau at step {step}: best I improved only {rel * 100:.2f}% over the "
            f"last {step - ref_step} steps (best={best:.6g})"
        )


def build_observables_logger(
    metrics_history: dict, H, N: int, wilson_loops=None, energy_stats=None
):
    """Build a NetKet training callback that records the energy, its
    variance and the V-score (and, when `wilson_loops` is given, the mean
    Wilson-loop and per-plaquette values, plus the infidelity if the driver
    logs one) into `metrics_history` at every step.

    Two quantities are recorded that used to be conflated under the single
    name 'variance':

    - `energy_variance` / `vscore`, from `driver.state.expect(H)`. These
      mean the same thing in both phases -- see `vscore_from_energy_stats`
      for why reading the driver's loss statistics instead was wrong in
      phase 2.
    - `loss_variance`, the driver's own loss variance. That *is* Var(E) in
      phase 1 but Var(Infidelity) in phase 2; it is kept because the
      infidelity estimator's spread is what `InfidelityCollapseGuard`
      diagnoses collapse from, so it is worth having on the curve.

    :param metrics_history: dict of lists to append to; must contain at
        least the keys 'step', 'energy', 'energy_error', 'variance', and
        (if `wilson_loops` is given) 'wp_mean'
    :param H: Hamiltonian operator to evaluate at each step
    :param N: number of spins, used to normalize the V-score
    :param wilson_loops: optional list of plaquette operators
    :param energy_stats: optional `StepEnergyStats` shared with the
        checkpoint callback, so `<H>` is evaluated once per step instead of
        once per callback. None keeps the original unshared `expect` call.
    """

    def log_metrics(step, log_data, driver):
        infid_stats = log_data.get("Infidelity")
        if infid_stats is not None:
            metrics_history.setdefault("infidelity", []).append(
                float(jnp.real(infid_stats.mean))
            )
            metrics_history.setdefault("infidelity_error", []).append(
                float(jnp.real(infid_stats.error_of_mean))
            )

        stats = _resolve_energy_stats(energy_stats, H, step, log_data, driver)
        energy = float(jnp.real(stats.mean))
        energy_error = float(jnp.real(stats.error_of_mean))
        energy_variance = float(jnp.real(stats.variance))
        vscore = vscore_from_energy_stats(stats, N)
        loss_variance = float(jnp.real(getattr(log_data[driver._loss_name], "variance")))

        metrics_history["step"].append(step)
        metrics_history["energy"].append(energy)
        metrics_history["energy_error"].append(energy_error)
        metrics_history["energy_variance"].append(energy_variance)
        metrics_history["vscore"].append(vscore)
        metrics_history["loss_variance"].append(loss_variance)

        if wilson_loops is None:
            print(
                f"Step {step}: Energy = {energy:.6f} +/- {energy_error:.2e}, "
                f"Var(E) = {energy_variance:.4f}, V-score = {vscore:.3e}"
            )
            return True

        wp_values = [float(np.real(driver.state.expect(op).mean)) for op in wilson_loops]
        wp_mean = float(np.mean(wp_values))
        metrics_history["wp_mean"].append(wp_mean)
        for idx, val in enumerate(wp_values):
            metrics_history.setdefault(f"Wp_{idx}", []).append(val)

        print(
            f"Step {step:4d} | E = {energy:.6f} | V-score = {vscore:.3e} | "
            f"Wp_avg = {wp_mean:.4f}"
        )
        return True

    return log_metrics
