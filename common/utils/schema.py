from typing import Iterable, Mapping, Sequence


IDENTITY_COLUMNS: Sequence[str] = (
    "run_id",
    "tag",
    "ansatz",
    "extent_x",
    "extent_y",
    "N",
    "layers",
    "Jz",
    "sector",
    "seed",
    "iteration",
    "git_commit",
)

CONFIG_COLUMNS: Sequence[str] = (
    "group",
    "projector",
    "optimizer",
    "n_samples",
    "diag_shift",
    "lr_phase1_peak",
    "epochs1",
    "epochs2",
    "vscore_threshold",
    "alpha",
    "heads",
    "dk",
    "dtype",
    "sampler",
    "clip_norm",
    "phase1_projected",
    "projection",
)

PHASE1_COLUMNS: Sequence[str] = (
    "energy_total_ph1",
    "energy_per_site_ph1",
    "vscore_ph1",
    "fidelity_sector_ph1",
    "fidelity_manifold_ph1",
    "n_steps_ph1",
    "n_iter_convergencia_ph1",
)

PHASE2_COLUMNS: Sequence[str] = (
    "energy_total_ph2",
    "energy_per_site_ph2",
    "energy_error_ph2",
    "vscore_ph2",
    "infidelity",
    "best_infidelity",
    "fidelity_sector",
    "fidelity_manifold",
    "collapsed",
    "collapse_reason",
    "stop_reason",
    "n_steps_ph2",
    "n_iter_convergencia_ph2",
)

REFERENCE_COLUMNS: Sequence[str] = (
    "E_ED",
    "delta_eps",
    "proj_norm",
    "target_n_minus",
    "target_flux_pattern",
    "target_vortex_plaquette_idx",
    "target_flux_ambiguous",
    "target_n_distinct_patterns",
    "target_n_equivalent_placements",
    "target_vortex_class",
    "target_n_vortex_classes",
)

OBSERVABLE_COLUMNS: Sequence[str] = (
    "m",
    "ms",
    "fluct",
    "fluct_s",
    "Wp",
)

COST_COLUMNS: Sequence[str] = (
    "n_params",
    "wall_clock_ph1",
    "wall_clock_ph2",
    "time_per_step_ph1",
    "time_per_step_ph2",
)

RESERVED_RUN_COLUMNS: Sequence[str] = (
    "corr_r1",
    "corr_r2",
    "corr_r3",
    "corr_r4",
    "w_x",
    "w_y",
    "wilson_x_expect",
    "wilson_y_expect",
    "hsym_eig",
    "lambda_sym",
    "fidelity_manifold_combinada",
    "mixing_strategy",
    "phase_gauge",
    "attention_weights_path",
    "peak_mem_MB",
)

RUN_COLUMNS: Sequence[str] = (
    *IDENTITY_COLUMNS,
    *CONFIG_COLUMNS,
    *PHASE1_COLUMNS,
    *PHASE2_COLUMNS,
    *REFERENCE_COLUMNS,
    *OBSERVABLE_COLUMNS,
    *COST_COLUMNS,
    *RESERVED_RUN_COLUMNS,
)


METRICS_PH1_COLUMNS: Sequence[str] = (
    "step",
    "energy",
    "energy_error",
    "energy_variance",
    "vscore",
)

METRICS_PH2_COLUMNS: Sequence[str] = (
    *METRICS_PH1_COLUMNS,
    "infidelity",
    "infidelity_error",
    "loss_variance",
    "wp_mean",
)


SECTORS_BY_JZ_COLUMNS: Sequence[str] = (
    "Jz",
    "extent_x",
    "extent_y",
    "N",
    "group",
    "tol",
    "E0",
    "n_eigenvalues",
    "manifold_dim",
    "gap_manifold",
    "hosting_sectors",
    "manifold_irrep_weights",
    "manifold_gaps",
    "manifold_tail_warning",
    "manifold_dim_vs_tol",
    "spectrum_head",
    "wilson_labels",
    "hsym_eigenvalues",
    "topological_degeneracy_ok",
)

RESERVED_SECTOR_COLUMNS: Sequence[str] = (
    "wilson_labels",
    "hsym_eigenvalues",
    "topological_degeneracy_ok",
)


VORTEX_SECTORS_COLUMNS: Sequence[str] = (
    "Jz",
    "extent_x",
    "extent_y",
    "N",
    "n_plaquettes",
    "group",
    "tol",
    "E0",
    "n_eigenvalues",
    "manifold_dim",
    "gap_manifold",
    "n_minus",
    "vortex_plaquette_idx",
    "n_distinct_patterns",
    "pattern_multiplicity",
    "all_same_n_minus",
    "representatives",
    "avg_Wp",
    "avg_Wp_spread",
    "hosting_sectors",
    "wp_pure",
    "wp_max_impurity",
    "wp_seed",
    "quotiented_by_symmetry",
    "source",
)


class SchemaError(ValueError):
    """Raised when a row does not match its declared column list."""


def validate_row(row: Mapping, columns: Iterable[str], *, context: str = "row") -> None:
    """Check that `row`'s keys are exactly `columns`.

    Reserved columns still have to be present -- carrying them as an
    explicit NaN is the whole point (see the module docstring), so a writer
    that silently drops them would defeat the alignment guarantee.

    :param row: mapping of column name to value
    :param columns: the declared column list this row must match
    :param context: label used in the error message
    :raises SchemaError: if any column is missing or unexpected
    """
    expected = set(columns)
    actual = set(row)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        parts = []
        if missing:
            parts.append(f"missing {missing}")
        if unexpected:
            parts.append(f"unexpected {unexpected}")
        raise SchemaError(f"{context} does not match the frozen schema: {'; '.join(parts)}")


def row_to_list(row: Mapping, columns: Iterable[str], *, context: str = "row") -> list:
    """Validate `row` and flatten it into a list ordered like `columns`,
    ready for `src.utils.io.append_observables_csv`."""
    columns = list(columns)
    validate_row(row, columns, context=context)
    return [row[c] for c in columns]


def empty_reserved(columns: Iterable[str] = RESERVED_RUN_COLUMNS) -> dict:
    """A dict mapping every reserved column to None, to splat into a row
    being built: `{**base, **empty_reserved()}`."""
    return {c: None for c in columns}


def first_step_below(values, steps, threshold: float):
    """Index into `steps` of the first entry whose `values` drops below
    `threshold`, or None if it never does.

    This is the guideline's `n_iter_convergencia` (§10.1), the x-axis of
    F7b's "wall-clock hasta umbral de V-score" bars. Defined here rather
    than in a training script so a live run and a backfill of an old one
    compute it the same way -- two definitions of "converged" would make
    the historical and new bars incomparable, which is the whole point of
    the figure.
    """
    for step, value in zip(steps, values):
        if value == value and value < threshold:
            return int(step)
    return None
