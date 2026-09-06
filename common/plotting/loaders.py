import json
from pathlib import Path

import pandas as pd

from common.utils.schema import (
    RUN_COLUMNS,
    SECTORS_BY_JZ_COLUMNS,
    VORTEX_SECTORS_COLUMNS,
)

RESULTS_DIR = Path("data/results")

_JSON_COLUMNS = (
    "hosting_sectors",
    "manifold_irrep_weights",
    "manifold_gaps",
    "manifold_dim_vs_tol",
    "spectrum_head",
    "wilson_labels",
    "hsym_eigenvalues",
)

_RUN_JSON_COLUMNS = (
    "target_flux_pattern",
    "target_vortex_plaquette_idx",
)

_VORTEX_JSON_COLUMNS = (
    "n_minus",
    "vortex_plaquette_idx",
    "pattern_multiplicity",
    "representatives",
    "hosting_sectors",
)


def _decode_json_column(series: pd.Series):
    """Deserialize a JSON-text column, leaving blanks as None."""
    def decode(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        if not isinstance(value, str):
            return value
        return json.loads(value)
    return series.map(decode)


def load_runs(path=RESULTS_DIR / "runs.csv") -> pd.DataFrame:
    """The per-run results table, reindexed to the frozen schema.

    Reindexing matters: a CSV written before a column existed comes back
    with that column present and NaN instead of absent, so a figure can ask
    for it unconditionally and let `has_data` decide whether to draw.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run `python Supervised_Infid_min/run_vmc.py` to produce new runs, "
            "or `python plots/supervised/backfill_metrics.py` to build it from the existing "
            "metrics/observables files."
        )
    df = pd.read_csv(path).reindex(columns=list(RUN_COLUMNS))
    for col in _RUN_JSON_COLUMNS:
        df[col] = _decode_json_column(df[col])
    return df


def load_sectors_by_jz(path=RESULTS_DIR / "sectors_by_jz.csv") -> pd.DataFrame:
    """The per-Jz ED table, with its JSON columns decoded into Python
    lists/dicts and rows sorted by Jz."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run `python common/analysis/build_sectors_by_jz.py` first."
        )
    df = pd.read_csv(path).reindex(columns=list(SECTORS_BY_JZ_COLUMNS))
    for col in _JSON_COLUMNS:
        df[col] = _decode_json_column(df[col])
    return df.sort_values("Jz").reset_index(drop=True)


def load_metrics(tag: str, phase: int, directory=RESULTS_DIR) -> pd.DataFrame:
    """Per-step training curve of one run.

    :param tag: the run's `tag` column, i.e. the filename stem shared by its
        checkpoints and metrics
    :param phase: 1 (variational VMC) or 2 (infidelity projection)
    """
    if phase not in (1, 2):
        raise ValueError(f"phase must be 1 or 2, got {phase!r}")
    path = Path(directory) / f"metrics_ph{phase}_{tag}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found (tag={tag!r}, phase={phase})")
    return pd.read_csv(path)


def irrep_weights_long(df: pd.DataFrame) -> pd.DataFrame:
    """`sectors_by_jz` irrep weights in long form `(Jz, irrep, weight)`.

    S4's heatmap is then a one-line `pivot`, and the same frame answers
    "which irreps carry weight at this Jz" without re-parsing JSON.
    """
    rows = []
    for _, row in df.iterrows():
        weights = row["manifold_irrep_weights"] or {}
        for irrep, weight in weights.items():
            rows.append({"Jz": row["Jz"], "irrep": int(irrep), "weight": float(weight)})
    return pd.DataFrame(rows, columns=["Jz", "irrep", "weight"])


def manifold_dim_vs_tol_long(df: pd.DataFrame) -> pd.DataFrame:
    """The S3 tolerance sweep in long form `(Jz, tol, manifold_dim)`."""
    rows = []
    for _, row in df.iterrows():
        sweep = row["manifold_dim_vs_tol"] or {}
        for tol, dim in sweep.items():
            rows.append({"Jz": row["Jz"], "tol": float(tol), "manifold_dim": int(dim)})
    return pd.DataFrame(rows, columns=["Jz", "tol", "manifold_dim"]).sort_values(["Jz", "tol"])


def n_irreps(df: pd.DataFrame) -> int:
    """Number of irreps in the group the ED table was built with."""
    counts = [len(w) for w in df["manifold_irrep_weights"] if w]
    return max(counts) if counts else 0


def load_vortex_sectors_by_jz(path=RESULTS_DIR / "vortex_sectors_by_jz.csv") -> pd.DataFrame:
    """The per-Jz vortex (flux) sector table, JSON columns decoded.

    Companion to `load_sectors_by_jz`, not a replacement: that one labels a
    level by the spatial irreps hosting it, this one by which plaquettes
    carry a vortex. Joining them on `Jz` is what shows the two labels moving
    independently -- see the module docstring of `common/analysis/map_vortex_sectors.py`.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run `python common/analysis/map_vortex_sectors.py` first."
        )
    df = pd.read_csv(path).reindex(columns=list(VORTEX_SECTORS_COLUMNS))
    for col in _VORTEX_JSON_COLUMNS:
        df[col] = _decode_json_column(df[col])
    return df.sort_values("Jz").reset_index(drop=True)


def reliable_vortex_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the Jz points whose flux census cannot be trusted.

    A row is unusable when the joint W_p diagonalization came back impure,
    which means the stored level was truncated rather than complete (the
    exactly-solvable limits Jz=0 and Jz=1 have degeneracies Lanczos does not
    resolve at any affordable k). Those rows carry a real `avg_Wp` number
    that means nothing, so a figure that plots them without filtering shows
    noise at both ends of the axis and invites exactly the wrong reading.
    """
    return df[df["wp_pure"].fillna(False).astype(bool)].reset_index(drop=True)


def has_data(df: pd.DataFrame, columns) -> bool:
    """True when every named column is present and has at least one value.

    The guard that lets a figure for not-yet-implemented physics ship
    today: it skips itself while its reserved columns are still empty, and
    starts drawing the moment they are filled -- no edit to the figure.
    """
    columns = [columns] if isinstance(columns, str) else list(columns)
    return all(c in df.columns and df[c].notna().any() for c in columns)
