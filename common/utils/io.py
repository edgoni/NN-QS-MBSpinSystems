import csv
import os
import pickle
from pathlib import Path

import flax
import jax.numpy as jnp
import netket as nk
import numpy as np


def load_target_state_from_vector(hi, sampler, psi, n_samples=2048, floor=1e-12):
    """Wrap an exact state vector as a sampled (`MCState`) infidelity target.

    :param floor: added to every amplitude before taking the log, so that
        exactly-zero amplitudes stay finite and Metropolis can move through
        them. A symmetry-projected target has many exactly-zero amplitudes,
        which this turns into a small uniform background -- prefer
        `load_target_state_exact` when the Hilbert space is small enough to
        enumerate, which needs no floor at all.
    """
    target_model = nk.models.LogStateVector(hi)
    vs_target = nk.vqs.MCState(sampler, target_model, n_samples=n_samples)
    vs_target.variables = {"params": {"logstate": jnp.log(psi.flatten() + floor + 0j)}}
    return vs_target


def load_target_state_exact(hi, psi):
    """Wrap an exact state vector as a `FullSumState` infidelity target.

    Enumerates the whole Hilbert space instead of sampling it, so no
    amplitude floor is needed: exactly-zero amplitudes of a
    symmetry-projected target stay exactly zero (`log(0) = -inf`,
    `exp(-inf) = 0`) rather than becoming a spurious uniform background.

    Requires the *variational* state to be a `FullSumState` too: NetKet's
    sampled infidelity estimator reads `target_state.samples`, which a
    `FullSumState` does not have. See `src.training.drivers` `exact=True`.
    """
    target_model = nk.models.LogStateVector(hi)
    vs_target = nk.vqs.FullSumState(hilbert=hi, model=target_model)
    with np.errstate(divide="ignore"):
        logstate = np.log(np.asarray(psi).flatten().astype(np.complex128))
    vs_target.variables = {"params": {"logstate": jnp.asarray(logstate)}}
    return vs_target


def save_checkpoint(state, mpack_path, pickle_path=None):
    """Persist a variational state's parameters as a Flax `.mpack`, and
    optionally also as a pickle (for quick unpickling without Flax)."""
    mpack_path = Path(mpack_path)
    mpack_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mpack_path, "wb") as f:
        f.write(flax.serialization.to_bytes(state.parameters))

    if pickle_path is not None:
        pickle_path = Path(pickle_path)
        pickle_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pickle_path, "wb") as f:
            pickle.dump(state.parameters, f)


def _migrate_csv_header(path: Path, header, delimiter: str) -> None:
    """Rewrite `path` under a new, wider `header` before appending to it.

    The frozen schema (`src.utils.schema`) grows when a run starts recording
    something it did not record before. Appending a row of the new width to a
    file whose header is the old one produces a table where every row has one
    more field than the header -- pandas either raises or silently shifts the
    columns, and either way the previous runs in that file are lost. So the
    file is brought up to the current header first, with the columns it never
    had left blank.

    Columns that exist on disk but not in `header` are a different situation:
    migrating would mean discarding real measurements, which is not this
    function's call to make, so it refuses instead.
    """
    with open(path, newline="") as f:
        rows = list(csv.reader(f, delimiter=delimiter))
    if not rows:
        return

    old_header, *data = rows
    if old_header == list(header):
        return

    dropped = [c for c in old_header if c not in set(header)]
    if dropped:
        raise ValueError(
            f"{path} has columns absent from the current schema: {dropped}. "
            "Rewriting it here would discard them -- migrate the file by hand "
            "(or move it aside) before appending new runs."
        )

    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(header)
        for row in data:
            values = dict(zip(old_header, row))
            writer.writerow([values.get(c, "") for c in header])
    os.replace(tmp, path)
    added = [c for c in header if c not in set(old_header)]
    print(f"[append_observables_csv] {path}: added column(s) {added} to {len(data)} existing row(s).")


def append_observables_csv(path, header, row, delimiter=","):
    """Append one results row to `path`, writing `header` first if the file
    doesn't exist yet, and widening an older header if the schema has grown
    since the file was written (see `_migrate_csv_header`).

    :param delimiter: field separator. Defaults to "," to match the `.csv`
        extension these files are written with; pass "\\t" to reproduce the
        original tab-separated output.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.is_file()
    if file_exists:
        _migrate_csv_header(path, list(header), delimiter)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)
