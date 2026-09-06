from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


CATEGORICAL = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)

SEQUENTIAL_RAMP = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)

SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("kitaev_blue", SEQUENTIAL_RAMP)

ORDINAL_STEPS = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

ANSATZ_COLORS = {"rbm": CATEGORICAL[0], "transformer": CATEGORICAL[1]}

PHASE_STYLE = {
    "A": {"color": CATEGORICAL[0], "marker": "o", "label": "Fase A (supervisada, con ED)"},
    "B": {"color": CATEGORICAL[1], "marker": "s", "label": "Fase B (variacional pura)"},
}

LINE_WIDTH = 2.0
MARKER_SIZE = 5.0
FIGURES_DIR = Path("data/figures")


def use_paper_style() -> None:
    """Apply the shared rcParams. Call once at the top of a plot script."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_PRIMARY,
        "axes.titlecolor": INK_PRIMARY,
        "axes.titlesize": 11,
        "axes.titleweight": "medium",
        "axes.labelsize": 10,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.6,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": LINE_WIDTH,
        "lines.markersize": MARKER_SIZE,
        "font.size": 10,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def jz_colors(jz_values):
    """Colors for a set of Jz values, from the sequential ramp.

    Jz is a continuous physical parameter, so it gets a one-hue ramp rather
    than categorical hues: the color then *means* "how far along the scan",
    and a reader can order the curves by eye. Up to five values are given
    the validated discrete steps; past that the continuous colormap is
    sampled and the figure carries a colorbar (see `add_jz_colorbar`),
    because no single hue yields six discrete steps a reader can tell apart.

    :return: (list of colors, whether a colorbar is required)
    """
    jz_values = list(jz_values)
    if len(jz_values) <= len(ORDINAL_STEPS):
        return list(ORDINAL_STEPS[: len(jz_values)]), False
    lo, hi = min(jz_values), max(jz_values)
    span = (hi - lo) or 1.0
    return [SEQUENTIAL_CMAP(0.15 + 0.80 * (jz - lo) / span) for jz in jz_values], True


def add_jz_colorbar(fig, ax, jz_values, label=r"$J_z$"):
    """Colorbar mapping the sequential ramp back to Jz, for figures with
    more ordered series than the discrete steps can carry."""
    lo, hi = min(jz_values), max(jz_values)
    norm = mpl.colors.Normalize(vmin=lo, vmax=hi)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=SEQUENTIAL_CMAP)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label(label, color=INK_PRIMARY)
    cbar.outline.set_edgecolor(AXIS)
    cbar.ax.tick_params(color=INK_MUTED, labelcolor=INK_MUTED)
    return cbar


def draw_hosting_band(ax, jz_values, hosting_per_jz, n_irreps, *, xlabel=r"$J_z$"):
    """Which irreps host the degenerate manifold, as a compact matrix strip:
    one row per irrep, one column per Jz, filled where that irrep hosts.

    This is the bottom panel F4 asks for and the same strip S8 puts under
    the plaquette curve, so it lives here rather than in either script.

    It is drawn as a presence *matrix* rather than the stacked color band
    the guideline sketches. Two reasons, and the second is the binding one:
    the reader can follow a single irrep horizontally across the whole Jz
    scan, which a stack cannot show; and a stacked band would need one
    distinguishable hue per irrep -- six at 3x3 -- which no single palette
    delivers, whereas presence is a one-hue encoding that stays readable
    however many irreps the group has.

    :param ax: axis to draw on; its x-axis is meant to be shared with the
        panel above
    :param jz_values: the Jz grid, ascending
    :param hosting_per_jz: list of hosting-irrep lists, aligned to `jz_values`
    :param n_irreps: number of irrep rows to show
    """
    jz_values = np.asarray(jz_values, dtype=float)
    grid = np.zeros((n_irreps, len(jz_values)))
    for col, hosting in enumerate(hosting_per_jz):
        for k in hosting:
            if 0 <= k < n_irreps:
                grid[k, col] = 1.0

    if len(jz_values) > 1:
        step = np.diff(jz_values).mean()
    else:
        step = 1.0
    x_edges = np.concatenate([jz_values - step / 2, [jz_values[-1] + step / 2]])
    y_edges = np.arange(n_irreps + 1) - 0.5

    ax.pcolormesh(
        x_edges, y_edges, grid,
        cmap=LinearSegmentedColormap.from_list("hosting", [SURFACE, CATEGORICAL[0]]),
        vmin=0, vmax=1, edgecolors=SURFACE, linewidth=1.5,
    )
    ax.set_yticks(range(n_irreps))
    ax.set_yticklabels([f"$k={k}$" for k in range(n_irreps)])
    ax.set_ylabel("sector", color=INK_SECONDARY)
    ax.set_xlabel(xlabel)
    ax.grid(False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def save_figure(fig, name: str, directory: Path = FIGURES_DIR) -> Path:
    """Write `fig` as both PNG (for review) and PDF (for the manuscript)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    png = directory / f"{name}.png"
    fig.savefig(png)
    fig.savefig(directory / f"{name}.pdf")
    plt.close(fig)
    print(f"  -> {png}  (+ .pdf)")
    return png
