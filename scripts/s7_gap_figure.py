"""Paper figure: encoding gap vs block granularity on one physical system.

(a) every ladder level at the FROZEN setting (lambda_D = 100, B = 250 MW);
(b) every level at its OWN operating point, chosen by the unchanged s2
    procedure (100 selection scenarios, seed 0; same lambda x bias grid;
    plateau rule k = 1). These per-level calibrations exist for this figure
    only -- the frozen point does not change.
Both panels are evaluated on the same 100 fresh scenarios (seed 4242) with
identical demand at every level.
"""

import csv
from pathlib import Path

import numpy as np

from qguc import REFINEMENT_LEVELS, make_scenario_set, refinement_fleet
from qguc.calibrate import select_operating_point, sweep_grid_parallel
from qguc.config import FROZEN

N_HOURS = 2
SELECT = (100, 0)
EVAL = (100, 4242)
LAMBDAS = np.logspace(0, 3.5, 29)
BIASES = np.array([0, 25, 50, 75, 100, 150, 200, 250, 300, 400], dtype=float)
WORKERS = 8
RESULTS = Path(__file__).resolve().parents[1] / "results"

BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1-2 (validated reference palette)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"


def stats(g):
    ad = g.adequate[0, 0]
    gaps = 100 * g.gap[0, 0][ad]
    return dict(adequacy=ad.mean(), mean=gaps.mean(), median=np.median(gaps), worst=gaps.max())


def compute():
    rows = []
    for k in REFINEMENT_LEVELS:
        sel = make_scenario_set(SELECT[0], n_hours=N_HOURS, seed=SELECT[1], refinement_level=k)
        grid = sweep_grid_parallel(sel, LAMBDAS, BIASES, WORKERS)
        b, li = select_operating_point(grid, plateau=1)
        lam, bias = float(grid.lambdas[li]), float(grid.biases[b])
        ev = make_scenario_set(EVAL[0], n_hours=N_HOURS, seed=EVAL[1], refinement_level=k)
        fixed = stats(sweep_grid_parallel(ev, [FROZEN.lambda_demand], [FROZEN.demand_bias_mw], WORKERS))
        own = stats(sweep_grid_parallel(ev, [lam], [bias], WORKERS))
        row = dict(level=f"L{k}", blocks=len(refinement_fleet(k)), qubits=sel[0].n_vars, own_lambda=lam,
                   own_bias=bias, **{f"fixed_{s}": v for s, v in fixed.items()},
                   **{f"own_{s}": v for s, v in own.items()})
        rows.append(row)
        print(f"  {row['level']} ({row['blocks']:2d} blocks): own point lambda {lam:.4g}, bias {bias:.0f} MW | "
              f"fixed: adeq {100 * fixed['adequacy']:.0f}%, mean {fixed['mean']:.2f}%, worst {fixed['worst']:.2f}% | "
              f"own: adeq {100 * own['adequacy']:.0f}%, mean {own['mean']:.2f}%, worst {own['worst']:.2f}%",
              flush=True)
    with (RESULTS / "s7_gap_ladder.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def plot(rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "font.family": "DejaVu Sans"})
    x = np.array([r["blocks"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True, facecolor=SURFACE)
    panels = (("fixed", f"(a) frozen setting: $\\lambda_D$ = {FROZEN.lambda_demand:g}, "
                        f"B = {FROZEN.demand_bias_mw:g} MW"),
              ("own", "(b) each level at its own calibrated ($\\lambda_D$, B)"))
    top = 1.18 * max(max(r["fixed_worst"], r["own_worst"]) for r in rows)
    for ax, (key, title) in zip(axes, panels):
        mean = [r[f"{key}_mean"] for r in rows]
        worst = [r[f"{key}_worst"] for r in rows]
        ax.plot(x, worst, color=ORANGE, lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5, label="worst case")
        ax.plot(x, mean, color=BLUE, lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5, label="mean")
        ax.text(x[-1] + 0.25, worst[-1], "worst", color=INK2, va="center", fontsize=8)
        ax.text(x[-1] + 0.25, mean[-1], "mean", color=INK2, va="center", fontsize=8)
        for xi, r, m in zip(x, rows, mean):
            if key == "own":
                ax.text(xi, top * 0.97, f"B={r['own_bias']:.0f}", ha="center", va="top", fontsize=7, color=INK2)
            if r[f"{key}_adequacy"] < 1:
                ax.text(xi + 0.15, m - 0.6, f"{100 * r[f'{key}_adequacy']:.0f}% adequate", ha="left", va="top",
                        fontsize=7, color=INK2)
        ax.set_title(title, fontsize=9, loc="left", color=INK)
        ax.set_xticks(x, [f"{r['level']}\n{r['blocks']:.0f}" for r in rows])
        ax.set_xlim(x[0] - 0.6, x[-1] + 1.6)
        ax.set_ylim(0, top)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("encoding gap vs MILP (%)")
    axes[1].legend(frameon=False, fontsize=8, loc="center left")
    fig.supxlabel("refinement level and number of blocks (4442 MW of capacity at every level)", fontsize=9,
                  color=INK)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"fig_encoding_gap_ladder.{ext}", dpi=200, facecolor=SURFACE)
    print("wrote results/fig_encoding_gap_ladder.png and .pdf")


def load():
    with (RESULTS / "s7_gap_ladder.csv").open() as f:
        return [{k: (v if k == "level" else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]


if __name__ == "__main__":
    import sys

    plot(load() if "--plot-only" in sys.argv else compute())
