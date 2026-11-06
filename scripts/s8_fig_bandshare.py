"""Band-share figure (the mechanism): P(best of 1024 = ground) against (a) the
initial ground-state overlap and (b) the ground state's share of the
low-energy band, overlap / (initial mass within 5% of the ground energy).

Data: Phase 1 (p = 1; arms B, C-milp, C-oracle, D, F x 7 eps + A; 30 held-out
scenarios x 3 seeds). Points = arm x eps x scenario means over seeds (psi_0,
hence both x values, does not depend on the seed). Curves = one logistic
least-squares fit pooled over ALL runs with overlap > 0 (D at eps = 0 has
overlap exactly 0 and P exactly 0: 69 runs, not drawable on a log axis,
noted on the figure). Colour carries the comparison, not the arm identity
alone: learned arms (blue; marker = which), LP relaxation D (orange), merit
order F (aqua), and every uniform start -- arm A and all arms at eps = 0.5
-- in neutral grey. Palette: the dataviz reference categorical slots 1-3,
validated all-pairs (worst CVD dE 9.2, normal-vision 24.0); aqua is below
3:1 contrast, so identity is also carried by the legend and marker shape.

    python scripts/s8_fig_bandshare.py  # -> results/fig_bandshare.png / .pdf
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "fig_bandshare"

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#9b9a95"
STYLE = {  # arm -> (legend label, colour, marker)
    "C-oracle": ("C-oracle (learned, ground-state labels)", BLUE, "o"),
    "C-milp": ("C-milp (learned, MILP labels)", BLUE, "s"),
    "B": ("B (learned start, X mixer)", BLUE, "^"),
    "D": ("D (LP relaxation)", ORANGE, "o"),
    "F": ("F (merit order)", AQUA, "D"),
    "uniform": ("uniform start (A, and any arm at eps = 0.5)", GREY, "o"),
}


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import s8_mass as M
    from s8_analysis import load
    from s8_eps_main import tasks

    keys = tasks(1)
    recs = [r for r in load() if (r["arm"], r["eps"], r["scenario"], r["seed"], r["p"]) in keys]
    feats = {tuple(json.loads(k)): v for k, v in json.loads(M.CACHE.read_text()).items()}
    zero = [r for r in recs if r["overlap0"] == 0]
    rs = [r for r in recs if r["overlap0"] > 0]
    for r in rs:
        r["m_e05"] = feats[(r["arm"], r["eps"], r["scenario"], -1)]["m_e05"]
    y = np.array([r["p_ground_K1024"] for r in rs])
    xa = np.log10([r["overlap0"] for r in rs])
    lme = np.log10([r["m_e05"] for r in rs])
    tss = ((y - y.mean()) ** 2).sum()
    # band-share index: the two-variable fit logit P = a + b log10 ov + c log10 m_e05 rewritten as
    # (b) * [log10 ov - k log10 m_e05], k = -c / b (fitted; k = 1 would be the plain ratio ov / m_e05)
    b2, rss2, _ = M.fit(np.column_stack([np.ones(len(y)), xa, lme]), y)
    k = -b2[2] / b2[1]
    xb = xa - k * lme
    _, rss_ratio, res_ratio = M.fit(np.column_stack([np.ones(len(y)), xa - lme]), y)
    cls = np.array(["uniform" if r["arm"] == "A" or r["eps"] == 0.5 else r["arm"] for r in rs])

    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "axes.labelcolor": INK, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0), sharey=True, facecolor=SURFACE)
    panels = (
        (axes[0], xa, "log10 initial ground-state overlap  |<ground|psi_0>|^2",
         "(a) Against ground-state overlap: the arms separate"),
        (axes[1], xb, f"band-share index  log10[ overlap / m5^{k:.2f} ]\n"
                      "m5 = initial mass within 5% of the ground energy; exponent fitted",
         "(b) Against band share: the arms collapse onto one curve"),
    )
    for ax, x, xlabel, title in panels:
        b, rss, res = M.fit(np.column_stack([np.ones(len(y)), x]), y)
        resid = {c: res[cls == c].mean() for c in STYLE}
        spread = max(resid.values()) - min(resid.values())
        # points: arm x eps x scenario means
        grp = {}
        for r, xi, c in zip(rs, x, cls):
            grp.setdefault((c, r["arm"], r["eps"], r["scenario"]), []).append((xi, r["p_ground_K1024"]))
        ax.set_facecolor(SURFACE)
        for c in ("uniform", "F", "D", "B", "C-milp", "C-oracle"):  # draw order: context first, learned on top
            pts = np.array([np.mean(v, axis=0) for k, v in grp.items() if k[0] == c])
            lab, col, mk = STYLE[c]
            ax.scatter(pts[:, 0], pts[:, 1], s=26 if mk != "D" else 22, marker=mk, color=col, alpha=0.75,
                       edgecolor=SURFACE, linewidth=0.6, label=lab, zorder=3)
        XMIN = -6.0
        for c in ("uniform", "F", "D", "B", "C-milp", "C-oracle"):  # off-scale points (D at eps = 0) at the edge
            off = [np.mean(v, axis=0) for kk, v in grp.items() if kk[0] == c and np.mean(v, axis=0)[0] < XMIN]
            if off:
                ax.scatter([XMIN - 0.12] * len(off), [o[1] for o in off], marker="<", s=30, color=STYLE[c][1],
                           edgecolor=SURFACE, linewidth=0.6, zorder=3)
        ax.set_xlim(XMIN - 0.3, 0.1)
        xs = np.linspace(XMIN, 0, 300)
        ax.plot(xs, M.sigmoid(b[0] + b[1] * xs), color=INK, lw=2, zorder=4, label="one logistic fit, all arms pooled")
        ax.text(0.03, 0.97, f"R² = {1 - rss / tss:.2f} (runs)\nmean residual by arm spans {spread:.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.35", fc=SURFACE, ec=GRID))
        ax.set_title(title, loc="left", fontsize=9.5, color=INK)
        ax.set_xlabel(xlabel)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("P(best of 1024 samples is the ground state)")
    h, l = axes[0].get_legend_handles_labels()
    order = [l.index(STYLE[c][0]) for c in ("C-oracle", "C-milp", "B", "D", "F", "uniform")] + [len(l) - 1]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="lower center", ncol=4, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.01))
    n_off = len({(r["arm"], r["eps"], r["scenario"]) for r, x0 in zip(rs, xa) if x0 < -6})
    spread_ratio = (max(res_ratio[cls == c].mean() for c in STYLE)
                    - min(res_ratio[cls == c].mean() for c in STYLE))
    fig.text(0.5, 0.145,
             f"Phase 1: p = 1, 30 held-out scenarios x 3 seeds. Points: arm x eps x scenario means. Curves and R^2: "
             f"fitted on all {len(y)} runs.\n"
             f"D at eps = 0: {len(zero)} runs with overlap exactly 0 (not drawable) and {n_off} scenario means below "
             f"10^-6 (drawn as < at the left edge); all have P = 0.\n"
             f"Plain ratio overlap / m5 (exponent 1): R^2 = {1 - rss_ratio / tss:.2f}, per-arm residuals span "
             f"{spread_ratio:.2f}. Held-out check: fitted on A, D, F and eps = 0.5 only, the index predicts the "
             "learned arms to within 0.015.",
             ha="center", va="center", fontsize=7.5, color=INK2, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.22, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=220, facecolor=SURFACE)
    print(f"wrote {OUT.relative_to(ROOT)}.png / .pdf")


if __name__ == "__main__":
    main()
