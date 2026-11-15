"""Every figure in the paper, from the result files, as PDF at final size.

IEEE two-column: one column = 3.5 in, full width (figure*) = 7.16 in. Fonts
are set at 8 pt at final size (body text is 10 pt). Colours: the validated
reference categorical slots (blue, orange, aqua; all-pairs CVD dE >= 9.2) for
learned / LP / merit order, neutral grey for uniform starts; identity is
also carried by marker shape and the legend.

    python scripts/make_figures.py   # -> paper/figures/*.pdf
"""

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "paper" / "figures"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#9b9a95"
COL, FULL = 3.5, 7.16
STYLE = {"C-oracle": ("C-oracle", BLUE, "o", "-"), "C-milp": ("C-milp", BLUE, "s", "--"),
         "B": ("B (X mixer)", BLUE, "^", ":"), "D": ("D (LP)", ORANGE, "o", "-"), "F": ("F (merit)", AQUA, "D", "-"),
         "uniform": ("uniform start", GREY, "o", "-")}


def setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "axes.labelcolor": INK, "font.family": "DejaVu Sans",
                         "pdf.fonttype": 42, "axes.linewidth": 0.6, "lines.linewidth": 1.2})
    return plt


PREVIEW = Path(sys.argv[sys.argv.index("--preview") + 1]) if "--preview" in sys.argv else None


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    if PREVIEW:
        PREVIEW.mkdir(parents=True, exist_ok=True)
        fig.savefig(PREVIEW / f"{name}.png", dpi=220)


def tidy(ax):
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def quiet(fn, *a):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a)


def fig_bandshare(plt):
    import s8_mass as M
    from s8_analysis import load
    from s8_eps_main import tasks

    k1 = tasks(1)
    recs = [r for r in quiet(load) if (r["arm"], r["eps"], r["scenario"], r["seed"], r["p"]) in k1 and r["overlap0"] > 0]
    feats = {tuple(json.loads(k)): v for k, v in json.loads(M.CACHE.read_text()).items()}
    for r in recs:
        r["m_e05"] = feats[(r["arm"], r["eps"], r["scenario"], -1)]["m_e05"]
    y = np.array([r["p_ground_K1024"] for r in recs])
    xa = np.log10([r["overlap0"] for r in recs])
    lme = np.log10([r["m_e05"] for r in recs])
    b2, _, _ = M.fit(np.column_stack([np.ones(len(y)), xa, lme]), y)
    k = -b2[2] / b2[1]
    xb = xa - k * lme
    cls = np.array(["uniform" if r["arm"] == "A" or r["eps"] == 0.5 else r["arm"] for r in recs])
    tss = ((y - y.mean()) ** 2).sum()
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.55), sharey=True)
    for ax, x, xl, title in ((axes[0], xa, r"$\log_{10}$ initial ground-state overlap", "(a) against overlap"),
                             (axes[1], xb, rf"band-share index $\log_{{10}}[\mathrm{{overlap}}/m_5^{{{k:.2f}}}]$",
                              "(b) against band share")):
        b, rss, res = M.fit(np.column_stack([np.ones(len(y)), x]), y)
        arm_of = np.array([r["arm"] for r in recs])  # per ARM, as s8_mass reports it (the paper's macro)
        spread = max(res[arm_of == a].mean() for a in set(arm_of)) - min(res[arm_of == a].mean() for a in set(arm_of))
        grp = {}
        for r, xi, c in zip(recs, x, cls):
            grp.setdefault((c, r["arm"], r["eps"], r["scenario"]), []).append((xi, r["p_ground_K1024"]))
        for c in ("uniform", "F", "D", "B", "C-milp", "C-oracle"):
            pts = np.array([np.mean(v, axis=0) for kk, v in grp.items() if kk[0] == c])
            lab, col, mk, _ = STYLE[c]
            inside = pts[:, 0] >= -6
            ax.scatter(pts[inside, 0], pts[inside, 1], s=11, marker=mk, color=col, alpha=0.75, lw=0.3,
                       edgecolor="white", label=lab, zorder=3)
            if (~inside).any():
                ax.scatter(np.full((~inside).sum(), -6.12), pts[~inside, 1], s=12, marker="<", color=col, zorder=3)
        xs = np.linspace(-6, 0, 300)
        ax.plot(xs, M.sigmoid(b[0] + b[1] * xs), color=INK, lw=1.6, zorder=4, label="pooled logistic fit")
        ax.text(0.03, 0.96, f"$R^2$ = {1 - rss / tss:.3f}\narm residual spread {spread:.3f}", transform=ax.transAxes,
                va="top", fontsize=7, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, lw=0.5))
        ax.set_xlim(-6.3, 0.1)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel(xl)
        ax.set_title(title, loc="left")
        tidy(ax)
    axes[0].set_ylabel(r"$P$(best of 1024 is ground)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 1.02), handletextpad=0.3,
               columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig_bandshare")
    plt.close(fig)


def fig_eps(plt):
    import s8_analysis as A8
    from s8_eps_main import tasks

    k1 = tasks(1)
    D = A8.Data([r for r in quiet(A8.load) if (r["arm"], r["eps"], r["scenario"], r["seed"], r["p"]) in k1])
    gm = A8.structural_groups()["GL"]
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    for a in ("C-oracle", "C-milp", "B", "D", "F"):
        lab, col, mk, ls = STYLE[a]
        m, lo, hi = [], [], []
        for e in A8.EPS:
            g = np.array(list(A8.to_groups(D.scen(a, e, 1, "p_ground_K1024", A8.SWEEP_SEEDS), gm).values()))
            h = stats.t.ppf(0.975, g.size - 1) * g.std(ddof=1) / np.sqrt(g.size)
            m.append(g.mean())
            lo.append(g.mean() - h)
            hi.append(g.mean() + h)
        ax.fill_between(A8.EPS, lo, hi, color=col, alpha=0.10, lw=0)
        ax.plot(A8.EPS, m, color=col, marker=mk, ms=3.5, ls=ls, label=lab)
    pa = np.mean(list(A8.to_groups(D.scen("A", 0.1, 1, "p_ground_K1024", A8.SWEEP_SEEDS), gm).values()))
    ax.axhline(pa, color=GREY, lw=1, ls="--")
    ax.text(0.01, pa + 0.02, "A (cold start)", color=INK2, fontsize=7)
    ax.set_xlabel(r"regularisation $\varepsilon$")
    ax.set_ylabel(r"$P$(best of 1024 is ground)")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(loc="lower left", ncol=2, frameon=False, bbox_to_anchor=(0.0, 0.08))
    tidy(ax)
    fig.tight_layout()
    save(fig, "fig_eps")
    plt.close(fig)


def fig_size(plt):
    def jl(p):
        return [json.loads(l) for l in (RES / p).read_text().splitlines() if l.strip()]

    H, HL, Gc = jl("s9_horizon_census.jsonl"), jl("s10_horizon_learned.jsonl"), jl("s9_census_ownpoint.jsonl")
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.3))
    ax = axes[0]
    for lv, mk in ((0, "o"), (1, "s"), (2, "^")):
        Ts = sorted({r["T"] for r in H if r["level"] == lv})
        n = [next(r["n"] for r in H if r["level"] == lv and r["T"] == T) for T in Ts]
        lp = [np.mean([r["n_wrong"] for r in H if r["level"] == lv and r["T"] == T]) for T in Ts]
        gc = [np.mean([r["C_wrong"] for r in HL if r["level"] == lv and r["T"] == T]) for T in Ts]
        ax.plot(n, lp, color=ORANGE, marker=mk, ms=4, label=f"LP, L{lv}")
        ax.plot(n, gc, color=BLUE, marker=mk, ms=4, ls="--", label=f"GCN, L{lv}")
    ax.set_xlabel("variables $n$ (more hours, same blocks)")
    ax.set_ylabel("mean wrongly pinned bits")
    ax.set_title("(a) horizon axis: L0/L1/L2 at $T = 1,\\ldots,4$", loc="left")
    ax.legend(ncol=2, frameon=False, loc="upper left")
    tidy(ax)
    ax = axes[1]
    lv = sorted({r["level"] for r in Gc})
    n = [next(r["n"] for r in Gc if r["level"] == k) for k in lv]
    mean = [np.mean([r["n_wrong"] for r in Gc if r["level"] == k]) for k in lv]
    frac = [np.mean([r["n_wrong"] >= 2 for r in Gc if r["level"] == k]) for k in lv]
    ax.plot(n, mean, color=ORANGE, marker="o", ms=4, label="LP mean wrong")
    for x, yv, k in zip(n, mean, lv):
        ax.annotate(f"L{k}", (x, yv), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=7, color=INK2)
    ax.set_xlabel("variables $n$ (finer blocks, $T = 2$)")
    ax.set_ylabel("mean wrongly pinned bits (LP)")
    ax.set_title("(b) granularity axis: L0 to L4 at $T = 2$", loc="left")
    ax.text(0.98, 0.04, "$P(\\geq 2$ wrong$)$: " + ", ".join(f"{f:.2f}" for f in frac), transform=ax.transAxes,
            ha="right", fontsize=7, color=INK2)
    axes[1].set_ylim(axes[0].get_ylim())
    tidy(ax)
    fig.tight_layout()
    save(fig, "fig_size")
    plt.close(fig)


def fig_classical(plt):
    R = [json.loads(l) for l in (RES / "s10_sa_compiled.jsonl").read_text().splitlines() if l.strip()]
    fig, ax = plt.subplots(figsize=(COL, 2.75))
    for level, mk in ((2, "o"), (4, "s")):
        rs = [r for r in R if r["level"] == level]
        keys = list(range(len(rs[0]["ladder"])))
        t = [np.mean([list(r["ladder"].values())[i]["run_s"] for r in rs]) for i in keys]
        p = [np.mean([list(r["ladder"].values())[i]["p_ground"] for r in rs]) for i in keys]
        ax.plot(t, p, color=INK, marker=mk, ms=4, label=f"compiled SA, $n={rs[0]['n']}$")
        for arm, col in (("C-milp", BLUE), ("D", ORANGE), ("A", GREY)):
            q = [r["qaoa"][arm] for r in rs]
            ax.scatter([np.mean([x[3] for x in q])], [np.mean([x[0] for x in q])], color=col, marker=mk, s=28,
                       zorder=3, edgecolor="white", lw=0.4)
    for arm, col, lab in (("C-milp", BLUE, "QAOA C-milp"), ("D", ORANGE, "QAOA D (LP)"), ("A", GREY, "QAOA A")):
        ax.scatter([], [], color=col, marker="o", s=20, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("time per run (s; QAOA on the exact simulator)")
    ax.set_ylabel("$P$(returned state is ground)")
    ax.set_ylim(-0.03, 1.05)
    tidy(ax)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=6.5,
               title="circles: $n = 16$; squares: $n = 22$", title_fontsize=6.5, bbox_to_anchor=(0.5, 0.0),
               handletextpad=0.3, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    save(fig, "fig_classical")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt = setup()
    fig_bandshare(plt)
    fig_eps(plt)
    fig_size(plt)
    fig_classical(plt)
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.pdf"))))


if __name__ == "__main__":
    main()
