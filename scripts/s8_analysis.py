"""Analysis of results/s8_runs.jsonl (scripts/s8_eps_main.py).

Statistics are SCENARIO-LEVEL: per-run values are first averaged over seeds
within each scenario, and paired comparisons are made across the 30
scenarios (the unit at which the instance-dependent effect lives). Reported
per comparison: mean difference, 95% CI (t and bootstrap), Wilcoxon
signed-rank p, and wins/ties/losses over scenarios.

    python scripts/s8_analysis.py --phase 1   # overlap fit + eps sweep (p = 1), figure -> results/
    python scripts/s8_analysis.py --phase 2   # main table (p = 1, 2, 3) + per-scenario detail
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

RESULTS = Path(__file__).resolve().parents[1] / "results"
STORE = RESULTS / "s8_runs.jsonl"
ARMS = ("A", "B", "C-milp", "C-oracle", "D", "E", "F")
SWEEP_ARMS = ("B", "C-milp", "C-oracle", "D", "F")
EPS = (0.0, 0.05, 0.1, 0.15, 0.25, 0.35, 0.5)
MAIN_EPS, MAIN_P, SWEEP_P = 0.1, (1, 2, 3), (1,)
MAIN_SEEDS, SWEEP_SEEDS = tuple(range(5)), tuple(range(3))
RNG = np.random.default_rng(0)
LOG_FLOOR = -300.0


def load():
    out, seen, dup = [], set(), 0
    for line in STORE.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = (r["arm"], r["eps"], r["scenario"], r["seed"], r["p"])
        if key in seen:  # two concurrent runners wrote the same run (deterministic, so identical but timing)
            dup += 1
            continue
        seen.add(key)
        out.append(r)
    if dup:
        print(f"(dropped {dup} duplicate records)")
    return out


def log_ov(r):
    return np.log10(r["overlap0"]) if r["overlap0"] > 0 else -np.inf


class Data:
    def __init__(self, recs):
        self.recs = recs
        self.idx = {}
        for r in recs:
            self.idx.setdefault((r["arm"], r["eps"], r["p"]), []).append(r)
        self.scenarios = sorted({r["scenario"] for r in recs})

    def runs(self, arm, eps, p, seeds):
        eps = MAIN_EPS if arm in ("A", "E") else eps
        return [r for r in self.idx.get((arm, eps, p), []) if r["seed"] in seeds]

    def scen(self, arm, eps, p, metric, seeds, how="mean"):
        """{scenario: aggregate over seeds}; evals-to metrics use the median of reached runs."""
        out = {}
        for r in self.runs(arm, eps, p, seeds):
            out.setdefault(r["scenario"], []).append(metric(r) if callable(metric) else r[metric])
        agg = {}
        for s, v in out.items():
            v = [x for x in v if x is not None]
            if not v:
                agg[s] = None
            elif how == "median":
                agg[s] = float(np.median(v))
            else:
                agg[s] = float(np.mean(v))
        return agg


def paired(x: dict, y: dict, scale=1.0):
    common = [s for s in x if s in y and x[s] is not None and y[s] is not None]
    d = scale * np.array([x[s] - y[s] for s in common], float)
    n = d.size
    if n < 3:
        return dict(n=n, text=f"n={n}: too few")
    m, sd = d.mean(), d.std(ddof=1)
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if sd > 0 else 0.0
    boot = d[RNG.integers(0, n, (10_000, n))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pw = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    w, t, l = int((d > 1e-12).sum()), int((np.abs(d) <= 1e-12).sum()), int((d < -1e-12).sum())
    return dict(n=n, mean=m, ci=(m - half, m + half), boot=(lo, hi), p=pw, wtl=(w, t, l),
                text=f"{m:+.3f} [{m - half:+.3f}, {m + half:+.3f}] (boot [{lo:+.3f}, {hi:+.3f}]) "
                     f"Wilcoxon p={pw:.1e}, scenarios +/=/-: {w}/{t}/{l} of {n}")


def mci(v, scale=1.0, fmt=".3f"):
    v = scale * np.array([x for x in v if x is not None], float)
    if v.size < 2:
        return "-"
    half = stats.t.ppf(0.975, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size)
    return f"{v.mean():{fmt}} [{v.mean() - half:{fmt}}, {v.mean() + half:{fmt}}]"


def reach(D, arm, eps, p, t, seeds):
    rs = D.runs(arm, eps, p, seeds)
    hit = [r[f"to_{t}"] for r in rs if r[f"to_{t}"] is not None]
    return (f"{np.median(hit):4.0f} ({len(hit) / len(rs):4.0%})" if hit else f"never ({0:4.0%})") if rs else "-"


# ---- 1. main table ---------------------------------------------------------------


def main_table(D):
    print("=" * 110)
    print(f"MAIN TABLE: eps = {MAIN_EPS}, {len(D.scenarios)} held-out scenarios x {len(MAIN_SEEDS)} seeds; "
          "scenario-level mean [95% CI over scenarios]")
    for p in MAIN_P:
        print(f"\n--- p = {p} ---")
        print(f"  {'arm':>8} | {'P(best1024=gnd)':>22} | {'P(gnd) 1 shot':>22} | {'gap K=1024 %':>20} | "
              f"{'g final':>22} | {'log10 overlap0':>15} | {'wrong pins':>10}")
        for a in ARMS:
            m = lambda k: list(D.scen(a, MAIN_EPS, p, k, MAIN_SEEDS).values())  # noqa: E731
            print(f"  {a:>8} | {mci(m('p_ground_K1024')):>22} | {mci(m('p_ground_K1'), fmt='.4f'):>22} | "
                  f"{mci(m('gap_repaired_K1024'), 100, '.2f'):>20} | {mci(m('g_final')):>22} | "
                  f"{np.mean([x for x in m(log_ov) if np.isfinite(x)]):15.2f} | {np.mean(m('n_pinned_wrong')):10.2f}")
        print(f"  {'arm':>8} | {'g state0':>7} {'g init':>7} | {'to g=.9':>12} {'to g=.95':>12} {'to g=.99':>12} | "
              f"{'cap':>4} | {'1-shot adeq':>11} | {'1-shot gap raw/rep %':>20} | {'wall s':>6} | {'src us':>6}")
        for a in ARMS:
            rs = D.runs(a, MAIN_EPS, p, MAIN_SEEDS)
            mm = lambda k: np.mean([r[k] for r in rs])  # noqa: E731
            print(f"  {a:>8} | {mm('g_state0'):7.3f} {mm('g_init'):7.3f} | "
                  + " ".join(f"{reach(D, a, MAIN_EPS, p, t, MAIN_SEEDS):>12}" for t in (0.9, 0.95, 0.99))
                  + f" | {mm('cap_hit'):4.0%} | {mm('adequate_K1'):11.3f} | "
                  f"{100 * mm('gap_raw_K1'):9.2f} / {100 * mm('gap_repaired_K1'):7.2f} | {mm('total_s'):6.2f} | "
                  f"{1e6 * np.median([r['source_s'] for r in rs]):6.0f}")
        print("  paired, scenario level (first - second):")
        pairs = [("C-oracle", "D"), ("C-milp", "D"), ("C-oracle", "C-milp"), ("B", "C-milp"), ("C-milp", "F"),
                 ("C-oracle", "F"), ("D", "F")] + [(a, "A") for a in ARMS[1:]]
        for a, b in pairs:
            for key, lab, sc, how in (("p_ground_K1024", "P(b1024=gnd)", 1, "mean"),
                                      ("p_ground_K1", "P(gnd) 1-shot", 1, "mean"),
                                      ("gap_repaired_K1024", "gap K1024 pp", 100, "mean"),
                                      ("g_final", "g final", 1, "mean"), ("to_0.9", "evals to g=.9", 1, "median")):
                res = paired(D.scen(a, MAIN_EPS, p, key, MAIN_SEEDS, how), D.scen(b, MAIN_EPS, p, key, MAIN_SEEDS, how), sc)
                print(f"    {a + ' - ' + b:>19} {lab:>14}: {res['text']}")


def per_scenario(D, p):
    print(f"\nPER-SCENARIO DETAIL, p = {p}, eps = {MAIN_EPS} (means over {len(MAIN_SEEDS)} seeds). "
          "wrong = pinned bits that disagree with the ground state; ov = log10 initial ground-state overlap")
    arms = ("A", "B", "C-milp", "C-oracle", "D", "F")
    print(f"  {'scen':>4} {'peak':>5} {'#gnd':>4} | {'D wrong':>7} {'D ov':>6} {'Cm ov':>6} {'Co ov':>6} {'F ov':>6} | "
          + " ".join(f"{a:>8}" for a in arms) + "   <- P(best of 1024 = ground)")
    P = {a: D.scen(a, MAIN_EPS, p, "p_ground_K1024", MAIN_SEEDS) for a in arms}
    ov = {a: D.scen(a, MAIN_EPS, p, log_ov, MAIN_SEEDS) for a in arms}
    wr = D.scen("D", MAIN_EPS, p, "n_pinned_wrong", MAIN_SEEDS)
    first = {r["scenario"]: r for r in D.runs("A", MAIN_EPS, p, MAIN_SEEDS)}
    for s in D.scenarios:
        print(f"  {s:4d} {first[s]['peak']:5.0f} {first[s]['n_ground']:4d} | {wr[s]:7.0f} {ov['D'][s]:6.1f} "
              f"{ov['C-milp'][s]:6.1f} {ov['C-oracle'][s]:6.1f} {ov['F'][s]:6.1f} | "
              + " ".join(f"{P[a][s]:8.3f}" for a in arms))
    for a in arms[1:]:
        low = [s for s in D.scenarios if P[a][s] < 0.9]
        print(f"  {a:>8}: P < 0.9 in {len(low)}/{len(D.scenarios)} scenarios {low}")


# ---- 2. eps sweep -------------------------------------------------------------------


def best_eps(D, arm, p, scen_subset=None):
    def score(e):
        P = D.scen(arm, e, p, "p_ground_K1024", SWEEP_SEEDS)
        G = D.scen(arm, e, p, "gap_repaired_K1024", SWEEP_SEEDS)
        ss = [s for s in P if scen_subset is None or s in scen_subset]
        return (np.mean([P[s] for s in ss]), -np.mean([G[s] for s in ss]))

    return max((e for e in EPS if e < 0.5), key=score)  # eps = 0.5 is the uniform limit, not a warm start


def sweep(D):
    print("\n" + "=" * 110)
    print(f"EPS SWEEP: {len(D.scenarios)} scenarios x {len(SWEEP_SEEDS)} seeds; scenario-level means "
          "(P(best1024=gnd) with 95% CI over scenarios)")
    for p in SWEEP_P:
        print(f"\n--- p = {p} ---")
        print(f"  {'arm':>8} {'eps':>5} | {'log10 ov':>8} {'wrong':>5} | {'g init':>6} {'g final':>7} | "
              f"{'P(gnd) 1sh':>10} | {'P(best1024=gnd)':>22} | {'gap K1024 %':>11} | {'to g=.9':>12} {'to g=.95':>12}")
        for a in SWEEP_ARMS:
            for e in EPS:
                m = lambda k: list(D.scen(a, e, p, k, SWEEP_SEEDS).values())  # noqa: E731
                lo = [x for x in m(log_ov)]
                lo_txt = f"{np.mean(lo):8.2f}" if all(np.isfinite(lo)) else f"{'-inf':>5}x{sum(~np.isfinite(lo)):<2d}"
                print(f"  {a:>8} {e:5.2f} | {lo_txt:>8} {np.mean(m('n_pinned_wrong')):5.2f} | "
                      f"{np.mean(m('g_init')):6.3f} {np.mean(m('g_final')):7.3f} | "
                      f"{np.mean(m('p_ground_K1')):10.4f} | {mci(m('p_ground_K1024')):>22} | "
                      f"{100 * np.mean(m('gap_repaired_K1024')):11.2f} | "
                      f"{reach(D, a, e, p, 0.9, SWEEP_SEEDS):>12} {reach(D, a, e, p, 0.95, SWEEP_SEEDS):>12}")
        A = D.scen("A", MAIN_EPS, p, "p_ground_K1024", SWEEP_SEEDS)
        print(f"  {'A':>8} {'-':>5} | (reference, same seeds) P(best1024=gnd) {mci(list(A.values()))}")

        best = {a: best_eps(D, a, p) for a in SWEEP_ARMS}
        print(f"\n  best eps (max scenario-mean P(best1024=gnd), ties -> lower cost gap; eps = 0.5 excluded): {best}")
        print("  per scenario at each arm's best eps: P(best of 1024 = ground) (mean over seeds); "
              "D wrong = pinned LP bits that disagree with the ground state; ov = log10 initial overlap")
        Pb = {a: D.scen(a, best[a], p, "p_ground_K1024", SWEEP_SEEDS) for a in SWEEP_ARMS}
        ovb = {a: D.scen(a, best[a], p, log_ov, SWEEP_SEEDS) for a in SWEEP_ARMS}
        wrD = D.scen("D", best["D"], p, "n_pinned_wrong", SWEEP_SEEDS)
        print(f"    {'scen':>4} {'D wrong':>7} {'D ov':>6} {'Cm ov':>6} {'Co ov':>6} | "
              + " ".join(f"{a + '@' + str(best[a]):>12}" for a in SWEEP_ARMS) + f" | {'Co - D':>7}")
        for s in D.scenarios:
            print(f"    {s:4d} {wrD[s]:7.0f} {ovb['D'][s]:6.1f} {ovb['C-milp'][s]:6.1f} {ovb['C-oracle'][s]:6.1f} | "
                  + " ".join(f"{Pb[a][s]:12.3f}" for a in SWEEP_ARMS) + f" | {Pb['C-oracle'][s] - Pb['D'][s]:+7.3f}")
        print("  DECISIVE: each arm at its best eps, paired over scenarios (first - second):")
        for a, b in (("D", "C-oracle"), ("D", "C-milp"), ("D", "F"), ("C-oracle", "C-milp"), ("B", "C-milp")):
            for key, lab, sc in (("p_ground_K1024", "P(b1024=gnd)", 1), ("p_ground_K1", "P(gnd) 1-shot", 1),
                                 ("gap_repaired_K1024", "gap K1024 pp", 100), ("g_final", "g final", 1)):
                res = paired(D.scen(a, best[a], p, key, SWEEP_SEEDS), D.scen(b, best[b], p, key, SWEEP_SEEDS), sc)
                print(f"    {a}@{best[a]} - {b}@{best[b]:<5} {lab:>14}: {res['text']}")
        # 2-fold cross-validated choice of eps (removes the winner's-curse bias of choosing on the same scenarios)
        folds = (set(D.scenarios[0::2]), set(D.scenarios[1::2]))
        cv = {a: {} for a in SWEEP_ARMS}
        chosen = {a: [] for a in SWEEP_ARMS}
        for k in (0, 1):
            for a in SWEEP_ARMS:
                e = best_eps(D, a, p, folds[k])
                chosen[a].append(e)
                vals = D.scen(a, e, p, "p_ground_K1024", SWEEP_SEEDS)
                cv[a].update({s: vals[s] for s in folds[1 - k]})
        print(f"  2-fold cross-validated eps choice (fold A -> B, B -> A): {chosen}")
        for a, b in (("D", "C-oracle"), ("D", "C-milp"), ("D", "F")):
            print(f"    CV {a} - {b}  P(b1024=gnd): {paired(cv[a], cv[b])['text']}")

        # eps = 0.5 limit
        print("  eps = 0.5 limit:")
        ref = {(r["scenario"], r["seed"]): r for r in D.runs("D", 0.5, p, SWEEP_SEEDS)}
        dev = max(abs(r[k] - ref[(r["scenario"], r["seed"])][k]) for a in ("C-milp", "C-oracle", "F")
                  for r in D.runs(a, 0.5, p, SWEEP_SEEDS) for k in ("p_ground_K1024", "g_final", "n_evals"))
        print(f"    warm-mixer arms C-milp, C-oracle, D, F identical to each other at eps 0.5: max |diff| = {dev:.1e}")
        Ar = {(r["scenario"], r["seed"]): r for r in D.runs("A", MAIN_EPS, p, SWEEP_SEEDS)}
        Bh = D.runs("B", 0.5, p, SWEEP_SEEDS)
        dB = {k: [abs(r[k] - Ar[(r["scenario"], r["seed"])][k]) for r in Bh]
              for k in ("overlap0", "g_init", "g_final", "p_ground_K1024", "n_evals")}
        print("    B (X mixer) at eps 0.5 vs arm A, run by run (same start to ~1e-16; the patience stop then fires a "
              "few evals apart): " + ", ".join(f"{k} max|d| {max(v):.1e}" for k, v in dB.items()))
        for key, lab in (("p_ground_K1024", "P(b1024=gnd)"), ("g_final", "g final"), ("overlap0", "overlap0")):
            res = paired(D.scen("D", 0.5, p, key, SWEEP_SEEDS), D.scen("A", MAIN_EPS, p, key, SWEEP_SEEDS))
            print(f"    warm mixer @0.5 - A {lab:>13}: {res['text']}")


# ---- 3. overlap -> sampling --------------------------------------------------------


def sigmoid(x, a, b):
    return 1.0 / (1.0 + np.exp(-(a + b * x)))


def overlap_fit(D):
    print("\n" + "=" * 110)
    print("OVERLAP vs SAMPLING: P(best of 1024 = ground) against log10 |<ground|psi_0>|^2, pooled over all arms and "
          f"eps at each p (every run present for this phase, {len(D.recs)} runs)")
    fits = {}
    for p in SWEEP_P:
        rs = [r for r in D.recs if r["p"] == p]
        zero = [r for r in rs if r["overlap0"] == 0]
        rs = [r for r in rs if r["overlap0"] > 0]
        if len(rs) < 10:
            print(f"\n  p = {p}: no data yet")
            continue
        x = np.array([np.log10(r["overlap0"]) for r in rs])
        y = np.array([r["p_ground_K1024"] for r in rs])
        arm = np.array([r["arm"] for r in rs])
        (a, b), _ = curve_fit(sigmoid, x, y, p0=(5.0, 1.0), maxfev=20000)
        res = y - sigmoid(x, a, b)
        tss = ((y - y.mean()) ** 2).sum()
        r2 = 1 - (res ** 2).sum() / tss
        rss_arm, k_arm = 0.0, 0
        for g in sorted(set(arm)):
            m = arm == g
            try:
                pa, _ = curve_fit(sigmoid, x[m], y[m], p0=(a, b), maxfev=20000)
            except RuntimeError:
                pa = (a, b)
            rss_arm += ((y[m] - sigmoid(x[m], *pa)) ** 2).sum()
            k_arm += 2
        r2_arm = 1 - rss_arm / tss
        F = ((res ** 2).sum() - rss_arm) / (k_arm - 2) / (rss_arm / (len(y) - k_arm))
        pF = stats.f.sf(F, k_arm - 2, len(y) - k_arm)
        ref = 1 - (1 - 10.0 ** x) ** 1024  # sampling the INITIAL state 1024 times (no QAOA layer)
        r2_ref = 1 - ((y - ref) ** 2).sum() / tss
        rho = stats.spearmanr(x, y).statistic
        # scenario-level (arm, eps, scenario means)
        grp = {}
        for r, xi in zip(rs, x):
            grp.setdefault((r["arm"], r["eps"], r["scenario"]), []).append((xi, r["p_ground_K1024"]))
        gx = np.array([np.mean([v[0] for v in g]) for g in grp.values()])
        gy = np.array([np.mean([v[1] for v in g]) for g in grp.values()])
        r2_grp = 1 - ((gy - sigmoid(gx, a, b)) ** 2).sum() / ((gy - gy.mean()) ** 2).sum()
        print(f"\n  p = {p}: {len(y)} runs with overlap > 0 ({len(zero)} runs with overlap exactly 0: "
              f"P(best1024=gnd) = {max([r['p_ground_K1024'] for r in zero], default=0):.1e} at most)")
        print(f"    pooled logistic  P = 1 / (1 + exp(-({a:.3f} + {b:.3f} * log10 ov))):  R^2 = {r2:.3f} (runs), "
              f"{r2_grp:.3f} (arm x eps x scenario means); Spearman rho = {rho:.3f}")
        print(f"    50% point at overlap = 10^{-a / b:.2f}; 90% at 10^{(np.log(9) - a) / b:.2f}")
        print(f"    separate curve per arm: R^2 = {r2_arm:.3f} (Delta R^2 = {r2_arm - r2:+.3f}; "
              f"F = {F:.1f}, p = {pF:.1e} with {len(y)} runs)")
        print(f"    no-QAOA reference 1 - (1 - ov)^1024: R^2 = {r2_ref:.3f}")
        print("    mean residual by arm (runs; 95% CI):")
        for g in sorted(set(arm)):
            m = arm == g
            print(f"      {g:>8}: {mci(res[m], fmt='+.3f'):>28}   (n = {m.sum()})")
        fits[p] = (a, b)
    # frozen bits -> overlap
    print("\n  FROZEN BITS -> OVERLAP (sweep arms, eps > 0; unique arm x scenario per eps; prediction: each wrong pin "
          "multiplies the overlap by about eps / (1 - eps))")
    for e in EPS[1:-1]:
        rows = {}
        for r in D.recs:
            if r["eps"] == e and r["arm"] in SWEEP_ARMS and r["p"] == 1 and r["overlap0"] > 0:
                rows[(r["arm"], r["scenario"])] = (r["n_pinned_wrong"], np.log10(r["overlap0"]))
        k = np.array([v[0] for v in rows.values()], float)
        lo = np.array([v[1] for v in rows.values()])
        slope, icpt, rv, _, se = stats.linregress(k, lo)
        print(f"    eps {e:4.2f}: log10 ov = {icpt:6.2f} + {slope:6.2f} * wrong_pins  (R^2 = {rv ** 2:.3f}, "
              f"slope SE {se:.2f}); predicted slope log10(eps/(1-eps)) = {np.log10(e / (1 - e)):6.2f}")
    return fits


def figure(D, fits):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, INK2, GRID, BLUE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "axes.labelcolor": INK})
    arms = [a for a in ARMS if any(r['arm'] == a and r['p'] in SWEEP_P for r in D.recs)]
    fig, axes = plt.subplots(len(SWEEP_P), len(arms), figsize=(11, 1.2 + 2.2 * len(SWEEP_P)), sharex=True,
                             sharey=True, squeeze=False)
    xs = np.linspace(-6, 0, 300)
    for i, p in enumerate(SWEEP_P):
        a, b = fits[p]
        for j, arm in enumerate(arms):
            ax = axes[i, j]
            grp = {}
            for r in D.recs:
                if r["arm"] == arm and r["p"] == p:
                    grp.setdefault((r["eps"], r["scenario"]), []).append(r)
            gx = [np.log10(max(np.mean([r["overlap0"] for r in g]), 1e-6)) for g in grp.values()]
            gy = [np.mean([r["p_ground_K1024"] for r in g]) for g in grp.values()]
            ax.plot(xs, 1 - (1 - 10.0 ** xs) ** 1024, color=INK2, lw=1, ls=(0, (3, 2)))
            ax.plot(xs, sigmoid(xs, a, b), color=INK, lw=1.5)
            ax.scatter(gx, gy, s=10, color=BLUE, alpha=0.55, lw=0)
            ax.grid(color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if i == 0:
                ax.set_title(arm, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"p = {p}\nP(best of 1024 = ground)")
    fig.supxlabel("log10 initial ground-state overlap |<ground|psi_0>|^2   (points: arm x eps x scenario means; "
                  "<= -6 drawn at -6)\nsolid: one logistic fit pooled over all arms and eps;  dashed: sampling psi_0 "
                  "itself 1024 times (no QAOA layer)", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"fig_overlap_vs_sampling.{ext}", dpi=200)
    print("\nwrote results/fig_overlap_vs_sampling.png and .pdf")


def main():
    try:
        phase = int(sys.argv[sys.argv.index("--phase") + 1])
    except (ValueError, IndexError):
        raise SystemExit("usage: s8_analysis.py --phase 1|2")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from s8_eps_main import tasks

    keys = tasks(phase)
    recs = [r for r in load() if (r["arm"], r["eps"], r["scenario"], r["seed"], r["p"]) in keys]
    D = Data(recs)
    print(f"phase {phase}: {len(D.recs)} of {len(keys)} runs present, {len(D.scenarios)} scenarios")
    if phase == 1:
        fits = overlap_fit(D)
        sweep(D)
        figure(D, fits)
    else:
        main_table(D)
        for p in (1, 3):
            per_scenario(D, p)


if __name__ == "__main__":
    main()
