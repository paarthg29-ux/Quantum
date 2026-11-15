"""Band share on 3-regular MaxCut (runs of scripts/s11_maxcut.py): the three
decisive questions. The fits, the features and the band (delta = 5% of the
spectral range, m_e05) are those of the UC analysis (scripts/s8_mass.py),
unchanged; nothing here is tuned on the MaxCut data.

  1. REFIT on MaxCut, per size and pooled: P(best of 1024 = optimal) on log10
     overlap alone vs log10 overlap + log10 m_e05, with the per-source
     intercept test. Held-out check: fit without the BAD sources, predict them.
  2. TRANSFER: the UC L2 (phase 1, p = 1) fits applied to MaxCut WITHOUT
     refitting -- the test the L2 curve passed at L4.
  3. DOSE-RESPONSE in wrongly pinned variables for the BAD sources (and for
     GW, by its own wrong pins).

Sensitivity, reported and labelled as such (not the pre-specified band): the
MaxCut spectrum is integer, so a 5% band holds few cut levels (at n = 12 only
the optimum itself); the same fits are repeated with the absolute bands
"within one / two edges of the maximum cut" (m_cut1, m_cut2).

    python scripts/s11_maxcut_analysis.py   # -> results/s11_maxcut.txt, .json
"""

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "s11_maxcut.txt"
ARMS = ("A", "GW", "GW-LS", "E", "BAD0", "BAD1", "BAD2", "BAD3", "BAD4")
MAIN_EPS = 0.1
Y = "p_ground_K1024"


def mci(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return f"{v.mean():.3f} (n={len(v)})"
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return f"{v.mean():.3f} [{v.mean() - h:.3f}, {v.mean() + h:.3f}] (n={len(v)})"


def by_instance(rs, key=Y):
    """mean over seeds within an instance -> array of per-instance values."""
    d = {}
    for r in rs:
        d.setdefault((r["n"], r["inst"]), []).append(r[key])
    return np.array([np.mean(v) for v in d.values()])


def main():
    import s8_mass as M
    import s11_maxcut as X

    recs = X.load()
    lines = []
    say = lines.append
    todo = X.tasks()
    have = {(r["n"], r["inst"], r["arm"], r["eps"], r["seed"]) for r in recs}
    say(f"3-regular MaxCut, p = 1: {len(recs)} runs in the store ({len(todo - have)} of {len(todo)} planned runs "
        f"missing); sizes {sorted({r['n'] for r in recs})}, {len({(r['n'], r['inst']) for r in recs})} graphs.")
    out = dict(n_runs=len(recs), missing=len(todo - have), sizes={}, transfer={}, dose={}, sources={})

    # ---- 0. the sources ------------------------------------------------------
    say("\n0. SOURCES (per graph; cut as a fraction of the maximum cut; wrong pins against the nearest optimum)")
    for n in X.SIZES:
        one = [r for r in recs if r["n"] == n and r["seed"] == 0 and r["eps"] == MAIN_EPS]
        row = []
        for a in ("GW", "GW-LS"):
            rs = [r for r in one if r["arm"] == a]
            frac = np.array([r["cut_source"] / r["cut_opt"] for r in rs])
            wrong = np.array([r["n_pinned_wrong"] for r in rs])
            row.append(f"{a}: cut/opt {frac.mean():.3f}, optimal in {int((frac == 1).sum())}/{len(rs)}, "
                       f"wrong pins mean {wrong.mean():.2f} (max {wrong.max()})")
            out["sources"][f"{n}/{a}"] = dict(cut_frac=float(frac.mean()), n_optimal=int((frac == 1).sum()),
                                             n=len(rs), wrong_mean=float(wrong.mean()))
        g = [r for r in one if r["arm"] == "GW"]
        say(f"  n = {n}: ground states per graph median {int(np.median([r['n_ground'] for r in g]))} "
            f"(range {min(r['n_ground'] for r in g)}-{max(r['n_ground'] for r in g)}); SDP dual-slack min eigenvalue "
            f">= {min(r['sdp_slack_min_eig'] for r in g):.1e}; " + "; ".join(row))

    # ---- mean P by source and eps ---------------------------------------------
    say(f"\n   mean P(best of 1024 = optimal) by source and eps (mean over graphs of the seed mean)")
    for n in X.SIZES:
        say(f"  n = {n}:   eps  " + " ".join(f"{e:>6}" for e in X.EPS))
        for a in ARMS:
            vals = []
            for e in X.EPS:
                rs = [r for r in recs if r["n"] == n and r["arm"] == a and r["eps"] == e]
                vals.append(f"{by_instance(rs).mean():6.3f}" if rs else "     -")
            say(f"    {a:>6}       " + " ".join(vals))
    say(f"   mean final gain g by source at eps = {MAIN_EPS} (for the approximation-ratio vs sampling comparison)")
    for n in X.SIZES:
        say(f"  n = {n}: " + ", ".join(
            f"{a} g {by_instance([r for r in recs if r['n'] == n and r['arm'] == a and r['eps'] == MAIN_EPS], 'g_final').mean():.3f}"
            f" / P {by_instance([r for r in recs if r['n'] == n and r['arm'] == a and r['eps'] == MAIN_EPS]).mean():.3f}"
            for a in ARMS))

    # ---- 1. refit --------------------------------------------------------------
    uc = json.loads((ROOT / "results" / "s8_mass_phase1.json").read_text())["1"]
    say("\n1. REFIT ON MAXCUT (logistic least squares, runs with overlap0 > 0; residual = observed - fitted, mean by "
        "source). UC reference at L2: overlap R^2 0.552, overlap + m_e05 R^2 0.952.")
    for label, sel in [(f"n = {n}", lambda r, n=n: r["n"] == n) for n in X.SIZES] + [("pooled", lambda r: True)]:
        rs = [r for r in recs if sel(r) and r["overlap0"] > 0]
        y = np.array([r[Y] for r in rs])
        arm = np.array([r["arm"] for r in rs])
        arms = [a for a in ARMS if a in set(arm)]
        tss = float(((y - y.mean()) ** 2).sum())
        lg = lambda k: np.log10(np.maximum([r[k] for r in rs], 1e-300))  # noqa: E731
        ov, me5 = lg("overlap0"), lg("m_e05")
        same = float(np.mean(np.isclose(ov, me5, rtol=0, atol=1e-12)))
        say(f"\n  --- {label}: {len(y)} runs ({sum(1 for r in recs if sel(r)) - len(y)} with overlap 0 excluded); "
            f"the 5% band is the optimum alone (m_e05 = overlap) in {100 * same:.0f}% of runs; Spearman with P: "
            f"log10 overlap {stats.spearmanr(ov, y).statistic:+.3f}, log10 m_e05 {stats.spearmanr(me5, y).statistic:+.3f}, "
            f"log10(ov/m_e05) {stats.spearmanr(ov - me5, y).statistic:+.3f}")
        res = {}
        models = [("log10 overlap", [ov]), ("ov + m_e05", [ov, me5]), ("band share log10(ov/m_e05)", [ov - me5]),
                  ("ov + m_e01 [sens.]", [ov, lg("m_e01")]), ("ov + m_cut1 [sens.]", [ov, lg("m_cut1")]),
                  ("ov + m_cut2 [sens.]", [ov, lg("m_cut2")])]
        for name, cols in models:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                res[name] = M.report(name, cols, y, arm, arms, tss)
            say(buf.getvalue().rstrip())
        # held-out sources: fit without the BAD arms, predict them
        bad = np.array([a.startswith("BAD") and a != "BAD0" for a in arm])
        for name, cols in models[:2] + models[4:5]:
            Xm = np.column_stack([np.ones(len(y))] + cols)
            b, _, _ = M.fit(Xm[~bad], y[~bad])
            pred = M.sigmoid(Xm @ b)
            bias = {a: float(np.mean(y[arm == a] - pred[arm == a])) for a in arms if a.startswith("BAD") and a != "BAD0"}
            r2 = 1 - float(((y[bad] - pred[bad]) ** 2).sum()) / float(((y[bad] - y[bad].mean()) ** 2).sum())
            say(f"    held-out BAD1-4 (fit on the other sources), {name}: R^2 on the held-out runs {r2:.3f}; "
                "observed - predicted " + ", ".join(f"{a} {v:+.3f}" for a, v in bias.items()))
            res[f"heldout {name}"] = dict(r2=r2, bias=bias)
        out["sizes"][label] = dict(n_runs=int(len(y)), frac_band_is_optimum=same, fits=res)

        # ---- 2. transfer (same subset) ----------------------------------------
        tr = {}
        for key, cols in (("baseline: log10 overlap0", [ov]), ("ov + m_e05", [ov, me5])):
            b = np.array(uc[key]["coef"])
            pred = M.sigmoid(np.column_stack([np.ones(len(y))] + cols) @ b)
            resid = y - pred
            tr[key] = dict(r2=float(1 - (resid @ resid) / tss), mae=float(np.abs(resid).mean()),
                           resid={a: float(resid[arm == a].mean()) for a in arms})
            say(f"    TRANSFER of the UC L2 fit '{key}' (coef {np.round(b, 2).tolist()}), unchanged: R^2 "
                f"{tr[key]['r2']:.3f}, mean |error| {tr[key]['mae']:.3f}; observed - predicted by source: "
                + ", ".join(f"{a} {v:+.3f}" for a, v in tr[key]["resid"].items()))
        out["transfer"][label] = tr

    # ---- 3. dose-response -------------------------------------------------------
    say("\n3. DOSE-RESPONSE IN WRONG PINS. BAD sources: P(best of 1024 = optimal) by the number of variables pinned "
        "against the nearest optimum (mean over graphs of the seed mean, 95% t interval). UC reference: the LP's "
        "deficit grows with its wrong pins.")
    for n in X.SIZES:
        for e in (0.05, MAIN_EPS, 0.25):
            row = []
            for d in range(0, 5):
                rs = [r for r in recs if r["n"] == n and r["eps"] == e and r["arm"].startswith("BAD")
                      and r["n_pinned_wrong"] == d]
                if rs:
                    v = by_instance(rs)
                    row.append(f"{d}: {mci(v)}")
                    out["dose"][f"{n}/{e}/{d}"] = dict(mean=float(v.mean()), n=int(len(v)))
            say(f"  n = {n}, eps = {e}: " + " | ".join(row))
    say("  Spearman of P with the wrong-pin count over BAD runs (per size, eps = 0.1), and with the features:")
    for n in X.SIZES:
        rs = [r for r in recs if r["n"] == n and r["eps"] == MAIN_EPS and r["arm"].startswith("BAD")]
        d = [r["n_pinned_wrong"] for r in rs]
        y = [r[Y] for r in rs]
        say(f"    n = {n}: wrong pins {stats.spearmanr(d, y).statistic:+.3f}; mean log10 overlap by dose "
            + ", ".join(f"{k}: {np.mean([np.log10(r['overlap0']) for r in rs if r['n_pinned_wrong'] == k]):.2f}"
                        for k in sorted(set(d)))
            + "; mean log10 m_e05 by dose "
            + ", ".join(f"{k}: {np.mean([np.log10(r['m_e05']) for r in rs if r['n_pinned_wrong'] == k]):.2f}"
                        for k in sorted(set(d))))
    say("  GW by its own wrong pins (eps = 0.1), all sizes pooled:")
    rs = [r for r in recs if r["arm"] == "GW" and r["eps"] == MAIN_EPS]
    for d in sorted({r["n_pinned_wrong"] for r in rs}):
        say(f"    {d} wrong: {mci(by_instance([r for r in rs if r['n_pinned_wrong'] == d]))}")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUT.with_suffix(".json").write_text(json.dumps(out, indent=1, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
