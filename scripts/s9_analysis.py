"""Phase 3 analysis: arms at L4 (22 qubits), 20 structurally diverse scenarios.

Each selected scenario is the only member of its (ground state, LP pin
pattern) signature in the set, so the unit of analysis is the scenario
(= structure; effective n = 20). Values are averaged over the 3 seeds first.
Two views of every comparison:

  per structure  unweighted: every structure counts once (the selection
                 over-represents rare structures by design);
  reweighted     each structure weighted by its signature's share of the
                 300-draw census pool (s9_selected.json "weight"), renormalised
                 over the 20; 95% CI by bootstrap over structures. This
                 estimates the effect for the demand distribution, but only
                 over the 34% of it that the 20 signatures cover.

    python scripts/s9_analysis.py           # -> results/s9_analysis.txt
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OWN_POINT = "--point" in sys.argv and sys.argv[sys.argv.index("--point") + 1] == "own"
OUT = ROOT / "results" / ("s9_analysis_ownpoint.txt" if OWN_POINT else "s9_analysis.txt")
RNG = np.random.default_rng(0)
COMBOS = (("A", 0.1), ("C-milp", 0.1), ("C-milp", 0.15), ("D", 0.1), ("D", 0.25), ("F", 0.1), ("F", 0.25))


def main():
    from s9_l4_arms import load, selected

    sel = {s["scenario"]: s for s in selected()}
    R = load()
    seen, recs = set(), []
    for r in R:
        k = (r["arm"], r["eps"], r["scenario"], r["seed"], r["p"])
        if k not in seen:
            seen.add(k)
            recs.append(r)
    S = sorted(sel)
    w = np.array([sel[s]["weight"] for s in S])
    w = w / w.sum()

    def scen(arm, eps, key, p=1):
        out = {}
        for r in recs:
            if r["arm"] == arm and r["eps"] == eps and r["p"] == p:
                out.setdefault(r["scenario"], []).append(r[key] if not callable(key) else key(r))
        return np.array([np.mean(out[s]) if s in out else np.nan for s in S])

    lines = []
    P = lambda s="": lines.append(s)  # noqa: E731
    n_runs = len(recs)
    P(f"PHASE 3, L4 (22 qubits), p = 1: {n_runs} of {len(S) * 3 * len(COMBOS)} runs present; {len(S)} structures "
      f"(one per signature), covering {sum(sel[s]['weight'] for s in S):.1%} of the demand distribution")

    def tci(v):
        v = v[~np.isnan(v)]
        if v.size < 3:
            return "-"
        h = stats.t.ppf(0.975, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size)
        return f"{v.mean():.3f} [{v.mean() - h:.3f}, {v.mean() + h:.3f}]"

    def wci(v):
        m = ~np.isnan(v)
        if m.sum() < 3:
            return "-"
        vv, ww = v[m], w[m] / w[m].sum()
        idx = RNG.integers(0, vv.size, (5000, vv.size))
        bs = (vv[idx] * ww[idx]).sum(1) / ww[idx].sum(1)
        return f"{(vv * ww).sum():.3f} [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]"

    P("\nARMS: P(best of 1024 = ground) per structure (mean [95% t CI over structures]) and reweighted to the demand "
      "distribution (bootstrap CI)")
    P(f"  {'arm':>8} {'eps':>5} | {'per structure':>22} | {'reweighted':>22} | {'P 1-shot':>8} | {'g final':>7} | "
      f"{'gap K1024 %':>11} | {'wrong pins':>10} | {'log10 ov':>8} | {'evals':>5} | {'cap':>4}")
    for a, e in COMBOS:
        pb = scen(a, e, "p_ground_K1024")
        lo = scen(a, e, lambda r: np.log10(max(r["overlap0"], 1e-300)))
        P(f"  {a:>8} {e:5.2f} | {tci(pb):>22} | {wci(pb):>22} | {np.nanmean(scen(a, e, 'p_ground_K1')):8.4f} | "
          f"{np.nanmean(scen(a, e, 'g_final')):7.3f} | {100 * np.nanmean(scen(a, e, 'gap_repaired_K1024')):11.2f} | "
          f"{np.nanmean(scen(a, e, 'n_pinned_wrong')):10.2f} | {np.nanmean(lo):8.2f} | "
          f"{np.nanmean(scen(a, e, 'n_evals')):5.0f} | {np.nanmean(scen(a, e, 'cap_hit')):4.0%}")

    def paired(x, y, label):
        m = ~np.isnan(x) & ~np.isnan(y)
        d = x[m] - y[m]
        if d.size < 3:
            P(f"    {label}: too few")
            return
        h = stats.t.ppf(0.975, d.size - 1) * d.std(ddof=1) / np.sqrt(d.size) if d.std() > 0 else 0.0
        bs = d[RNG.integers(0, d.size, (10000, d.size))].mean(1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pw = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        ww = w[m] / w[m].sum()
        idx = RNG.integers(0, d.size, (5000, d.size))
        wbs = (d[idx] * ww[idx]).sum(1) / ww[idx].sum(1)
        P(f"    {label}: per structure {d.mean():+.3f} [{d.mean() - h:+.3f}, {d.mean() + h:+.3f}] "
          f"(boot [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]), Wilcoxon p={pw:.1e}, "
          f"+/=/-: {(d > 1e-12).sum()}/{(np.abs(d) <= 1e-12).sum()}/{(d < -1e-12).sum()} of {d.size} | reweighted "
          f"{(d * ww).sum():+.3f} [{np.percentile(wbs, 2.5):+.3f}, {np.percentile(wbs, 97.5):+.3f}]")

    P("\nPAIRED over structures (first - second), P(best of 1024 = ground):")
    for (a, e), (b, f) in ((("C-milp", 0.1), ("D", 0.1)), (("C-milp", 0.15), ("D", 0.25)),
                           (("C-milp", 0.1), ("F", 0.1)), (("C-milp", 0.15), ("F", 0.25)),
                           (("D", 0.25), ("F", 0.25)), (("C-milp", 0.1), ("A", 0.1)), (("D", 0.1), ("A", 0.1)),
                           (("D", 0.25), ("D", 0.1)), (("C-milp", 0.15), ("C-milp", 0.1))):
        paired(scen(a, e, "p_ground_K1024"), scen(b, f, "p_ground_K1024"), f"{a}@{e} - {b}@{f}".ljust(26))
    P("\nPAIRED, other metrics (C-milp@0.1 - D@0.1 and at each arm's Phase-1 best eps):")
    for key, lab in (("p_ground_K1", "P 1-shot"), ("g_final", "g final"), ("gap_repaired_K1024", "gap K1024")):
        for (a, e), (b, f) in ((("C-milp", 0.1), ("D", 0.1)), (("C-milp", 0.15), ("D", 0.25))):
            paired(scen(a, e, key), scen(b, f, key), f"{lab:>9} {a}@{e} - {b}@{f}".ljust(26))

    P("\nMECHANISM AT 22 QUBITS: split by how many bits the LP pins wrongly at THIS operating point (from the runs)")
    wrongD = scen("D", 0.1, "n_pinned_wrong")  # = the census count at the frozen point; differs at the own point
    for lab, m in (("D wrong <= 1", wrongD <= 1), ("D wrong >= 2", wrongD >= 2), ("D wrong >= 3", wrongD >= 3)):
        cm, d1 = scen("C-milp", 0.15, "p_ground_K1024")[m], scen("D", 0.25, "p_ground_K1024")[m]
        cm0, d0 = scen("C-milp", 0.1, "p_ground_K1024")[m], scen("D", 0.1, "p_ground_K1024")[m]
        P(f"  {lab:>13} ({m.sum():2d} structures): at best eps C-milp {np.nanmean(cm):.3f} vs D {np.nanmean(d1):.3f} "
          f"(diff {np.nanmean(cm - d1):+.3f}); at eps 0.1 C-milp {np.nanmean(cm0):.3f} vs D {np.nanmean(d0):.3f} "
          f"(diff {np.nanmean(cm0 - d0):+.3f})")
    wc = scen("C-milp", 0.1, "n_pinned_wrong")
    wd = scen("D", 0.1, "n_pinned_wrong")
    P(f"  wrong pins per structure: C-milp mean {np.nanmean(wc):.2f} vs D {np.nanmean(wd):.2f}; C-milp pins >= 2 wrong "
      f"in {np.mean(wc >= 2):.0%} of structures, D in {np.mean(wd >= 2):.0%}")

    P("\nPER STRUCTURE (means over 3 seeds): P(best of 1024 = ground)")
    P(f"  {'scen':>4} {'weight':>6} {'D wr':>4} {'Cm wr':>5} | " + " ".join(f"{a + '@' + str(e):>11}" for a, e in COMBOS))
    cols = {c: scen(*c, "p_ground_K1024") for c in COMBOS}
    for i, s in enumerate(S):
        P(f"  {s:4d} {sel[s]['weight']:6.3f} {wd[i]:4.0f} {wc[i]:5.1f} | "
          + " ".join(f"{cols[c][i]:11.3f}" for c in COMBOS))

    if OWN_POINT:  # G1: same 20 structures, frozen (L2-calibrated) vs L4's own operating point, paired
        frozen = [json.loads(l) for l in (ROOT / "results" / "s9_runs.jsonl").read_text().splitlines() if l.strip()]

        def scen_f(arm, eps, key):
            out = {}
            for r in frozen:
                if r["arm"] == arm and r["eps"] == eps and r["p"] == 1:
                    out.setdefault(r["scenario"], []).append(r[key])
            return np.array([np.mean(out[s]) for s in S])

        P("\nG1: FROZEN (lambda 100, B 250) vs L4's OWN point (lambda 100, B 150), same 20 structures, paired")
        P(f"  {'arm':>8} {'eps':>5} | {'P(b1024) frozen':>15} {'own':>7} {'own - frozen':>22}")
        for a, e in COMBOS:
            fz, ow = scen_f(a, e, "p_ground_K1024"), scen(a, e, "p_ground_K1024")
            d = ow - fz
            h = stats.t.ppf(0.975, d.size - 1) * d.std(ddof=1) / np.sqrt(d.size)
            P(f"  {a:>8} {e:5.2f} | {fz.mean():15.3f} {ow.mean():7.3f} {d.mean():+.3f} [{d.mean() - h:+.3f}, {d.mean() + h:+.3f}]")
        P("  the advantage at each point, and its change (difference of paired differences, per structure):")
        for (a, e), (b, f) in ((("C-milp", 0.1), ("D", 0.1)), (("C-milp", 0.15), ("D", 0.25)),
                               (("C-milp", 0.1), ("F", 0.1)), (("C-milp", 0.1), ("A", 0.1))):
            dz = scen_f(a, e, "p_ground_K1024") - scen_f(b, f, "p_ground_K1024")
            do = scen(a, e, "p_ground_K1024") - scen(b, f, "p_ground_K1024")
            dd = do - dz
            h = stats.t.ppf(0.975, dd.size - 1) * dd.std(ddof=1) / np.sqrt(dd.size)
            P(f"    {a}@{e} - {b}@{f}: frozen {dz.mean():+.3f}, own {do.mean():+.3f}; change {dd.mean():+.3f} "
              f"[{dd.mean() - h:+.3f}, {dd.mean() + h:+.3f}]")
        wf = scen_f("D", 0.1, "n_pinned_wrong")
        P(f"  D wrong pins: frozen {wf.mean():.2f} -> own {wd.mean():.2f} per structure; C-milp: "
          f"{scen_f('C-milp', 0.1, 'n_pinned_wrong').mean():.2f} -> {wc.mean():.2f}")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
