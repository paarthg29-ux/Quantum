"""Head-to-head against the published heuristic: as a RANKING rule over
warm-start sources, how often does each depth-0 predictor pick the source
that actually samples the optimum best?

Bhattacharyya, Capriotti and Tate (arXiv:2504.06253) rank warm starts by the
depth-0 values of the initial state: its overlap with the optimal states and
its approximation ratio. Both are made explicit baselines here, against band
share. No new runs: UC from results/s8_runs.jsonl (16 qubits) and
results/s9_runs.jsonl (22 qubits), MaxCut from results/s11_maxcut_runs.jsonl.

Rules (each scores a candidate from its INITIAL state only; highest wins)
  overlap     log10 overlap0                       (their depth-0 P)
  ratio0      g_state0, the initial approximation gain (their depth-0 alpha)
  bandshare   b1 log10 overlap0 + b2 log10 m_e05, with (b1, b2) the UC L2
              phase-1 fit, FIXED: in-sample at UC 16 qubits (two fitted
              exponents), out-of-sample at UC 22 qubits and on MaxCut
  share       log10(overlap0 / m_e05), the plain ratio: no fitted number

Truth: P(best of 1024 samples is optimal) after QAOA, p = 1, mean over seeds.
A ranking problem is one instance with a set of candidates:
  (a) the sources at eps = 0.1                       [primary]
  (b) the sources at each eps of the sweep, one problem per (instance, eps)
  (c) every (source, eps) configuration of an instance jointly
Scores:
  top-1    the rule's choice is a best candidate (within TOL = 0.001 of the
           maximum). Predictor ties are split evenly (expected value of a
           random tie-break), so counts can be fractional.
  regret   P(best) - P(chosen), averaged over problems
  pairs    share of candidate pairs whose P differ by more than 0.01 that
           the rule orders correctly (a predictor tie counts one half)
Problems in which all candidates are within TOL of each other are counted
separately as "trivial" and excluded: any rule is right there.

    python scripts/s11_ranking.py   # -> results/s11_ranking.txt, .json
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "s11_ranking.txt"
TOL, PAIR_MIN = 1e-3, 0.01
Y = "p_ground_K1024"
RULES = ("overlap", "ratio0", "bandshare", "share")
FLOOR = -300.0


def scores(r, b):
    lo = np.log10(r["overlap0"]) if r["overlap0"] > 0 else FLOOR
    lm = np.log10(r["m_e05"]) if r["m_e05"] > 0 else FLOOR
    return dict(overlap=lo, ratio0=r["g_state0"], bandshare=b[1] * lo + b[2] * lm, share=lo - lm)


def candidates(recs, inst_key, cand_key, b):
    """{instance: {candidate: (P mean over seeds, {rule: score mean over seeds})}}"""
    acc = {}
    for r in recs:
        acc.setdefault(inst_key(r), {}).setdefault(cand_key(r), []).append((r[Y], scores(r, b)))
    return {i: {c: (float(np.mean([p for p, _ in v])), {k: float(np.mean([s[k] for _, s in v])) for k in RULES})
                for c, v in cs.items()} for i, cs in acc.items()}


def evaluate(problems):
    """problems: list of {candidate: (P, scores)}; returns per-rule stats."""
    out = {k: dict(top1=0.0, regret=0.0, pair_ok=0.0, pair_n=0) for k in RULES}
    n = trivial = 0
    for cs in problems:
        P = np.array([v[0] for v in cs.values()])
        if len(P) < 2 or P.max() - P.min() <= TOL:
            trivial += 1
            continue
        n += 1
        for k in RULES:
            s = np.array([v[1][k] for v in cs.values()])
            top = np.isclose(s, s.max(), rtol=0, atol=1e-9)
            out[k]["top1"] += float(np.mean(P[top] >= P.max() - TOL))
            out[k]["regret"] += float(np.mean(P.max() - P[top]))
            i, j = np.triu_indices(len(P), 1)
            m = np.abs(P[i] - P[j]) > PAIR_MIN
            ds, dp = (s[i] - s[j])[m], (P[i] - P[j])[m]
            tie = np.abs(ds) <= 1e-9
            out[k]["pair_ok"] += float((np.sign(ds[~tie]) == np.sign(dp[~tie])).sum() + 0.5 * tie.sum())
            out[k]["pair_n"] += int(m.sum())
    for k in RULES:
        o = out[k]
        o["regret"] = o["regret"] / n if n else float("nan")
        o["pairs"] = o["pair_ok"] / o["pair_n"] if o["pair_n"] else float("nan")
        o["top1_frac"] = o["top1"] / n if n else float("nan")
    return dict(n=n, trivial=trivial, rules=out)


def show(lines, label, res, store):
    store[label] = res
    lines.append(f"  {label}: {res['n']} problems ({res['trivial']} trivial excluded)")
    for k in RULES:
        o = res["rules"][k]
        lines.append(f"      {k:>9}: top-1 {o['top1']:6.1f}/{res['n']} ({100 * o['top1_frac']:5.1f}%) | regret "
                     f"{o['regret']:.3f} | pairs ordered correctly {100 * o['pairs']:5.1f}% of {o['pair_n']}")


def run_set(lines, store, name, recs, inst_key, b, main_eps=0.1, note=""):
    lines.append(f"\n=== {name} {note}")
    arms = sorted({r["arm"] for r in recs})
    lines.append(f"  sources: {arms}")
    a = candidates([r for r in recs if r["eps"] == main_eps], inst_key, lambda r: r["arm"], b)
    show(lines, f"{name} (a) sources at eps = {main_eps}", evaluate(list(a.values())), store)
    eps_all = sorted({r["eps"] for r in recs})
    if len(eps_all) > 2:
        probs = []
        for e in eps_all:
            probs += list(candidates([r for r in recs if r["eps"] == e], inst_key, lambda r: r["arm"], b).values())
        show(lines, f"{name} (b) sources at each eps ({len(eps_all)} values)", evaluate(probs), store)
    c = candidates(recs, inst_key, lambda r: (r["arm"], r["eps"]), b)
    show(lines, f"{name} (c) all (source, eps) configurations jointly", evaluate(list(c.values())), store)


def main():
    import s8_analysis
    import s8_eps_main
    import s9_l4_arms
    import s11_maxcut as X

    b = np.array(json.loads((ROOT / "results" / "s8_mass_phase1.json").read_text())["1"]["ov + m_e05"]["coef"])
    lines = [f"Ranking warm-start sources by a depth-0 rule. Truth: P(best of 1024 = optimal), p = 1, mean over seeds. "
             f"bandshare = {b[1]:.2f} log10 overlap {b[2]:+.2f} log10 m_e05 (UC L2 fit, fixed). TOL {TOL}."]
    store = {}

    # ---- UC, 16 qubits ----------------------------------------------------------
    f8 = {tuple(json.loads(k)): v for k, v in
          json.loads((ROOT / "results" / "s8_mass_features.json").read_text()).items()}
    r8 = [r for r in s8_analysis.load() if r["p"] == 1]
    for r in r8:
        r["m_e05"] = f8[(r["arm"], r["eps"], r["scenario"], r["seed"] if r["arm"] == "E" else -1)]["m_e05"] \
            if (r["arm"], r["eps"], r["scenario"], r["seed"] if r["arm"] == "E" else -1) in f8 else None
    ph2 = s8_eps_main.tasks(2)
    main8 = [r for r in r8 if (r["arm"], r["eps"], r["scenario"], r["seed"], r["p"]) in ph2 and r["arm"] != "B"]
    assert all(r["m_e05"] is not None for r in main8)
    lines.append("\n=== UC, 16 qubits (L2), 30 scenarios. Arm B is left out: it has C-milp's initial state and differs "
                 "only in the mixer, so no depth-0 rule can separate them. IN-SAMPLE for bandshare's two exponents.")
    a = candidates(main8, lambda r: r["scenario"], lambda r: r["arm"], b)
    lines.append(f"  sources: {sorted({r['arm'] for r in main8})} (main table, eps = 0.1, 5 seeds)")
    show(lines, "UC-16 (a) sources at eps = 0.1", evaluate(list(a.values())), store)
    ph1 = s8_eps_main.tasks(1)
    sw8 = [r for r in r8 if (r["arm"], r["eps"], r["scenario"], r["seed"], r["p"]) in ph1 and r["arm"] != "B"]
    assert all(r["m_e05"] is not None for r in sw8)
    probs = []
    for e in sorted({r["eps"] for r in sw8}):
        probs += list(candidates([r for r in sw8 if r["eps"] == e and r["arm"] != "A"], lambda r: r["scenario"],
                                 lambda r: r["arm"], b).values())
    show(lines, "UC-16 (b) sweep sources (C-milp, C-oracle, D, F) at each eps (7 values)", evaluate(probs), store)
    c = candidates(sw8, lambda r: r["scenario"], lambda r: (r["arm"], r["eps"]), b)
    show(lines, "UC-16 (c) all (source, eps) configurations jointly (incl. A)", evaluate(list(c.values())), store)

    # ---- UC, 22 qubits ----------------------------------------------------------
    f9 = {tuple(json.loads(k)): v for k, v in
          json.loads((ROOT / "results" / "s9_mass_features.json").read_text()).items()}
    r9 = s9_l4_arms.load()
    for r in r9:
        r["m_e05"] = f9[(r["arm"], r["eps"], r["scenario"])]["m_e05"]
    run_set(lines, store, "UC-22", r9, lambda r: r["scenario"], b,
            note="(L4, 20 structurally distinct scenarios; OUT-OF-SAMPLE for bandshare)")

    # ---- MaxCut -------------------------------------------------------------------
    mc = X.load()
    for n in X.SIZES:
        rs = [r for r in mc if r["n"] == n]
        if not rs:
            continue
        run_set(lines, store, f"MaxCut-{n} all non-oracle sources", [r for r in rs if r["arm"] != "BAD0"],
                lambda r: r["inst"], b, note="(OUT-OF-SAMPLE; BAD0, the optimum itself, is left out)")
        run_set(lines, store, f"MaxCut-{n} realistic sources", [r for r in rs if r["arm"] in ("A", "GW", "GW-LS", "E")],
                lambda r: r["inst"], b, note="(A, GW, GW-LS, E only)")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUT.with_suffix(".json").write_text(json.dumps(store, indent=1, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
