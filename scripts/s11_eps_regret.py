"""The cost of choosing epsilon by a depth-0 rule (existing UC runs only).

For one warm-start source and one instance, the candidates are the values of
the regularisation eps. A rule picks the eps whose INITIAL state maximises
its score (scripts/s11_ranking.scores):

  overlap     log10 overlap0   -- the published heuristic (Bhattacharyya et
              al., arXiv:2504.06253, depth-0 P)
  ratio0      initial approximation gain (their depth-0 alpha)
  bandshare   the UC L2 band-share index, exponents fixed (in-sample at 16
              qubits, out-of-sample at 22)
  share       log10(overlap0 / m_e05), no fitted number
and, for reference, two rules that use no initial-state information:
  fixed 0.1   the default eps
  pooled      the single eps that is best on average for that source (the
              rule behind the paper's eps-regret table; in-sample)

Decision regret = P(best of 1024 = ground) at the instance's own best eps
(the oracle) minus P at the chosen eps; P is the mean over seeds. Predictor
ties are split evenly. Reported per source: mean over instances, mean over
structural groups (16 qubits), worst instance, instances with regret > 0.01,
and the eps the rule picks.

16 qubits: the phase-1 sweep, 7 eps values, 30 scenarios x 3 seeds.
22 qubits: each source was run at only TWO eps values (0.1 and its 16-qubit
best), so the choice there is binary -- reported, but thin.

    python scripts/s11_eps_regret.py   # -> results/s11_eps_regret.txt, .json
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "s11_eps_regret.txt"
Y = "p_ground_K1024"
RULES = ("overlap", "ratio0", "bandshare", "share")


def table(recs, b, inst_key, arms, eps_set, groups=None):
    """-> {arm: {rule: stats}}"""
    from s11_ranking import scores

    out = {}
    for arm in arms:
        acc = {}
        for r in recs:
            if r["arm"] == arm and r["eps"] in eps_set:
                acc.setdefault(inst_key(r), {}).setdefault(r["eps"], []).append((r[Y], scores(r, b)))
        eps = sorted(eps_set)
        insts = sorted(i for i, d in acc.items() if len(d) == len(eps))
        P = np.array([[np.mean([p for p, _ in acc[i][e]]) for e in eps] for i in insts])  # (instances, eps)
        best = P.max(1)
        res = {}
        choices = {k: np.array([[np.mean([s[k] for _, s in acc[i][e]]) for e in eps] for i in insts]) for k in RULES}
        picks = {}
        for k, S in choices.items():
            top = np.isclose(S, S.max(1, keepdims=True), rtol=0, atol=1e-9)
            picks[k] = top / top.sum(1, keepdims=True)  # tie-split weights
        if 0.1 in eps:
            w = np.zeros_like(P)
            w[:, eps.index(0.1)] = 1
            picks["fixed 0.1"] = w
        w = np.zeros_like(P)
        w[:, int(np.argmax(P.mean(0)))] = 1
        picks["pooled"] = w
        for k, w in picks.items():
            reg = best - (w * P).sum(1)
            g = None
            if groups is not None:
                gid = np.array([groups[i] for i in insts])
                g = float(np.mean([reg[gid == x].mean() for x in np.unique(gid)]))
            res[k] = dict(mean=float(reg.mean()), group_mean=g, worst=float(reg.max()), n_bad=int((reg > 0.01).sum()),
                          n=len(insts), p_chosen=float((w * P).sum(1).mean()),
                          picked={str(e): float(w[:, j].mean()) for j, e in enumerate(eps)})
        res["_oracle_p"] = float(best.mean())
        out[arm] = res
    return out


def show(lines, title, tab, eps):
    lines.append(f"\n{title}")
    lines.append(f"  {'source':>8} {'rule':>10} | regret mean | group mean |  worst | >0.01 | mean P chosen (oracle) | "
                 "share of instances picking eps = " + " ".join(f"{e:g}" for e in sorted(eps)))
    for arm, res in tab.items():
        for k in RULES + ("fixed 0.1", "pooled"):
            if k not in res:
                continue
            o = res[k]
            gm = f"{o['group_mean']:10.3f}" if o["group_mean"] is not None else "         -"
            lines.append(f"  {arm:>8} {k:>10} | {o['mean']:11.3f} | {gm} | {o['worst']:6.3f} | {o['n_bad']:2d}/{o['n']:<2d} | "
                         f"{o['p_chosen']:.3f} ({res['_oracle_p']:.3f})          | "
                         + " ".join(f"{o['picked'][str(e)]:.2f}" for e in sorted(eps)))


def main():
    import s8_analysis
    import s8_eps_main
    import s9_l4_arms

    b = np.array(json.loads((ROOT / "results" / "s8_mass_phase1.json").read_text())["1"]["ov + m_e05"]["coef"])
    lines = ["Decision regret of choosing eps by a depth-0 rule: P(best of 1024 = ground) at the instance's own best "
             "eps minus P at the chosen eps (p = 1, mean over seeds). "
             f"bandshare = {b[1]:.2f} log10 overlap {b[2]:+.2f} log10 m_e05 (UC L2 fit, fixed)."]
    store = {}
    f8 = {tuple(json.loads(k)): v for k, v in
          json.loads((ROOT / "results" / "s8_mass_features.json").read_text()).items()}
    ph1 = s8_eps_main.tasks(1)
    r8 = [r for r in s8_analysis.load() if (r["arm"], r["eps"], r["scenario"], r["seed"], r["p"]) in ph1
          and r["arm"] != "A"]
    for r in r8:
        r["m_e05"] = f8[(r["arm"], r["eps"], r["scenario"], -1)]["m_e05"]
    groups = s8_analysis.structural_groups()["GL"]
    arms = ("C-milp", "C-oracle", "D", "F", "B")
    for label, eps in (("full sweep, eps in [0, 0.5]", s8_eps_main.EPS),
                       ("eps in [0.05, 0.35] (as the paper's eps-regret table: without eps = 0, which freezes wrong "
                        "pins, and eps = 0.5, the cold limit)", s8_eps_main.EPS[1:-1])):
        tab = table(r8, b, lambda r: r["scenario"], arms, set(eps), groups)
        show(lines, f"=== UC, 16 qubits (L2), 30 scenarios in 17 structural groups; {label}. bandshare is IN-SAMPLE "
                    "here (two fitted exponents). B has C-milp's initial state but the X mixer.", tab, eps)
        store[f"UC-16 {label[:10]}"] = tab

    f9 = {tuple(json.loads(k)): v for k, v in
          json.loads((ROOT / "results" / "s9_mass_features.json").read_text()).items()}
    r9 = s9_l4_arms.load()
    for r in r9:
        r["m_e05"] = f9[(r["arm"], r["eps"], r["scenario"])]["m_e05"]
    for arm in ("C-milp", "D", "F"):
        eps = sorted({r["eps"] for r in r9 if r["arm"] == arm})
        tab = table(r9, b, lambda r: r["scenario"], (arm,), set(eps))
        show(lines, f"=== UC, 22 qubits (L4), 20 structurally distinct scenarios; {arm}: only eps in {eps} were run, "
                    "so this is a binary choice. bandshare is OUT-OF-SAMPLE.", tab, eps)
        store[f"UC-22 {arm}"] = tab
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUT.with_suffix(".json").write_text(json.dumps(store, indent=1, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
