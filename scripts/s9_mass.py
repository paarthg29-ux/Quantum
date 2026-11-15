"""Does the band-share mechanism hold at 22 qubits, and does the L2 curve
transfer to L4 without refitting?

For every Phase-3 run (L4, p = 1) the initial-state features of psi_0 are
computed exactly from the 2^22 distribution (scripts/s8_mass.features), and:

  1. refit at L4: P(best of 1024 = ground) on log10 overlap alone vs log10
     overlap + log10 m_e05, with the per-arm intercept test;
  2. transfer: the Phase-1 (L2, p = 1) fit of overlap + m_e05 applied to the
     L4 runs unchanged -- an out-of-size prediction (R^2 and mean residual by
     arm). The L2 overlap-only fit is transferred too, for comparison.

    python scripts/s9_mass.py   # -> results/s9_mass.txt
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "s9_mass.txt"
CACHE = ROOT / "results" / "s9_mass_features.json"


def main():
    import s8_mass as M
    from s9_l4_arms import COMBOS, get_problem, load, selected

    from qguc.arms import ARMS
    from qguc.fastsim import probabilities, uniform_state
    from qguc.warmstart import regularise, warm_initial_state

    feats = {tuple(json.loads(k)): v for k, v in json.loads(CACHE.read_text()).items()} if CACHE.exists() else {}
    for s in (x["scenario"] for x in selected()):
        need = [(a, e) for a, e in COMBOS if (a, e, s) not in feats]
        if not need:
            continue
        pb = get_problem(s)
        for a, e in need:
            arm = ARMS[a]
            psi = uniform_state(pb.n) if arm.c_source is None else warm_initial_state(
                regularise(arm.c_source(pb, 0, e).c, e))
            feats[(a, e, s)] = M.features(probabilities(psi), pb.table, pb.ev)
        CACHE.write_text(json.dumps({json.dumps(list(k)): v for k, v in feats.items()}))
        print(f"  features for scenario {s} done", flush=True)

    recs = load()
    for r in recs:
        f = feats[(r["arm"], r["eps"], r["scenario"])]
        assert abs(f["overlap0"] - r["overlap0"]) <= 1e-6 * max(r["overlap0"], 1e-30) + 1e-12, (r, f["overlap0"])
        r.update({k: v for k, v in f.items() if k != "overlap0"})
    rs = [r for r in recs if r["overlap0"] > 0]
    y = np.array([r["p_ground_K1024"] for r in rs])
    arm = np.array([r["arm"] for r in rs])
    arms = [a for a in ("A", "C-milp", "D", "F") if a in set(arm)]
    tss = float(((y - y.mean()) ** 2).sum())
    ov = np.log10([r["overlap0"] for r in rs])
    me5 = np.log10([r["m_e05"] for r in rs])
    lines = [f"L4 (22 qubits), p = 1: {len(y)} runs ({len(recs) - len(rs)} with overlap 0 excluded); features of psi_0 "
             "checked against the stored overlap0 (complex64 runs vs complex128 features, rel. tol 1e-6)"]
    lines.append(f"  Spearman with P(best of 1024 = ground): log10 overlap {stats.spearmanr(ov, y).statistic:+.3f}, "
                 f"log10 m_e05 {stats.spearmanr(me5, y).statistic:+.3f}, "
                 f"log10(ov / m_e05) {stats.spearmanr(ov - me5, y).statistic:+.3f}")
    import io
    import contextlib

    out = dict(n_runs=int(len(y)), refit={}, transfer={})
    lines.append("\n1. REFIT AT L4 (logistic least squares; per-arm intercept test):")
    for lab, cols in (("log10 overlap", [ov]), ("log10 overlap + log10 m_e05", [ov, me5]),
                      ("band share log10(ov / m_e05)", [ov - me5])):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out["refit"][lab] = M.report(lab, cols, y, arm, arms, tss)
        lines.append(buf.getvalue().rstrip())

    lines.append("\n2. TRANSFER: the L2 (Phase 1, p = 1) fits applied to the L4 runs WITHOUT refitting:")
    l2 = json.loads((ROOT / "results" / "s8_mass_phase1.json").read_text())["1"]
    for key, cols in (("baseline: log10 overlap0", [ov]), ("ov + m_e05", [ov, me5])):
        b = np.array(l2[key]["coef"])
        X = np.column_stack([np.ones(len(y))] + cols)
        pred = M.sigmoid(X @ b)
        res = y - pred
        out["transfer"][key] = dict(r2=float(1 - (res @ res) / tss), coef=b.tolist(),
                                    resid={a: float(res[arm == a].mean()) for a in arms})
        lines.append(f"  L2 fit '{key}' (coef {np.round(b, 2).tolist()}): R^2 on L4 = {1 - (res @ res) / tss:.3f}; "
                     "mean observed - predicted by arm: "
                     + ", ".join(f"{a} {res[arm == a].mean():+.3f}" for a in arms))
    lines.append("\n  mean initial-state features by arm at L4 (log10):")
    for a in arms:
        m = arm == a
        lines.append(f"    {a:>7}: overlap {ov[m].mean():6.2f}  m_e05 {me5[m].mean():6.2f}  share {(ov - me5)[m].mean():6.2f}"
                     f"  | P(b1024) {y[m].mean():.3f}")
    OUT.write_text("\n".join(lines) + "\n")
    OUT.with_suffix(".json").write_text(json.dumps(out, indent=1, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
