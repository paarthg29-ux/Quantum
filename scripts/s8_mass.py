"""Why do the learned arms beat their own ground-state overlap?

Hypothesis: a warm start that spreads its initial probability over SEVERAL
low-energy states samples the ground state better after QAOA than one with
the same ground-state overlap concentrated elsewhere. For every initial
state psi_0 (it depends on (arm, eps, scenario) only; arm E also on the seed)
we compute, exactly from the 2^n distribution:

    m_top{k}   mass on the k lowest-energy states, k in {1, 5, 10, 50, 200}
               (sorted by QUBO energy, ties by state number; m_top1 is ONE
               ground state, overlap0 is all degenerate ground states)
    m_e01/m_e05  mass on states with E - E_min <= 1% / 5% of (E_max - E_min)
               (normalised spectrum: the QUBO energy has an arbitrary offset,
               so "1% of E_min" would not be meaningful)
    m_adeq     mass on adequate schedules (sum_i P_i x_it >= D_t)
    entropy    Shannon entropy of |psi_0|^2, bits (n = uniform)

Then, pooled over all arms and eps (the same runs as the overlap fit:
overlap0 > 0), logistic least-squares fits of P(best of 1024 = ground) on
  (a) log10 overlap0 alone (the baseline)
  (b) each feature alone
  (c) log10 overlap0 + each feature
and for every model: does adding a per-arm intercept still help (F-test,
Delta R^2), and the mean residual by arm. If a feature closes the per-arm
gap, the arm intercepts stop mattering.

    python scripts/s8_mass.py --phase 1      # p = 1 eps sweep (+ A)
    python scripts/s8_mass.py --phase 2      # main table, eps = 0.1, p = 1, 2, 3
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

KS_TOP = (1, 5, 10, 50, 200)
FEATURES = tuple(f"m_top{k}" for k in KS_TOP) + ("m_e01", "m_e05", "m_adeq", "entropy")
CACHE = ROOT / "results" / "s8_mass_features.json"


def features(probs, table, ev) -> dict:
    ps = probs[table.order]
    Es = ev.E[table.order]
    rng = ev.e_max - ev.e_min
    out = {f"m_top{k}": float(ps[:k].sum()) for k in KS_TOP}
    out["m_e01"] = float(ps[Es - ev.e_min <= 0.01 * rng].sum())
    out["m_e05"] = float(ps[Es - ev.e_min <= 0.05 * rng].sum())
    out["m_adeq"] = float(ps[table.adequate].sum())
    nz = ps[ps > 0]
    out["entropy"] = float(-(nz * np.log2(nz)).sum())
    out["overlap0"] = float(ps[: table.n_ground].sum())
    return out


def compute(keys):
    """keys: set of (arm, eps, scenario, seed); seed matters only for arm E."""
    from s8_eps_main import get_problem

    from qguc.arms import ARMS
    from qguc.fastsim import probabilities, uniform_state
    from qguc.warmstart import regularise, warm_initial_state

    cache = {}
    if CACHE.exists():
        cache = {tuple(json.loads(k)): v for k, v in json.loads(CACHE.read_text()).items()}
    need = sorted({(a, e, s, sd if a == "E" else -1) for a, e, s, sd in keys} - set(cache))
    for i, (a, e, s, sd) in enumerate(need):
        pb = get_problem(s)
        arm = ARMS[a]
        if arm.c_source is None or not arm.warm_state:  # arm A: |+>^n
            psi = uniform_state(pb.n)
        else:  # B shares C-milp's warm state (it differs only in the mixer)
            psi = warm_initial_state(regularise(arm.c_source(pb, max(sd, 0), e).c, e))
        cache[(a, e, s, sd)] = features(probabilities(psi), pb.table, pb.ev)
        if (i + 1) % 100 == 0:
            print(f"  features {i + 1}/{len(need)}", flush=True)
    CACHE.write_text(json.dumps({json.dumps(list(k)): v for k, v in cache.items()}))
    return cache


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit(X, y):
    """Least-squares logistic P = sigmoid(X @ b); X includes the constant column."""
    b0 = np.zeros(X.shape[1])
    b0[0] = 5.0
    r = least_squares(lambda b: sigmoid(X @ b) - y, b0, method="lm", max_nfev=20000)
    res = sigmoid(X @ r.x) - y
    return r.x, float(res @ res), -res  # residual = observed - fitted


def report(name, cols, y, arm, arms, tss):
    n = len(y)
    X = np.column_stack([np.ones(n)] + cols)
    b, rss, res = fit(X, y)
    D = np.column_stack([(arm == a).astype(float) for a in arms[1:]])  # per-arm intercepts (first arm = reference)
    _, rss_a, _ = fit(np.column_stack([X, D]), y)
    k = D.shape[1]
    F = ((rss - rss_a) / k) / (rss_a / (n - X.shape[1] - k))
    pF = stats.f.sf(F, k, n - X.shape[1] - k)
    r2, r2a = 1 - rss / tss, 1 - rss_a / tss
    resid = {a: res[arm == a].mean() for a in arms}
    spread = max(resid.values()) - min(resid.values())
    print(f"  {name:<26} R^2 {r2:6.3f} | + arm intercepts R^2 {r2a:6.3f} (dR^2 {r2a - r2:+.3f}, F {F:6.1f}, "
          f"p {pF:.1e}) | resid range {spread:.3f} | " + " ".join(f"{a} {resid[a]:+.3f}" for a in arms))
    return dict(r2=r2, r2_arm=r2a, dr2=r2a - r2, F=F, p=pF, resid=resid, spread=spread, coef=b.tolist())


def main():
    try:
        phase = int(sys.argv[sys.argv.index("--phase") + 1])
    except (ValueError, IndexError):
        raise SystemExit("usage: s8_mass.py --phase 1|2")
    from s8_analysis import load
    from s8_eps_main import tasks

    keys = tasks(phase)
    recs = [r for r in load() if (r["arm"], r["eps"], r["scenario"], r["seed"], r["p"]) in keys]
    feats = compute({(r["arm"], r["eps"], r["scenario"], r["seed"]) for r in recs})
    for r in recs:
        f = feats[(r["arm"], r["eps"], r["scenario"], r["seed"] if r["arm"] == "E" else -1)]
        assert abs(f["overlap0"] - r["overlap0"]) <= 1e-9 * max(r["overlap0"], 1e-300) + 1e-15, (r, f["overlap0"])
        r.update({k: v for k, v in f.items() if k != "overlap0"})
    print(f"phase {phase}: {len(recs)} runs; features recomputed from psi_0 and checked against the stored overlap0")
    out = {}
    for p in sorted({r["p"] for r in recs}):
        rs = [r for r in recs if r["p"] == p and r["overlap0"] > 0]
        y = np.array([r["p_ground_K1024"] for r in rs])
        arm = np.array([r["arm"] for r in rs])
        arms = [a for a in ("A", "B", "C-milp", "C-oracle", "D", "E", "F") if a in set(arm)]
        tss = float(((y - y.mean()) ** 2).sum())
        lg = lambda k: np.log10(np.maximum([r[k] for r in rs], 1e-300))  # noqa: E731
        ov = np.log10([r["overlap0"] for r in rs])
        print(f"\n=== p = {p}: {len(y)} runs with overlap0 > 0, arms {arms} (residual = observed - fitted, "
              "mean per arm) ===")
        print("  Spearman rho with P(best of 1024 = ground): "
              + ", ".join(f"{k} {stats.spearmanr(lg(k) if k != 'entropy' else [r[k] for r in rs], y).statistic:+.3f}"
                          for k in FEATURES) + f", log10 overlap0 {stats.spearmanr(ov, y).statistic:+.3f}")
        res = {"baseline: log10 overlap0": report("log10 overlap0 (baseline)", [ov], y, arm, arms, tss)}
        for k in FEATURES:
            x = np.array([r[k] for r in rs]) if k == "entropy" else lg(k)
            lab = k if k == "entropy" else f"log10 {k}"
            res[lab] = report(lab, [x], y, arm, arms, tss)
        for k in FEATURES:
            x = np.array([r[k] for r in rs]) if k == "entropy" else lg(k)
            lab = f"ov + {k}"
            res[lab] = report(lab, [ov, x], y, arm, arms, tss)
        res["ov + all"] = report("ov + all features", [ov] + [np.array([r[k] for r in rs]) if k == "entropy"
                                                               else lg(k) for k in FEATURES], y, arm, arms, tss)
        out[p] = res
        # held-out checks: (1) fit on the non-learned runs (A, D, F, E and every arm at eps = 0.5), predict the
        # learned arms (B, C-milp, C-oracle at eps < 0.5); (2) leave one structural group out at a time.
        from s8_analysis import structural_groups

        eps = np.array([r["eps"] for r in rs])
        learned = np.isin(arm, ["B", "C-milp", "C-oracle"]) & (eps < 0.5)
        gid = np.array([structural_groups()["GL"][r["scenario"]] for r in rs])
        me5 = lg("m_e05")
        print("  HELD-OUT CHECKS (log10 overlap alone vs log10 overlap + log10 m_e05):")
        for lab, cols in (("overlap only", [ov]), ("overlap + m_e05", [ov, me5])):
            X = np.column_stack([np.ones(len(y))] + cols)
            b, _, _ = fit(X[~learned], y[~learned])
            pred = sigmoid(X @ b)
            bias = {a: float(np.mean(y[learned & (arm == a)] - pred[learned & (arm == a)]))
                    for a in ("B", "C-milp", "C-oracle") if (learned & (arm == a)).any()}
            cv = np.empty_like(y)
            for g in np.unique(gid):
                m = gid == g
                bg, _, _ = fit(X[~m], y[~m])
                cv[m] = sigmoid(X[m] @ bg)
            r2cv = 1 - ((y - cv) ** 2).sum() / tss
            print(f"    {lab:>16}: fit on non-learned runs, learned arms observed - predicted: "
                  + ", ".join(f"{a} {v:+.3f}" for a, v in bias.items())
                  + f" | leave-one-group-out CV R^2 {r2cv:.3f} ({len(np.unique(gid))} groups)")
            res[f"heldout {lab}"] = dict(learned_bias=bias, logo_cv_r2=r2cv)
        # where the learned arms put their mass, vs D, at matched overlap
        print("  mean initial-state features by arm (log10 masses; entropy in bits):")
        for a in arms:
            m = arm == a
            print(f"    {a:>8}: log10 ov {ov[m].mean():6.2f} | " + " ".join(
                f"{k} {(lg(k)[m].mean() if k != 'entropy' else np.mean([r[k] for r, mm in zip(rs, m) if mm])):6.2f}"
                for k in FEATURES))
    (ROOT / "results" / f"s8_mass_phase{phase}.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
