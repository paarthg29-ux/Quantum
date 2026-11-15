"""G3 (scoped) and G11: robustness of the arm comparison to (a) finite-shot
expectation estimates during optimisation and (b) a second optimiser.

L2 (16 qubits), the 30 held-out Phase-2 scenarios (seed 9100), seeds 0-2,
p = 1, eps = 0.1, arms A, C-milp, D, F:

  (a) shots in {128, 1024, 8192}: the optimiser (COBYLA, unchanged stopping
      rule) sees the mean energy of that many samples from the exact output
      distribution; the reported parameters are the best MEASURED ones; every
      metric is EXACT at those parameters (qguc.qaoa.run_qaoa(shots=...)).
  (b) Nelder-Mead with exact expectations, same evaluation cap and patience.

Reference: the exact-COBYLA Phase-2 runs of the same (arm, scenario, seed)
in results/s8_runs.jsonl, so every comparison is paired.

Full noise modelling (density matrices, 4^n) is out of scope.

    python scripts/s10_robust.py          # -> results/s10_runs.jsonl, results/s10_robust.txt
    python scripts/s10_robust.py --report
"""

import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
ARMS = ("A", "C-milp", "D", "F")
SHOTS = (128, 1024, 8192)
SEEDS, EPS, P = (0, 1, 2), 0.1, 1
WORKERS = 4
STORE = ROOT / "results" / "s10_runs.jsonl"
REPORT = ROOT / "results" / "s10_robust.txt"
THREAD_ENV = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")


def variants():
    return [("COBYLA", s) for s in SHOTS] + [("Nelder-Mead", 0)]


def tasks():
    from s8_eps_main import N_SCEN

    return {(a, s, sd, m, sh) for a in ARMS for s in range(N_SCEN) for sd in SEEDS for m, sh in variants()}


def job(args):
    scenario, seed, combos = args
    os.environ.update(THREAD_ENV)
    from s8_eps_main import get_problem

    from qguc.arms import run_arm

    pb = get_problem(scenario)
    out = []
    for arm, method, shots in combos:
        r = run_arm(arm, pb, P, seed, eps=EPS, shots=shots or None, method=method)
        g_exact = r.extra.get("exact_g_chosen", r.g_final)
        rec = dict(arm=arm, scenario=scenario, seed=seed, p=P, eps=EPS, method=method, shots=shots,
                   n_evals=r.n_evals, cap_hit=r.cap_hit, early=r.early_stopped, g_exact=g_exact,
                   g_init_exact=r.extra.get("exact_g_init", r.g_init), g_measured=r.g_final,
                   g_traj_max_exact=r.extra.get("exact_g_trajectory_max", r.g_final), wall_s=r.wall_s,
                   **{f"{k}_K{K}": v for K, m in r.sampling.items() for k, v in m.items()})
        out.append(rec)
    return out


def load():
    if not STORE.exists():
        return []
    return [json.loads(l) for l in STORE.read_text().splitlines() if l.strip()]


def run():
    done = {(r["arm"], r["scenario"], r["seed"], r["method"], r["shots"]) for r in load()}
    left = tasks() - done
    groups = {}
    for a, s, sd, m, sh in sorted(left):
        groups.setdefault((s, sd), []).append((a, m, sh))
    jobs = [(s, sd, c) for (s, sd), c in sorted(groups.items())]
    print(f"{len(tasks())} runs ({len(done)} done, {len(left)} left) in {len(jobs)} jobs; {WORKERS} workers", flush=True)
    if not jobs:
        return
    t0 = time.perf_counter()
    with ProcessPoolExecutor(WORKERS) as pool, STORE.open("a") as f:
        futs = {pool.submit(job, j): j for j in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                recs = fut.result()
            except Exception as exc:
                print(f"  FAILED job {futs[fut][:2]}: {exc!r}", flush=True)
                continue
            for rec in recs:
                f.write(json.dumps(rec) + "\n")
            f.flush()
            if k % 15 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} jobs, {time.perf_counter() - t0:.0f} s", flush=True)


def report():
    from scipy import stats

    sys.path.insert(0, str(ROOT / "scripts"))
    import s8_analysis as A8

    R = load()
    gm = A8.structural_groups()["GL"]
    ref = {}
    for r in A8.load():
        if r["p"] == P and r["eps"] == EPS and r["seed"] in SEEDS and r["arm"] in ARMS:
            ref[(r["arm"], r["scenario"], r["seed"])] = r

    def scen(rows, key):
        out = {}
        for r in rows:
            out.setdefault(r["scenario"], []).append(r[key])
        return {s: float(np.mean(v)) for s, v in out.items()}

    def variant_rows(arm, method, shots):
        if method == "exact":
            return [r for (a, s, sd), r in ref.items() if a == arm]
        return [r for r in R if r["arm"] == arm and r["method"] == method and r["shots"] == shots]

    def paired(x, y):
        common = sorted(set(x) & set(y))
        d = np.array([x[s] - y[s] for s in common])
        h = stats.t.ppf(0.975, d.size - 1) * d.std(ddof=1) / np.sqrt(d.size) if d.std() > 0 else 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pw = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        gx, gy = A8.to_groups(x, gm), A8.to_groups(y, gm)
        dg = np.array([gx[g] - gy[g] for g in sorted(gx)])
        hg = stats.t.ppf(0.975, dg.size - 1) * dg.std(ddof=1) / np.sqrt(dg.size) if dg.std() > 0 else 0.0
        return (f"{d.mean():+.3f} [{d.mean() - h:+.3f}, {d.mean() + h:+.3f}] (Wilcoxon p={pw:.1e}, "
                f"+/=/- {(d > 1e-12).sum()}/{(np.abs(d) <= 1e-12).sum()}/{(d < -1e-12).sum()}) | 17 groups "
                f"{dg.mean():+.3f} [{dg.mean() - hg:+.3f}, {dg.mean() + hg:+.3f}]")

    L = []
    W = lambda s="": L.append(s)  # noqa: E731
    W("G3 (scoped) + G11: L2, 30 held-out scenarios x 3 seeds, p = 1, eps = 0.1. Metrics are EXACT at the chosen "
      "parameters; scenario means, then paired over scenarios (and over the 17 structural groups).")
    rows = [("exact", 0, "exact <H>, COBYLA (Phase 2)")] + [("COBYLA", s, f"{s} shots, COBYLA") for s in
                                                            sorted(SHOTS, reverse=True)] + \
           [("Nelder-Mead", 0, "exact <H>, Nelder-Mead")]
    W("")
    W(f"  {'optimiser view':<28} {'arm':>7} | {'P(b1024=gnd)':>12} {'P 1-shot':>8} {'g (exact)':>9} {'evals':>6} "
      f"{'cap':>5} {'gap K1024 %':>11}")
    for method, shots, lab in rows:
        for a in ARMS:
            rs = variant_rows(a, method, shots)
            if not rs:
                continue
            gkey = "g_final" if method == "exact" else "g_exact"
            W(f"  {lab:<28} {a:>7} | {np.mean([r['p_ground_K1024'] for r in rs]):12.3f} "
              f"{np.mean([r['p_ground_K1'] for r in rs]):8.4f} {np.mean([r[gkey] for r in rs]):9.3f} "
              f"{np.mean([r['n_evals'] for r in rs]):6.0f} {np.mean([r['cap_hit'] for r in rs]):5.0%} "
              f"{100 * np.mean([r['gap_repaired_K1024'] for r in rs]):11.2f}")
        W("")
    W("PAIRED: the learned advantage under each optimiser view, P(best of 1024 = ground), C-milp - D and C-milp - F")
    for method, shots, lab in rows:
        cm = scen(variant_rows("C-milp", method, shots), "p_ground_K1024")
        if not cm:
            continue
        for b in ("D", "F", "A"):
            W(f"  {lab:<28} C-milp - {b}: {paired(cm, scen(variant_rows(b, method, shots), 'p_ground_K1024'))}")
    W("")
    W("DOES THE ADVANTAGE GROW AS SHOTS FALL? (change of C-milp - D relative to the exact-COBYLA run, per scenario)")
    base = {s: v for s, v in scen(variant_rows("C-milp", "exact", 0), "p_ground_K1024").items()}
    based = scen(variant_rows("D", "exact", 0), "p_ground_K1024")
    adv0 = {s: base[s] - based[s] for s in base}
    for shots in sorted(SHOTS, reverse=True):
        c = scen(variant_rows("C-milp", "COBYLA", shots), "p_ground_K1024")
        d = scen(variant_rows("D", "COBYLA", shots), "p_ground_K1024")
        adv = {s: c[s] - d[s] for s in c}
        W(f"  {shots:>5} shots: advantage minus exact-run advantage {paired(adv, adv0)}")
    W("")
    W("HOW MUCH DOES EACH ARM LOSE TO SHOT NOISE? (P(b1024) at 128 shots minus exact, per scenario)")
    for a in ARMS:
        W(f"  {a:>7}: {paired(scen(variant_rows(a, 'COBYLA', 128), 'p_ground_K1024'), scen(variant_rows(a, 'exact', 0), 'p_ground_K1024'))}")
    REPORT.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    if "--report" not in sys.argv:
        run()
    report()
