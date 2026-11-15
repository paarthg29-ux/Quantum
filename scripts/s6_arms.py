"""Pilot of the warm-start arms A, D, E, F at L2 (n = 16).

5 scenarios x 5 seeds x p in {1, 2, 3}, frozen encoding, identical stopping
rule, paired on (scenario, seed): every (scenario, seed, p) job runs all
arms in one worker, from identical initial parameters. Then a side sweep of
the regulariser eps in {0, 0.05, 0.1, 0.25} for the warm arms.

Scenarios: a fresh set (seed 7001), not used in any calibration.
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

LEVEL, N_HOURS = 2, 2
SCEN_SEED, N_SCEN, SEEDS = 7001, 5, (0, 1, 2, 3, 4)
P_VALUES = (1, 2, 3)
ARM_NAMES = ("A", "D", "D-dem", "E", "F")
TAG = "s6b"  # v1 (cap 100, arm D at true demand) is results/s6_arms.*
G_TARGETS = (0.9, 0.95, 0.99)
EPS_SWEEP = (0.0, 0.05, 0.1, 0.25)
EPS_ARMS, EPS_P, EPS_SEEDS = ("D", "D-dem", "E", "F"), (1, 2), (0, 1)
WORKERS = 4
RESULTS = Path(__file__).resolve().parents[1] / "results"
THREAD_ENV = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

_PROBLEMS = {}


def get_problem(scenario):
    from qguc import make_scenario_set
    from qguc.qaoa import make_problem

    if scenario not in _PROBLEMS:
        inst = make_scenario_set(N_SCEN, n_hours=N_HOURS, seed=SCEN_SEED, refinement_level=LEVEL)[scenario]
        pb = make_problem(inst)
        pb.table  # build the per-state table once per worker
        _PROBLEMS[scenario] = pb
    return _PROBLEMS[scenario]


def record(r, scenario, eps):
    rec = dict(arm=r.arm, scenario=scenario, seed=r.seed, p=r.p, eps=eps, n_evals=r.n_evals,
               cap_hit=r.cap_hit, early=r.early_stopped, g_state0=r.g_state0, g_init=r.g_init,
               g_final=r.g_final, ratio=r.best_ratio,
               **{f"to_{t}": r.evals_to_g(t) for t in G_TARGETS},
               wall_s=r.wall_s, source_s=r.source_s, total_s=r.total_wall_s,
               lp_solve_s=r.extra.get("lp_solve_s"), n_fractional=r.extra.get("n_fractional"),
               gains=np.round(r.gains, 5).tolist())
    for K, m in r.sampling.items():
        rec.update({f"{k}_K{K}": v for k, v in m.items()})
    return rec


def job(args):
    scenario, seed, p, arms, eps = args
    os.environ.update(THREAD_ENV)
    from qguc.arms import run_arm

    pb = get_problem(scenario)
    return [record(run_arm(a, pb, p, seed, eps=eps), scenario, eps) for a in arms]


def ms(v):
    v = np.asarray([x for x in v if x is not None], float)
    return (f"{v.mean():.3f} +/- {v.std(ddof=1):.3f}" if v.size > 1 else f"{v.mean():.3f}") if v.size else "-"


def paired(recs, a, b, p, key, sign=1.0):
    """Paired differences key[a] - key[b] over (scenario, seed) where both exist."""
    idx = {(r["scenario"], r["seed"]): r for r in recs if r["arm"] == b and r["p"] == p}
    d = [r[key] - idx[(r["scenario"], r["seed"])][key] for r in recs if r["arm"] == a and r["p"] == p
         and r[key] is not None and idx[(r["scenario"], r["seed"])][key] is not None]
    d = np.array(d, float)
    if d.size == 0:
        return "no pairs", None
    pval = wilcoxon(d).pvalue if d.size > 5 and np.any(d != 0) else float("nan")
    return f"{d.mean():+8.3f} +/- {d.std(ddof=1):6.3f} (n={d.size:2d}, Wilcoxon p={pval:.1e})", d


def summarise(recs):
    for p in P_VALUES:
        print(f"\n--- p = {p}: mean +/- std over {N_SCEN * len(SEEDS)} paired runs ---")
        print(f"  {'arm':>5} | {'g state0':>15} | {'g init':>15} | {'g final':>15} | {'evals':>17} | "
              f"{'cap hit':>7} | " + " | ".join(f"{'evals to g=' + str(t):>18}" for t in G_TARGETS))
        for a in ARM_NAMES:
            rs = [r for r in recs if r["arm"] == a and r["p"] == p]
            to = []
            for t in G_TARGETS:
                v = [r[f"to_{t}"] for r in rs if r[f"to_{t}"] is not None]
                to.append(f"{np.median(v):6.0f} med ({len(v):2d}/{len(rs)})" if v else f"{'never':>9} (0/{len(rs)})")
            print(f"  {a:>5} | {ms([r['g_state0'] for r in rs]):>15} | {ms([r['g_init'] for r in rs]):>15} | "
                  f"{ms([r['g_final'] for r in rs]):>15} | {ms([r['n_evals'] for r in rs]):>17} | "
                  f"{np.mean([r['cap_hit'] for r in rs]):6.0%} | " + " | ".join(f"{x:>18}" for x in to))
        print(f"  {'arm':>5} | {'P(ground)':>15} | {'P(best1024=gnd)':>15} | {'1-shot adeq':>11} | "
              f"{'gap raw %':>15} | {'gap repaired %':>15} | {'wall s/run':>15} | {'LP solve ms':>11}")
        for a in ARM_NAMES:
            rs = [r for r in recs if r["arm"] == a and r["p"] == p]
            lp = [1e3 * r["lp_solve_s"] for r in rs if r["lp_solve_s"] is not None]
            print(f"  {a:>5} | {ms([r['p_ground_K1'] for r in rs]):>15} | {ms([r['p_ground_K1024'] for r in rs]):>15} | "
                  f"{np.mean([r['adequate_K1'] for r in rs]):11.3f} | "
                  f"{ms([100 * r['gap_raw_K1024'] for r in rs]):>15} | {ms([100 * r['gap_repaired_K1024'] for r in rs]):>15} | "
                  f"{ms([r['total_s'] for r in rs]):>15} | {(f'{np.mean(lp):.2f}' if lp else '-'):>11}")
        print("  paired differences vs A (arm - A):")
        for a in ARM_NAMES[1:]:
            for key, lab in (("g_init", "g init"), ("g_final", "g final"), ("n_evals", "evals"),
                             ("to_0.9", "evals to g=0.9"), ("p_ground_K1024", "P(best1024=gnd)"),
                             ("gap_repaired_K1024", "gap repaired")):
                print(f"    {a} {lab:>16}: {paired(recs, a, 'A', p, key)[0]}")
        print("  warm arms against each other:")
        for a, b in (("D", "E"), ("F", "E"), ("D", "F"), ("D", "D-dem")):
            for key, lab in (("g_init", "g init"), ("g_final", "g final"), ("n_evals", "evals"),
                             ("p_ground_K1024", "P(b1024=gnd)"), ("gap_repaired_K1024", "gap repaired")):
                print(f"    {a}-{b} {lab:>14}: {paired(recs, a, b, p, key)[0]}")


def main():
    import psutil

    from qguc.config import FROZEN
    from qguc.qaoa import OptimiserSettings

    print(f"L{LEVEL} (n = {2 * 8}), T = {N_HOURS}, scenarios {N_SCEN} (seed {SCEN_SEED}), seeds {list(SEEDS)}, "
          f"p {list(P_VALUES)}; frozen point {FROZEN}")
    print(f"optimiser {OptimiserSettings()}; eps = 0.1; {WORKERS} worker processes, 1 thread each; "
          f"CPU load at start {psutil.cpu_percent(interval=2.0):.0f}%")
    t0 = time.perf_counter()
    jobs = [(s, seed, p, ARM_NAMES, 0.1) for s in range(N_SCEN) for seed in SEEDS for p in P_VALUES]
    with ProcessPoolExecutor(WORKERS) as pool:
        recs = [r for out in pool.map(job, jobs) for r in out]
    print(f"{len(recs)} runs in {time.perf_counter() - t0:.0f} s")
    (RESULTS / f"{TAG}_arms.json").write_text(json.dumps(recs))
    summarise(recs)

    if "--no-eps" in sys.argv:
        return
    print(f"\n===== side sweep over eps (arms {list(EPS_ARMS)}, p {list(EPS_P)}, {N_SCEN} scenarios x "
          f"{len(EPS_SEEDS)} seeds) =====")
    jobs = [(s, seed, p, EPS_ARMS, eps) for eps in EPS_SWEEP for s in range(N_SCEN) for seed in EPS_SEEDS
            for p in EPS_P]
    with ProcessPoolExecutor(WORKERS) as pool:
        erecs = [r for out in pool.map(job, jobs) for r in out]
    (RESULTS / f"{TAG}_eps_sweep.json").write_text(json.dumps(erecs))
    a_ref = {(r["scenario"], r["seed"], r["p"]): r for r in recs if r["arm"] == "A"}
    print(f"  {'arm':>5} {'p':>2} {'eps':>5} | {'g state0':>15} | {'g init':>15} | {'g final':>15} | "
          f"{'evals':>17} | {'cap':>4} | {'P(best1024=gnd)':>15} | {'gap repaired %':>15}")
    for a in EPS_ARMS:
        for p in EPS_P:
            ar = [a_ref[(s, sd, p)] for s in range(N_SCEN) for sd in EPS_SEEDS]
            print(f"  {'A':>5} {p:2d} {'-':>5} | {ms([r['g_state0'] for r in ar]):>15} | "
                  f"{ms([r['g_init'] for r in ar]):>15} | {ms([r['g_final'] for r in ar]):>15} | "
                  f"{ms([r['n_evals'] for r in ar]):>17} | {np.mean([r['cap_hit'] for r in ar]):4.0%} | "
                  f"{ms([r['p_ground_K1024'] for r in ar]):>15} | {ms([100 * r['gap_repaired_K1024'] for r in ar]):>15}")
            for eps in EPS_SWEEP:
                rs = [r for r in erecs if r["arm"] == a and r["p"] == p and r["eps"] == eps]
                print(f"  {a:>5} {p:2d} {eps:5.2f} | {ms([r['g_state0'] for r in rs]):>15} | "
                      f"{ms([r['g_init'] for r in rs]):>15} | {ms([r['g_final'] for r in rs]):>15} | "
                      f"{ms([r['n_evals'] for r in rs]):>17} | {np.mean([r['cap_hit'] for r in rs]):4.0%} | "
                      f"{ms([r['p_ground_K1024'] for r in rs]):>15} | {ms([100 * r['gap_repaired_K1024'] for r in rs]):>15}")


if __name__ == "__main__":
    main()
