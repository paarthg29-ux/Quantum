"""Full arm comparison at L2 (n = 16).

Arms A, B, C-milp, C-oracle, D, E, F; 10 held-out scenarios (seed 9001:
never used for calibration, GNN training/validation/accuracy, or the pilot)
x 10 seeds x p in {1, 2, 3, 5}; paired on (scenario, seed): each (scenario,
seed, p) job runs every arm in one worker from identical initial
parameters. Frozen encoding, frozen stopping rule (cap 200 per parameter).

Results are appended to results/s7_arms_full.jsonl as jobs finish, and a
rerun skips completed jobs, so an interrupted run resumes.
    python scripts/s7_arms_full.py              # run (or resume), then summarise
    python scripts/s7_arms_full.py --summary    # summarise only
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

LEVEL, N_HOURS = 2, 2
SCEN_SEED, N_SCEN, SEEDS = 9001, 10, tuple(range(10))
P_VALUES = (1, 2, 3, 5)
ARMS = ("A", "B", "C-milp", "C-oracle", "D", "E", "F")
G_TARGETS = (0.9, 0.95, 0.99)
WORKERS = 6
OUT = Path(__file__).resolve().parents[1] / "results" / "s7_arms_full.jsonl"
THREAD_ENV = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
_PROBLEMS = {}


def get_problem(scenario):
    from qguc import make_scenario_set
    from qguc.qaoa import make_problem

    if scenario not in _PROBLEMS:
        inst = make_scenario_set(N_SCEN, n_hours=N_HOURS, seed=SCEN_SEED, refinement_level=LEVEL)[scenario]
        pb = make_problem(inst)
        pb.table
        _PROBLEMS[scenario] = pb
    return _PROBLEMS[scenario]


def job(args):
    scenario, seed, p = args
    os.environ.update(THREAD_ENV)
    from qguc.arms import run_arm

    pb = get_problem(scenario)
    out = []
    for a in ARMS:
        r = run_arm(a, pb, p, seed)
        rec = dict(arm=a, scenario=scenario, seed=seed, p=p, n_evals=r.n_evals, max_evals=r.max_evals,
                   cap_hit=r.cap_hit, early=r.early_stopped, g_state0=r.g_state0, g_init=r.g_init,
                   g_final=r.g_final, ratio=r.best_ratio, **{f"to_{t}": r.evals_to_g(t) for t in G_TARGETS},
                   wall_s=r.wall_s, source_s=r.source_s, total_s=r.total_wall_s,
                   gains=np.round(r.gains[:: max(1, r.n_evals // 400)], 5).tolist())
        for K, m in r.sampling.items():
            rec.update({f"{k}_K{K}": v for k, v in m.items()})
        out.append(rec)
    return out


def load():
    if not OUT.exists():
        return []
    return [json.loads(line) for line in OUT.read_text().splitlines() if line.strip()]


def run():
    import psutil

    done = {(r["scenario"], r["seed"], r["p"]) for r in load()}
    jobs = [(s, sd, p) for p in P_VALUES[::-1] for s in range(N_SCEN) for sd in SEEDS if (s, sd, p) not in done]
    print(f"{len(done)} jobs already done, {len(jobs)} to run with {WORKERS} workers; "
          f"CPU load {psutil.cpu_percent(interval=2.0):.0f}%", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(WORKERS) as pool, OUT.open("a") as f:
        futs = [pool.submit(job, j) for j in jobs]
        for k, fut in enumerate(as_completed(futs), 1):
            for rec in fut.result():
                f.write(json.dumps(rec) + "\n")
            f.flush()
            if k % 20 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} jobs, {time.perf_counter() - t0:.0f} s", flush=True)


def ms(v, scale=1.0, fmt=".3f"):
    v = scale * np.asarray([x for x in v if x is not None], float)
    if not v.size:
        return "-"
    return f"{v.mean():{fmt}} +/- {v.std(ddof=1):{fmt}}" if v.size > 1 else f"{v.mean():{fmt}}"


def paired(recs, a, b, p, key, scale=1.0):
    idx = {(r["scenario"], r["seed"]): r for r in recs if r["arm"] == b and r["p"] == p}
    d = np.array([scale * (r[key] - idx[(r["scenario"], r["seed"])][key]) for r in recs
                  if r["arm"] == a and r["p"] == p and r[key] is not None
                  and idx.get((r["scenario"], r["seed"]), {}).get(key) is not None], float)
    if d.size == 0:
        return "no pairs"
    pv = wilcoxon(d).pvalue if d.size > 5 and np.any(d != 0) else float("nan")
    return f"{d.mean():+8.3f} +/- {d.std(ddof=1):6.3f} (n={d.size:3d}, Wilcoxon p={pv:.1e})"


def summarise(recs):
    n_pairs = N_SCEN * len(SEEDS)
    for p in P_VALUES:
        rp = [r for r in recs if r["p"] == p]
        if not rp:
            continue
        print(f"\n=============== p = {p}: mean +/- std over {n_pairs} paired runs (cap {rp[0]['max_evals']}) "
              f"===============")
        print(f"  {'arm':>8} | {'g state0':>15} | {'g init':>16} | {'g final':>15} | {'evals':>19} | {'cap':>4} | "
              + " | ".join(f"{'to g=' + str(t):>15}" for t in G_TARGETS))
        for a in ARMS:
            rs = [r for r in rp if r["arm"] == a]
            to = []
            for t in G_TARGETS:
                v = [r[f"to_{t}"] for r in rs if r[f"to_{t}"] is not None]
                to.append(f"{np.median(v):5.0f} ({len(v):3d}/{len(rs)})" if v else f"never (0/{len(rs)})")
            print(f"  {a:>8} | {ms([r['g_state0'] for r in rs]):>15} | {ms([r['g_init'] for r in rs]):>16} | "
                  f"{ms([r['g_final'] for r in rs]):>15} | {ms([r['n_evals'] for r in rs], fmt='.1f'):>19} | "
                  f"{np.mean([r['cap_hit'] for r in rs]):4.0%} | " + " | ".join(f"{x:>15}" for x in to))
        print(f"  {'arm':>8} | {'P(ground) K=1':>15} | {'P(gnd) K=128':>15} | {'P(gnd) K=1024':>15} | "
              f"{'1-shot adeq':>11} | {'gap K=1 raw/rep %':>19} | {'gap K=1024 raw %':>16} | {'rep %':>15} | "
              f"{'wall s/run':>15} | {'source us':>9}")
        for a in ARMS:
            rs = [r for r in rp if r["arm"] == a]
            print(f"  {a:>8} | {ms([r['p_ground_K1'] for r in rs], fmt='.4f'):>15} | "
                  f"{ms([r['p_ground_K128'] for r in rs]):>15} | {ms([r['p_ground_K1024'] for r in rs]):>15} | "
                  f"{np.mean([r['adequate_K1'] for r in rs]):11.3f} | "
                  f"{np.mean([r['gap_raw_K1'] for r in rs]) * 100:8.2f} / {np.mean([r['gap_repaired_K1'] for r in rs]) * 100:6.2f} | "
                  f"{ms([r['gap_raw_K1024'] for r in rs], 100, '.2f'):>16} | "
                  f"{ms([r['gap_repaired_K1024'] for r in rs], 100, '.2f'):>15} | "
                  f"{ms([r['total_s'] for r in rs], fmt='.2f'):>15} | "
                  f"{1e6 * np.median([r['source_s'] for r in rs]):9.0f}")
        keys = (("p_ground_K1", "P(gnd) K=1", 1), ("p_ground_K1024", "P(gnd) K=1024", 1),
                ("gap_repaired_K1024", "gap K=1024 (pp)", 100), ("g_final", "g final", 1),
                ("to_0.9", "evals to g=0.9", 1), ("to_0.95", "evals to g=0.95", 1))
        for title, pairs in (("Q1 learned vs D", (("C-milp", "D"), ("C-oracle", "D"))),
                             ("Q2 better labels", (("C-oracle", "C-milp"),)),
                             ("Q3 warm-mixer ablation", (("B", "C-milp"),)),
                             ("Q4 vs the trivial heuristic F", (("C-milp", "F"), ("C-oracle", "F"), ("D", "F"))),
                             ("vs cold start A", tuple((a, "A") for a in ARMS[1:]))):
            print(f"  {title}:")
            for a, b in pairs:
                for key, lab, sc in keys:
                    print(f"    {a + ' - ' + b:>19} {lab:>16}: {paired(rp, a, b, p, key, sc)}")


if __name__ == "__main__":
    if "--summary" not in sys.argv:
        run()
    summarise(load())
