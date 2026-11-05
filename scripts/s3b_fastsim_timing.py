"""Re-time QAOA on the fast NumPy simulator (qguc.fastsim) and re-plan the
benchmark.

* Sizes: the aggregation ladder at T = 2, n = 10, 12, 16, 18, 22 (L0..L4),
  frozen operating point, scenario 0 of the verification set.
* Per energy evaluation (COBYLA needs no gradients), p in {1, 2, 3, 5},
  complex128 and complex64; each level in its own single-thread subprocess
  so peak memory is clean.
* COBYLA's own per-evaluation overhead, measured on a trivial objective.
* Parallel throughput: K simultaneous single-thread processes.
* Evaluations per run: arm A (cold start) actually run with the frozen
  optimiser settings and early stopping, at L0, L2 and L3.

Estimate: 5 arms x 10 scenarios x 10 seeds x p in {1, 2, 3, 5}, evaluations
per run = measured arm-A mean (other arms assumed to cost the same).
"""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

LEVELS = (0, 1, 2, 3, 4)
P_VALUES = (1, 2, 3, 5)
ARMS, SCENARIOS, SEEDS = 5, 10, 10
SCEN_SEED = 4242
PAR_K = (1, 4, 8)
THREAD_ENV = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
COUNT_PLAN = {0: (3, 3), 2: (3, 3), 3: (1, 2)}  # level -> (scenarios, seeds) for arm-A runs


def peak_mb():
    import psutil

    mi = psutil.Process().memory_info()
    return getattr(mi, "peak_wset", mi.rss) / 2**20


def problem(level, scenario=0):
    from qguc import make_scenario_set
    from qguc.qaoa import make_problem

    inst = make_scenario_set(scenario + 1, n_hours=2, seed=SCEN_SEED, refinement_level=level)[scenario]
    return make_problem(inst)


def median_time(fn, budget_s=3.0, max_reps=50):
    fn()
    ts = []
    while len(ts) < 3 or (sum(ts) < budget_s and len(ts) < max_reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def time_worker(level):
    from qguc.fastsim import QAOASimulator

    base = peak_mb()
    t0 = time.perf_counter()
    pb = problem(level)
    build_s = time.perf_counter() - t0  # E by brute force (+ MILP, which is tiny)
    rng = np.random.default_rng(0)
    out = dict(level=level, n=pb.n, build_s=build_s, p={})
    sims = {dt: QAOASimulator(pb.ev.E, dtype=dt) for dt in ("complex128", "complex64")}
    for p in P_VALUES:
        g, b = rng.uniform(0, 1, p) / pb.ev.sigma, rng.uniform(-0.8, 0.8, p)
        rec = {dt: median_time(lambda s=s: s.energy(g, b)) for dt, s in sims.items()}
        diffs = []
        for _ in range(5):
            g, b = rng.uniform(0, 1, p) / pb.ev.sigma, rng.uniform(-0.8, 0.8, p)
            diffs.append(abs(sims["complex128"].energy(g, b) - sims["complex64"].energy(g, b)))
        rec["c64_ratio_err"] = float(max(diffs) / (pb.ev.e_max - pb.ev.e_min))
        out["p"][p] = rec
    out["peak_mb"], out["base_mb"] = peak_mb(), base
    print(json.dumps(out))


def cobyla_overhead():
    from scipy.optimize import minimize

    res = {}
    for npar in (2, 4, 6, 10):
        cnt = [0]

        def f(x):
            cnt[0] += 1
            return float(np.sum((x - 0.3) ** 2) + 0.01 * np.sum(np.cos(7 * x)))

        t0 = time.perf_counter()
        minimize(f, np.zeros(npar), method="COBYLA", options=dict(rhobeg=0.25, tol=1e-6, maxiter=5000))
        res[npar] = (time.perf_counter() - t0) / cnt[0]
    print(json.dumps(res))


def par_worker(level, reps):
    from qguc.fastsim import QAOASimulator

    pb = problem(level)
    sim = QAOASimulator(pb.ev.E)
    g, b = np.full(3, 0.5) / pb.ev.sigma, np.full(3, 0.3)
    sim.energy(g, b)
    t0 = time.perf_counter()
    for _ in range(reps):
        sim.energy(g, b)
    print(json.dumps(dict(per_eval=(time.perf_counter() - t0) / reps)))


def count_job(args):
    level, scenario, seed, p = args
    os.environ.update(THREAD_ENV)
    from qguc.qaoa import arm_a

    pb = problem(level, scenario)
    r = arm_a(pb, p, seed)
    return dict(level=level, n=pb.n, scenario=scenario, seed=seed, p=p, evals=r.n_evals, wall=r.wall_s,
                early=r.early_stopped, cap=r.cap_hit, ratio=r.best_ratio, init=r.initial_ratio)


def sub(*args):
    res = subprocess.run([sys.executable, os.path.abspath(__file__), *map(str, args)],
                         env=dict(os.environ, **THREAD_ENV), capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr[-3000:])
    return json.loads(res.stdout.strip().splitlines()[-1])


def main():
    import psutil

    phys = psutil.cpu_count(logical=False)
    load = psutil.cpu_percent(interval=2.0)
    print(f"{phys} physical / {os.cpu_count()} logical cores; CPU load before start {load:.0f}% "
          f"(other users' jobs included); 1 thread per process; fast simulator group = 4")

    print("\n=== per energy evaluation, single thread (median, s) ===")
    print(f"  {'lvl':>3} {'n':>3} {'E build':>8} | " + " | ".join(f"p={p}: {'c128':>8} {'c64':>8}" for p in P_VALUES)
          + f" | {'peak MB':>7} | c64 ratio err")
    timing = {}
    for lvl in LEVELS:
        r = sub("--time", lvl)
        timing[r["n"]] = r
        cells = " | ".join(f"     {r['p'][str(p)]['complex128']:8.4f} {r['p'][str(p)]['complex64']:8.4f}"
                           for p in P_VALUES)
        err = max(r["p"][str(p)]["c64_ratio_err"] for p in P_VALUES)
        print(f"  L{lvl:<2} {r['n']:3d} {r['build_s']:7.2f}s | {cells} | {r['peak_mb']:7.0f} | {err:.1e}", flush=True)

    ovh = {int(k): v for k, v in sub("--cobyla").items()}
    print("\nCOBYLA overhead per evaluation (trivial objective): "
          + ", ".join(f"{k} params {1e3 * v:.2f} ms" for k, v in ovh.items()))

    print("\n=== parallel throughput: K simultaneous processes, p = 3, complex128 ===")
    par = {}
    by_level = {t["level"]: t for t in timing.values()}
    for lvl in (2, 3, 4):
        n, single = by_level[lvl]["n"], by_level[lvl]["p"]["3"]["complex128"]
        reps = max(3, int(20.0 / single))
        line = f"  L{lvl} n={n}:"
        for K in PAR_K:
            procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__), "--par", str(lvl), str(reps)],
                                      env=dict(os.environ, **THREAD_ENV), stdout=subprocess.PIPE, text=True)
                     for _ in range(K)]
            per = [json.loads(pr.communicate()[0].strip().splitlines()[-1])["per_eval"] for pr in procs]
            eff = single / np.mean(per)
            par[(lvl, K)] = eff
            line += f"  K={K}: {1e3 * np.mean(per):.1f} ms/eval/process (efficiency {100 * eff:.0f}%)"
        print(line, flush=True)

    print("\n=== evaluations per run: arm A (cold start), COBYLA + early stopping, frozen settings ===")
    jobs = [(lvl, s, seed, p) for lvl, (ns, nseed) in COUNT_PLAN.items() for s in range(ns)
            for seed in range(nseed) for p in P_VALUES]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(4) as pool:
        counts = list(pool.map(count_job, jobs))
    print(f"  {len(jobs)} runs in {time.perf_counter() - t0:.0f} s (4 workers)")
    print(f"  {'lvl':>3} {'n':>3} {'p':>2} {'runs':>4} | {'evals mean':>10} {'min':>5} {'max':>5} | "
          f"{'early-stop':>10} | {'ratio init -> best':>19}")
    evals = {}
    for lvl in COUNT_PLAN:
        for p in P_VALUES:
            rs = [c for c in counts if c["level"] == lvl and c["p"] == p]
            e = np.array([c["evals"] for c in rs])
            evals[(lvl, p)] = e.mean()
            print(f"  L{lvl:<2} {rs[0]['n']:3d} {p:2d} {len(rs):4d} | {e.mean():10.0f} {e.min():5d} {e.max():5d} | "
                  f"{sum(c['early'] for c in rs):5d}/{len(rs):<4} | "
                  f"{np.mean([c['init'] for c in rs]):.3f} -> {np.mean([c['ratio'] for c in rs]):.3f}")

    runs = ARMS * SCENARIOS * SEEDS
    ev_p = {p: max(evals[(lvl, p)] for lvl in COUNT_PLAN) for p in P_VALUES}  # largest observed mean
    print(f"\n=== benchmark estimate: {ARMS} arms x {SCENARIOS} scenarios x {SEEDS} seeds = {runs} runs per p, "
          f"p in {list(P_VALUES)} ===")
    print("  evaluations per run = largest arm-A mean over L0/L2/L3: "
          + ", ".join(f"p={p}: {ev_p[p]:.0f}" for p in P_VALUES)
          + f" -> {runs * sum(ev_p.values()):,.0f} evaluations in total")
    print("  time per evaluation = measured single-thread energy time + COBYLA overhead; "
          "+ one brute-force E build per scenario")
    print(f"  {'n':>3} | {'core-h c128':>11} {'core-h c64':>10} | {'wall h, 8 procs x measured eff.':>31} | "
          f"{'wall h, ideal 8':>15}")
    for n, r in sorted(timing.items()):
        lvl = r["level"]
        core = {}
        for dt in ("complex128", "complex64"):
            sec = sum(runs * ev_p[p] * (r["p"][str(p)][dt] + ovh[min(ovh, key=lambda k: abs(k - 2 * p))])
                      for p in P_VALUES) + SCENARIOS * r["build_s"]
            core[dt] = sec / 3600
        eff8 = par.get((lvl, 8))
        wall_eff = f"{core['complex128'] / (8 * eff8):9.1f} (eff {100 * eff8:.0f}%)" if eff8 else "-"
        print(f"  {n:3d} | {core['complex128']:11.1f} {core['complex64']:10.1f} | {wall_eff:>31} | "
              f"{core['complex128'] / 8:15.2f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--time":
        time_worker(int(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == "--cobyla":
        cobyla_overhead()
    elif len(sys.argv) > 1 and sys.argv[1] == "--par":
        par_worker(int(sys.argv[2]), int(sys.argv[3]))
    else:
        main()
