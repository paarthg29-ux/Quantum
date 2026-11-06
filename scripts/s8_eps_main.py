"""Epsilon sweep (phase 1) and main table (phase 2) on 30 held-out scenarios at L2 (n = 16).

    phase 1:  eps sweep, p = 1: arms B, C-milp, C-oracle, D, F x eps in EPS x 30 scenarios x 3 seeds,
              plus arm A at p = 1 on the same seeds (the reference for the eps = 0.5 limit)
    phase 2:  main table: arms A, B, C-milp, C-oracle, D, E, F; eps = 0.1; 30 scenarios x 5 seeds x p in {1, 2, 3}

p = 5 was dropped (step-7 rescope: ~60% of the cost, no arm reached g = 0.99,
final g moved by ~0.003). The 1050 p = 5 runs and the p = 3 sweep runs already
in the store are kept but belong to neither phase.

Scenarios: seed 9100 -- never used for calibration, GNN training (8001),
validation (8002), accuracy (9002), the pilot (7001) or the previous full
run (9001). One resumable store, results/s8_runs.jsonl, keyed by
(arm, eps, scenario, seed, p); a run is computed once and shared by both
phases (phase 1's eps = 0.1 runs are phase-2 runs). Each (scenario, seed, p)
job runs every (arm, eps) it needs in one worker from identical initial
parameters, so all comparisons are paired. One runner at a time (lock file).

    python scripts/s8_eps_main.py --phase 1          # run / resume phase 1
    python scripts/s8_eps_main.py --phase 2          # run / resume phase 2
    python scripts/s8_eps_main.py --phase 1 --plan   # task counts only
Analysis: scripts/s8_analysis.py --phase 1 | 2.
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

LEVEL, N_HOURS = 2, 2
SCEN_SEED, N_SCEN = 9100, 30
MAIN_ARMS = ("A", "B", "C-milp", "C-oracle", "D", "E", "F")
MAIN_EPS, MAIN_SEEDS, MAIN_P = 0.1, tuple(range(5)), (1, 2, 3)
SWEEP_ARMS = ("B", "C-milp", "C-oracle", "D", "F")
EPS = (0.0, 0.05, 0.1, 0.15, 0.25, 0.35, 0.5)
SWEEP_SEEDS, SWEEP_P = tuple(range(3)), (1,)
G_TARGETS = (0.9, 0.95, 0.99)
WORKERS = 8
STORE = Path(__file__).resolve().parents[1] / "results" / "s8_runs.jsonl"
THREAD_ENV = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
_PROBLEMS = {}


def tasks(phase):
    if phase == 1:
        t = {(a, e, s, sd, p) for a in SWEEP_ARMS for e in EPS for s in range(N_SCEN) for sd in SWEEP_SEEDS
             for p in SWEEP_P}
        return t | {("A", MAIN_EPS, s, sd, p) for s in range(N_SCEN) for sd in SWEEP_SEEDS for p in SWEEP_P}
    if phase == 2:
        return {(a, MAIN_EPS, s, sd, p) for a in MAIN_ARMS for s in range(N_SCEN) for sd in MAIN_SEEDS
                for p in MAIN_P}
    raise SystemExit("usage: s8_eps_main.py --phase 1|2 [--plan]")


def phase_arg():
    try:
        return int(sys.argv[sys.argv.index("--phase") + 1])
    except (ValueError, IndexError):
        raise SystemExit("usage: s8_eps_main.py --phase 1|2 [--plan]")


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
    scenario, seed, p, combos = args
    os.environ.update(THREAD_ENV)
    from qguc.arms import run_arm

    pb = get_problem(scenario)
    out = []
    for arm, eps in combos:
        r = run_arm(arm, pb, p, seed, eps=eps)
        rec = dict(arm=arm, eps=eps, scenario=scenario, seed=seed, p=p, peak=pb.instance.meta["peak_mw"],
                   n_ground=len(pb.ev.ground), r_unif=pb.uniform_ratio, n_evals=r.n_evals, max_evals=r.max_evals,
                   cap_hit=r.cap_hit, early=r.early_stopped, overlap0=r.overlap0,
                   n_pinned=r.extra["n_pinned"], n_pinned_wrong=r.extra["n_pinned_wrong"],
                   g_state0=r.g_state0, g_init=r.g_init, g_final=r.g_final, ratio=r.best_ratio,
                   **{f"to_{t}": r.evals_to_g(t) for t in G_TARGETS},
                   wall_s=r.wall_s, source_s=r.source_s, total_s=r.total_wall_s)
        for K, m in r.sampling.items():
            rec.update({f"{k}_K{K}": v for k, v in m.items()})
        out.append(rec)
    return out


def load():
    if not STORE.exists():
        return []
    out = []
    for line in STORE.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:  # a line cut off by an interrupted run; that job is redone
            pass
    return out


def main():
    import psutil

    phase = phase_arg()
    todo = tasks(phase)
    done = {(r["arm"], r["eps"], r["scenario"], r["seed"], r["p"]) for r in load()}
    left = todo - done
    groups = {}
    for a, e, s, sd, p in sorted(left):
        groups.setdefault((s, sd, p), []).append((a, e))
    jobs = sorted(((s, sd, p, c) for (s, sd, p), c in groups.items()), key=lambda j: (-j[2], j[0], j[1]))
    print(f"phase {phase}: {len(todo)} runs in total ({len(done & todo)} done, {len(left)} left) in {len(jobs)} jobs; "
          f"{WORKERS} workers; CPU load {psutil.cpu_percent(interval=2.0):.0f}%", flush=True)
    if "--plan" in sys.argv or not jobs:
        return
    lock = STORE.with_suffix(".lock")  # one runner at a time: two would redo the same jobs and interleave writes
    if lock.exists():
        try:
            other = int(lock.read_text().strip())
        except ValueError:
            other = -1
        if other != os.getpid() and psutil.pid_exists(other):
            sys.exit(f"another runner (pid {other}) is using {STORE.name}; stop it first (or delete {lock.name} "
                     "if that pid is not this script)")
    lock.write_text(str(os.getpid()))
    try:
        run(jobs)
    finally:
        lock.unlink(missing_ok=True)


def run(jobs):
    t0 = time.perf_counter()
    text = STORE.read_text() if STORE.exists() else ""
    if text and not text.endswith("\n"):  # drop a partial last line before appending
        STORE.write_text(text[: text.rfind("\n") + 1])
    with ProcessPoolExecutor(WORKERS) as pool, STORE.open("a") as f:
        futs = {pool.submit(job, j): j for j in jobs}
        failed = 0
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                recs = fut.result()
            except Exception as exc:  # log and carry on; the job stays undone and is retried on resume
                failed += 1
                s, sd, p, _ = futs[fut]
                print(f"  FAILED job (scenario {s}, seed {sd}, p {p}): {exc!r}", flush=True)
                continue
            for rec in recs:
                f.write(json.dumps(rec) + "\n")
            f.flush()
            if k % 25 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} jobs, {failed} failed, {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
