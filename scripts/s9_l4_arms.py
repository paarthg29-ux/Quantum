"""Phase 3: does the learned advantage survive at L4 (22 qubits)?

Arms A (cold), C-milp (GCN trained at L4 on MILP labels), D (LP at the QUBO
target), F (merit order), each warm arm at eps = 0.1 and at its Phase-1 best
eps (C-milp 0.15, D 0.25, F 0.25); the 20 structurally diverse L4 scenarios
of results/s9_selected.json (one per (ground state, LP pin pattern)
signature, drawn from the census pool, seed 9300) x 3 seeds x p = 1;
complex64. 7 runs per (scenario, seed, p) job, paired by identical initial
parameters. p = 2 is not run unless p = 1 gives a reason (it bought nothing
at L2).

22-qubit evaluation is memory-bandwidth bound: 4 workers give the same
throughput as 8 (measured 11.8 vs 12.0 evals/s at p = 1). Each worker keeps
ONE problem (energy vector + state table, ~200 MB) and rebuilds it (~10 s)
when its next job is a different scenario.

    python scripts/s9_l4_arms.py           # run / resume -> results/s9_runs.jsonl
    python scripts/s9_l4_arms.py --plan
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
LEVEL, N_HOURS = 4, 2
COMBOS = (("A", 0.1), ("C-milp", 0.1), ("C-milp", 0.15), ("D", 0.1), ("D", 0.25), ("F", 0.1), ("F", 0.25))
SEEDS, PS = (0, 1, 2), (1,)
G_TARGETS = (0.9, 0.95, 0.99)
WORKERS = 4
DTYPE = np.complex64
STORE = ROOT / "results" / "s9_runs.jsonl"
SELECTED = ROOT / "results" / "s9_selected.json"
THREAD_ENV = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
_CACHE = {}


def selected():
    return json.loads(SELECTED.read_text())["selected"]


def tasks():
    return {(a, e, s["scenario"], sd, p) for s in selected() for a, e in COMBOS for sd in SEEDS for p in PS}


def get_problem(scenario):
    from s9_census import scenario as census_scenario

    from qguc.qaoa import make_problem

    if scenario not in _CACHE:
        _CACHE.clear()  # one 22-qubit problem per worker
        pb = make_problem(census_scenario(LEVEL, scenario))
        pb.table
        _CACHE[scenario] = pb
    return _CACHE[scenario]


def job(args):
    scenario, seed, p, combos = args
    os.environ.update(THREAD_ENV)
    from qguc.arms import run_arm

    pb = get_problem(scenario)
    out = []
    for arm, eps in combos:
        r = run_arm(arm, pb, p, seed, eps=eps, dtype=DTYPE)
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
        except json.JSONDecodeError:
            pass
    return out


def main():
    import psutil

    todo = tasks()
    done = {(r["arm"], r["eps"], r["scenario"], r["seed"], r["p"]) for r in load()}
    left = todo - done
    groups = {}
    for a, e, s, sd, p in sorted(left):
        groups.setdefault((s, sd, p), []).append((a, e))
    jobs = sorted(((s, sd, p, c) for (s, sd, p), c in groups.items()), key=lambda j: (j[0], j[1]))
    print(f"phase 3 (L4): {len(todo)} runs in total ({len(done & todo)} done, {len(left)} left) in {len(jobs)} jobs; "
          f"{WORKERS} workers; CPU load {psutil.cpu_percent(interval=2.0):.0f}%", flush=True)
    if "--plan" in sys.argv or not jobs:
        return
    lock = STORE.with_suffix(".lock")
    if lock.exists():
        try:
            other = int(lock.read_text().strip())
        except ValueError:
            other = -1
        if other != os.getpid() and psutil.pid_exists(other):
            sys.exit(f"another runner (pid {other}) is using {STORE.name}; stop it first")
    lock.write_text(str(os.getpid()))
    try:
        t0 = time.perf_counter()
        text = STORE.read_text() if STORE.exists() else ""
        if text and not text.endswith("\n"):
            STORE.write_text(text[: text.rfind("\n") + 1])
        with ProcessPoolExecutor(WORKERS) as pool, STORE.open("a") as f:
            futs = {pool.submit(job, j): j for j in jobs}
            failed = 0
            for k, fut in enumerate(as_completed(futs), 1):
                try:
                    recs = fut.result()
                except Exception as exc:
                    failed += 1
                    s, sd, p, _ = futs[fut]
                    print(f"  FAILED job (scenario {s}, seed {sd}, p {p}): {exc!r}", flush=True)
                    continue
                for rec in recs:
                    f.write(json.dumps(rec) + "\n")
                f.flush()
                el = time.perf_counter() - t0
                print(f"  {k}/{len(jobs)} jobs, {failed} failed, {el:.0f} s (eta {el / k * (len(jobs) - k) / 60:.0f} min)",
                      flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
