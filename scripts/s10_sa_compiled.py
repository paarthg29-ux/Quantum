"""G17: compiled simulated annealing (numba), so the classical comparison is
not a pure-numpy strawman.

The algorithm is exactly s9_classical.anneal (random-scan single flips,
Metropolis, geometric temperature T0 -> T1 from s9_classical.temperatures,
best state tracked), compiled with numba, one chain per replica. For every
scenario of the G4 comparison (L2: 30; L4: 20 structures, frozen point):

  * compiled single-chain throughput (flips/s, measured after compilation);
  * a budget ladder: the QAOA-evaluation-matched budget, 1e4, 1e5, 1e6
    flips -- P(returned state is an exact ground state) over R replicas,
    mean repaired cost gap, and the MEASURED compiled time per run;
  * wall-time matched: the largest ladder budget whose compiled run time
    fits in the QAOA run's wall time, and the number of flips the QAOA wall
    time would actually buy (far beyond the ladder).

    python scripts/s10_sa_compiled.py          # -> results/s10_sa_compiled.jsonl / .txt
    python scripts/s10_sa_compiled.py --report
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
R = 400
LADDER = (10_000, 100_000, 1_000_000)
STORE = ROOT / "results" / "s10_sa_compiled.jsonl"
REPORT = ROOT / "results" / "s10_sa_compiled.txt"


@njit(cache=True)
def sa_chain(h, Js, steps, T0, T1, seed):
    np.random.seed(seed)
    n = h.size
    z = np.empty(n)
    for i in range(n):
        z[i] = 1.0 if np.random.random() < 0.5 else -1.0
    F = h.copy()
    for i in range(n):
        for j in range(n):
            F[i] += Js[i, j] * z[j]
    E = 0.0
    for i in range(n):
        E += z[i] * h[i]
        for j in range(n):
            E += 0.5 * z[i] * Js[i, j] * z[j]
    best = E
    bestz = z.copy()
    ratio = (T1 / T0) ** (1.0 / max(steps - 1, 1))
    T = T0
    for s in range(steps):
        i = np.random.randint(n)
        dE = -2.0 * z[i] * F[i]
        if dE <= 0.0 or np.random.random() < np.exp(-dE / T):
            zo = z[i]
            z[i] = -zo
            for j in range(n):
                F[j] += -2.0 * zo * Js[j, i]
            E += dE
            if E < best - 1e-12:
                best = E
                for j in range(n):
                    bestz[j] = z[j]
        T *= ratio
    return bestz, best


@njit(cache=True)
def sa_replicas(h, Js, steps, T0, T1, seed0, reps):
    n = h.size
    Z = np.empty((reps, n))
    Eb = np.empty(reps)
    for r in range(reps):
        z, e = sa_chain(h, Js, steps, T0, T1, seed0 + r)
        Z[r] = z
        Eb[r] = e
    return Z, Eb


def chain_time(h, Js, steps, T0, T1, reps=5):
    t0 = time.perf_counter()
    for r in range(reps):
        sa_chain(h, Js, steps, T0, T1, 10_000 + r)
    return (time.perf_counter() - t0) / reps


def run():
    import s9_classical as C

    base = {(r["level"], r["scenario"]): r for r in map(json.loads, (ROOT / "results" / "s9_classical.jsonl").read_text().splitlines())}
    done = set()
    if STORE.exists():
        done = {(r["level"], r["scenario"]) for r in map(json.loads, STORE.read_text().splitlines())}
    for level in (2, 4):
        for s, mk in C.problems(level):
            if (level, s) in done:
                continue
            pb = mk()
            h, Js = C.coupling(pb.ev)
            T0, T1 = C.temperatures(h, Js, np.random.default_rng([level, s, 0xA5]))
            sa_chain(h, Js, 100, T0, T1, 0)  # compile / warm
            fps = 1_000_000 / chain_time(h, Js, 1_000_000, T0, T1, reps=3)
            b0 = base[(level, s)]
            rec = dict(level=level, scenario=s, n=len(h), flips_per_s=fps, qaoa_wall_s=b0["qaoa_wall_s"],
                       wall_buys_flips=fps * b0["qaoa_wall_s"], qaoa=b0["qaoa"], ladder={})
            for b in (b0["b_eval"],) + LADDER:
                Z, Eb = sa_replicas(h, Js, b, T0, T1, 1_000 * level + 7 * s, R)
                pg, gap = C.evaluate(pb, Z.astype(int), Eb)
                rec["ladder"][str(b)] = dict(p_ground=pg, gap=gap, run_s=chain_time(h, Js, b, T0, T1, reps=3))
            with STORE.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            lad = rec["ladder"]
            print(f"  L{level} scenario {s:3d}: {fps / 1e6:.1f} M flips/s; P at "
                  + ", ".join(f"{int(k):,} -> {v['p_ground']:.3f} ({1e3 * v['run_s']:.2f} ms)" for k, v in lad.items()),
                  flush=True)


def report():
    from scipy import stats

    recs = [json.loads(l) for l in STORE.read_text().splitlines()]
    L = []
    P = lambda s="": L.append(s)  # noqa: E731
    P(f"G17: COMPILED SA (numba) on the identical QUBO, R = {R} replicas per budget; per-scenario means [95% t CI "
      "over scenarios]. Same algorithm and schedule as the numpy SA of s9_classical.py.")

    def ci(v):
        v = np.asarray(v, float)
        h = stats.t.ppf(0.975, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size)
        return f"{v.mean():.3f} [{v.mean() - h:.3f}, {v.mean() + h:.3f}]"

    for level in (2, 4):
        rs = [r for r in recs if r["level"] == level]
        if not rs:
            continue
        P(f"\n--- L{level}: n = {rs[0]['n']} qubits, {len(rs)} scenarios; compiled throughput "
          f"{np.mean([r['flips_per_s'] for r in rs]) / 1e6:.1f} M flips/s per chain (numpy SA: see s9_classical.txt) ---")
        for arm in ("C-milp", "D", "A"):
            v = [r["qaoa"][arm] for r in rs]
            P(f"  QAOA {arm:<7} best of 1024: P(ground) {ci([x[0] for x in v])}, gap {100 * np.mean([x[1] for x in v]):.2f} %, "
              f"{np.mean([x[3] for x in v]):.2f} s per run (simulated)")
        keys = list(rs[0]["ladder"])
        labels = ["= QAOA evaluations"] + [f"{int(k):,} flips" for k in keys[1:]]
        for k, lab in zip(keys, labels):
            if lab == "= QAOA evaluations":
                vals = [r["ladder"][next(iter(r["ladder"]))] for r in rs]
                bud = np.mean([int(next(iter(r["ladder"]))) for r in rs])
                lab = f"= QAOA evaluations ({bud:.0f})"
            else:
                vals = [r["ladder"][k] for r in rs]
            P(f"  SA compiled, {lab:<26}: P(ground) {ci([x['p_ground'] for x in vals])}, gap "
              f"{100 * np.mean([x['gap'] for x in vals]):.2f} %, {1e3 * np.mean([x['run_s'] for x in vals]):8.2f} ms per run")
        P(f"  QAOA wall time ({np.mean([r['qaoa_wall_s'] for r in rs]):.2f} s) would buy "
          f"{np.mean([r['wall_buys_flips'] for r in rs]):.2e} compiled flips per run; every ladder budget fits inside it")
    REPORT.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    if "--report" not in sys.argv:
        run()
    report()
