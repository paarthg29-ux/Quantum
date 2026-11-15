"""G4: classical samplers on the IDENTICAL QUBO -- simulated annealing and tabu search.

Both minimise the same normalised Ising energy the QAOA cost layer uses,
E(z) = h.z + sum_{i<j} J_ij z_i z_j with z = 1 - 2x (qguc.ising), so every
state they return has exactly the energy of the brute-force energy vector
(checked for every returned state). One "objective evaluation" = evaluating
the energy change of one single-bit flip.

Budgets, per scenario, matched to the QAOA run they are compared with
(C-milp at eps = 0.1, p = 1, mean over its seeds; arm A shown alongside):

  evals   QAOA's objective evaluations: optimiser evaluations (n_evals) + the
          1024 final samples (each a bitstring whose energy is looked at).
          This is GENEROUS to QAOA: on hardware each expectation estimate
          costs many shots, here it counts as one evaluation.
  wall    as many flips as this numpy SA does in the QAOA run's wall time
          (optimisation + exact metrics on the simulator). Measured
          single-chain throughput of THIS implementation (pure numpy, one flip
          per Python step) -- a compiled SA is ~100x faster, so this is
          conservative for SA. Capped at WALL_CAP flips; when the cap binds the
          result is reported as ">=".

SA: random-scan single flips, Metropolis, geometric temperature from T0
(median |dE| of random flips at random states) to T1 (5th percentile of the
positive |dE| / ln 100), best state tracked. T1 was chosen among {smallest
|dE|, 5th percentile, T0 / 100} on 4 L2 scenarios at 1,200 and 10,000 flips
(it was best in every case) -- light tuning, disclosed, in SA's favour. Tabu: steepest single flip among
non-tabu moves (aspiration on a new best), tenure max(2, n // 2), random
start, RESTART from a new random state after 5n steps without improving the
current restart's best; one step evaluates all n flips, so it gets budget // n
steps. Tenure and restart were chosen among {n/4, n/2} x {none, 2n, 5n} on
the 3 hardest L2 scenarios (20, 11, 0); without restarts tabu cycles and does
not improve with budget at all (P 0.15 at both 1,191 and 57,031 flips) --
disclosed, in tabu's favour. Each
(scenario, method, budget) is run as R independent replicas: P(ground) =
fraction of replicas whose best state is an exact ground state; cost gap =
mean repaired operating-cost gap of the replicas' best states (as the QAOA
best-of-1024 metric).

Scenarios: L2 -- the 30 held-out Phase-2 scenarios (frozen point); L4 --
the 20 Phase-3 structures (frozen point).

    python scripts/s9_classical.py           # -> results/s9_classical.jsonl, results/s9_classical.txt
    python scripts/s9_classical.py --report
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
R = 400
WALL_CAP = 200_000
STORE = ROOT / "results" / "s9_classical.jsonl"
REPORT = ROOT / "results" / "s9_classical.txt"
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")


def coupling(ev):
    J = ev.ising.J
    return ev.ising.h.astype(float), (J + J.T).astype(float)


def index_of(Z):
    """Spins (R, n) -> state index (bit j = (k >> j) & 1, x = (1 - z) / 2)."""
    X = (1 - Z) // 2
    return (X.astype(np.int64) << np.arange(Z.shape[1], dtype=np.int64)).sum(1)


def temperatures(h, Js, rng):
    n = len(h)
    Z = rng.choice([-1, 1], (2000, n))
    F = h + Z @ Js
    dE = np.abs(-2 * Z * F).ravel()
    pos = dE[dE > 1e-12]
    return float(np.median(pos)), float(np.percentile(pos, 5) / np.log(100.0))


def anneal(h, Js, steps, rng, reps=R):
    """Simulated annealing, reps replicas in lockstep; returns best states (reps, n) and best energies."""
    n = len(h)
    T0, T1 = temperatures(h, Js, rng)
    Z = rng.choice([-1, 1], (reps, n))
    F = h + Z @ Js  # local fields
    E = Z @ h + 0.5 * np.einsum("ri,ij,rj->r", Z, Js, Z)
    bestE, bestZ = E.copy(), Z.copy()
    temps = T0 * (T1 / T0) ** (np.arange(steps) / max(steps - 1, 1))
    rows = np.arange(reps)
    for s in range(steps):
        i = rng.integers(0, n, reps)
        zi = Z[rows, i]
        dE = -2.0 * zi * F[rows, i]
        acc = (dE <= 0) | (rng.random(reps) < np.exp(-np.maximum(dE, 0) / temps[s]))
        if acc.any():
            r, ii, z_old = rows[acc], i[acc], zi[acc]
            Z[r, ii] = -z_old
            F[r] += (-2.0 * z_old)[:, None] * Js[ii]
            E[r] += dE[acc]
            better = E[r] < bestE[r] - 1e-12
            if better.any():
                rb = r[better]
                bestE[rb] = E[rb]
                bestZ[rb] = Z[rb]
    return bestZ, bestE


def tabu(h, Js, steps, rng, reps=R):
    n = len(h)
    tenure = max(2, n // 2)
    restart_after = 5 * n
    Z = rng.choice([-1, 1], (reps, n))
    F = h + Z @ Js
    E = Z @ h + 0.5 * np.einsum("ri,ij,rj->r", Z, Js, Z)
    bestE, bestZ = E.copy(), Z.copy()
    run_best, last_improved = E.copy(), np.zeros(reps, int)
    until = np.zeros((reps, n), int)  # tabu until step
    rows = np.arange(reps)
    for s in range(steps):
        stale = np.flatnonzero(s - last_improved > restart_after)
        if stale.size:  # restart from a fresh random state
            Z[stale] = rng.choice([-1, 1], (stale.size, n))
            F[stale] = h + Z[stale] @ Js
            E[stale] = Z[stale] @ h + 0.5 * np.einsum("ri,ij,rj->r", Z[stale], Js, Z[stale])
            run_best[stale], last_improved[stale], until[stale] = E[stale], s, 0
        dE0 = -2.0 * Z * F  # exact energy change of every single flip
        dE = dE0 + 1e-9 * rng.random((reps, n))  # tiny jitter for selection only: random tie-breaking
        allowed = (until <= s) | (E[:, None] + dE0 < bestE[:, None] - 1e-12)  # aspiration on a new best
        dE_m = np.where(allowed, dE, np.inf)
        i = np.argmin(dE_m, axis=1)
        ok = np.isfinite(dE_m[rows, i])
        r, ii = rows[ok], i[ok]
        z_old = Z[r, ii]
        E[r] += dE0[r, ii]
        Z[r, ii] = -z_old
        F[r] += (-2.0 * z_old)[:, None] * Js[ii]
        until[r, ii] = s + 1 + tenure
        imp = E[r] < run_best[r] - 1e-9
        run_best[r[imp]], last_improved[r[imp]] = E[r[imp]], s
        better = E[r] < bestE[r] - 1e-9
        if better.any():
            rb = r[better]
            bestE[rb], bestZ[rb] = E[rb], Z[rb]
    return bestZ, bestE


def throughput(h, Js, rng, steps=20_000):
    """Flips per second of ONE SA chain in this implementation."""
    t0 = time.perf_counter()
    anneal(h, Js, steps, rng, reps=1)
    return steps / (time.perf_counter() - t0)


def evaluate(pb, Z, Eb):
    k = index_of(Z)
    assert np.allclose(pb.ev.E[k], Eb, atol=1e-7), "energy mismatch with the brute-force energy vector"
    pos = np.empty_like(pb.table.order)
    pos[pb.table.order] = np.arange(len(pos))
    ground = np.isin(k, pb.ev.ground)
    return float(ground.mean()), float(pb.table.gap_repaired[pos[k]].mean())


def qaoa_reference(level):
    """{scenario: {arm: (P(b1024), gap K1024, n_evals, wall+source s)}} at p = 1, eps = 0.1 (mean over seeds)."""
    if level == 2:
        from s8_analysis import load

        recs = [r for r in load() if r["p"] == 1 and r["eps"] == 0.1 and r["seed"] < 5]
    else:
        from s9_l4_arms import load

        recs = [r for r in load() if r["p"] == 1 and r["eps"] == 0.1]
    out = {}
    for r in recs:
        out.setdefault(r["scenario"], {}).setdefault(r["arm"], []).append(
            (r["p_ground_K1024"], r["gap_repaired_K1024"], r["n_evals"], r["wall_s"] + r["source_s"]))
    return {s: {a: tuple(np.mean(v, axis=0)) for a, v in d.items()} for s, d in out.items()}


def problems(level):
    if level == 2:
        from s8_eps_main import N_SCEN, get_problem

        return [(s, lambda s=s: get_problem(s)) for s in range(N_SCEN)]
    from s9_l4_arms import get_problem, selected

    return [(x["scenario"], lambda s=x["scenario"]: get_problem(s)) for x in selected()]


def run():
    done = set()
    if STORE.exists():
        done = {(r["level"], r["scenario"]) for r in map(json.loads, STORE.read_text().splitlines())}
    for level in (2, 4):
        ref = qaoa_reference(level)
        for s, mk in problems(level):
            if (level, s) in done:
                continue
            t0 = time.perf_counter()
            pb = mk()
            h, Js = coupling(pb.ev)
            rng = np.random.default_rng([level, s, 0xA5])
            n = len(h)
            q = ref[s]["C-milp"]
            b_eval = int(round(q[2])) + 1024
            fps = throughput(h, Js, rng)
            b_wall_uncapped = int(fps * q[3])
            b_wall = min(b_wall_uncapped, WALL_CAP)
            rec = dict(level=level, scenario=s, n=n, qaoa=ref[s], b_eval=b_eval, sa_flips_per_s=fps,
                       qaoa_wall_s=q[3], b_wall=b_wall, b_wall_uncapped=b_wall_uncapped, R=R)
            for name, b in (("eval", b_eval), ("wall", b_wall)):
                ts = time.perf_counter()
                Zs, Es = anneal(h, Js, b, rng)
                rec[f"sa_{name}"] = dict(zip(("p_ground", "gap"), evaluate(pb, Zs, Es)),
                                         chain_s=b / fps, sim_s=time.perf_counter() - ts)
                ts = time.perf_counter()
                Zt, Et = tabu(h, Js, max(1, b // n), rng)
                rec[f"tabu_{name}"] = dict(zip(("p_ground", "gap"), evaluate(pb, Zt, Et)),
                                           steps=max(1, b // n), sim_s=time.perf_counter() - ts)
            rec["total_s"] = time.perf_counter() - t0
            with STORE.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"  L{level} scenario {s:3d}: n {n}, eval budget {b_eval}, wall budget {b_wall} "
                  f"({'capped' if b_wall < b_wall_uncapped else 'uncapped'}); SA P {rec['sa_eval']['p_ground']:.3f} / "
                  f"{rec['sa_wall']['p_ground']:.3f}, tabu P {rec['tabu_eval']['p_ground']:.3f} / "
                  f"{rec['tabu_wall']['p_ground']:.3f}; QAOA C-milp P {q[0]:.3f} ({rec['total_s']:.0f} s)", flush=True)


def report():
    from scipy import stats

    recs = [json.loads(l) for l in STORE.read_text().splitlines()]
    lines = []
    P = lambda s="": lines.append(s)  # noqa: E731
    P("G4: CLASSICAL SAMPLERS ON THE IDENTICAL QUBO vs QAOA (p = 1, eps = 0.1; per-scenario means; 95% t CI over "
      f"scenarios). R = {R} independent replicas per (scenario, method, budget).")
    P("  P(ground) = probability that the method's returned (best) state is an exact ground state: for QAOA the "
      "best of 1024 samples, for SA / tabu the best state of one run. gap = repaired operating-cost gap vs MILP, %.")
    for level in (2, 4):
        rs = [r for r in recs if r["level"] == level]
        if not rs:
            continue

        def ci(v):
            v = np.asarray(v, float)
            h = stats.t.ppf(0.975, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else np.nan
            return f"{v.mean():.3f} [{v.mean() - h:.3f}, {v.mean() + h:.3f}]"

        P(f"\n--- L{level}: n = {rs[0]['n']} qubits, {len(rs)} scenarios ---")
        P(f"  {'method':<34} | {'P(ground)':>22} | {'gap %':>6} | {'budget (objective evaluations)':>32} | time")
        for arm in ("C-milp", "D", "A"):
            if all(arm in r["qaoa"] for r in rs):
                v = [r["qaoa"][arm] for r in rs]
                P(f"  {'QAOA ' + arm + ' (best of 1024)':<34} | {ci([x[0] for x in v]):>22} | "
                  f"{100 * np.mean([x[1] for x in v]):6.2f} | {np.mean([x[2] for x in v]):12.0f} evals + 1024 samples "
                  f"| {np.mean([x[3] for x in v]):7.2f} s simulated")
        for meth in ("sa", "tabu"):
            for b in ("eval", "wall"):
                v = [r[f"{meth}_{b}"] for r in rs]
                budget = np.mean([r[f"b_{b}"] for r in rs])
                capped = sum(r["b_wall"] < r["b_wall_uncapped"] for r in rs) if b == "wall" else 0
                tm = (f"{1e3 * np.mean([x['chain_s'] for x in v]):7.1f} ms per run" if meth == "sa"
                      else f"{np.mean([x['sim_s'] for x in v]) / R * 1e3:7.1f} ms per run")
                P(f"  {meth.upper() + (' (= QAOA evaluations)' if b == 'eval' else ' (= QAOA wall time)'):<34} | "
                  f"{ci([x['p_ground'] for x in v]):>22} | {100 * np.mean([x['gap'] for x in v]):6.2f} | "
                  f"{budget:12.0f} flip evaluations{(' (cap hit in %d)' % capped) if capped else '':<14} | {tm}")
        P(f"  SA throughput of this implementation: {np.mean([r['sa_flips_per_s'] for r in rs]):,.0f} flips/s per chain; "
          f"QAOA wall time per run {np.mean([r['qaoa_wall_s'] for r in rs]):.2f} s")
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def rerun_tabu():
    """Recompute only the tabu fields of every stored record, with the stored budgets (SA is untouched)."""
    recs = [json.loads(l) for l in STORE.read_text().splitlines()]
    pbs = {lv: dict(problems(lv)) for lv in (2, 4)}
    for rec in recs:
        pb = pbs[rec["level"]][rec["scenario"]]()
        h, Js = coupling(pb.ev)
        rng = np.random.default_rng([rec["level"], rec["scenario"], 0x7AB])
        for name in ("eval", "wall"):
            b = rec[f"b_{name}"]
            ts = time.perf_counter()
            Zt, Et = tabu(h, Js, max(1, b // rec["n"]), rng)
            rec[f"tabu_{name}"] = dict(zip(("p_ground", "gap"), evaluate(pb, Zt, Et)), steps=max(1, b // rec["n"]),
                                       sim_s=time.perf_counter() - ts)
        print(f"  L{rec['level']} scenario {rec['scenario']:3d}: tabu P {rec['tabu_eval']['p_ground']:.3f} / "
              f"{rec['tabu_wall']['p_ground']:.3f}", flush=True)
    STORE.write_text("".join(json.dumps(r) + "\n" for r in recs))


if __name__ == "__main__":
    if "--tabu-only" in sys.argv:
        rerun_tabu()
    elif "--report" not in sys.argv:
        run()
    report()
