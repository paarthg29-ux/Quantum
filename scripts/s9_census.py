"""Wrong-pin census across system size (Phase 3, step 1). No QAOA.

Does the LP relaxation pin more bits wrongly as the system grows? At every
refinement level L0..L4 (5, 6, 8, 9, 11 blocks x 2 hours = 10..22 qubits)
the SAME 300 i.i.d. demand draws (seed 9300; the ladder keeps total capacity
at 4442 MW, so the demand profiles are identical across levels and the
comparison is paired) are encoded at the frozen operating point. For each:

  * exact QUBO ground state(s) by brute force (2^n energies),
  * LP relaxation at the QUBO target D_t + B (arm D's source),
  * pinned bits (raw c <= 0.1 or >= 0.9) and WRONGLY pinned bits: pinned
    with the rounded value disagreeing with the ground state (the
    degenerate ground state that disagrees least is used; arms.frozen_bits),
  * which (block, hour) the wrong pins fall on.

The L4 records are also the candidate pool for the Phase-3 scenario
selection (scripts/s9_select.py). Seed 9300 is unused elsewhere
(calibration, training 8001/8002, accuracy 9002, s7 9001, s8 9100).

    python scripts/s9_census.py           # -> results/s9_census.jsonl + results/s9_census.txt
    python scripts/s9_census.py --report  # summary only, from the stored records
    python scripts/s9_census.py --point own   # each level at its OWN calibrated (lambda, B) from
                                              # results/s7_gap_ladder.csv -> results/s9_census_ownpoint.*
                                              # (robustness: is the trend an artefact of the L2-calibrated point?)
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LEVELS, N_HOURS = (0, 1, 2, 3, 4), 2
CENSUS_SEED, N = 9300, 300
PIN_TAU = 0.1
WORKERS = 4  # 22-qubit brute force is memory-bandwidth bound: 4 processes ~ 8
OWN_POINT = "--point" in sys.argv and sys.argv[sys.argv.index("--point") + 1] == "own"
_SUFFIX = "_ownpoint" if OWN_POINT else ""
STORE = ROOT / "results" / f"s9_census{_SUFFIX}.jsonl"
REPORT = ROOT / "results" / f"s9_census{_SUFFIX}.txt"
THREAD_ENV = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
_SETS = {}


def scenario(level, k):
    from qguc import make_scenario_set

    if level not in _SETS:
        _SETS[level] = make_scenario_set(N, n_hours=N_HOURS, seed=CENSUS_SEED, refinement_level=level)
    return _SETS[level][k]


def point(level):
    """The operating point used at ``level``: the frozen L2 point, or the level's own calibrated one."""
    from qguc.config import FROZEN, OperatingPoint

    if not OWN_POINT:
        return FROZEN
    import csv

    with open(ROOT / "results" / "s7_gap_ladder.csv") as f:
        row = next(r for r in csv.DictReader(f) if r["level"] == f"L{level}")
    return OperatingPoint(model=f"L{level}", lambda_demand=float(row["own_lambda"]),
                          demand_bias_mw=float(row["own_bias"]))


def census_one(args):
    level, k = args
    os.environ.update(THREAD_ENV)
    from qguc.exact import index_to_bits, solve_lp_relaxation
    from qguc.qaoa import make_problem

    t0 = time.perf_counter()
    pb = make_problem(scenario(level, k), point(level))
    inst = pb.instance
    lp = solve_lp_relaxation(inst, target="qubo")
    c = lp.x
    G = index_to_bits(pb.ev.ground, pb.n).astype(int)
    pinned = (c <= PIN_TAU) | (c >= 1 - PIN_TAU)
    W = [(pinned & (np.round(c) != g)) for g in G]
    best = int(np.argmin([w.sum() for w in W]))
    wrong = W[best]
    idx = inst.unflatten(np.arange(pb.n))  # (blocks, hours) -> variable index
    pos = {int(idx[i, t]): (i, t) for i in range(inst.n_plants) for t in range(inst.n_hours)}
    return dict(level=level, scenario=k, n=pb.n, peak=inst.meta["peak_mw"], lambda_demand=pb.point.lambda_demand,
                bias=pb.point.demand_bias_mw,
                demand=[float(d) for d in inst.demand_mw], target=[float(d) for d in inst.target_mw],
                n_ground=len(G), ground=["".join(map(str, g)) for g in G],
                lp_pins="".join("1" if x >= 1 - PIN_TAU else "0" if x <= PIN_TAU else "." for x in c),
                n_fractional=int(((c > 1e-9) & (c < 1 - 1e-9)).sum()), n_pinned=int(pinned.sum()),
                n_wrong=int(wrong.sum()),
                wrong_at=[[inst.plants[pos[j][0]].name, pos[j][1]] for j in np.flatnonzero(wrong)],
                ground_used=best, wall_s=time.perf_counter() - t0)


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


def run():
    done = {(r["level"], r["scenario"]) for r in load()}
    todo = [(lv, k) for lv in sorted(LEVELS, reverse=True) for k in range(N) if (lv, k) not in done]
    print(f"census: {len(done)} done, {len(todo)} to do ({WORKERS} workers)", flush=True)
    if not todo:
        return
    text = STORE.read_text() if STORE.exists() else ""
    if text and not text.endswith("\n"):
        STORE.write_text(text[: text.rfind("\n") + 1])
    t0 = time.perf_counter()
    with ProcessPoolExecutor(WORKERS) as pool, STORE.open("a") as f:
        for i, rec in enumerate(pool.map(census_one, todo, chunksize=4), 1):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 100 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}, {time.perf_counter() - t0:.0f} s", flush=True)


# ---- report -------------------------------------------------------------------------


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def report():
    from scipy import stats

    from qguc.instance import refinement_fleet
    from qguc.repair import cost_per_mw_order

    R = load()
    by = {lv: sorted((r for r in R if r["level"] == lv), key=lambda r: r["scenario"]) for lv in LEVELS}
    lines = []
    P = lambda *a: lines.append(" ".join(str(x) for x in a))  # noqa: E731
    P(f"WRONG-PIN CENSUS: the same {N} i.i.d. demand draws (seed {CENSUS_SEED}) at every level, "
      + ("each level at its OWN calibrated operating point (" + ", ".join(
          f"L{lv} lambda {point(lv).lambda_demand:.3g} B {point(lv).demand_bias_mw:.0f}" for lv in LEVELS) + ")"
         if OWN_POINT else "frozen operating point (L2-calibrated: lambda 100, B 250 MW)")
      + ", LP at the QUBO target; pinned = c <= 0.1 or >= 0.9; wrong = pinned and disagreeing with the exact "
      "ground state (least-disagreeing degenerate ground state)")
    P("")
    P(f"{'level':>5} {'n':>3} {'runs':>4} | {'P(wrong >= 2)':>22} | {'mean wrong':>20} | "
      f"{'0':>6} {'1':>6} {'2':>6} {'3+':>6} | {'mean pinned':>11} {'wrong/pinned':>12} | {'#gnd > 1':>8}")
    rng = np.random.default_rng(0)
    for lv in LEVELS:
        rs = by[lv]
        if not rs:
            continue
        w = np.array([r["n_wrong"] for r in rs])
        k2 = int((w >= 2).sum())
        lo, hi = wilson(k2, len(w))
        boot = w[rng.integers(0, len(w), (10_000, len(w)))].mean(axis=1)
        dist = [np.mean(w == 0), np.mean(w == 1), np.mean(w == 2), np.mean(w >= 3)]
        pins = np.array([r["n_pinned"] for r in rs])
        P(f"{'L' + str(lv):>5} {rs[0]['n']:3d} {len(rs):4d} | {k2 / len(w):6.3f} [{lo:.3f}, {hi:.3f}] | "
          f"{w.mean():5.2f} [{np.percentile(boot, 2.5):.2f}, {np.percentile(boot, 97.5):.2f}] | "
          + " ".join(f"{d:6.3f}" for d in dist)
          + f" | {pins.mean():11.2f} {w.sum() / max(pins.sum(), 1):12.3f} | {np.mean([r['n_ground'] > 1 for r in rs]):8.3f}")
    # trend, paired over the same demand draws
    common = sorted(set.intersection(*[{r["scenario"] for r in by[lv]} for lv in LEVELS if by[lv]]))
    if len(common) >= 10 and all(by[lv] for lv in LEVELS):
        n_of = {lv: by[lv][0]["n"] for lv in LEVELS}
        Wm = np.array([[next(r["n_wrong"] for r in by[lv] if r["scenario"] == s) for lv in LEVELS] for s in common])
        ns = np.array([n_of[lv] for lv in LEVELS], float)

        def slopes(M):
            return np.polyfit(ns, M.mean(axis=0), 1)[0], np.polyfit(ns, (M >= 2).mean(axis=0), 1)[0]

        s_mean, s_frac = slopes(Wm)
        bs = np.array([slopes(Wm[rng.integers(0, len(common), len(common))]) for _ in range(5000)])
        P("")
        P(f"TREND over n (paired: {len(common)} demand draws present at every level; bootstrap over draws):")
        P(f"  mean wrong pins per qubit added: {s_mean:+.4f}  95% CI [{np.percentile(bs[:, 0], 2.5):+.4f}, "
          f"{np.percentile(bs[:, 0], 97.5):+.4f}]")
        P(f"  P(wrong >= 2) per qubit added:   {s_frac:+.4f}  95% CI [{np.percentile(bs[:, 1], 2.5):+.4f}, "
          f"{np.percentile(bs[:, 1], 97.5):+.4f}]")
        rho = stats.spearmanr(np.repeat(ns[None], len(common), 0).ravel(), Wm.ravel()).statistic
        P(f"  Spearman(n, wrong pins) over all (draw, level) pairs: {rho:+.3f}")
        for a, b in ((2, 4), (0, 4)):
            x, y = Wm[:, LEVELS.index(a)] >= 2, Wm[:, LEVELS.index(b)] >= 2
            n01, n10 = int((~x & y).sum()), int((x & ~y).sum())
            pm = stats.binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0
            P(f"  L{a} -> L{b}: draws with >= 2 wrong pins {x.sum()} -> {y.sum()} (became >= 2: {n01}, stopped: "
              f"{n10}; exact McNemar p = {pm:.2g}); mean wrong {Wm[:, LEVELS.index(a)].mean():.2f} -> "
              f"{Wm[:, LEVELS.index(b)].mean():.2f}, paired Wilcoxon p = "
              f"{stats.wilcoxon(Wm[:, LEVELS.index(b)] - Wm[:, LEVELS.index(a)]).pvalue if np.any(Wm[:, LEVELS.index(b)] != Wm[:, LEVELS.index(a)]) else 1.0:.2g}")
    # where the wrong pins fall
    P("")
    P("WHERE THE WRONG PINS FALL: per block, the fraction of draws in which it is wrongly pinned (either hour), "
      "with its capacity, merit-order rank (1 = cheapest per MW) and how often the LP pins it at all")
    for lv in LEVELS:
        rs = by[lv]
        if not rs:
            continue
        fleet = refinement_fleet(lv)
        inst = scenario(lv, 0)
        rank = {inst.plants[i].name: r + 1 for r, i in enumerate(cost_per_mw_order(inst))}
        cnt = {p.name: 0 for p in fleet}
        for r in rs:
            for name in {w[0] for w in r["wrong_at"]}:
                cnt[name] += 1
        pin_rate = {p.name: 0.0 for p in fleet}
        names = [p.name for p in inst.plants]
        idx = inst.unflatten(np.arange(inst.n_vars))
        for r in rs:
            for i, nm in enumerate(names):
                pin_rate[nm] += np.mean([r["lp_pins"][idx[i, t]] != "." for t in range(N_HOURS)]) / len(rs)
        total = sum(cnt.values())
        P(f"  L{lv} (n = {rs[0]['n']}; {total} wrong-pinned blocks in {len(rs)} draws):")
        for p in sorted(fleet, key=lambda p: -cnt[p.name]):
            P(f"    {p.name:>12} {p.capacity_mw:7.0f} MW  rank {rank[p.name]:2d}/{len(fleet)} | wrong in "
              f"{cnt[p.name] / len(rs):6.3f} of draws ({cnt[p.name] / max(total, 1):5.1%} of all wrong pins) | "
              f"pinned {pin_rate[p.name]:5.2f}")
        caps = np.array([p.capacity_mw for p in fleet])
        rates = np.array([cnt[p.name] / len(rs) for p in fleet])
        rk = np.array([rank[p.name] for p in fleet])
        P(f"    Spearman(wrong rate, capacity) = {stats.spearmanr(rates, caps).statistic:+.2f}; "
          f"Spearman(wrong rate, merit rank) = {stats.spearmanr(rates, rk).statistic:+.2f} "
          f"(over {len(fleet)} blocks; descriptive)")
    P("")
    P(f"wall time per instance (median, s): " + ", ".join(
        f"L{lv} {np.median([r['wall_s'] for r in by[lv]]):.2f}" for lv in LEVELS if by[lv]))
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    if "--report" not in sys.argv:
        run()
    report()
