"""G2: do wrong LP pins grow with problem SIZE, or with commitment GRANULARITY?

The ladder L0..L4 adds variables by splitting stations into smaller units,
which is exactly what adds the small marginal units that attract wrong
pins, so the L0..L4 census cannot separate the two. Here the HORIZON is
varied at fixed granularity -- the same blocks, no new marginal units, more
variables:

    L0 (5 blocks) x T = 1, 2, 3, 4  ->  5, 10, 15, 20 variables
    L1 (6 blocks) x T = 1, 2, 3     ->  6, 12, 18
    L2 (8 blocks) x T = 1, 2        ->  8, 16

and the granularity axis is the T = 2 row L0..L4 (10..22 variables, from
results/s9_census_ownpoint.jsonl). Every cell is at its OWN calibrated
operating point, chosen by the unchanged s2/s7 procedure (100 selection
scenarios, seed 0; lambda x bias grid; plateau rule k = 1) -- for this
analysis only; the frozen point does not change. T = 2 cells reuse the
s7_gap_ladder.csv points. Census: the same 300 demand draws (seed 9300:
identical peaks at every T; the hourly profile follows T), exact ground
state, LP at the QUBO target, wrong pins as in s9_census.py.

    python scripts/s9_horizon.py           # -> results/s9_horizon_points.csv, s9_horizon_census.jsonl, s9_horizon.txt
    python scripts/s9_horizon.py --report
"""

import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
CELLS = ((0, 1), (0, 2), (0, 3), (0, 4), (1, 1), (1, 2), (1, 3), (2, 1), (2, 2))
SELECT = (100, 0)
LAMBDAS = np.logspace(0, 3.5, 29)
BIASES = np.array([0, 25, 50, 75, 100, 150, 200, 250, 300, 400], dtype=float)
CENSUS_SEED, N = 9300, 300
PIN_TAU = 0.1
WORKERS = 4
POINTS = ROOT / "results" / "s9_horizon_points.csv"
STORE = ROOT / "results" / "s9_horizon_census.jsonl"
REPORT = ROOT / "results" / "s9_horizon.txt"
THREAD_ENV = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
_SETS = {}


def calibrate():
    """Own (lambda, B) per cell; T = 2 from s7_gap_ladder.csv, the rest by the s7 procedure. Cached."""
    from qguc import make_scenario_set
    from qguc.calibrate import select_operating_point, sweep_grid_parallel

    pts = {}
    if POINTS.exists():
        with POINTS.open() as f:
            pts = {(int(r["level"]), int(r["T"])): r for r in csv.DictReader(f)}
    with open(ROOT / "results" / "s7_gap_ladder.csv") as f:
        ladder = {int(r["level"][1:]): r for r in csv.DictReader(f)}
    for lv, T in CELLS:
        if (lv, T) in pts:
            continue
        t0 = time.perf_counter()
        if T == 2:
            lam, bias, src = float(ladder[lv]["own_lambda"]), float(ladder[lv]["own_bias"]), "s7_gap_ladder.csv"
        else:
            sel = make_scenario_set(SELECT[0], n_hours=T, seed=SELECT[1], refinement_level=lv)
            grid = sweep_grid_parallel(sel, LAMBDAS, BIASES, WORKERS)
            chosen = select_operating_point(grid, plateau=1)
            if chosen is None:
                print(f"  L{lv} T={T}: NO fully adequate (lambda, B) cell -- cell skipped", flush=True)
                pts[(lv, T)] = dict(level=lv, T=T, own_lambda="", own_bias="", source="no adequate cell")
                continue
            b, li = chosen
            lam, bias, src = float(grid.lambdas[li]), float(grid.biases[b]), "calibrated (s7 procedure)"
        pts[(lv, T)] = dict(level=lv, T=T, own_lambda=lam, own_bias=bias, source=src)
        print(f"  L{lv} T={T}: lambda {lam:.4g}, B {bias:.0f} MW ({src}, {time.perf_counter() - t0:.0f} s)", flush=True)
        with POINTS.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["level", "T", "own_lambda", "own_bias", "source"])
            w.writeheader()
            w.writerows(pts[k] for k in sorted(pts))
    return pts


def scenario(level, T, k):
    from qguc import make_scenario_set

    if (level, T) not in _SETS:
        _SETS[(level, T)] = make_scenario_set(N, n_hours=T, seed=CENSUS_SEED, refinement_level=level)
    return _SETS[(level, T)][k]


def census_one(args):
    level, T, k, lam, bias = args
    os.environ.update(THREAD_ENV)
    from qguc.config import OperatingPoint
    from qguc.exact import index_to_bits, solve_lp_relaxation
    from qguc.qaoa import make_problem

    pb = make_problem(scenario(level, T, k), OperatingPoint(model=f"L{level}", lambda_demand=lam,
                                                            demand_bias_mw=bias, calibrated_n_hours=T))
    inst = pb.instance
    c = solve_lp_relaxation(inst, target="qubo").x
    G = index_to_bits(pb.ev.ground, pb.n).astype(int)
    pinned = (c <= PIN_TAU) | (c >= 1 - PIN_TAU)
    W = [(pinned & (np.round(c) != g)) for g in G]
    wrong = W[int(np.argmin([w.sum() for w in W]))]
    idx = inst.unflatten(np.arange(pb.n))
    pos = {int(idx[i, t]): (i, t) for i in range(inst.n_plants) for t in range(inst.n_hours)}
    return dict(level=level, T=T, scenario=k, n=pb.n, blocks=inst.n_plants, peak=inst.meta["peak_mw"],
                lambda_demand=lam, bias=bias, n_ground=len(G), n_pinned=int(pinned.sum()), n_wrong=int(wrong.sum()),
                wrong_at=[[inst.plants[pos[j][0]].name, pos[j][1]] for j in np.flatnonzero(wrong)])


def load(path=STORE):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def run(pts):
    done = {(r["level"], r["T"], r["scenario"]) for r in load()}
    todo = [(lv, T, k, float(pts[(lv, T)]["own_lambda"]), float(pts[(lv, T)]["own_bias"]))
            for lv, T in CELLS if pts[(lv, T)]["own_lambda"] != "" for k in range(N) if (lv, T, k) not in done]
    print(f"census: {len(done)} done, {len(todo)} to do", flush=True)
    if not todo:
        return
    t0 = time.perf_counter()
    with ProcessPoolExecutor(WORKERS) as pool, STORE.open("a") as f:
        for i, rec in enumerate(pool.map(census_one, todo, chunksize=8), 1):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 300 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}, {time.perf_counter() - t0:.0f} s", flush=True)


def report(pts):
    from scipy import stats

    rng = np.random.default_rng(0)
    H = load()
    G = [dict(r, T=2) for r in load(ROOT / "results" / "s9_census_ownpoint.jsonl")]  # granularity axis, T = 2
    cells = {}
    for r in H + [g for g in G if g["level"] in (3, 4)]:
        cells.setdefault((r["level"], r["T"]), {})[r["scenario"]] = r
    lines = []
    P = lambda s="": lines.append(s)  # noqa: E731
    P(f"G2: SIZE vs GRANULARITY. Wrong LP pins (pinned c <= 0.1 or >= 0.9 disagreeing with the exact ground state), "
      f"{N} demand draws per cell (seed {CENSUS_SEED}), every cell at its own calibrated operating point.")
    P("")
    P(f"{'cell':>8} {'blocks':>6} {'T':>2} {'n':>3} | {'lambda':>7} {'B':>4} | {'P(wrong >= 2)':>22} | {'mean wrong':>19} "
      f"| {'wrong per hour':>14} | {'0':>5} {'1':>5} {'2':>5} {'3+':>5} | {'pinned':>6}")
    for (lv, T) in sorted(cells):
        rs = list(cells[(lv, T)].values())
        w = np.array([r["n_wrong"] for r in rs])
        k2 = int((w >= 2).sum())
        p = k2 / len(w)
        z = 1.96
        d = 1 + z * z / len(w)
        c = (p + z * z / (2 * len(w))) / d
        h = z * np.sqrt(p * (1 - p) / len(w) + z * z / (4 * len(w) ** 2)) / d
        bs = w[rng.integers(0, len(w), (5000, len(w)))].mean(1)
        P(f"{'L' + str(lv) + ' T' + str(T):>8} {rs[0].get('blocks', rs[0]['n'] // T):6d} {T:2d} {rs[0]['n']:3d} | "
          f"{rs[0].get('lambda_demand', float('nan')):7.3g} {rs[0].get('bias', float('nan')):4.0f} | "
          f"{p:6.3f} [{c - h:.3f}, {c + h:.3f}] | {w.mean():5.2f} [{np.percentile(bs, 2.5):.2f}, "
          f"{np.percentile(bs, 97.5):.2f}] | {w.mean() / T:14.2f} | "
          + " ".join(f"{v:5.2f}" for v in (np.mean(w == 0), np.mean(w == 1), np.mean(w == 2), np.mean(w >= 3)))
          + f" | {np.mean([r['n_pinned'] for r in rs]):6.1f}")

    def axis(name, keys):
        keys = [k for k in keys if k in cells]
        common = sorted(set.intersection(*[set(cells[k]) for k in keys]))
        n = np.array([cells[k][common[0]]["n"] for k in keys], float)
        M = np.array([[cells[k][s]["n_wrong"] for k in keys] for s in common], float)

        def slopes(A):
            return (np.polyfit(n, A.mean(0), 1)[0], np.polyfit(n, (A >= 2).mean(0), 1)[0])

        s = slopes(M)
        B = np.array([slopes(M[rng.integers(0, len(common), len(common))]) for _ in range(3000)])
        P(f"  {name:<44} n {n.astype(int).tolist()}: mean wrong per added variable {s[0]:+.4f} "
          f"[{np.percentile(B[:, 0], 2.5):+.4f}, {np.percentile(B[:, 0], 97.5):+.4f}]; P(>= 2) per added variable "
          f"{s[1]:+.4f} [{np.percentile(B[:, 1], 2.5):+.4f}, {np.percentile(B[:, 1], 97.5):+.4f}]  "
          f"(paired over {len(common)} draws)")
        return s

    P("")
    P("TRENDS (slope of the cell means against n; 95% CI by bootstrap over the demand draws, paired across cells):")
    P(" granularity axis (T = 2, finer blocks):")
    g = axis("L0..L4, T = 2", [(lv, 2) for lv in range(5)])
    axis("L0..L2, T = 2 (the levels with a horizon axis)", [(lv, 2) for lv in range(3)])
    P(" horizon axis (fixed blocks, more hours):")
    hz = [axis(f"L{lv}, T = 1..{Tm}", [(lv, T) for T in range(1, Tm + 1)]) for lv, Tm in ((0, 4), (1, 3), (2, 2))]
    P("")
    P("WRONG PINS PER HOUR (if they grow with T only because there are more hours, this stays flat along the "
      "horizon axis):")
    for lv, Tm in ((0, 4), (1, 3), (2, 2)):
        P(f"  L{lv}: " + ", ".join(f"T={T} {np.mean([r['n_wrong'] for r in cells[(lv, T)].values()]) / T:.2f}"
                                   for T in range(1, Tm + 1) if (lv, T) in cells))
    P("  T = 2 granularity axis: " + ", ".join(
        f"L{lv} {np.mean([r['n_wrong'] for r in cells[(lv, 2)].values()]) / 2:.2f}" for lv in range(5) if (lv, 2) in cells))
    P("")
    P(f"granularity slope (L0..L4) {g[0]:+.4f} wrong pins per added variable vs horizon slopes "
      + ", ".join(f"{s[0]:+.4f}" for s in hz) + " (L0, L1, L2)")
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    pts = calibrate()
    if "--report" not in sys.argv:
        run(pts)
    report(pts)
