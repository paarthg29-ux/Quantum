"""G16: do the LEARNED start's wrong pins grow with the horizon, like the LP's?

Matches the G2 horizon census (s9_horizon.py): cells L0 x T = 1..4, L1 x
T = 1..3, L2 x T = 1..2, the same 300 demand draws per cell (seed 9300),
each cell at its own calibrated operating point (s9_horizon_points.csv),
wrong pins = pinned (c <= 0.1 or >= 0.9, raw c) and disagreeing with the
exact ground state (least-disagreeing degenerate ground state).

C-milp per cell: a GCN trained AT THAT HORIZON by the unchanged protocol
(MILP labels; train 300 seed 8001, validation 50 seed 8002; default
hyperparameters, no tuning) -> models/horizon/gnn_milp_L<l>_T<T>.pt. Also,
as a horizon-generalisation check, each level's T = 2 model applied
unchanged at the other horizons ("C-milp T2-model"). The LP's wrong pins
come from the same draws, so C-milp vs D is paired per draw.

    python scripts/s10_horizon_learned.py   # -> results/s10_horizon_learned.jsonl / .txt
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
MODEL_DIR = ROOT / "models" / "horizon"
STORE = ROOT / "results" / "s10_horizon_learned.jsonl"
REPORT = ROOT / "results" / "s10_horizon_learned.txt"
PIN_TAU = 0.1
WORKERS = 4
THREAD_ENV = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")


def model_path(level, T):
    return MODEL_DIR / f"gnn_milp_L{level}_T{T}.pt"


def train_all():
    import torch

    from qguc import make_scenario_set
    from qguc.config import FROZEN
    from qguc.gnn import LABELERS, dataset, train

    torch.set_num_threads(1)
    log = []
    for lv, T in CELLS:
        p = model_path(lv, T)
        if p.exists():
            continue
        mk = lambda n, s: make_scenario_set(n, n_hours=T, seed=s, refinement_level=lv,  # noqa: E731
                                            demand_bias_mw=FROZEN.demand_bias_mw)
        tr, va = mk(300, 8001), mk(50, 8002)
        t0 = time.perf_counter()
        labels = (dataset(tr, LABELERS["milp"]), dataset(va, LABELERS["milp"]))
        tl = time.perf_counter() - t0
        t0 = time.perf_counter()
        m = train(tr, va, "milp", seed=0, labels=labels)
        tt = time.perf_counter() - t0
        m.meta.update(level=lv, n_hours=T, label_s=tl, train_s=tt)
        m.save(p)
        msg = (f"  trained L{lv} T={T}: labels {tl:.1f} s, training {tt:.1f} s, best epoch {m.meta['best_epoch']} "
               f"of {m.meta['epochs_run']}, val BCE {m.meta['best_val_bce']:.4f}")
        print(msg, flush=True)
        log.append(msg)
    return log


def points():
    with open(ROOT / "results" / "s9_horizon_points.csv") as f:
        return {(int(r["level"]), int(r["T"])): (float(r["own_lambda"]), float(r["own_bias"])) for r in csv.DictReader(f)}


_MODELS = {}


def wrong(c, G):
    c = np.asarray(c, float)
    pinned = (c <= PIN_TAU) | (c >= 1 - PIN_TAU)
    return int(min((pinned & (np.round(c) != g)).sum() for g in G)), int(pinned.sum())


def one(args):
    lv, T, k, lam, bias = args
    os.environ.update(THREAD_ENV)
    import torch

    from s9_horizon import scenario

    from qguc.config import OperatingPoint
    from qguc.exact import index_to_bits, solve_lp_relaxation
    from qguc.gnn import WarmStartModel
    from qguc.qaoa import make_problem

    torch.set_num_threads(1)
    pb = make_problem(scenario(lv, T, k), OperatingPoint(model=f"L{lv}", lambda_demand=lam, demand_bias_mw=bias,
                                                         calibrated_n_hours=T))
    inst = pb.instance
    G = index_to_bits(pb.ev.ground, pb.n).astype(int)
    out = dict(level=lv, T=T, scenario=k, n=pb.n)
    out["D_wrong"], out["D_pinned"] = wrong(solve_lp_relaxation(inst, target="qubo").x, G)
    for key, (ml, mT) in (("C", (lv, T)), ("C_T2", (lv, 2))):
        if (ml, mT) not in _MODELS:
            _MODELS[(ml, mT)] = WarmStartModel.load(model_path(ml, mT))
        out[f"{key}_wrong"], out[f"{key}_pinned"] = wrong(_MODELS[(ml, mT)].predict_flat(inst), G)
    return out


def run():
    pts = points()
    done = set()
    if STORE.exists():
        done = {(r["level"], r["T"], r["scenario"]) for r in map(json.loads, STORE.read_text().splitlines())}
    todo = [(lv, T, k, *pts[(lv, T)]) for lv, T in CELLS for k in range(300) if (lv, T, k) not in done]
    print(f"census: {len(done)} done, {len(todo)} to do", flush=True)
    if not todo:
        return
    with ProcessPoolExecutor(WORKERS) as pool, STORE.open("a") as f:
        for i, rec in enumerate(pool.map(one, todo, chunksize=10), 1):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 300 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}", flush=True)


def report(train_log):
    from scipy import stats

    rng = np.random.default_rng(0)
    R = [json.loads(l) for l in STORE.read_text().splitlines()]
    cells = {}
    for r in R:
        cells.setdefault((r["level"], r["T"]), []).append(r)
    L = []
    P = lambda s="": L.append(s)  # noqa: E731
    P("G16: wrong pins of the LEARNED start (C-milp, trained at each horizon) vs the LP (D), along the horizon at "
      "fixed granularity; 300 paired demand draws per cell (seed 9300), each cell at its own calibrated point.")
    P("C T2-model = the level's T = 2 model applied unchanged (horizon generalisation).")
    P("")
    P(f"{'cell':>7} {'n':>3} | {'D mean wrong':>12} {'C mean wrong':>12} {'C(T2) wrong':>11} | {'D P(>=2)':>8} "
      f"{'C P(>=2)':>8} {'C(T2) P(>=2)':>12} | {'D per hr':>8} {'C per hr':>8} | {'D pinned':>8} {'C pinned':>8} | "
      f"{'paired C - D mean wrong [95% CI]':>32}")
    for (lv, T) in sorted(cells):
        rs = cells[(lv, T)]
        d = np.array([r["D_wrong"] for r in rs])
        c = np.array([r["C_wrong"] for r in rs])
        c2 = np.array([r["C_T2_wrong"] for r in rs])
        diff = c - d
        h = stats.t.ppf(0.975, diff.size - 1) * diff.std(ddof=1) / np.sqrt(diff.size)
        P(f"{'L' + str(lv) + ' T' + str(T):>7} {rs[0]['n']:3d} | {d.mean():12.2f} {c.mean():12.2f} {c2.mean():11.2f} | "
          f"{np.mean(d >= 2):8.3f} {np.mean(c >= 2):8.3f} {np.mean(c2 >= 2):12.3f} | {d.mean() / T:8.2f} "
          f"{c.mean() / T:8.2f} | {np.mean([r['D_pinned'] for r in rs]):8.1f} {np.mean([r['C_pinned'] for r in rs]):8.1f} "
          f"| {diff.mean():+.2f} [{diff.mean() - h:+.2f}, {diff.mean() + h:+.2f}]")

    P("")
    P("SLOPES along the horizon (per added variable; bootstrap over draws, paired across cells):")
    for lv, Tm in ((0, 4), (1, 3), (2, 2)):
        ks = [(lv, T) for T in range(1, Tm + 1)]
        common = sorted(set.intersection(*[{r["scenario"] for r in cells[k]} for k in ks]))
        n = np.array([cells[k][0]["n"] for k in ks], float)
        idx = {k: {r["scenario"]: r for r in cells[k]} for k in ks}
        for key, lab in (("D_wrong", "LP (D)"), ("C_wrong", "C-milp"), ("C_T2_wrong", "C-milp T2-model")):
            M = np.array([[idx[k][s][key] for k in ks] for s in common], float)
            sl = lambda A: (np.polyfit(n, A.mean(0), 1)[0], np.polyfit(n, (A >= 2).mean(0), 1)[0])  # noqa: E731
            s = sl(M)
            B = np.array([sl(M[rng.integers(0, len(common), len(common))]) for _ in range(3000)])
            P(f"  L{lv} {lab:<16}: mean wrong {s[0]:+.4f} [{np.percentile(B[:, 0], 2.5):+.4f}, "
              f"{np.percentile(B[:, 0], 97.5):+.4f}] per variable; P(>= 2) {s[1]:+.4f} "
              f"[{np.percentile(B[:, 1], 2.5):+.4f}, {np.percentile(B[:, 1], 97.5):+.4f}] per variable")
    P("")
    P("training log:" if train_log else "training: models already present (see models/horizon/*.pt meta)")
    for m in train_log:
        P(m)
    REPORT.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    log = train_all()
    run()
    report(log)
