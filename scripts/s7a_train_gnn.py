"""Train the GCN warm-start models (C-milp and C-oracle) at L2.

Scenario sets (disjoint seeds; none used by any calibration or benchmark):
    train 300 (seed 8001), validation 50 (seed 8002), held-out accuracy 100
    (seed 9002). The arm benchmark (s7_arms_full.py) uses seed 9001.
A model that fails the load-conditioning check is not saved.
"""

import json
import time

import numpy as np

from qguc import make_instance, make_scenario_set
from qguc.config import FROZEN
from qguc.exact import solve_lp_relaxation
from qguc.gnn import (
    LABELERS,
    MODEL_DIR,
    assert_load_conditioned,
    dataset,
    train,
)

LEVEL, N_HOURS = 2, 2
TRAIN, VAL, HELD = (300, 8001), (50, 8002), (100, 9002)
SEED = 0


def sets():
    mk = lambda n, s: make_scenario_set(n, n_hours=N_HOURS, seed=s, refinement_level=LEVEL,  # noqa: E731
                                        demand_bias_mw=FROZEN.demand_bias_mw)
    return mk(*TRAIN), mk(*VAL), mk(*HELD)


def probe_pair():
    mk = lambda peak: make_instance(peak, n_hours=N_HOURS, seed=0, refinement_level=LEVEL,  # noqa: E731
                                    demand_bias_mw=FROZEN.demand_bias_mw)
    return mk(2500.0), mk(4200.0)


def main():
    import torch

    torch.set_num_threads(1)
    tr, va, held = sets()
    print(f"L{LEVEL}: {tr[0].n_plants} blocks x {N_HOURS} h; train {len(tr)}, val {len(va)}, held-out {len(held)} "
          f"scenarios; frozen point {FROZEN}")
    names = [p.name for p in tr[0].plants]
    low, high = probe_pair()
    labelled, out = {}, {}
    for variant in ("milp", "oracle"):
        t0 = time.perf_counter()
        labelled[variant] = tuple(dataset(s, LABELERS[variant]) for s in (tr, va, held))
        print(f"\n===== C-{variant}: labels in {time.perf_counter() - t0:.1f} s =====")
        (Xtr, Ytr), (Xva, Yva), (Xh, Yh) = labelled[variant]
        t0 = time.perf_counter()
        m = train(tr, va, variant, seed=SEED, labels=((Xtr, Ytr), (Xva, Yva)))
        print(f"trained in {time.perf_counter() - t0:.1f} s: best epoch {m.meta['best_epoch']} of "
              f"{m.meta['epochs_run']}, val BCE {m.meta['best_val_bce']:.4f}")
        preds = np.stack([m.predict(i) for i in held])  # (S, N, T)
        labels = Yh.reshape(len(held), N_HOURS, -1).transpose(0, 2, 1)  # (S, N, T)
        hit = (preds > 0.5) == (labels > 0.5)
        mae = np.abs(preds - labels)
        # majority-class baseline: the most frequent TRAINING label of each (block, hour)
        ytr = Ytr.reshape(len(tr), N_HOURS, -1).transpose(0, 2, 1)
        base = (labels > 0.5) == ((ytr > 0.5).mean(axis=0) > 0.5)[None]
        print("held-out accuracy (threshold 0.5 vs label > 0.5), mean |c - label| and the majority-class "
              "baseline accuracy, per block and hour:")
        print(f"  {'block':>12} | " + " | ".join(f"h{t} acc   mae  base" for t in range(N_HOURS)) + " | all acc  base")
        for i, n in enumerate(names):
            print(f"  {n:>12} | " + " | ".join(f"{hit[:, i, t].mean():6.3f} {mae[:, i, t].mean():5.3f} "
                                               f"{base[:, i, t].mean():5.2f}" for t in range(N_HOURS))
                  + f" | {hit[:, i].mean():6.3f} {base[:, i].mean():5.2f}")
        print(f"  {'ALL':>12} | " + " | ".join(f"{hit[:, :, t].mean():6.3f} {mae[:, :, t].mean():5.3f} "
                                             f"{base[:, :, t].mean():5.2f}" for t in range(N_HOURS))
              + f" | {hit.mean():6.3f} {base.mean():5.2f}")
        cond = assert_load_conditioned(m, low, high)
        print(f"load conditioning (2500 vs 4200 MW peak): mean |dc| = {cond['mean_abs_diff']:.3f}; expected "
              f"committed MW rises by {np.round(cond['committed_mw_diff']).astype(int).tolist()} for load "
              f"+{np.round(cond['load_diff_mw']).astype(int).tolist()} -> PASS")
        tm = m.time_forward(held[0])
        print(f"timing (1 thread): features + forward {1e6 * tm['features_and_forward_s']:.0f} us, "
              f"forward only {1e6 * tm['forward_s']:.0f} us")
        m.meta.update(heldout_accuracy=float(hit.mean()), heldout_mae=float(mae.mean()), load_check=cond,
                      timing=tm, sets=dict(train=TRAIN, val=VAL, held=HELD))
        path = MODEL_DIR / f"gnn_{variant}_L{LEVEL}.pt"
        m.save(path)
        print(f"saved {path.relative_to(path.parents[1])}")
        out[variant] = dict(acc=float(hit.mean()), mae=float(mae.mean()), **tm)

    lp = [solve_lp_relaxation(i, target="qubo") for i in held[:50]]
    print(f"\nreference: LP relaxation (arm D) solve {1e3 * np.median([x.solve_time_s for x in lp]):.2f} ms median, "
          f"build + solve {1e3 * np.median([x.total_time_s for x in lp]):.2f} ms")
    ym, yo = labelled["milp"][2][1], labelled["oracle"][2][1]
    print(f"labels: mean |MILP - oracle| on held-out = {np.abs(ym - yo).mean():.3f}; fraction of (block, hour) "
          f"where they round differently = {((ym > 0.5) != (yo > 0.5)).mean():.3f}")
    (MODEL_DIR / "gnn_summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
