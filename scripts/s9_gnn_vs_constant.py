"""GCN warm start vs a constant predictor, by probabilistic error (T4 support).

Thresholded accuracy (c > 0.5 vs label > 0.5) is crude here: labels are
symmetrised over interchangeable units, so two identical units with one
committed both get label 0.5 (and three with two committed get 0.67), and a
prediction near 0.5 is scored as a coin flip. The fair comparison is mean
absolute error and binary cross-entropy against the constant predictor that
outputs each variable's training-set mean label.

    python scripts/s9_gnn_vs_constant.py   # -> results/s9_gnn_vs_constant.txt
"""

from pathlib import Path

import numpy as np

from qguc import make_scenario_set
from qguc.config import FROZEN
from qguc.gnn import LABELERS, MODEL_DIR, WarmStartModel, dataset

OUT = Path(__file__).resolve().parents[1] / "results" / "s9_gnn_vs_constant.txt"


def bce(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main():
    import torch

    torch.set_num_threads(1)
    lines = ["GCN (C-milp) vs constant predictor (per-variable training mean), held-out 100 scenarios (seed 9002); "
             "train 300 (seed 8001)"]
    lines.append(f"  {'level':>5} {'n':>3} | {'GCN mae':>8} {'const mae':>9} | {'GCN bce':>8} {'const bce':>9} | "
                 f"{'GCN acc':>7} {'majority acc':>12} | label values")
    for L in (2, 4):
        mk = lambda n, s: make_scenario_set(n, n_hours=2, seed=s, refinement_level=L,  # noqa: E731
                                            demand_bias_mw=FROZEN.demand_bias_mw)
        tr, held = mk(300, 8001), mk(100, 9002)
        _, Ytr = dataset(tr, LABELERS["milp"])
        _, Yh = dataset(held, LABELERS["milp"])
        m = WarmStartModel.load(MODEL_DIR / f"gnn_milp_L{L}.pt")
        P = np.stack([m.predict_flat(i) for i in held])
        Yh, Ytr = Yh.reshape(P.shape), Ytr.reshape(len(tr), -1)
        base = np.broadcast_to(Ytr.mean(0), Yh.shape)
        maj = np.broadcast_to((Ytr > 0.5).mean(0) > 0.5, Yh.shape)
        lines.append(f"  {'L' + str(L):>5} {P.shape[1]:3d} | {np.abs(P - Yh).mean():8.4f} {np.abs(base - Yh).mean():9.4f} | "
                     f"{bce(P, Yh):8.4f} {bce(base, Yh):9.4f} | {((P > 0.5) == (Yh > 0.5)).mean():7.3f} "
                     f"{(maj == (Yh > 0.5)).mean():12.3f} | {sorted(set(np.round(Yh.ravel(), 2)))}")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
