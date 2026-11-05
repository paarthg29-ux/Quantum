"""Re-verify (NOT re-tune) the frozen operating point on the aggregation
ladder and measure the encoding gap vs block granularity on one physical
system (4442 MW at every level).

Same scenarios (identical demand) at every level, fresh seed never used in
calibration. If L2 adequacy < 95%, the rule is: re-calibrate ONCE on L2.
"""

import csv
import time
from pathlib import Path

import numpy as np

from qguc import REFINEMENT_LEVELS, make_scenario_set, refinement_fleet
from qguc.calibrate import sweep_grid_parallel
from qguc.config import FROZEN

N_HOURS = FROZEN.calibrated_n_hours
N_SCEN, SEED = 100, 4242  # fresh: calibration used seeds 0 (selection) and 2026 (test)
WORKERS = 8
OUT = Path(__file__).resolve().parents[1] / "results" / "s4_ladder.csv"


def main():
    lam, bias = FROZEN.lambda_demand, FROZEN.demand_bias_mw
    print(f"Frozen point: lambda_D = {lam:g}, demand_bias = {bias:g} MW, lambda_switch = {FROZEN.lambda_switch:g}; "
          f"T = {N_HOURS}; {N_SCEN} scenarios (seed {SEED}), identical demand at every level")
    print("gap = (C(QUBO ground) - C(MILP)) / C(MILP), C = fuel + start-up, over scenarios whose ground state is "
          "adequate;\n=MILP: ground state is the MILP schedule up to swapping identical blocks.\n")
    head = (f"  {'level':>5} {'blocks':>6} {'qubits':>6} | {'adequate':>8} | {'gap mean':>8} {'median':>7} "
            f"{'p90':>7} {'worst':>7} | {'=MILP':>6} | {'dynIsing':>8} {'degen':>5} | {'time':>5}")
    print(head)
    rows = []
    for k in REFINEMENT_LEVELS:
        insts = make_scenario_set(N_SCEN, n_hours=N_HOURS, seed=SEED, refinement_level=k)
        t0 = time.perf_counter()
        g = sweep_grid_parallel(insts, [lam], [bias], WORKERS)
        dt = time.perf_counter() - t0
        ad = g.adequate[0, 0]
        gaps = g.gap[0, 0][ad]
        row = dict(level=f"L{k}", blocks=len(refinement_fleet(k)), qubits=insts[0].n_vars,
                   adequacy=ad.mean(), gap_mean=gaps.mean(), gap_median=np.median(gaps),
                   gap_p90=np.percentile(gaps, 90), gap_worst=gaps.max(), match=g.match[0, 0].mean(),
                   dyn_worst=g.dyn_range[0, 0].max(), degen_max=int(g.degeneracy[0, 0].max()),
                   inadequate=[g.labels[i] for i in np.flatnonzero(~ad)])
        rows.append(row)
        print(f"  {row['level']:>5} {row['blocks']:6d} {row['qubits']:6d} | {100 * row['adequacy']:7.0f}% | "
              f"{100 * row['gap_mean']:7.2f}% {100 * row['gap_median']:6.2f}% {100 * row['gap_p90']:6.2f}% "
              f"{100 * row['gap_worst']:6.2f}% | {100 * row['match']:5.0f}% | {row['dyn_worst']:8.0f} "
              f"{row['degen_max']:5d} | {dt:4.0f}s", flush=True)
    for r in rows:
        if r["inadequate"]:
            print(f"  {r['level']} inadequate ground states: {', '.join(r['inadequate'])}")

    l2 = next(r for r in rows if r["level"] == "L2")
    verdict = "PASS" if l2["adequacy"] >= 0.95 else "FAIL -> re-calibrate once on L2"
    print(f"\nRe-verification rule (L2 adequacy >= 95%): {100 * l2['adequacy']:.0f}% -> {verdict}")

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[c for c in rows[0] if c != "inadequate"])
        w.writeheader()
        for r in rows:
            w.writerow({c: v for c, v in r.items() if c != "inadequate"})
    print(f"wrote {OUT.relative_to(OUT.parents[1])}")


if __name__ == "__main__":
    main()
