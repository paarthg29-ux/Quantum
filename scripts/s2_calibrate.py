"""Step 2: station (5 blocks) vs unit (11 blocks) model, swept over lambda_D and
demand bias B at T = 2, measuring the encoding gap. Selects one operating
point per model on a selection set and validates it on an untouched test set.

    python scripts/s2_calibrate.py            # station and unit models
    python scripts/s2_calibrate.py L2         # any models / ladder levels
"""

import sys
import time

import numpy as np

from qguc import build_qubo, make_scenario_set
from qguc.calibrate import cross_check, select_operating_point, sweep_grid, sweep_grid_parallel
from qguc.exact import brute_force

WORKERS = 8  # scenario-parallel processes; enumeration s/scenario below is per busy worker

N_HOURS = 2
SELECT = (100, 0)  # (n_scenarios, seed): chooses the operating point
TEST = (40, 2026)  # fresh; evaluated once, after the selection rule was fixed
PLATEAU = 1  # frozen rule: the neighbouring lambdas must also be fully adequate
LAMBDAS = np.logspace(0, 3.5, 29)  # 1 .. 3162 INR/MW^2, 8 per decade
BIASES = np.array([0, 25, 50, 75, 100, 150, 200, 250, 300, 400], dtype=float)
MODELS = tuple(sys.argv[1:]) or ("station", "unit")


def pct(v):
    return "   -  " if np.isnan(v) else f"{100 * v:6.2f}"


def time_brute_force_22():
    inst = make_scenario_set(1, n_hours=N_HOURS, seed=0, plants="unit")[0]
    q = build_qubo(inst, 100.0)
    runs = [brute_force(q).elapsed_s for _ in range(3)]
    spec = brute_force(q, return_spectrum=True).elapsed_s
    print(f"Brute force, unit model T={N_HOURS}: n = {q.n_vars} vars, 2^{q.n_vars} = {1 << q.n_vars:,} states: "
          f"{min(runs):.2f} s (best of 3; runs {', '.join(f'{r:.2f}' for r in runs)}); "
          f"with sorted spectrum {spec:.2f} s; energy vector {8 * (1 << q.n_vars) / 2**20:.0f} MB")


def per_bias_table(grid):
    print(f"  {'bias':>5} {'min lam 100%':>12} | best fully adequate cell: {'lam':>7} {'gap mean%':>9} "
          f"{'gap worst%':>10} {'=MILP%':>6} {'dynIsing':>9}")
    for b, bias in enumerate(grid.biases):
        full = np.flatnonzero(grid.adequacy_rate[b] == 1.0)
        if full.size == 0:
            best_ad = grid.adequacy_rate[b].max()
            print(f"  {bias:5.0f} {'none':>12} | (max adequacy over lambda {100 * best_ad:.0f}%)")
            continue
        li = min(full, key=lambda i: (grid.gap_worst[b, i], grid.gap_mean[b, i], grid.dyn_worst[b, i]))
        print(f"  {bias:5.0f} {grid.lambdas[full[0]]:12.3g} | {'':>26}{grid.lambdas[li]:7.3g} "
              f"{pct(grid.gap_mean[b, li]):>9} {pct(grid.gap_worst[b, li]):>10} "
              f"{100 * grid.match_rate[b, li]:6.0f} {grid.dyn_worst[b, li]:9.3g}")


def lambda_table(grid, b):
    print(f"  lambda sweep at bias {grid.biases[b]:.0f} MW:")
    print(f"  {'lambda':>8} {'adeq%':>6} {'gap mean%':>9} {'gap worst%':>10} {'=MILP%':>6} {'dynIsing':>9} {'degen':>6}")
    for li, lam in enumerate(grid.lambdas):
        print(f"  {lam:8.3g} {100 * grid.adequacy_rate[b, li]:6.0f} {pct(grid.gap_mean[b, li]):>9} "
              f"{pct(grid.gap_worst[b, li]):>10} {100 * grid.match_rate[b, li]:6.0f} "
              f"{grid.dyn_worst[b, li]:9.3g} {grid.degeneracy[b, li].max():6d}")


def main():
    print(f"T = {N_HOURS}; selection {SELECT[0]} scenarios (seed {SELECT[1]}), test {TEST[0]} (seed {TEST[1]}); "
          f"lambda_D {len(LAMBDAS)} points in [{LAMBDAS[0]:g}, {LAMBDAS[-1]:.0f}]; biases {BIASES.astype(int).tolist()} MW")
    print("encoding gap = (C(QUBO ground) - C(MILP)) / C(MILP), C = fuel + start-up, MILP enforces gen >= D;\n"
          "gap statistics are over scenarios whose ground state is adequate. =MILP% compares schedules up to\n"
          "swapping identical units. dynIsing = max|c| / min nonzero |c| over (h, J), worst scenario.\n")
    time_brute_force_22()

    rng = np.random.default_rng(0)
    chosen = {}
    for model in MODELS:
        cal = make_scenario_set(SELECT[0], n_hours=N_HOURS, seed=SELECT[1], plants=model)
        t0 = time.perf_counter()
        grid = sweep_grid_parallel(cal, LAMBDAS, BIASES, WORKERS)
        sweep_s = time.perf_counter() - t0
        n_checks = 6 if cal[0].n_vars > 20 else 20
        bf_mean = cross_check(cal, grid, n_checks, rng)
        print(f"\n===== {model} model: {cal[0].n_plants} blocks x {N_HOURS} h = {cal[0].n_vars} vars =====")
        print(f"sweep {sweep_s:.0f} s (enumeration {grid.enum_times.mean():.2f} s/scenario); "
              f"MILP median {1e3 * np.median(grid.milp_times):.0f} ms, max {1e3 * grid.milp_times.max():.0f} ms; "
              f"cross-check: {n_checks} cells re-solved by full brute force ({bf_mean:.2f} s each), all identical")
        per_bias_table(grid)
        held = make_scenario_set(TEST[0], n_hours=N_HOURS, seed=TEST[1], plants=model)
        for plateau in (PLATEAU, 0):
            cell = select_operating_point(grid, plateau=plateau)
            if cell is None:
                print(f"  plateau={plateau}: no qualifying cell")
                continue
            b, li = cell
            if plateau == PLATEAU:
                lambda_table(grid, b)
            bias, lam = grid.biases[b], grid.lambdas[li]
            hg = sweep_grid(held, [lam], [bias])
            tag = "frozen rule" if plateau == PLATEAU else "reference, no plateau"
            chosen[(model, plateau)] = dict(bias=bias, lam=lam, cal=(grid, b, li), held=hg, n=cal[0].n_vars)
            print(f"  SELECTED [{tag}, plateau={plateau}]: lambda_D = {lam:.4g}, bias = {bias:.0f} MW -> adequacy "
                  f"{100 * grid.adequacy_rate[b, li]:.0f}%, gap mean {pct(grid.gap_mean[b, li]).strip()}%, "
                  f"worst {pct(grid.gap_worst[b, li]).strip()}%, =MILP {100 * grid.match_rate[b, li]:.0f}%, "
                  f"dynIsing {grid.dyn_worst[b, li]:.3g}")
            print(f"    TEST ({TEST[0]} fresh scenarios): adequacy {100 * hg.adequacy_rate[0, 0]:.0f}%, "
                  f"gap mean {pct(hg.gap_mean[0, 0]).strip()}%, worst {pct(hg.gap_worst[0, 0]).strip()}%, "
                  f"=MILP {100 * hg.match_rate[0, 0]:.0f}%, dynIsing {hg.dyn_worst[0, 0]:.3g}")

    print("\n===== comparison at each model's selected point =====")
    print(f"  {'model':>8} {'rule':>10} {'vars':>5} {'lambda':>7} {'bias':>5} | {'sel adeq':>8} {'mean%':>6} "
          f"{'worst%':>7} {'=MILP%':>6} | {'test adeq':>9} {'mean%':>6} {'worst%':>7} {'=MILP%':>6} | {'dynIsing':>8}")
    for (model, plateau), c in chosen.items():
        g, b, li = c["cal"]
        h = c["held"]
        print(f"  {model:>8} {'plateau=' + str(plateau):>10} {c['n']:5d} "
              f"{c['lam']:7.3g} {c['bias']:5.0f} | "
              f"{100 * g.adequacy_rate[b, li]:7.0f}% {pct(g.gap_mean[b, li])} {pct(g.gap_worst[b, li]):>7} "
              f"{100 * g.match_rate[b, li]:6.0f} | {100 * h.adequacy_rate[0, 0]:8.0f}% {pct(h.gap_mean[0, 0])} "
              f"{pct(h.gap_worst[0, 0]):>7} {100 * h.match_rate[0, 0]:6.0f} | "
              f"{max(g.dyn_worst[b, li], h.dyn_worst[0, 0]):8.3g}")


if __name__ == "__main__":
    main()
