"""Smoke test of QAOA arm A (cold start) on the fast simulator.

One scenario, seeds {0, 1}, p in {1, 2}, at n = 10 (L0) and n = 16 (L2),
frozen operating point and optimiser settings. Checks:

1. the energy actually DECREASES during optimisation;
2. the best approximation ratio is well above the random-state baseline
   (the uniform superposition = a uniformly random bitstring): it must close
   at least half of the distance from the baseline to 1;
3. p = 1 lands in a sane range: above the baseline, below 1, and without
   concentrating on the ground state (p_ground < 0.5).
"""

import sys

import numpy as np

from qguc import make_scenario_set
from qguc.config import FROZEN
from qguc.qaoa import OptimiserSettings, arm_a, make_problem

SCEN_SEED, SCENARIO = 4242, 0
SEEDS = (0, 1)
P_VALUES = (1, 2)
LEVELS = (0, 2)
CHECKPOINTS = (1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1000, 1500)


def trace(r):
    best_e = np.minimum.accumulate(r.energies)
    best_r = np.maximum.accumulate(r.ratios)
    pts = [c for c in CHECKPOINTS if c < r.n_evals] + [r.n_evals]
    print(f"      {'eval':>5} {'<H_C>':>9} {'best <H_C>':>10} {'ratio':>7} {'best ratio':>10}")
    for c in pts:
        print(f"      {c:5d} {r.energies[c - 1]:9.4f} {best_e[c - 1]:10.4f} {r.ratios[c - 1]:7.4f} {best_r[c - 1]:10.4f}")


def main():
    s = OptimiserSettings()
    print(f"Frozen point {FROZEN}; optimiser {s}")
    failures = []
    for lvl in LEVELS:
        inst = make_scenario_set(SCENARIO + 1, n_hours=2, seed=SCEN_SEED, refinement_level=lvl)[SCENARIO]
        pb = make_problem(inst)
        base = pb.uniform_ratio
        print(f"\n===== L{lvl}: {inst.n_plants} blocks x {inst.n_hours} h = {pb.n} qubits; {inst.label}; "
              f"demand {np.round(inst.demand_mw).astype(int).tolist()} MW =====")
        print(f"  E (normalised) in [{pb.ev.e_min:.3f}, {pb.ev.e_max:.3f}], {len(pb.ev.ground)} ground state(s), "
              f"sigma {pb.ev.sigma:.3f}; random-state baseline: <H_C> = {pb.ev.E.mean():.4f}, ratio {base:.4f}; "
              f"MILP cost {pb.milp.operating_cost:,.0f} INR")
        results = {}
        for p in P_VALUES:
            for seed in SEEDS:
                r = arm_a(pb, p, seed)
                results[(p, seed)] = r
                print(f"\n  -- p={p} seed={seed}: {r.n_evals} evals, {r.wall_s:.2f} s, stop: {r.stop_reason}")
                print(f"     gamma = {np.round(r.gammas, 4).tolist()}, beta = {np.round(r.betas, 4).tolist()}")
                trace(r)
        print(f"\n  {'p':>2} {'seed':>4} | {'E init':>8} {'E best':>8} {'decr.':>5} | {'ratio':>6} | "
              f"{'g init':>6} {'g final':>7} | {'evals to g=.9/.95/.99':>21} | {'cap':>3} | {'P(gnd)':>7} | "
              f"best-of-1024: {'P(gnd)':>6} {'adeq':>5} {'gap raw':>7} {'repaired':>8}")
        for p in P_VALUES:
            for seed in SEEDS:
                r = results[(p, seed)]
                dec = r.energies.min() < r.energies[0] - 1e-9
                ev = "/".join(str(r.evals_to_g(t) or "-") for t in (0.9, 0.95, 0.99))
                b = r.sampling[1024]
                print(f"  {p:2d} {seed:4d} | {r.energies[0]:8.4f} {r.energies.min():8.4f} {'yes' if dec else 'NO':>5} | "
                      f"{r.best_ratio:6.4f} | {r.g_init:6.3f} {r.g_final:7.3f} | {ev:>21} | "
                      f"{'yes' if r.cap_hit else 'no':>3} | {r.p_ground:7.4f} | {b['p_ground']:19.4f} "
                      f"{b['adequate']:5.2f} {100 * b['gap_raw']:6.2f}% {100 * b['gap_repaired']:7.2f}%")
                if not dec:
                    failures.append(f"L{lvl} p={p} seed={seed}: energy did not decrease")
                if r.g_final < 0.5:
                    failures.append(f"L{lvl} p={p} seed={seed}: g = {r.g_final:.3f} < 0.5")
                if p == 1 and not (base < r.best_ratio < 1 and r.p_ground < 0.5):
                    failures.append(f"L{lvl} p=1 seed={seed}: outside the sane range")
        print("  g = (r - r_unif) / (1 - r_unif): share of the distance from the random-state ratio to 1")

    print("\nSMOKE TEST:", "PASS" if not failures else "FAIL")
    for f in failures:
        print("  -", f)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
