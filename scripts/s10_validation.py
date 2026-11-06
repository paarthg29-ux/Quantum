"""G12: simulator validation table (the evidence previously lived only in tests).

Every component of qguc.fastsim is compared against PennyLane's default.qubit,
an independent implementation, with the helpers of tests/test_fastsim.py
(variable j on wire n-1-j, states compared up to a global phase):

  1. standard QAOA (|+>^n, X mixer), random Ising, n = 4..12, p = 1..3;
  2. warm-start initial state prod RY(theta)|0>, X mixer;
  3. the ACTUAL warm-start mixer of every warm arm (qguc.warmstart.WarmStartMixer,
     c regularised at eps = 0.1), applied in PennyLane as QubitUnitary
     exp(-i beta H_M) with H_M built independently (scipy expm);
  4. a real unit-commitment instance (L0, 10 qubits, frozen point): the
     normalised Ising Hamiltonian from the QUBO, energy vector and circuit.

Plus the exact best-of-K formula against Monte Carlo (40,000 trials,
K = 1, 8, 64): the largest |frequency - exact| in units of its standard error.

    python scripts/s10_validation.py   # -> results/s10_simulator_validation.txt
"""

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
OUT = ROOT / "results" / "s10_simulator_validation.txt"


def state_err(a, b):
    phase = np.vdot(a, b)
    return float(np.abs(a * phase / abs(phase) - b).max())


def main():
    import pennylane as qml
    import test_fastsim as T

    from qguc import make_scenario_set
    from qguc.fastsim import QAOASimulator, ising_energy_vector, ry_product_state, sample, uniform_state
    from qguc.qaoa import make_problem
    from qguc.sampling import best_of_k_weights
    from qguc.warmstart import WarmStartMixer, mixer_hamiltonians, regularise, thetas_from_c

    rows = []
    for n in (4, 6, 8, 10, 12):
        for p in (1, 2, 3):
            rng = np.random.default_rng(1000 + 10 * n + p)
            h, J = T.random_ising(n, rng)
            H = T.pl_hamiltonian(n, h, J)
            gam, bet = rng.uniform(-np.pi, np.pi, p), rng.uniform(-np.pi, np.pi, p)
            E = ising_energy_vector(h, J)
            # 1. standard
            ref, ref_e = T.run_pl(n, lambda: [qml.Hadamard(w) for w in range(n)], T.standard_layers(n, H, gam, bet), H)
            sim = QAOASimulator(E)
            e1 = (state_err(sim.state(gam, bet), ref), abs(sim.energy(gam, bet) - ref_e))
            # 2. warm initial state, X mixer
            c = regularise(rng.uniform(0, 1, n), 0.1)
            th = thetas_from_c(c)
            prep = lambda: [qml.RY(th[j], wires=T.wire(n, j)) for j in range(n)]  # noqa: E731
            ref, ref_e = T.run_pl(n, prep, T.standard_layers(n, H, gam, bet), H)
            sim = QAOASimulator(E, initial_state=ry_product_state(th))
            e2 = (state_err(sim.state(gam, bet), ref), abs(sim.energy(gam, bet) - ref_e))
            # 3. actual warm-start mixer
            HM = mixer_hamiltonians(c)

            def layers():
                for g, b in zip(gam, bet):
                    qml.qaoa.cost_layer(g, H)
                    for j in range(n):
                        qml.QubitUnitary(expm(-1j * b * HM[j]), wires=T.wire(n, j))

            ref, ref_e = T.run_pl(n, prep, layers, H)
            sim = QAOASimulator(E, initial_state=ry_product_state(th), mixer=WarmStartMixer(c))
            e3 = (state_err(sim.state(gam, bet), ref), abs(sim.energy(gam, bet) - ref_e))
            rows.append((n, p, e1, e2, e3))
    # 4. real UC instance
    pb = make_problem(make_scenario_set(1, n_hours=2, seed=3, refinement_level=0)[0])
    n = pb.n
    h, J = pb.ev.ising.h, pb.ev.ising.J
    H = T.pl_hamiltonian(n, h, J)
    e_vec = float(np.abs(ising_energy_vector(h, J) - pb.ev.E).max())
    uc = []
    for p in (1, 2, 3):
        rng = np.random.default_rng(7 + p)
        gam, bet = rng.uniform(-1, 1, p), rng.uniform(-np.pi / 4, np.pi / 4, p)
        ref, ref_e = T.run_pl(n, lambda: [qml.Hadamard(w) for w in range(n)], T.standard_layers(n, H, gam, bet), H)
        sim = QAOASimulator(pb.ev.E)
        uc.append((p, state_err(sim.state(gam, bet), ref), abs(sim.energy(gam, bet) - ref_e)))
    # 5. best-of-K vs Monte Carlo
    mc = []
    for K in (1, 8, 64):
        rng = np.random.default_rng(K)
        nn = 5
        psi = rng.normal(size=1 << nn) + 1j * rng.normal(size=1 << nn)
        psi /= np.linalg.norm(psi)
        probs = np.abs(psi) ** 2
        Ev = rng.permutation(1 << nn).astype(float)
        order = np.argsort(Ev, kind="stable")
        w = best_of_k_weights(probs[order], K)
        trials = 40_000
        shots = sample(psi, trials * K, rng).reshape(trials, K)
        best = shots[np.arange(trials), np.argmin(Ev[shots], axis=1)]
        freq = np.bincount(best, minlength=1 << nn)[order] / trials
        sd = np.sqrt(np.maximum(w * (1 - w), 1e-300) / trials)
        z = np.abs(freq - w) / np.where(sd > 0, sd, np.inf)
        mc.append((K, float(z.max()), float(np.abs(freq - w).max())))

    L = ["SIMULATOR VALIDATION: qguc.fastsim vs PennyLane default.qubit (independent implementation). Max over the "
         "2^n amplitudes of |psi_fast - psi_PL| (global phase removed) and |<H>_fast - <H>_PL|; random Ising "
         "(30% couplings zeroed), random angles in [-pi, pi]."]
    L.append(f"  {'n':>3} {'p':>2} | {'standard QAOA: state, energy':>30} | {'warm state + X mixer':>24} | "
             f"{'warm state + warm mixer (arms C, D, F)':>40}")
    for n, p, e1, e2, e3 in rows:
        L.append(f"  {n:3d} {p:2d} | {e1[0]:14.1e} {e1[1]:14.1e} | {e2[0]:11.1e} {e2[1]:11.1e} | "
                 f"{e3[0]:19.1e} {e3[1]:19.1e}")
    worst = max(max(e1 + e2 + e3) for *_, e1, e2, e3 in rows)
    L.append(f"  worst over all {len(rows) * 3} cases: {worst:.1e} (test tolerance 1e-10)")
    L.append("")
    L.append(f"REAL UNIT-COMMITMENT INSTANCE (L0, {pb.n} qubits, frozen point): energy vector from the normalised Ising "
             f"model vs the brute-force QUBO energies max |diff| = {e_vec:.1e}; circuit vs PennyLane:")
    for p, es, ee in uc:
        L.append(f"  p = {p}: state {es:.1e}, energy {ee:.1e}")
    L.append("")
    L.append("EXACT BEST-OF-K vs MONTE CARLO (40,000 trials of K samples; 5-qubit random state, distinct energies):")
    for K, z, d in mc:
        L.append(f"  K = {K:3d}: max |frequency - exact| = {d:.4f}, = {z:.2f} standard errors (largest over 32 states)")
    L.append("")
    L.append("complex64 vs complex128 (used at 22 qubits): max approximation-ratio error 2.6e-7 (results/s3b_fastsim_timing.txt)")
    L.append("Source: tests/test_fastsim.py and tests/test_sampling_repair_arms.py run the same checks as unit tests.")
    OUT.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
