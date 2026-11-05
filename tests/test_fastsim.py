"""Correctness oracle for qguc.fastsim: every piece is checked against
PennyLane's default.qubit (an independent implementation) to 1e-10.

PennyLane orders wire 0 as the most significant bit; fastsim uses
x_j = (k >> j) & 1. Variable j is therefore put on wire n-1-j, which makes
the two statevectors directly comparable index by index.
"""

import dataclasses
import tracemalloc

import numpy as np
import pytest
from scipy.linalg import expm

qml = pytest.importorskip("pennylane")

from qguc import build_qubo, make_scenario_set, qubo_energy_batch  # noqa: E402
from qguc.exact import index_to_bits  # noqa: E402
from qguc.fastsim import (  # noqa: E402
    CustomMixer,
    QAOASimulator,
    RotatedMixer,
    XMixer,
    apply_diagonal_phase,
    apply_single_qubit,
    energy_vector,
    expectation,
    ising_energy_vector,
    probabilities,
    product_state,
    ry_product_state,
    sample,
    uniform_state,
)
from qguc.ising import bits_to_spins  # noqa: E402

TOL = 1e-10
NS = (4, 6, 8, 10, 12)
PS = (1, 2, 3)
X = np.array([[0, 1], [1, 0]], complex)
Z = np.diag([1.0, -1.0]).astype(complex)


def wire(n, j):
    return n - 1 - j


def random_ising(n, rng):
    h = rng.normal(size=n)
    J = np.triu(rng.normal(size=(n, n)), 1)
    J[rng.random((n, n)) < 0.3] = 0.0  # some missing couplings
    return h, np.triu(J, 1)


def pl_hamiltonian(n, h, J):
    coeffs, ops = [], []
    for j in range(n):
        coeffs.append(h[j])
        ops.append(qml.Z(wire(n, j)))
    for i, j in zip(*np.nonzero(J)):
        coeffs.append(J[i, j])
        ops.append(qml.Z(wire(n, i)) @ qml.Z(wire(n, j)))
    return qml.Hamiltonian(coeffs, ops)


def random_state(n, rng):
    v = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    return v / np.linalg.norm(v)


def assert_same_state(a, b, tol=TOL):
    """Equal up to a global phase."""
    phase = np.vdot(a, b)
    assert abs(abs(phase) - 1) < tol
    np.testing.assert_allclose(a * phase / abs(phase), b, atol=tol, rtol=0)


def run_pl(n, prep, layers, H):
    """prep(): state preparation; layers(): the QAOA layers. Returns (state, <H>)."""
    dev = qml.device("default.qubit", wires=n)

    def body():
        prep()
        layers()

    @qml.qnode(dev)
    def st():
        body()
        return qml.state()

    @qml.qnode(dev)
    def ex():
        body()
        return qml.expval(H)

    return np.asarray(st()), float(ex())


def standard_layers(n, H, gammas, betas):
    mixer_h = qml.qaoa.x_mixer(range(n))

    def layers():
        for g, b in zip(gammas, betas):
            qml.qaoa.cost_layer(g, H)
            qml.qaoa.mixer_layer(b, mixer_h)

    return layers


# ---- 1. standard QAOA (|+>^n, X mixer) ------------------------------------


@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("p", PS)
def test_standard_qaoa_matches_pennylane(n, p):
    rng = np.random.default_rng(100 * n + p)
    h, J = random_ising(n, rng)
    gammas, betas = rng.uniform(-np.pi, np.pi, p), rng.uniform(-np.pi, np.pi, p)
    H = pl_hamiltonian(n, h, J)

    def prep():
        for w in range(n):
            qml.Hadamard(w)

    ref_state, ref_e = run_pl(n, prep, standard_layers(n, H, gammas, betas), H)
    sim = QAOASimulator(ising_energy_vector(h, J))
    psi = sim.state(gammas, betas)
    assert_same_state(psi, ref_state)
    assert abs(sim.energy(gammas, betas) - ref_e) < TOL


# ---- 2. non-uniform initial states ----------------------------------------


@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("p", PS)
@pytest.mark.parametrize("kind", ["ry_product", "arbitrary"])
def test_nonuniform_initial_state_matches_pennylane(n, p, kind):
    rng = np.random.default_rng(7 + 100 * n + p)
    h, J = random_ising(n, rng)
    gammas, betas = rng.uniform(-np.pi, np.pi, p), rng.uniform(-np.pi, np.pi, p)
    H = pl_hamiltonian(n, h, J)
    if kind == "ry_product":
        thetas = rng.uniform(0, np.pi, n)
        psi0 = ry_product_state(thetas)

        def prep():
            for j in range(n):
                qml.RY(thetas[j], wires=wire(n, j))
    else:
        psi0 = random_state(n, rng)

        def prep():
            qml.StatePrep(psi0, wires=range(n))

    ref_state, ref_e = run_pl(n, prep, standard_layers(n, H, gammas, betas), H)
    sim = QAOASimulator(ising_energy_vector(h, J), initial_state=psi0)
    assert_same_state(sim.state(gammas, betas), ref_state)
    assert abs(sim.energy(gammas, betas) - ref_e) < TOL


# ---- 3. custom per-qubit mixers -------------------------------------------


@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("p", PS)
@pytest.mark.parametrize("kind", ["rotated", "random_generator"])
def test_custom_mixer_matches_pennylane(n, p, kind):
    """The mixer's per-qubit 2x2s are built independently with scipy expm and
    applied in PennyLane via QubitUnitary; a warm-start product state is used."""
    rng = np.random.default_rng(11 + 100 * n + p)
    h, J = random_ising(n, rng)
    gammas, betas = rng.uniform(-np.pi, np.pi, p), rng.uniform(-np.pi, np.pi, p)
    thetas = rng.uniform(0, np.pi, n)
    H = pl_hamiltonian(n, h, J)
    if kind == "rotated":
        gens = [np.cos(t) * Z + np.sin(t) * X for t in thetas]
        mixer = RotatedMixer(thetas)
    else:  # random Hermitian generators: U is not symmetric, so a transposed
        gens = []  # or mis-ordered application would be caught
        for _ in range(n):
            A = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
            gens.append((A + A.conj().T) / 2)
        mixer = CustomMixer(n, lambda beta: np.stack([expm(-1j * beta * G) for G in gens]))

    def prep():
        for j in range(n):
            qml.RY(thetas[j], wires=wire(n, j))

    def layers():
        for g, b in zip(gammas, betas):
            qml.qaoa.cost_layer(g, H)
            for j in range(n):
                qml.QubitUnitary(expm(-1j * b * gens[j]), wires=wire(n, j))

    ref_state, ref_e = run_pl(n, prep, layers, H)
    sim = QAOASimulator(ising_energy_vector(h, J), initial_state=ry_product_state(thetas), mixer=mixer)
    assert_same_state(sim.state(gammas, betas), ref_state)
    assert abs(sim.energy(gammas, betas) - ref_e) < TOL


def test_rotated_mixer_at_half_pi_is_the_x_mixer():
    n = 5
    np.testing.assert_allclose(RotatedMixer(np.full(n, np.pi / 2)).unitaries(0.37), XMixer(n).unitaries(0.37),
                               atol=1e-15)
    np.testing.assert_allclose(ry_product_state(np.full(n, np.pi / 2)), uniform_state(n), atol=1e-15)


# ---- 4. the diagonal shortcut vs gates --------------------------------------


@pytest.mark.parametrize("n", NS)
def test_diagonal_phase_equals_rz_rzz_circuit(n):
    """exp(-i gamma H_C) as elementwise phase == RZ(2 gamma h) and
    IsingZZ(2 gamma J) gates, exactly (no global phase is involved)."""
    rng = np.random.default_rng(n)
    h, J = random_ising(n, rng)
    gamma = rng.uniform(-2, 2)
    psi0 = random_state(n, rng)

    @qml.qnode(qml.device("default.qubit", wires=n))
    def circuit():
        qml.StatePrep(psi0, wires=range(n))
        for j in range(n):
            qml.RZ(2 * gamma * h[j], wires=wire(n, j))
        for i, j in zip(*np.nonzero(J)):
            qml.IsingZZ(2 * gamma * J[i, j], wires=[wire(n, i), wire(n, j)])
        return qml.state()

    np.testing.assert_allclose(apply_diagonal_phase(psi0, ising_energy_vector(h, J), gamma), circuit(),
                               atol=TOL, rtol=0)


@pytest.mark.parametrize("n", (3, 6))
def test_single_qubit_application_and_ordering(n):
    rng = np.random.default_rng(n)
    psi0 = random_state(n, rng)
    for q in range(n):
        A = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        U, _ = np.linalg.qr(A)

        @qml.qnode(qml.device("default.qubit", wires=n))
        def circuit():
            qml.StatePrep(psi0, wires=range(n))
            qml.QubitUnitary(U, wires=wire(n, q))
            return qml.state()

        np.testing.assert_allclose(apply_single_qubit(psi0, q, U), circuit(), atol=TOL, rtol=0)


# ---- 5. no 4^n allocation -------------------------------------------------


@pytest.mark.parametrize("n", (12, 14))
def test_no_4_to_the_n_allocation(n):
    """Peak traced memory of a full p = 3 simulation stays a small multiple of
    one state vector. A single 4^n array would be 4^n * 16 B (128 MB at
    n = 12); even 4^n bytes would exceed the bound."""
    rng = np.random.default_rng(0)
    h, J = random_ising(n, rng)
    E = ising_energy_vector(h, J)
    sim = QAOASimulator(E, mixer=RotatedMixer(rng.uniform(0, np.pi, n)))
    gammas, betas = rng.uniform(0, 1, 3), rng.uniform(0, 1, 3)
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        sim.energy(gammas, betas)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    state_bytes = 16 * (1 << n)
    assert peak < 8 * state_bytes, f"peak {peak} B vs state {state_bytes} B"
    assert peak < (4**n) // 4


# ---- 6. batching, precision, helpers -----------------------------------------


def test_batch_matches_individual_runs():
    rng = np.random.default_rng(3)
    n, p, B = 8, 3, 5
    h, J = random_ising(n, rng)
    sim = QAOASimulator(ising_energy_vector(h, J), initial_state=ry_product_state(rng.uniform(0, np.pi, n)),
                        mixer=RotatedMixer(rng.uniform(0, np.pi, n)))
    G, Bt = rng.uniform(-1, 1, (B, p)), rng.uniform(-1, 1, (B, p))
    batch = sim.state(G, Bt)
    for r in range(B):
        np.testing.assert_allclose(batch[r], sim.state(G[r], Bt[r]), atol=1e-14, rtol=0)
    np.testing.assert_allclose(sim.energy(G, Bt), [sim.energy(G[r], Bt[r]) for r in range(B)], atol=1e-12)


def test_complex64_agrees_to_single_precision():
    rng = np.random.default_rng(4)
    n, p = 12, 5
    h, J = random_ising(n, rng)
    E = ising_energy_vector(h, J)
    g, b = rng.uniform(0, 1, p), rng.uniform(0, 1, p)
    e64 = QAOASimulator(E, dtype=np.complex128).energy(g, b)
    e32 = QAOASimulator(E, dtype=np.complex64).energy(g, b)
    assert abs(e32 - e64) < 1e-4 * np.abs(E).max()


def test_product_state_ordering():
    amps = np.array([[1, 0], [0, 1], [1, 0]], complex)  # x = (0, 1, 0) -> k = 2
    assert np.flatnonzero(product_state(amps)).tolist() == [2]


def test_sampling_follows_probabilities():
    rng = np.random.default_rng(5)
    psi = random_state(4, rng)
    counts = np.bincount(sample(psi, 200_000, rng), minlength=16) / 200_000
    np.testing.assert_allclose(counts, probabilities(psi), atol=5e-3)
    assert expectation(psi, np.arange(16.0)) == pytest.approx(probabilities(psi) @ np.arange(16.0))


def test_energy_vector_matches_qubo_and_ising_and_is_cached():
    inst = make_scenario_set(1, n_hours=2, seed=0, refinement_level=0, demand_bias_mw=150.0)[0]
    ev = energy_vector(inst, 100.0)
    assert energy_vector(inst, 100.0) is ev  # cached on the instance
    assert energy_vector(inst, 50.0) is not ev  # keyed by the QUBO parameters
    q = build_qubo(inst, 100.0)
    k = np.arange(1 << inst.n_vars)
    xs = index_to_bits(k, inst.n_vars)
    # same numbering as brute force, normalised Ising units
    np.testing.assert_allclose(ev.ising.to_physical(ev.E), qubo_energy_batch(q, xs), rtol=1e-12)
    np.testing.assert_allclose(ev.E, ev.ising.normalised_energy(bits_to_spins(xs)), atol=1e-9)
    assert ev.E[ev.ground].max() == pytest.approx(ev.E.min(), abs=1e-9)
    assert ev.sigma == pytest.approx(ev.E.std(), rel=1e-9)
    assert dataclasses.replace(inst).cache == {}


@pytest.mark.parametrize("group", [1, 2, 3, 4, 5, 7])
def test_grouped_mixer_equals_qubit_by_qubit(group):
    """The Kronecker-grouped mixer is the same operation as n calls of
    apply_single_qubit (checked for group sizes that do and do not divide n)."""
    rng = np.random.default_rng(group)
    n = 9
    E = rng.normal(size=1 << n)
    mixer = RotatedMixer(rng.uniform(0, np.pi, n))
    psi0 = random_state(n, rng)
    g, b = rng.uniform(-1, 1, 2), rng.uniform(-1, 1, 2)
    ref = psi0.copy()
    for k in range(2):
        ref = apply_diagonal_phase(ref, E, g[k])
        U = mixer.unitaries(b[k])
        for q in range(n):
            ref = apply_single_qubit(ref, q, U[q])
    sim = QAOASimulator(E, initial_state=psi0, mixer=mixer, group=group)
    np.testing.assert_allclose(sim.state(g, b), ref, atol=1e-13, rtol=0)
    G = rng.uniform(-1, 1, (3, 2))
    np.testing.assert_allclose(sim.state(G, G)[1], QAOASimulator(E, initial_state=psi0, mixer=mixer,
                               group=1).state(G[1], G[1]), atol=1e-13, rtol=0)
