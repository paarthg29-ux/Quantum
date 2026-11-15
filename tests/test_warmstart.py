"""Numerical checks of the warm-start construction (not trusting the algebra)."""

import numpy as np
import pytest
from scipy.linalg import expm

from qguc.fastsim import QAOASimulator, XMixer, ising_energy_vector, uniform_state
from qguc.warmstart import (
    WarmStartMixer,
    mixer_hamiltonians,
    regularise,
    thetas_from_c,
    warm_initial_state,
)

X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.diag([1.0, -1.0]).astype(complex)


def RY(t):
    return expm(-1j * t / 2 * Y)


def RZ(t):
    return expm(-1j * t / 2 * Z)


C_VALUES = np.array([0.1, 0.25, 0.5, 0.63, 0.9, 0.02])


@pytest.mark.parametrize("c", C_VALUES)
def test_mixer_hamiltonian_is_minus_rotated_z(c):
    th = thetas_from_c([c])[0]
    H = mixer_hamiltonians([c])[0]
    np.testing.assert_allclose(H, -RY(th) @ Z @ RY(-th), atol=1e-14)
    np.testing.assert_allclose(H, [[2 * c - 1, -2 * np.sqrt(c * (1 - c))], [-2 * np.sqrt(c * (1 - c)), 1 - 2 * c]],
                               atol=1e-15)


@pytest.mark.parametrize("c", C_VALUES)
@pytest.mark.parametrize("beta", [-1.3, 0.0, 0.4, 2.2])
def test_layer_unitary_equals_ry_rz_ry(c, beta):
    th = thetas_from_c([c])[0]
    U_expm = expm(-1j * beta * mixer_hamiltonians([c])[0])
    np.testing.assert_allclose(U_expm, RY(th) @ RZ(-2 * beta) @ RY(-th), atol=1e-13)
    np.testing.assert_allclose(WarmStartMixer([c]).unitaries(beta)[0], U_expm, atol=1e-13)
    # batched beta gives the same matrices
    np.testing.assert_allclose(WarmStartMixer([c]).unitaries(np.array([beta, beta]))[1, 0], U_expm, atol=1e-13)


def kron_sum(Hs):
    """sum_i H_i acting on qubit i (qubit i = bit i: H_0 is the LAST kron factor)."""
    n = len(Hs)
    total = np.zeros((1 << n, 1 << n), complex)
    for i, H in enumerate(Hs):
        ops = [np.eye(2)] * n
        ops[n - 1 - i] = H
        M = ops[0]
        for o in ops[1:]:
            M = np.kron(M, o)
        total += M
    return total


def test_initial_state_is_the_unique_ground_state_of_the_mixer():
    rng = np.random.default_rng(0)
    c = regularise(rng.uniform(0, 1, 5), 0.1)
    w, V = np.linalg.eigh(kron_sum(mixer_hamiltonians(c)))
    assert w[0] == pytest.approx(-5.0, abs=1e-12) and w[1] > w[0] + 1.0  # each H_i has eigenvalues +-1
    psi0 = warm_initial_state(c)
    assert abs(np.vdot(V[:, 0], psi0)) == pytest.approx(1.0, abs=1e-12)
    # and its single-qubit marginals are c
    probs = np.abs(psi0) ** 2
    k = np.arange(1 << 5)
    np.testing.assert_allclose([probs[(k >> i) & 1 == 1].sum() for i in range(5)], c, atol=1e-14)


def test_half_reduces_to_x_mixer_and_uniform_state():
    n = 6
    c = np.full(n, 0.5)
    np.testing.assert_allclose(mixer_hamiltonians(c)[0], -X, atol=1e-15)
    np.testing.assert_allclose(warm_initial_state(c), uniform_state(n), atol=1e-15)
    for beta in (-0.7, 0.3, 1.1):  # exp(-i beta (-X)) = standard X mixer at -beta
        np.testing.assert_allclose(WarmStartMixer(c).unitaries(beta), XMixer(n).unitaries(-beta), atol=1e-15)
    # whole circuit: warm machinery at c = 1/2 with (gamma, beta) == arm A with (gamma, -beta)
    rng = np.random.default_rng(1)
    h, J = rng.normal(size=n), np.triu(rng.normal(size=(n, n)), 1)
    E = ising_energy_vector(h, J)
    g, b = rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3)
    warm = QAOASimulator(E, initial_state=warm_initial_state(c), mixer=WarmStartMixer(c)).state(g, b)
    np.testing.assert_allclose(warm, QAOASimulator(E).state(g, -b), atol=1e-13)


def test_unregularised_extremes_freeze_the_qubit_and_eps_unfreezes_it():
    for c in (0.0, 1.0):
        U = WarmStartMixer([c]).unitaries(0.83)[0]
        assert abs(U[0, 1]) < 1e-15 and abs(U[1, 0]) < 1e-15  # diagonal: can never flip
    U = WarmStartMixer(regularise([0.0], 0.1)).unitaries(0.83)[0]
    assert abs(U[0, 1]) > 0.1
    np.testing.assert_allclose(regularise([0.0, 0.03, 0.5, 0.97, 1.0], 0.05), [0.05, 0.05, 0.5, 0.95, 0.95])


def test_warm_qaoa_matches_pennylane_gate_decomposition():
    """Independent check: PennyLane circuit with RY prep and RY(-th) RZ(-2b) RY(th) gates."""
    qml = pytest.importorskip("pennylane")
    n, p = 6, 2
    rng = np.random.default_rng(2)
    h, J = rng.normal(size=n), np.triu(rng.normal(size=(n, n)), 1)
    c = regularise(rng.uniform(0, 1, n), 0.1)
    th = thetas_from_c(c)
    g, b = rng.uniform(-1, 1, p), rng.uniform(-1, 1, p)
    wire = lambda j: n - 1 - j  # noqa: E731
    coeffs = [h[j] for j in range(n)] + [J[i, j] for i in range(n) for j in range(i + 1, n)]
    ops = [qml.Z(wire(j)) for j in range(n)] + [qml.Z(wire(i)) @ qml.Z(wire(j))
                                                for i in range(n) for j in range(i + 1, n)]
    H = qml.Hamiltonian(coeffs, ops)

    @qml.qnode(qml.device("default.qubit", wires=n))
    def circuit():
        for j in range(n):
            qml.RY(th[j], wires=wire(j))
        for k in range(p):
            qml.qaoa.cost_layer(g[k], H)
            for j in range(n):  # matrix RY(th) RZ(-2b) RY(-th): RY(-th) acts first
                qml.RY(-th[j], wires=wire(j))
                qml.RZ(-2 * b[k], wires=wire(j))
                qml.RY(th[j], wires=wire(j))
        return qml.state()

    ref = np.asarray(circuit())
    psi = QAOASimulator(ising_energy_vector(h, J), initial_state=warm_initial_state(c),
                        mixer=WarmStartMixer(c)).state(g, b)
    phase = np.vdot(psi, ref)
    np.testing.assert_allclose(psi * phase / abs(phase), ref, atol=1e-10)


def test_eps_half_is_the_cold_limit():
    """eps = 0.5 (the sweep's sanity point) is accepted: c -> 1/2 everywhere,
    the warm state is |+>^n and the warm mixer is the X mixer with beta -> -beta."""
    from qguc.fastsim import XMixer

    c = regularise([0.0, 0.2, 0.97, 1.0], 0.5)
    np.testing.assert_array_equal(c, 0.5)
    np.testing.assert_allclose(warm_initial_state(c), np.full(16, 0.25), atol=1e-15)
    np.testing.assert_allclose(WarmStartMixer(c).unitaries(0.37), XMixer(4).unitaries(-0.37), atol=1e-15)
    with pytest.raises(ValueError):
        regularise([0.3], 0.51)


def test_complex64_warm_state_accepted_at_22_qubits():
    """Regression: the normalisation check summed |psi|^2 in float32 and rejected a
    correct complex64 warm state at n >= 20 (drift ~7e-4 at n = 22)."""
    n = 22
    c = regularise(np.random.default_rng(0).uniform(0, 1, n), 0.1)
    psi = warm_initial_state(c, np.complex64)
    sim = QAOASimulator(np.zeros(1 << n), initial_state=psi, mixer=WarmStartMixer(c), dtype=np.complex64)
    assert sim.initial_state.dtype == np.complex64
    with pytest.raises(ValueError):
        QAOASimulator(np.zeros(1 << n), initial_state=psi * 1.001, dtype=np.complex64)
