import numpy as np
import pytest

from qguc import build_qubo, make_instance, qubo_energy_batch
from qguc.exact import iter_bitstrings
from qguc.metrics import approximation_ratio
from qguc.ising import bits_to_spins, normalise, qubo_to_ising, spins_to_bits


def _qubos(seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for n_hours in (1, 2, 3, 4):
        for _ in range(3):
            inst = make_instance(rng.uniform(2500, 4200), n_hours=n_hours,
                                 initial_on=rng.integers(0, 2, size=5),
                                 seed=int(rng.integers(1 << 30)), demand_bias_mw=rng.choice([0.0, 200.0]))
            out.append(build_qubo(inst, rng.uniform(0.1, 200), lambda_switch=rng.choice([0.0, 3e5]),
                                  include_startup=bool(rng.integers(0, 2))))
    return out


def _all_energies(qubo, model):
    """QUBO energies and normalised Ising energies of every state, in state order."""
    eq, ei = [], []
    for _, X in iter_bitstrings(qubo.n_vars):
        eq.append(qubo_energy_batch(qubo, X))
        ei.append(model.normalised_energy(bits_to_spins(X)))
    return np.concatenate(eq), np.concatenate(ei)


def test_spin_convention():
    assert list(bits_to_spins([0, 1])) == [1, -1]
    assert list(spins_to_bits([1, -1])) == [0, 1]


def test_ising_energy_equals_qubo_energy():
    rng = np.random.default_rng(1)
    for q in _qubos():
        X = rng.integers(0, 2, size=(500, q.n_vars))
        e_qubo = qubo_energy_batch(q, X)
        raw = qubo_to_ising(q)
        norm = normalise(raw)
        np.testing.assert_allclose(raw.energy_from_bits(X), e_qubo, rtol=1e-9, atol=0)
        np.testing.assert_allclose(norm.energy_from_bits(X), e_qubo, rtol=1e-9, atol=0)
        for x, e in zip(X[:20], e_qubo[:20]):  # single-state path
            assert raw.energy_from_bits(x) == pytest.approx(e, rel=1e-9)


def test_ising_coefficients_match_hand_derivation():
    # 2-variable QUBO: E = c + a x0 + b x1 + w x0 x1.
    from qguc.qubo import QUBO
    c, a, b, w = 3.0, 5.0, -7.0, 11.0
    m = qubo_to_ising(QUBO(Q=np.array([[a, w], [0.0, b]]), offset=c, n_vars=2))
    assert np.allclose(m.h, [-a / 2 - w / 4, -b / 2 - w / 4])
    assert m.J[0, 1] == w / 4 and m.J[1, 0] == 0.0
    assert m.const == c + (a + b) / 2 + w / 4
    for x0 in (0, 1):
        for x1 in (0, 1):
            assert m.energy_from_bits([x0, x1]) == c + a * x0 + b * x1 + w * x0 * x1


def test_normalised_coefficients_bounded():
    for q in _qubos():
        m = normalise(qubo_to_ising(q))
        top = max(np.abs(m.h).max(), np.abs(m.J).max())
        assert top <= 1.0
        assert top == pytest.approx(1.0)
        assert np.all(np.tril(m.J) == 0.0)


def test_energy_unit_round_trip():
    q = _qubos()[5]
    m = normalise(qubo_to_ising(q))
    e = np.array([-1.0, 0.0, 2.5])
    np.testing.assert_allclose(m.to_normalised(m.to_physical(e)), e, rtol=0, atol=1e-12)


def test_normalisation_preserves_argmin_and_approximation_ratio():
    rng = np.random.default_rng(2)
    for q in _qubos(seed=3):
        if q.n_vars > 15:
            continue
        norm = normalise(qubo_to_ising(q))
        assert norm.scale > 1e3  # the test is only meaningful if the rescale is large
        e_phys, e_norm = _all_energies(q, norm)

        assert np.argmin(e_phys) == np.argmin(e_norm)
        assert np.array_equal(np.argsort(e_phys, kind="stable")[:5], np.argsort(e_norm, kind="stable")[:5])

        ar_phys = approximation_ratio(e_phys, e_phys.min(), e_phys.max())
        ar_norm = approximation_ratio(e_norm, e_norm.min(), e_norm.max())
        np.testing.assert_allclose(ar_phys, ar_norm, rtol=0, atol=1e-9)

        # Fixed states, energies converted through the model's unit maps.
        for k in rng.integers(0, e_phys.size, size=10):
            a1 = approximation_ratio(e_phys[k], e_phys.min(), e_phys.max())
            a2 = approximation_ratio(norm.to_normalised(e_phys[k]), norm.to_normalised(e_phys.min()),
                                     norm.to_normalised(e_phys.max()))
            assert a1 == pytest.approx(a2, abs=1e-9)
        # The naive ratio E / E_min is not invariant; that is why it is not used.
        k = int(np.argmax(e_phys))
        assert abs(e_phys[k] / e_phys.min() - e_norm[k] / e_norm.min()) > 1e-3
