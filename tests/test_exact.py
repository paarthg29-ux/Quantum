import itertools

import numpy as np
import pytest

from qguc import build_qubo, make_instance, make_scenario_set, qubo_energy, select_plants
from qguc.exact import brute_force, brute_force_operational, index_to_bits, solve_milp
from qguc.metrics import approximation_ratio
from qguc.qubo import QUBO


def test_brute_force_matches_naive_enumeration():
    inst = make_instance(3300.0, n_hours=2, initial_on=[0, 1, 0, 1, 0], seed=3, demand_bias_mw=200.0)
    q = build_qubo(inst, 20.0)
    bf = brute_force(q, return_spectrum=True, return_energies=True)
    naive = {}
    for bits in itertools.product((0, 1), repeat=q.n_vars):
        x = np.array(bits)
        k = int(sum(int(b) << j for j, b in enumerate(bits)))  # x_j is bit j of k
        naive[k] = qubo_energy(q, x)
    e = np.array([naive[k] for k in range(1 << q.n_vars)])
    np.testing.assert_allclose(bf.energies, e, rtol=1e-12)
    np.testing.assert_allclose(bf.spectrum, np.sort(e), rtol=1e-12)
    assert bf.ground_energy == pytest.approx(e.min(), rel=1e-12)
    assert bf.max_energy == pytest.approx(e.max(), rel=1e-12)
    assert qubo_energy(q, bf.ground_state) == pytest.approx(e.min(), rel=1e-12)
    assert np.array_equal(index_to_bits(bf.ground_indices[0], q.n_vars), bf.ground_state)


def test_brute_force_reports_degeneracy():
    flat = QUBO(Q=np.zeros((4, 4)), offset=1.0, n_vars=4)
    assert brute_force(flat).degeneracy == 16

    # E = (x0 - x1)^2 = x0 + x1 - 2 x0 x1: ground states 00 and 11.
    q = QUBO(Q=np.array([[1.0, -2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 5.0]]), offset=0.0, n_vars=3)
    bf = brute_force(q)
    assert bf.is_degenerate and bf.degeneracy == 2
    assert {tuple(s) for s in bf.ground_states} == {(0, 0, 0), (1, 1, 0)}

    unique = brute_force(build_qubo(make_instance(3600.0, n_hours=2, seed=1), 10.0))
    assert not unique.is_degenerate


def test_brute_force_twenty_variables():
    q = build_qubo(make_instance(3900.0, n_hours=4, seed=7), 10.0)
    bf = brute_force(q, return_spectrum=True)
    assert bf.spectrum.shape == (1 << 20,)
    assert np.all(np.diff(bf.spectrum) >= 0)
    assert bf.spectrum[0] == bf.ground_energy
    assert bf.elapsed_s < 30.0


def test_brute_force_rejects_too_many_vars():
    with pytest.raises(ValueError):
        brute_force(QUBO(Q=np.zeros((25, 25)), offset=0.0, n_vars=25))


def test_approximation_ratio_endpoints():
    assert approximation_ratio(-5.0, -5.0, 10.0) == 1.0
    assert approximation_ratio(10.0, -5.0, 10.0) == 0.0
    assert approximation_ratio(2.5, -5.0, 10.0) == 0.5


@pytest.mark.parametrize("model,hours", [("station", (1, 2, 3)), ("unit", (1,))])
def test_milp_matches_exhaustive_operational_optimum(model, hours):
    rng = np.random.default_rng(11)
    n_blocks = len(select_plants(model))
    for n_hours in hours:
        for _ in range(4):
            inst = make_instance(rng.uniform(2500, 4200), n_hours=n_hours, plants=model,
                                 initial_on=rng.integers(0, 2, size=n_blocks),
                                 seed=int(rng.integers(1 << 30)), demand_bias_mw=rng.choice([0.0, 200.0]))
            m = solve_milp(inst)
            ref = brute_force_operational(inst)
            assert m.operating_cost == pytest.approx(ref.operating_cost, rel=1e-12), inst.label
            assert m.status == 0
            assert inst.is_adequate(m.x)
            assert m.operating_cost == pytest.approx(m.solver_objective, rel=1e-9)


def test_milp_four_hour_scenarios():
    for inst in make_scenario_set(4, n_hours=4, seed=5) + make_scenario_set(2, n_hours=4, seed=5, plants="unit"):
        m = solve_milp(inst)
        assert inst.is_adequate(m.x)
        assert m.fuel_cost == inst.fuel_cost(m.x)
        assert m.startup_cost == inst.startup_expense(m.x)  # aux y is exact at the optimum
        assert m.solve_time_s > 0
