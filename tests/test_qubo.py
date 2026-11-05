import dataclasses

import numpy as np
import pytest

from qguc import (
    build_qubo,
    make_instance,
    make_scenario_set,
    qubo_energy,
    qubo_energy_batch,
    select_plants,
)


def _random_instances(n=6, seed=0, model="station", hours=(1, 2, 4, 6)):
    """Instances with varied horizons, peaks, hour -1 states and biases."""
    rng = np.random.default_rng(seed)
    n_blocks = len(select_plants(model))
    out = []
    for n_hours in hours:
        for _ in range(n):
            peak = rng.uniform(2500, 4200)
            init = rng.integers(0, 2, size=n_blocks)
            bias = rng.choice([0.0, 150.0, rng.uniform(0, 400)])
            out.append(make_instance(peak, n_hours=n_hours, initial_on=init, plants=model,
                                     seed=int(rng.integers(1 << 30)), demand_bias_mw=bias))
    return out


def _reference_energy(inst, x, lam_d, lam_s, include_startup):
    """Objective recomputed from the instance methods, without touching Q."""
    u = inst.unflatten(x).astype(float)
    prev = np.concatenate([inst.initial_on[:, None].astype(float), u[:, :-1]], axis=1)
    switching = lam_s * np.sum((u - prev) ** 2)
    startup = inst.startup_expense(x) if include_startup else 0.0
    demand = lam_d * np.sum((inst.generation(x) - inst.demand_mw - inst.demand_bias_mw) ** 2)
    return inst.fuel_cost(x) + startup + demand + switching


def test_matrix_energy_matches_term_by_term_objective():
    rng = np.random.default_rng(1)
    instances = _random_instances() + _random_instances(n=3, model="unit", hours=(1, 2, 3))
    for inst in instances:
        for lam_d, lam_s, inc in [(0.0, 0.0, True), (37.5, 0.0, True), (100.0, 2.5e5, True),
                                  (12.0, 1e6, False), (rng.uniform(1, 500), rng.uniform(0, 1e6), True)]:
            q = build_qubo(inst, lam_d, lambda_switch=lam_s, include_startup=inc)
            X = rng.integers(0, 2, size=(200, inst.n_vars))
            batch = qubo_energy_batch(q, X)
            for x, e_batch in zip(X, batch):
                ref = _reference_energy(inst, x, lam_d, lam_s, inc)
                e = qubo_energy(q, x)
                assert e == pytest.approx(ref, rel=1e-6), (inst.label, lam_d, lam_s, inc, x)
                assert e_batch == pytest.approx(e, rel=1e-12)


def test_q_is_upper_triangular():
    for inst in _random_instances(n=2):
        q = build_qubo(inst, 50.0, lambda_switch=1e5)
        assert np.all(np.tril(q.Q, k=-1) == 0.0)
        assert q.Q.shape == (inst.n_vars, inst.n_vars)


def test_fuel_only_energy_equals_fuel_cost():
    rng = np.random.default_rng(3)
    for inst in _random_instances(n=2):
        q = build_qubo(inst, 0.0, lambda_switch=0.0, include_startup=False)
        assert q.offset == 0.0
        for x in rng.integers(0, 2, size=(50, inst.n_vars)):
            assert qubo_energy(q, x) == inst.fuel_cost(x)


def test_exact_demand_gives_zero_penalty():
    rng = np.random.default_rng(4)
    base = make_instance(3500.0, n_hours=4, seed=0)
    for _ in range(20):
        x = rng.integers(0, 2, size=base.n_vars)
        inst = dataclasses.replace(base, demand_mw=base.generation(x), meta={})
        assert np.all(inst.mismatch(x) == 0.0)
        no_pen = build_qubo(inst, 0.0)
        with_pen = build_qubo(inst, 10.0)
        assert qubo_energy(with_pen, x) - qubo_energy(no_pen, x) == pytest.approx(0.0, abs=1e-6)
    # Sanity: a different state does pay a penalty on the same instance.
    y = 1 - x
    if np.any(inst.mismatch(y) != 0):
        assert qubo_energy(with_pen, y) > qubo_energy(no_pen, y)


def test_var_index_unflatten_roundtrip():
    inst = make_instance(3800.0, n_hours=5, seed=0)
    N, T = inst.n_plants, inst.n_hours
    seen = set()
    for t in range(T):
        for i in range(N):
            k = inst.var_index(i, t)
            assert 0 <= k < inst.n_vars
            seen.add(k)
            e = np.zeros(inst.n_vars, dtype=int)
            e[k] = 1
            u = inst.unflatten(e)
            assert u.shape == (N, T)
            assert u[i, t] == 1 and u.sum() == 1
    assert seen == set(range(inst.n_vars))

    # Time-major: one hour's units occupy a contiguous block, hours in order.
    for t in range(T):
        block = [inst.var_index(i, t) for i in range(N)]
        assert block == list(range(block[0], block[0] + N))
        if t + 1 < T:
            assert inst.var_index(0, t + 1) == block[-1] + 1

    x = np.random.default_rng(5).integers(0, 2, size=inst.n_vars)
    assert np.array_equal(inst.flatten(inst.unflatten(x)), x)


def test_startup_respects_initial_on():
    inst = make_instance(3500.0, n_hours=4, initial_on=[1, 0, 0, 0, 0], seed=0)
    u = np.zeros((inst.n_plants, inst.n_hours), dtype=int)
    u[0, 0] = 1  # already on in hour -1: no start-up
    u[1, 0] = 1  # off in hour -1: one start-up
    u[2, 0] = u[2, 2] = 1  # off, on, off, on: two start-ups
    x = inst.flatten(u)

    assert list(inst.startup_count(x)) == [0, 1, 2, 0, 0]
    S = inst.startup_cost
    expected = S[1] + 2 * S[2]
    assert inst.startup_expense(x) == expected

    with_su = build_qubo(inst, 0.0, include_startup=True)
    without = build_qubo(inst, 0.0, include_startup=False)
    assert qubo_energy(with_su, x) - qubo_energy(without, x) == expected

    # Flipping unit 0's hour -1 state makes it pay a start-up at t=0.
    inst_off = dataclasses.replace(inst, initial_on=np.zeros(5, dtype=int), meta={})
    d = qubo_energy(build_qubo(inst_off, 0.0), x) - qubo_energy(build_qubo(inst_off, 0.0, include_startup=False), x)
    assert d == expected + S[0]


def test_demand_bias_shifts_penalty_target_but_not_adequacy():
    base = make_instance(3500.0, n_hours=4, seed=0)
    u = np.zeros((base.n_plants, base.n_hours), dtype=int)
    u[[2, 3, 4], :] = 1  # 3300 MW every hour
    x = base.flatten(u)
    inst = dataclasses.replace(base, demand_mw=np.full(4, 3200.0), demand_bias_mw=200.0, meta={})
    assert inst.is_adequate(x)  # 3300 >= 3200 = D; the bias plays no part
    assert np.all(inst.target_mismatch(x) == -100.0)
    pen = qubo_energy(build_qubo(inst, 1.0, include_startup=False), x) - inst.fuel_cost(x)
    assert pen == 4 * 100.0**2  # penalty measured from D + B, not D


def test_switching_penalty_off_by_default():
    inst = make_instance(3500.0, n_hours=4, seed=0)
    assert build_qubo(inst, 10.0).params["lambda_switch"] == 0.0
    assert np.array_equal(build_qubo(inst, 10.0).Q, build_qubo(inst, 10.0, lambda_switch=0.0).Q)


def test_batch_energy_chunking():
    inst = make_instance(3900.0, n_hours=4, seed=2)
    q = build_qubo(inst, 80.0, lambda_switch=3e5)
    X = np.random.default_rng(6).integers(0, 2, size=(1000, inst.n_vars), dtype=np.uint8)
    full = qubo_energy_batch(q, X)
    chunked = qubo_energy_batch(q, X, chunk_size=37)
    assert np.allclose(full, chunked, rtol=1e-12, atol=0)
    assert np.allclose(full, [qubo_energy(q, x) for x in X], rtol=1e-12, atol=0)


def test_scenario_set_spans_load_range():
    insts = make_scenario_set(200, n_hours=4, seed=0)
    peaks = np.array([i.meta["peak_mw"] for i in insts])
    assert peaks.min() < 2700 and peaks.max() > 4000
    for inst in insts:
        assert np.all((inst.demand_mw >= 2500) & (inst.demand_mw <= 4200))
        # Shaped, not noise: the last hour is above the first (evening ramp).
        assert inst.demand_mw[-1] >= inst.demand_mw[0]
