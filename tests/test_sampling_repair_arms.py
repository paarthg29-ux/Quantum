"""Exact best-of-K sampling, the repair heuristic, the LP relaxation and the arm registry."""

import numpy as np
import pytest

from qguc import make_scenario_set
from qguc.arms import ARMS, c_merit_order, c_random, run_arm
from qguc.exact import index_to_bits, solve_lp_relaxation, solve_milp
from qguc.fastsim import sample
from qguc.qaoa import OptimiserSettings, initial_parameters, make_problem
from qguc.repair import cost_per_mw_order, repair
from qguc.sampling import StateTable, best_of_k_weights, exact_sampling_metrics


# ---- exact best-of-K --------------------------------------------------------


def test_best_of_k_weights_basic():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(20))
    np.testing.assert_allclose(best_of_k_weights(p, 1), p, atol=1e-15)
    for K in (2, 17, 1024):
        w = best_of_k_weights(p, K)
        assert w.min() >= -1e-15 and w.sum() == pytest.approx(1.0, abs=1e-12)
    # brute-force check for K = 3 over all ordered triples
    q = p[:6] / p[:6].sum()
    exact = np.zeros(6)
    for a in range(6):
        for b in range(6):
            for c in range(6):
                exact[min(a, b, c)] += q[a] * q[b] * q[c]
    np.testing.assert_allclose(best_of_k_weights(q, 3), exact, atol=1e-15)


@pytest.mark.parametrize("K", [1, 8, 64])
def test_best_of_k_matches_monte_carlo(K):
    """Exact distribution vs Monte Carlo sampling within sampling error (4 sigma)."""
    rng = np.random.default_rng(K)
    n = 5
    psi = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    psi /= np.linalg.norm(psi)
    probs = np.abs(psi) ** 2
    E = rng.permutation(1 << n).astype(float)  # distinct energies
    cost = rng.uniform(0, 1, 1 << n)
    order = np.argsort(E, kind="stable")
    w = best_of_k_weights(probs[order], K)
    trials = 40_000
    shots = sample(psi, trials * K, rng).reshape(trials, K)
    best = shots[np.arange(trials), np.argmin(E[shots], axis=1)]
    freq = np.bincount(best, minlength=1 << n)[order] / trials
    sd = np.sqrt(w * (1 - w) / trials)
    assert np.all(np.abs(freq - w) <= 4 * sd + 1e-4)
    mc_cost = cost[best]
    exp_cost = w @ cost[order]
    assert abs(mc_cost.mean() - exp_cost) <= 4 * mc_cost.std() / np.sqrt(trials)


def test_exact_metrics_on_an_instance():
    inst = make_scenario_set(1, n_hours=2, seed=3, refinement_level=0)[0]
    pb = make_problem(inst)
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(1 << pb.n) * 0.3)
    m = exact_sampling_metrics(probs, pb.table)
    assert m[1]["p_ground"] == pytest.approx(probs[pb.ev.ground].sum(), abs=1e-14)
    assert m[1024]["p_ground"] == pytest.approx(1 - (1 - probs[pb.ev.ground].sum()) ** 1024, abs=1e-9)
    # a point mass on the ground state: every K returns it
    delta = np.zeros(1 << pb.n)
    delta[pb.ev.ground[0]] = 1.0
    xg = index_to_bits(pb.ev.ground[:1], pb.n)[0]
    for K, v in exact_sampling_metrics(delta, pb.table).items():
        assert v["p_ground"] == pytest.approx(1.0)
        assert v["gap_raw"] == pytest.approx((inst.operating_cost(xg) - pb.milp.operating_cost)
                                             / pb.milp.operating_cost)
    assert m[1024]["gap_repaired"] >= -1e-12  # repaired schedules are adequate, so never below the MILP


# ---- repair -----------------------------------------------------------------


@pytest.mark.parametrize("level", [0, 2, 4])
def test_repair_is_adequate_monotone_and_idempotent(level):
    inst = make_scenario_set(1, n_hours=2, seed=5, refinement_level=level)[0]
    rng = np.random.default_rng(level)
    X = rng.integers(0, 2, (500, inst.n_vars))
    R = repair(inst, X)
    assert inst.is_adequate(R).all()
    assert (R >= X).all()  # only adds
    ok = inst.is_adequate(X)
    np.testing.assert_array_equal(R[ok], X[ok])  # adequate schedules untouched
    np.testing.assert_array_equal(repair(inst, R), R)
    for r in range(20):
        np.testing.assert_array_equal(repair(inst, X[r]), R[r])


def test_repair_adds_cheapest_per_mw_first():
    inst = make_scenario_set(1, n_hours=1, seed=1, refinement_level=0)[0]
    order = cost_per_mw_order(inst)
    x = repair(inst, np.zeros(inst.n_vars, int))
    P = inst.capacity
    # committed set is exactly the shortest prefix of the order that covers demand
    k = np.searchsorted(np.cumsum(P[order]), inst.demand_mw[0]) + 1
    assert set(np.flatnonzero(x)) == set(order[:k])


# ---- LP relaxation ----------------------------------------------------------


@pytest.mark.parametrize("level", [0, 2, 4])
def test_lp_relaxation_bounds_the_milp(level):
    for inst in make_scenario_set(5, n_hours=3, seed=11, refinement_level=level):
        lp, milp = solve_lp_relaxation(inst), solve_milp(inst)
        assert lp.x.min() >= 0 and lp.x.max() <= 1
        assert lp.objective <= milp.operating_cost * (1 + 1e-9)
        assert (inst.capacity @ inst.unflatten(lp.x) >= inst.demand_mw - 1e-6).all()


# ---- arms -------------------------------------------------------------------


def test_arm_sources_and_pairing():
    inst = make_scenario_set(2, n_hours=2, seed=8, refinement_level=0)[1]
    pb = make_problem(inst)
    F = c_merit_order(pb, 0, 0.1)
    assert set(np.unique(F.c)) <= {0.1, 0.9}
    assert (inst.capacity @ inst.unflatten(F.info["schedule"]) >= inst.target_mw).all()
    E1, E2 = c_random(pb, 0, 0.1), c_random(pb, 1, 0.1)
    assert E1.c.min() >= 0.1 and E1.c.max() <= 0.9 and not np.allclose(E1.c, E2.c)
    np.testing.assert_array_equal(c_random(pb, 0, 0.1).c, E1.c)  # reproducible
    s = OptimiserSettings(cap_per_param=10)
    runs = {a: run_arm(a, pb, 1, seed=3, settings=s) for a in ARMS if not a.startswith(('B', 'C'))}
    th, be = initial_parameters(1, 3)
    for a, r in runs.items():
        assert r.n_evals <= s.max_evals(2)
        assert r.cap_hit == (not r.early_stopped and r.n_evals == s.max_evals(2))
        assert 0 <= r.sampling[1024]["p_ground"] <= 1
    assert runs["A"].g_state0 == pytest.approx(0.0, abs=1e-12)  # uniform start has g = 0 by definition
    assert runs["D"].source_s > 0 and runs["D"].extra["lp_bound"] <= pb.milp.operating_cost * (1 + 1e-9)
    # identical initial parameters: arm A's first evaluation is the circuit at (theta0, beta0)
    from qguc.fastsim import QAOASimulator
    e0 = QAOASimulator(pb.ev.E).energy(th / pb.ev.sigma, be)
    assert runs["A"].energies[0] == pytest.approx(e0, abs=1e-12)


def test_patience_stop_triggers():
    inst = make_scenario_set(1, n_hours=2, seed=8, refinement_level=0)[0]
    pb = make_problem(inst)
    r = run_arm("A", pb, 1, seed=0, settings=OptimiserSettings(patience=5, min_delta=0.5))
    assert r.early_stopped and r.n_evals == 6  # best set at eval 1, then 5 evals without +0.5


def test_lp_relaxation_at_qubo_target():
    for inst in make_scenario_set(4, n_hours=2, seed=12, refinement_level=2, demand_bias_mw=250.0):
        lp_d, lp_q = solve_lp_relaxation(inst, "demand"), solve_lp_relaxation(inst, "qubo")
        gen_q = inst.capacity @ inst.unflatten(lp_q.x)
        assert (gen_q >= inst.target_mw - 1e-6).all()
        assert lp_q.objective >= lp_d.objective - 1e-6  # tighter constraint, higher bound
    with pytest.raises(ValueError):
        solve_lp_relaxation(inst, "reserve")


def test_learned_arms_at_L2():
    from qguc.gnn import MODEL_DIR

    if not (MODEL_DIR / "gnn_milp_L2.pt").exists():
        pytest.skip("GNN models not trained yet")
    pb = make_problem(make_scenario_set(1, n_hours=2, seed=8, refinement_level=2)[0])
    s = OptimiserSettings(cap_per_param=10)
    rB, rC = run_arm("B", pb, 1, seed=3, settings=s), run_arm("C-milp", pb, 1, seed=3, settings=s)
    np.testing.assert_allclose(rB.extra["c"], rC.extra["c"])  # same warm start, different mixer
    assert rB.g_state0 == pytest.approx(rC.g_state0)  # identical initial state
    assert 0 < rC.source_s < 0.05
