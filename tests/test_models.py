import dataclasses

import numpy as np
import pytest

from qguc import (
    HARYANA_STATIONS,
    HARYANA_UNITS,
    default_initial_on,
    encoding_gap,
    make_instance,
    make_scenario_set,
    select_plants,
)
from qguc.calibrate import cross_check, select_operating_point, sweep_grid, sweep_grid_parallel
from qguc.exact import solve_milp


def test_unit_fleet_matches_station_fleet():
    assert len(HARYANA_UNITS) == 11
    assert sum(u.capacity_mw for u in HARYANA_UNITS) == 4442
    for st in HARYANA_STATIONS:
        units = [u for u in HARYANA_UNITS if u.station == st.station]
        assert sum(u.capacity_mw for u in units) == st.capacity_mw
        assert sum(u.hourly_cost for u in units) == pytest.approx(st.hourly_cost, rel=1e-12)
        assert sum(u.startup for u in units) == pytest.approx(st.startup, rel=1e-12)
        assert all(u.b == st.b for u in units)


def test_whole_station_schedule_costs_the_same_in_both_models():
    rng = np.random.default_rng(0)
    st = make_instance(3500.0, n_hours=4, seed=3, plants="station")
    un = make_instance(3500.0, n_hours=4, seed=3, plants="unit")
    assert np.array_equal(st.demand_mw, un.demand_mw)
    # Map station rows onto all of that station's units.
    rows = [[j for j, u in enumerate(un.plants) if u.station == p.station] for p in st.plants]
    for _ in range(50):
        us = rng.integers(0, 2, size=(5, 4))
        uu = np.zeros((11, 4), dtype=int)
        for i, r in enumerate(rows):
            uu[r] = us[i]
        xs, xu = st.flatten(us), un.flatten(uu)
        assert np.array_equal(st.generation(xs), un.generation(xu))
        assert st.fuel_cost(xs) == pytest.approx(un.fuel_cost(xu), rel=1e-12)
        assert st.startup_expense(xs) == pytest.approx(un.startup_expense(xu), rel=1e-12)


def test_default_initial_on_agrees_across_models():
    st = default_initial_on(HARYANA_STATIONS)
    un = default_initial_on(HARYANA_UNITS)
    on_st = {p.station for p, v in zip(HARYANA_STATIONS, st) if v}
    on_un = {p.station for p, v in zip(HARYANA_UNITS, un) if v}
    assert on_st == on_un == {"IGSTPP", "RGTPP"}
    assert all(v for p, v in zip(HARYANA_UNITS, un) if p.station in on_st)


def test_scenario_sets_share_demand_across_models():
    a = make_scenario_set(5, n_hours=2, seed=4, plants="station")
    b = make_scenario_set(5, n_hours=2, seed=4, plants="unit")
    for x, y in zip(a, b):
        assert np.array_equal(x.demand_mw, y.demand_mw)
        assert y.n_vars == 22 and x.n_vars == 10


def test_unit_subset_selection_and_scaled_load():
    plants = select_plants("unit", ["IGSTPP", "RGTPP-1", "FGPP-1"])
    assert [p.name for p in plants] == ["FGPP-1", "RGTPP-1", "IGSTPP-1", "IGSTPP-2", "IGSTPP-3"]
    cap = sum(p.capacity_mw for p in plants)
    for inst in make_scenario_set(20, n_hours=3, seed=1, plants=plants):
        assert inst.n_vars == 5 * 3
        assert np.all(inst.demand_mw <= 4200 * cap / 4442 + 1e-9)
        assert inst.is_adequate(solve_milp(inst).x)
    with pytest.raises(ValueError):
        select_plants("unit", ["PTPS-9"])


def test_symmetry_canonical_form():
    inst = make_instance(3500.0, n_hours=2, seed=0, plants="unit")
    groups = inst.symmetry_groups()
    names = sorted(sorted(inst.plants[i].name for i in g) for g in groups)
    assert ["IGSTPP-1", "IGSTPP-2", "IGSTPP-3"] in names
    assert ["PTPS-7", "PTPS-8"] in names and ["RGTPP-1", "RGTPP-2"] in names
    u = np.zeros((11, 2), dtype=int)
    i1, i2 = (next(i for i, p in enumerate(inst.plants) if p.name == n) for n in ("IGSTPP-1", "IGSTPP-3"))
    u[i1] = [1, 0]
    v = u.copy()
    v[i1], v[i2] = [0, 0], [1, 0]
    assert inst.same_schedule(inst.flatten(u), inst.flatten(v))
    w = u.copy()
    w[i2] = [0, 1]  # a different schedule, not a relabelling
    assert not inst.same_schedule(inst.flatten(u), inst.flatten(w))


def test_milp_ignores_demand_bias():
    inst = make_instance(3600.0, n_hours=3, seed=2)
    a = solve_milp(inst)
    b = solve_milp(dataclasses.replace(inst, demand_bias_mw=300.0))
    assert a.operating_cost == b.operating_cost


def test_encoding_gap_definition():
    inst = make_instance(3600.0, n_hours=3, seed=2)
    milp = solve_milp(inst)
    assert encoding_gap(inst, milp.x, milp) == 0.0
    all_on = np.ones(inst.n_vars, dtype=int)
    expected = (inst.operating_cost(all_on) - milp.operating_cost) / milp.operating_cost
    assert encoding_gap(inst, all_on) == pytest.approx(expected, rel=1e-12)
    assert expected > 0


def test_grid_engine_agrees_with_full_brute_force():
    scen = make_scenario_set(3, n_hours=2, seed=9, plants="station")
    grid = sweep_grid(scen, lambdas=[3.0, 30.0, 300.0], biases=[0.0, 150.0])
    cross_check(scen, grid, n_checks=12, rng=np.random.default_rng(0))
    small_units = make_scenario_set(2, n_hours=2, seed=9, plants=select_plants("unit", ["IGSTPP", "PTPS", "FGPP"]))
    grid_u = sweep_grid(small_units, lambdas=[10.0, 100.0], biases=[0.0, 100.0])
    cross_check(small_units, grid_u, n_checks=8, rng=np.random.default_rng(1))
    assert np.all(grid_u.degeneracy >= grid_u.classes)
    cell = select_operating_point(grid)
    if cell is not None:
        assert grid.adequacy_rate[cell] == 1.0


def test_parallel_sweep_matches_sequential():
    insts = make_scenario_set(6, n_hours=2, seed=3, plants="station")
    lams, biases = [10.0, 100.0], [0.0, 200.0]
    seq = sweep_grid(insts, lams, biases)
    par = sweep_grid_parallel(insts, lams, biases, workers=3)
    for name in ("adequate", "gap", "match", "degeneracy", "classes", "dyn_range"):
        np.testing.assert_array_equal(getattr(seq, name), getattr(par, name))
    assert seq.labels == par.labels and seq.ground.keys() == par.ground.keys()
    for k in seq.ground:
        np.testing.assert_array_equal(seq.ground[k], par.ground[k])


# ---- aggregation ladder ---------------------------------------------------

from qguc import HARYANA_STATIONS, HARYANA_UNITS, REFINEMENT_LEVELS, refinement_fleet  # noqa: E402


@pytest.mark.parametrize("level,n_blocks", [(0, 5), (1, 6), (2, 8), (3, 9), (4, 11)])
def test_ladder_sizes_and_constant_capacity(level, n_blocks):
    fleet = refinement_fleet(level)
    assert len(fleet) == n_blocks
    assert sum(p.capacity_mw for p in fleet) == 4442
    # every unit is covered exactly once: either as itself, or inside its
    # station's aggregate block (which then is the only block of that station)
    covered = []
    for p in fleet:
        units = [u for u in HARYANA_UNITS if u.station == p.station]
        covered += [p.name] if p in units else [u.name for u in units]
        if p not in units:
            assert sum(q.station == p.station for q in fleet) == 1
    assert sorted(covered) == sorted(u.name for u in HARYANA_UNITS)


@pytest.mark.parametrize("level", REFINEMENT_LEVELS)
def test_aggregated_block_coefficients_are_sums_of_constituents(level):
    for block in refinement_fleet(level):
        units = [u for u in HARYANA_UNITS if u.station == block.station]
        if block in units:
            continue  # a split unit is its own constituent
        assert block.capacity_mw == pytest.approx(sum(u.capacity_mw for u in units), rel=1e-15)
        assert block.a == pytest.approx(sum(u.a for u in units), rel=1e-15)
        assert block.startup == pytest.approx(sum(u.startup for u in units), rel=1e-15)
        assert all(block.b == u.b for u in units)
        assert block.hourly_cost == pytest.approx(sum(u.hourly_cost for u in units), rel=1e-12)


def test_ladder_endpoints_are_the_station_and_unit_models():
    assert refinement_fleet(4) == HARYANA_UNITS
    l0 = refinement_fleet(0)
    for blk, st in zip(l0, HARYANA_STATIONS):
        assert (blk.name, blk.station, blk.capacity_mw, blk.b) == (st.name, st.station, st.capacity_mw, st.b)
        assert blk.a == pytest.approx(st.a, rel=1e-12) and blk.startup == pytest.approx(st.startup, rel=1e-12)


def test_ladder_split_order_and_units():
    names = [{p.name for p in refinement_fleet(k)} for k in REFINEMENT_LEVELS]
    assert {"RGTPP-1", "RGTPP-2"} <= names[1] and "Jhajjar" in names[1]
    assert {"IGSTPP-1", "IGSTPP-2", "IGSTPP-3"} <= names[2] and "Yamunanagar" in names[2]
    assert {"DCRTPP-1", "DCRTPP-2"} <= names[3] and "Panipat" in names[3]
    assert {"PTPS-6", "PTPS-7", "PTPS-8"} <= names[4]


def test_ladder_levels_share_demand_and_initial_state():
    sets = [make_scenario_set(4, n_hours=3, seed=9, refinement_level=k) for k in REFINEMENT_LEVELS]
    for k, insts in zip(REFINEMENT_LEVELS, sets):
        for a, b in zip(sets[0], insts):
            np.testing.assert_array_equal(a.demand_mw, b.demand_mw)
            assert b.label.split("_")[1] == f"L{k}"
            # hour -1: Jhajjar and Hisar on, whatever the granularity
            on = {p.station for p, s in zip(b.plants, b.initial_on) if s}
            assert on == {"IGSTPP", "RGTPP"} and all(
                s == (p.station in on) for p, s in zip(b.plants, b.initial_on))


@pytest.mark.parametrize("level", REFINEMENT_LEVELS)
def test_whole_station_schedules_cost_the_same_at_every_level(level):
    """Committing whole stations gives identical generation and operating
    cost at every refinement level."""
    rng = np.random.default_rng(level)
    base = make_scenario_set(1, n_hours=3, seed=1, refinement_level=0)[0]
    inst = make_scenario_set(1, n_hours=3, seed=1, refinement_level=level)[0]
    for _ in range(20):
        u0 = rng.integers(0, 2, (base.n_plants, base.n_hours))
        st = {p.station: u0[i] for i, p in enumerate(base.plants)}
        u = np.array([st[p.station] for p in inst.plants])
        x0, x = base.flatten(u0), inst.flatten(u)
        np.testing.assert_allclose(inst.generation(x), base.generation(x0))
        assert inst.operating_cost(x) == pytest.approx(base.operating_cost(x0), rel=1e-12)
