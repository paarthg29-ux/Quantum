"""GCN warm start: graph mapping, features, labels, and the HARD load-conditioning test."""

import numpy as np
import pytest

pytest.importorskip("torch")

from qguc import make_instance, make_scenario_set, refinement_fleet  # noqa: E402
from qguc.gnn import (  # noqa: E402
    FEATURES,
    MODEL_DIR,
    WarmStartModel,
    assert_load_conditioned,
    milp_labels,
    node_features,
    normalised_adjacency,
    oracle_labels,
    train,
    unit_adjacency,
)


def probe_pair(level=2):
    return (make_instance(2500.0, n_hours=2, seed=0, refinement_level=level, demand_bias_mw=250.0),
            make_instance(4200.0, n_hours=2, seed=0, refinement_level=level, demand_bias_mw=250.0))


def test_unit_graph_follows_the_ieee14_mapping():
    plants = refinement_fleet(2)  # Panipat, Faridabad, Hisar x2, Yamunanagar, Jhajjar x3
    names = [p.name for p in plants]
    A = unit_adjacency(plants)
    ix = {n: i for i, n in enumerate(names)}
    assert np.array_equal(A, A.T) and np.all(np.diag(A) == 0)
    assert A[ix["RGTPP-1"], ix["RGTPP-2"]] == 1  # same bus
    assert A[ix["IGSTPP-1"], ix["RGTPP-1"]] == 1  # line 1-2
    assert A[ix["RGTPP-2"], ix["Panipat"]] == 1  # line 2-3
    assert A[ix["IGSTPP-2"], ix["Panipat"]] == 0  # buses 1 and 3: no direct line
    assert A[ix["Faridabad"]].sum() == 0 and A[ix["Yamunanagar"]].sum() == 0  # buses 8 and 6 isolated
    Ah = normalised_adjacency(A)
    np.testing.assert_allclose(Ah, Ah.T)
    assert np.all(np.linalg.eigvalsh(Ah) <= 1 + 1e-12)


def test_features_shape_and_temporal_context():
    inst = make_scenario_set(1, n_hours=3, seed=0, refinement_level=2)[0]
    X = node_features(inst)
    assert X.shape == (3, 8, len(FEATURES))
    f = {k: i for i, k in enumerate(FEATURES)}
    D = inst.demand_mw / 4200.0
    np.testing.assert_allclose(X[1, 0, [f["load_prev"], f["load"], f["load_next"]]], D[[0, 1, 2]], rtol=1e-6)
    np.testing.assert_allclose(X[0, 0, f["load_prev"]], D[0], rtol=1e-6)  # edge padding
    np.testing.assert_allclose(X[:, 0, f["t_frac"]], [0, 0.5, 1])


def test_labels_are_symmetric_and_match_solvers():
    inst = make_scenario_set(3, n_hours=2, seed=4, refinement_level=2, demand_bias_mw=250.0)[2]
    for lab in (milp_labels(inst), oracle_labels(inst)):
        assert lab.shape == (8, 2) and lab.min() >= 0 and lab.max() <= 1
        for g in inst.symmetry_groups():
            assert np.allclose(lab[g], lab[g][0])
        # symmetrisation preserves committed capacity per hour
    from qguc.exact import solve_milp
    np.testing.assert_allclose(inst.capacity @ milp_labels(inst),
                               inst.capacity @ inst.unflatten(solve_milp(inst).x))


@pytest.fixture(scope="module")
def small_model():
    tr = make_scenario_set(120, n_hours=2, seed=101, refinement_level=2, demand_bias_mw=250.0)
    va = make_scenario_set(30, n_hours=2, seed=102, refinement_level=2, demand_bias_mw=250.0)
    return train(tr, va, "milp", epochs=600, patience=150, seed=0)


def test_trained_model_conditions_on_load(small_model):
    """HARD test (a real defect of the mid-term version): a 2500 MW and a
    4200 MW instance must give substantially different warm starts."""
    low, high = probe_pair()
    m = assert_load_conditioned(small_model, low, high)
    assert m["mean_abs_diff"] >= 0.15


def test_load_check_rejects_a_constant_model(small_model):
    """The check itself must be able to fail: zero the weights that read the
    load features and it must raise."""
    import copy

    import torch

    const = copy.deepcopy(small_model)
    with torch.no_grad():
        for k in ("load_prev", "load", "load_next", "dem_cap"):
            const.model.l1.weight[:, FEATURES.index(k)] = 0.0
    low, high = probe_pair()
    with pytest.raises(AssertionError):
        assert_load_conditioned(const, low, high)


def test_predict_save_load_roundtrip(small_model, tmp_path):
    inst = make_scenario_set(1, n_hours=2, seed=7, refinement_level=2)[0]
    c = small_model.predict(inst)
    assert c.shape == (8, 2) and np.all((c > 0) & (c < 1))
    small_model.save(tmp_path / "m.pt")
    np.testing.assert_allclose(WarmStartModel.load(tmp_path / "m.pt").predict(inst), c, atol=1e-7)
    with pytest.raises(ValueError):
        small_model.predict(make_scenario_set(1, n_hours=2, seed=7, refinement_level=3)[0])


@pytest.mark.parametrize("variant", ["milp", "oracle"])
def test_saved_production_models_condition_on_load(variant):
    path = MODEL_DIR / f"gnn_{variant}_L2.pt"
    if not path.exists():
        pytest.skip("production model not trained yet (scripts/s7a_train_gnn.py)")
    assert_load_conditioned(WarmStartModel.load(path), *probe_pair())


def test_predict_graph_matches_predict_and_handles_outage():
    """predict_graph == predict on the training block set; on a fleet with one block removed
    it runs on the reduced graph (predict refuses it)."""
    import dataclasses

    import numpy as np

    from qguc import make_instance
    from qguc.gnn import MODEL_DIR, WarmStartModel

    path = MODEL_DIR / "gnn_milp_L2.pt"
    if not path.exists():
        pytest.skip("L2 model not trained")
    m = WarmStartModel.load(path)
    inst = make_instance(3300.0, n_hours=2, seed=1, refinement_level=2)
    np.testing.assert_allclose(m.predict_graph(inst), m.predict(inst), atol=1e-6)
    keep = np.array([p.name != "Yamunanagar" for p in inst.plants])
    out = dataclasses.replace(inst, plants=tuple(p for p, k in zip(inst.plants, keep) if k),
                              initial_on=inst.initial_on[keep])
    c = m.predict_graph(out)
    assert c.shape == (out.n_plants, out.n_hours) and np.all((c >= 0) & (c <= 1))
    with pytest.raises(ValueError):
        m.predict(out)
