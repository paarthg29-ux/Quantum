"""Learned warm start: a 2-layer GCN over the unit graph, one pass per hour.

Plain PyTorch (no torch_geometric): with 5-11 nodes a GCN layer is
H' = act(A_hat @ H @ W), A_hat = D^-1/2 (A + I) D^-1/2.

Unit graph (IEEE 14-bus mapping)
--------------------------------
The IEEE 14-bus test system has exactly five generator buses, 1, 2, 3, 6
and 8; we place the five Haryana stations on them by size (largest at the
slack bus):

    bus 1  IGSTPP Jhajjar      1500 MW      bus 6  DCRTPP Yamunanagar  600 MW
    bus 2  RGTPP  Hisar        1200 MW      bus 8  FGPP   Faridabad    432 MW
    bus 3  PTPS   Panipat       710 MW

Every unit (or aggregated block) of a station sits on the station's bus.
Two blocks are adjacent if they share a bus or their buses are joined by a
line of the standard IEEE 14-bus branch list (IEEE14_LINES). Among the
generator buses only 1-2 and 2-3 are directly connected, so Yamunanagar's
and Faridabad's blocks connect only to their own station's other blocks.
The mapping is a stated modelling assumption, not Haryana's real network.
Every node also carries the hour's system-wide load features, so even an
isolated node conditions on load.

Node features (one graph per hour t; FEATURES order)
----------------------------------------------------
    cap        P_i / 1500 MW               (largest block in any model)
    a          a_i / 300 000 INR/h
    b          b_i / 6200 INR/MWh
    startup    S_i / 4 000 000 INR
    init_on    x_i[-1]
    load_prev  D_{t-1} / 4200 MW           (t = 0: D_0, edge padding)
    load       D_t / 4200 MW
    load_next  D_{t+1} / 4200 MW           (t = T-1: D_{T-1}, edge padding)
    t_frac     t / (T - 1)                 (0 if T = 1)
    dem_cap    D_t / total capacity of the instance
The constants are fleet-wide maxima, fixed so features mean the same thing
for every instance and refinement level. (load and dem_cap are collinear
here because every level has the same capacity; both are kept as specified.)

Labels (per block and hour, in [0, 1])
--------------------------------------
* "milp":   the MILP optimal commitments (scalable to any system size);
* "oracle": the exact QUBO ground state(s) at the frozen operating point,
            averaged over degenerate ground states (not scalable: needs
            brute force; an upper bound on what a learned start can target).
Interchangeable blocks (identical data, see UCInstance.symmetry_groups) get
identical features and graph positions, so the model cannot tell them apart;
their labels are averaged within the group per hour (e.g. 2 of 3 Jhajjar
units on -> 0.667 each), which is the symmetric product state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .instance import UCInstance

GENERATOR_BUS = {"IGSTPP": 1, "RGTPP": 2, "PTPS": 3, "DCRTPP": 6, "FGPP": 8}
IEEE14_LINES = ((1, 2), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (4, 5), (4, 7), (4, 9), (5, 6),
                (6, 11), (6, 12), (6, 13), (7, 8), (7, 9), (9, 10), (9, 14), (10, 11), (12, 13), (13, 14))
FEATURES = ("cap", "a", "b", "startup", "init_on", "load_prev", "load", "load_next", "t_frac", "dem_cap")
_NORM = dict(cap=1500.0, a=300_000.0, b=6200.0, startup=4_000_000.0, load=4200.0)
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def unit_adjacency(plants) -> np.ndarray:
    """0/1 adjacency (no self loops): same bus, or buses joined by an IEEE 14-bus line."""
    lines = {frozenset(e) for e in IEEE14_LINES}
    bus = [GENERATOR_BUS[p.station] for p in plants]
    N = len(plants)
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j and (bus[i] == bus[j] or frozenset((bus[i], bus[j])) in lines):
                A[i, j] = 1.0
    return A


def normalised_adjacency(A) -> np.ndarray:
    A = np.asarray(A, float) + np.eye(len(A))
    d = 1.0 / np.sqrt(A.sum(axis=1))
    return d[:, None] * A * d[None, :]


def node_features(instance: UCInstance) -> np.ndarray:
    """(T, N, len(FEATURES)) float32."""
    T, N = instance.n_hours, instance.n_plants
    D = instance.demand_mw
    X = np.zeros((T, N, len(FEATURES)), np.float32)
    static = np.stack([instance.capacity / _NORM["cap"], np.array([p.a for p in instance.plants]) / _NORM["a"],
                       np.array([p.b for p in instance.plants]) / _NORM["b"], instance.startup_cost / _NORM["startup"],
                       instance.initial_on.astype(float)], axis=1)
    total = instance.capacity.sum()
    for t in range(T):
        X[t, :, :5] = static
        X[t, :, 5] = D[max(t - 1, 0)] / _NORM["load"]
        X[t, :, 6] = D[t] / _NORM["load"]
        X[t, :, 7] = D[min(t + 1, T - 1)] / _NORM["load"]
        X[t, :, 8] = t / (T - 1) if T > 1 else 0.0
        X[t, :, 9] = D[t] / total
    return X


def symmetrise(instance: UCInstance, U: np.ndarray) -> np.ndarray:
    """Average (N, T) labels within each group of interchangeable blocks."""
    U = np.array(U, float, copy=True)
    for g in instance.symmetry_groups():
        U[g] = U[g].mean(axis=0, keepdims=True)
    return U


def milp_labels(instance: UCInstance) -> np.ndarray:
    from .exact import solve_milp

    return symmetrise(instance, instance.unflatten(solve_milp(instance).x))


def oracle_labels(instance: UCInstance, point=None) -> np.ndarray:
    """Mean over the exact QUBO ground states at ``point`` (default FROZEN)."""
    import dataclasses

    from .config import FROZEN
    from .exact import index_to_bits
    from .fastsim import energy_vector

    point = FROZEN if point is None else point
    inst = dataclasses.replace(instance, demand_bias_mw=point.demand_bias_mw)
    ev = energy_vector(inst, point.lambda_demand, point.lambda_switch)
    bits = index_to_bits(ev.ground, inst.n_vars).astype(float)
    return symmetrise(inst, inst.unflatten(bits.mean(axis=0)))


LABELERS = {"milp": milp_labels, "oracle": oracle_labels}


def _torch():
    import torch

    return torch


def make_model(hidden: int = 32):
    torch = _torch()
    nn = torch.nn

    class GCN(nn.Module):
        """logits = A_hat @ relu(A_hat @ X @ W1 + b1) @ W2 + b2, per node."""

        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(len(FEATURES), hidden)
            self.l2 = nn.Linear(hidden, 1)

        def forward(self, A_hat, X):  # A_hat (N, N); X (G, N, F) for G graphs
            H = torch.relu(A_hat @ self.l1(X))
            return (A_hat @ self.l2(H)).squeeze(-1)  # (G, N)

    return GCN()


@dataclass
class WarmStartModel:
    """A trained GCN for one block set (refinement level)."""

    model: object
    A_hat: np.ndarray
    plant_names: tuple
    variant: str
    meta: dict = field(default_factory=dict)

    def _check(self, instance: UCInstance):
        if tuple(p.name for p in instance.plants) != self.plant_names:
            raise ValueError("model was trained on a different block set")

    def predict(self, instance: UCInstance) -> np.ndarray:
        """c as an (N, T) array of P(x[i, t] = 1)."""
        torch = _torch()
        self._check(instance)
        with torch.no_grad():
            logits = self.model(torch.as_tensor(self.A_hat, dtype=torch.float32),
                                torch.as_tensor(node_features(instance)))  # (T, N)
        return torch.sigmoid(logits).numpy().T.astype(float)

    def predict_flat(self, instance: UCInstance) -> np.ndarray:
        return instance.flatten(self.predict(instance))

    def predict_graph(self, instance: UCInstance) -> np.ndarray:
        """c as (N, T) for an instance whose block set MAY DIFFER from training
        (e.g. a forced outage removes a node): the GCN weights are shared across
        nodes, so the trained model is applied on the instance's own normalised
        adjacency. Out-of-distribution use only; ``predict`` keeps the check."""
        torch = _torch()
        A_hat = normalised_adjacency(unit_adjacency(instance.plants))
        with torch.no_grad():
            logits = self.model(torch.as_tensor(A_hat, dtype=torch.float32),
                                torch.as_tensor(node_features(instance)))
        return torch.sigmoid(logits).numpy().T.astype(float)

    def time_forward(self, instance: UCInstance, reps: int = 2000) -> dict:
        """Median wall time (s) of features + forward pass, and of the forward pass alone."""
        torch = _torch()
        A = torch.as_tensor(self.A_hat, dtype=torch.float32)
        Xt = torch.as_tensor(node_features(instance))
        full, fwd = [], []
        with torch.no_grad():
            for _ in range(reps):
                t0 = time.perf_counter()
                self.predict(instance)
                full.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                self.model(A, Xt)
                fwd.append(time.perf_counter() - t0)
        return dict(features_and_forward_s=float(np.median(full)), forward_s=float(np.median(fwd)))

    def save(self, path: Path):
        torch = _torch()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state=self.model.state_dict(), A_hat=self.A_hat, plant_names=self.plant_names,
                        variant=self.variant, meta=self.meta), path)

    @classmethod
    def load(cls, path: Path) -> "WarmStartModel":
        torch = _torch()
        d = torch.load(path, weights_only=False)
        model = make_model(d["meta"].get("hidden", 32))
        model.load_state_dict(d["state"])
        model.eval()
        return cls(model, d["A_hat"], tuple(d["plant_names"]), d["variant"], d["meta"])


def dataset(instances, labeler):
    """Stacked hourly graphs: X (S*T, N, F), Y (S*T, N)."""
    X = np.concatenate([node_features(i) for i in instances])
    Y = np.concatenate([labeler(i).T for i in instances]).astype(np.float32)
    return X, Y


def train(train_set, val_set, variant: str, *, hidden: int = 32, lr: float = 1e-2, weight_decay: float = 1e-4,
          epochs: int = 6000, patience: int = 400, seed: int = 0, labels=None) -> WarmStartModel:
    """Full-batch Adam on BCE; keeps the weights with the lowest validation loss.
    ``labels`` = precomputed ((Xtr, Ytr), (Xva, Yva)) to skip labelling."""
    torch = _torch()
    torch.manual_seed(seed)
    plants = train_set[0].plants
    A_hat = normalised_adjacency(unit_adjacency(plants))
    if labels is None:
        labels = (dataset(train_set, LABELERS[variant]), dataset(val_set, LABELERS[variant]))
    (Xtr, Ytr), (Xva, Yva) = labels
    A = torch.as_tensor(A_hat, dtype=torch.float32)
    Xtr, Ytr, Xva, Yva = (torch.as_tensor(v) for v in (Xtr, Ytr, Xva, Yva))
    model = make_model(hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best, best_state, best_epoch, history = np.inf, None, 0, []
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(A, Xtr), Ytr)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(A, Xva), Yva))
        history.append((float(loss.detach()), vl))
        if vl < best - 1e-6:
            best, best_epoch = vl, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch > patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    meta = dict(hidden=hidden, lr=lr, weight_decay=weight_decay, best_epoch=best_epoch, best_val_bce=best,
                epochs_run=len(history), seed=seed, n_train=len(train_set), n_val=len(val_set),
                history=history[:: max(1, len(history) // 200)])
    return WarmStartModel(model, A_hat, tuple(p.name for p in plants), variant, meta)


def load_conditioning(model: WarmStartModel, low: UCInstance, high: UCInstance) -> dict:
    """How strongly the prediction depends on load: mean |c_low - c_high| and
    the difference in expected committed capacity sum_i P_i c_i per hour."""
    cl, ch = model.predict(low), model.predict(high)
    cap_l, cap_h = low.capacity @ cl, high.capacity @ ch
    return dict(mean_abs_diff=float(np.abs(cl - ch).mean()),
                committed_mw_diff=(cap_h - cap_l).tolist(),
                load_diff_mw=(high.demand_mw - low.demand_mw).tolist())


def assert_load_conditioned(model: WarmStartModel, low: UCInstance, high: UCInstance,
                            min_abs_diff: float = 0.15, min_capacity_fraction: float = 0.5) -> dict:
    """HARD check that the model conditions on load (a constant model fails):
    outputs must differ by >= ``min_abs_diff`` on average, and the expected
    committed capacity must rise by >= ``min_capacity_fraction`` of the load
    increase in EVERY hour."""
    m = load_conditioning(model, low, high)
    ok_cap = all(dc >= min_capacity_fraction * dl for dc, dl in zip(m["committed_mw_diff"], m["load_diff_mw"]))
    if m["mean_abs_diff"] < min_abs_diff or not ok_cap:
        raise AssertionError(f"warm-start model does not condition on load: {m}")
    return m
