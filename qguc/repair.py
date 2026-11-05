"""Greedy adequacy repair of a schedule (post-processing; the encoding is untouched).

For every hour that is short of the TRUE demand D_t, uncommitted blocks are
committed in order of lowest full-load cost per MW, (a_i + b_i P_i) / P_i,
until the hour is adequate. Blocks are only ever added, never removed, and
start-up costs are not part of the ordering (they are part of the repaired
schedule's cost). Since the fleet capacity exceeds every demand, the result
is always adequate.
"""

from __future__ import annotations

import numpy as np

from .instance import UCInstance


def cost_per_mw_order(instance: UCInstance) -> np.ndarray:
    """Block indices by full-load cost per MW, cheapest first (stable)."""
    return np.argsort(instance.hourly_cost / instance.capacity, kind="stable")


def repair(instance: UCInstance, x) -> np.ndarray:
    """Repaired copy of schedule(s) x, shape (n_vars,) or (m, n_vars)."""
    x = np.asarray(x)
    single = x.ndim == 1
    X = np.atleast_2d(x)
    U = np.array(instance.unflatten(X), dtype=np.int8, copy=True)  # (m, N, T)
    P = instance.capacity
    gen = np.einsum("i,mit->mt", P, U)
    for t in range(instance.n_hours):
        need = instance.demand_mw[t]
        for i in cost_per_mw_order(instance):
            add = (gen[:, t] < need) & (U[:, i, t] == 0)
            U[add, i, t] = 1
            gen[add, t] += P[i]
    out = instance.flatten(U).astype(int)
    if not instance.is_adequate(out).all():
        raise RuntimeError("repair failed: demand exceeds fleet capacity")
    return out[0] if single else out
