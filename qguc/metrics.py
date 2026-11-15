"""Named quantities reported in the paper."""

from __future__ import annotations

import numpy as np

from .exact import OperationalResult, solve_milp
from .instance import UCInstance


def encoding_gap(instance: UCInstance, x, milp: OperationalResult | None = None) -> float:
    """Relative operating-cost gap of a schedule to the operational optimum.

        encoding_gap = (C(x) - C(x_MILP)) / C(x_MILP)

    where C is the operating cost, fuel + start-up ONLY (no penalty term),
    and x_MILP is the MILP optimum enforcing generation >= demand D_t in
    every hour (:func:`qguc.exact.solve_milp`). Returned as a fraction.

    Evaluated at x = the exact ground state of the QUBO, this is the
    ENCODING GAP: the cost of replacing the hard adequacy constraint by a
    soft quadratic penalty (with its weight lambda_D and demand bias B).
    It is fixed before any quantum computation and is a property of the
    encoding, not of QAOA. A solver that returns the QUBO ground state
    exactly still pays it, so it bounds how close any end-to-end result
    that minimises this QUBO can get to the operational optimum.

    The quantity is meaningful only for an adequate x. An inadequate
    schedule can be cheaper than the optimum (negative gap) precisely
    because it does not serve demand; such cases must be reported as
    adequacy failures, not as gaps.
    """
    if milp is None:
        milp = solve_milp(instance)
    cost = float(instance.operating_cost(np.asarray(x)))
    return (cost - milp.operating_cost) / milp.operating_cost


def approximation_ratio(energy, e_min: float, e_max: float):
    """(E_max - E) / (E_max - E_min): 1 at the ground state, 0 at the worst.

    This form is invariant under any positive affine map E -> a E + b
    (a > 0), so it gives the same value in physical INR and in normalised
    Ising units. The naive ratio E / E_min is NOT invariant to the constant
    shift and must not be used.
    """
    width = e_max - e_min
    if width == 0:
        return np.ones_like(np.asarray(energy, dtype=float))
    return (e_max - np.asarray(energy, dtype=float)) / width
