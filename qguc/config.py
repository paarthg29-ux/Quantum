"""Frozen operating point of the penalty encoding.

Chosen once in step 2 (scripts/s2_calibrate.py) and not tuned afterwards;
all downstream experiments use it.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class OperatingPoint:
    model: str  # 'station' or 'unit' (see qguc.instance)
    lambda_demand: float  # INR / MW^2
    demand_bias_mw: float  # penalty-target bias, NOT a reserve
    lambda_switch: float = 0.0
    calibrated_n_hours: int = 2


# s2_calibrate.py, plateau rule (k = 1), 100 selection scenarios (seed 0), T = 2:
# lambda grid point 16 of logspace(0, 3.5, 29) = 10**2.0. Test set (40, seed 2026):
# adequacy 100%, encoding gap mean 2.48%, worst 7.80%.
FROZEN = OperatingPoint(model="unit", lambda_demand=100.0, demand_bias_mw=150.0)
