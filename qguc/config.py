"""Frozen operating point of the penalty encoding.

Chosen once and not tuned afterwards; all downstream experiments use it.

History (each step is recorded in results/):
1. s2_calibrate.py, unit model (= ladder L4), plateau rule k = 1, 100
   selection scenarios (seed 0), T = 2: lambda_D = 100, bias = 150 MW
   (test, 40 scenarios, seed 2026: adequacy 100%, gap mean 2.48%, worst 7.80%).
2. s4_ladder.py re-verified that point on the main problem size, ladder L2
   (8 blocks, 16 qubits), over 100 fresh scenarios (seed 4242): adequacy
   89% < 95%, so by the pre-agreed rule it was re-calibrated ONCE on L2.
3. s2_calibrate.py L2 (same grid, sets and plateau rule): lambda_D = 100,
   bias = 250 MW (selection 100% adequate, gap mean 4.95%, worst 18.27%;
   test 100% adequate, gap mean 5.66%, worst 20.67%). This is FROZEN.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class OperatingPoint:
    model: str  # fleet model it was calibrated on ('station', 'unit', 'L0'..'L4'; see qguc.instance)
    lambda_demand: float  # INR / MW^2
    demand_bias_mw: float  # penalty-target bias, NOT a reserve
    lambda_switch: float = 0.0
    calibrated_n_hours: int = 2

    @property
    def refinement_level(self) -> int | None:
        return int(self.model[1:]) if self.model.startswith("L") else None


FROZEN = OperatingPoint(model="L2", lambda_demand=100.0, demand_bias_mw=250.0)
SUPERSEDED = (OperatingPoint(model="unit", lambda_demand=100.0, demand_bias_mw=150.0),)
