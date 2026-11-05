"""Multi-period unit-commitment instances for the Haryana thermal fleet.

Two fleet models describe the same 4442 MW system at different granularity.

Station-level model (5 blocks)
------------------------------
Installed capacities P_i (MW) come from public plant listings. The cost
coefficients are NOT measured heat-rate or tariff data. They are DOCUMENTED
ASSUMPTIONS, chosen to be plausible in order of magnitude and to give a
sensible merit order (large modern coal cheapest, old subcritical coal
dearer, gas CCGT dearest per MWh but cheapest to start). Results that depend
on these numbers are results on a stylised instance, not a cost estimate for
the Haryana grid.

    idx  name         station  P_i(MW)  a_i(INR/h)  b_i(INR/MWh)  S_i(INR)
    q0   Panipat      PTPS       710      180000       4300        2000000
    q1   Faridabad    FGPP       432      120000       6200         600000
    q2   Hisar        RGTPP     1200      250000       3600        3200000
    q3   Yamunanagar  DCRTPP     600      160000       3900        1800000
    q4   Jhajjar      IGSTPP    1500      300000       3200        4000000

Unit-level model (11 blocks)
----------------------------
Real unit commitment decides individual generating units, not stations.
The unit split below is likewise a stated assumption:

    station  unit       MW   type
    PTPS     PTPS-6    210   subcritical coal
    PTPS     PTPS-7    250   subcritical coal
    PTPS     PTPS-8    250   subcritical coal
    FGPP     FGPP-1    432   gas CCGT (single block)
    RGTPP    RGTPP-1   600   coal
    RGTPP    RGTPP-2   600   coal
    DCRTPP   DCRTPP-1  300   coal
    DCRTPP   DCRTPP-2  300   coal
    IGSTPP   IGSTPP-1  500   subcritical coal
    IGSTPP   IGSTPP-2  500   subcritical coal
    IGSTPP   IGSTPP-3  500   subcritical coal

Unit costs are derived from the parent station by capacity share
s_u = P_u / P_station:

    a_u = s_u * a_station        (INR/h)
    b_u = b_station              (INR/MWh; a per-MWh rate needs no scaling)
    S_u = s_u * S_station        (INR per start)

so a unit's full-block hourly cost a_u + b_u P_u = s_u (a_s + b_s P_s), and
committing every unit of a station costs exactly what committing the
station costs in the station model, hourly and at start-up. The two models
therefore differ only in granularity. The derivation ignores the real
economies of scale between unit sizes and station-level fixed costs.

Cost model
----------
Decision x[i,t] in {0,1}: block i committed in hour t. A committed block
produces its FULL RATED OUTPUT P_i, so

    hourly fuel cost of block i = a_i + b_i * P_i      (if committed)
    start-up cost               = S_i * x[i,t] * (1 - x[i,t-1])

with x[i,-1] given by ``initial_on``.

Demand bias
-----------
``demand_bias_mw`` (B, default 0) shifts the TARGET of the QUBO demand
penalty from D_t to D_t + B. It is a bias on the penalty target, nothing
more. With fixed output blocks, total generation takes only finitely many
values per hour and the symmetric penalty (g - D)^2 settles on the nearest
achievable level, which is often below demand. Shifting the target upward
moves the nearest achievable level up and improves adequacy.

It is NOT spinning reserve. A committed block runs at full output and has
no headroom, so the surplus a bias produces is wasted generation, not
reserve, and the QUBO ground state does not in general meet D_t + B
anyway. Adequacy (:meth:`UCInstance.is_adequate`) is always judged against
the true demand D_t, and the MILP baseline enforces generation >= D_t.

Limitations (to be stated in the paper)
---------------------------------------
* Fixed-block output. Blocks are not dispatchable between Pmin and Pmax;
  economic dispatch is not modelled.
* Cost coefficients and the unit split are assumptions (see above).
* No ramp limits, reserve requirement, network constraints or must-run
  status. Minimum up/down time is not enforced; :mod:`qguc.qubo` offers an
  optional quadratic switching penalty (off by default).
* IGSTPP Jhajjar's output is shared with Delhi; here the whole station is
  treated as available to Haryana.

Fleet selection
---------------
:func:`select_plants` returns either model or any subset of its blocks; the
problem size is n_vars = n_blocks * n_hours. Generated loads span
[2500, 4200] MW for the full 4442 MW fleet and are scaled by capacity
share for a subset, so every selection sees the same relative loading.

Variable indexing
-----------------
Variables are flattened TIME-MAJOR so that the blocks coupled by one hour's
demand penalty are contiguous. :meth:`UCInstance.var_index`,
:meth:`UCInstance.unflatten` and :meth:`UCInstance.flatten` are the only
places that know the layout; all other code must go through them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LOAD_MIN_MW = 2500.0
LOAD_MAX_MW = 4200.0


@dataclass(frozen=True)
class Plant:
    """One commitment block: a whole station or a single unit."""

    name: str
    station: str
    capacity_mw: float
    a: float  # no-load cost while committed, INR/h
    b: float  # incremental cost, INR/MWh
    startup: float  # INR per start
    kind: str

    @property
    def hourly_cost(self) -> float:
        """Cost of one committed hour at the full block, INR/h."""
        return self.a + self.b * self.capacity_mw


HARYANA_STATIONS: tuple[Plant, ...] = (
    Plant("Panipat", "PTPS", 710, 180_000, 4300, 2_000_000, "old subcritical coal"),
    Plant("Faridabad", "FGPP", 432, 120_000, 6200, 600_000, "gas CCGT"),
    Plant("Hisar", "RGTPP", 1200, 250_000, 3600, 3_200_000, "coal"),
    Plant("Yamunanagar", "DCRTPP", 600, 160_000, 3900, 1_800_000, "coal"),
    Plant("Jhajjar", "IGSTPP", 1500, 300_000, 3200, 4_000_000, "subcritical coal (3 x 500 MW)"),
)

_UNIT_SPECS = (  # (station code, unit name, MW, type)
    ("PTPS", "PTPS-6", 210, "subcritical coal"),
    ("PTPS", "PTPS-7", 250, "subcritical coal"),
    ("PTPS", "PTPS-8", 250, "subcritical coal"),
    ("FGPP", "FGPP-1", 432, "gas CCGT (single block)"),
    ("RGTPP", "RGTPP-1", 600, "coal"),
    ("RGTPP", "RGTPP-2", 600, "coal"),
    ("DCRTPP", "DCRTPP-1", 300, "coal"),
    ("DCRTPP", "DCRTPP-2", 300, "coal"),
    ("IGSTPP", "IGSTPP-1", 500, "subcritical coal"),
    ("IGSTPP", "IGSTPP-2", 500, "subcritical coal"),
    ("IGSTPP", "IGSTPP-3", 500, "subcritical coal"),
)


def _derive_units(stations, specs) -> tuple[Plant, ...]:
    """Unit blocks with costs scaled from the parent station by capacity share."""
    by_code = {s.station: s for s in stations}
    units = []
    for code, name, mw, kind in specs:
        st = by_code[code]
        share = mw / st.capacity_mw
        units.append(Plant(name, code, mw, share * st.a, st.b, share * st.startup, kind))
    for st in stations:
        total = sum(u.capacity_mw for u in units if u.station == st.station)
        if total != st.capacity_mw:
            raise ValueError(f"units of {st.station} sum to {total} MW, station is {st.capacity_mw}")
    return tuple(units)


HARYANA_UNITS: tuple[Plant, ...] = _derive_units(HARYANA_STATIONS, _UNIT_SPECS)
FLEET_CAPACITY_MW = float(sum(p.capacity_mw for p in HARYANA_STATIONS))
MODELS = {"station": HARYANA_STATIONS, "unit": HARYANA_UNITS}


def select_plants(model: str = "station", names=None) -> tuple[Plant, ...]:
    """Blocks of ``model`` ('station' or 'unit'), optionally only those named.

    ``names`` may hold block names (e.g. 'IGSTPP-2', 'Hisar') or station
    codes (e.g. 'IGSTPP', selecting all its blocks). Fleet order is kept.
    """
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {sorted(MODELS)}")
    fleet = MODELS[model]
    if names is None:
        return fleet
    wanted = set(names)
    chosen = tuple(p for p in fleet if p.name in wanted or p.station in wanted)
    known = {p.name for p in fleet} | {p.station for p in fleet}
    if wanted - known:
        raise ValueError(f"unknown blocks for model {model!r}: {sorted(wanted - known)}")
    return chosen


def _resolve_plants(plants) -> tuple[tuple[Plant, ...], str]:
    if isinstance(plants, str):
        return select_plants(plants), plants
    plants = tuple(plants)
    for name, fleet in MODELS.items():
        if plants == fleet:
            return plants, name
    return plants, "custom"


@dataclass
class UCInstance:
    """One multi-period unit-commitment problem.

    ``demand_mw`` has shape (T,), ``initial_on`` shape (N,) and holds the
    binary commitment state in hour -1. ``demand_bias_mw`` shifts the QUBO
    penalty target (see module docstring); it never affects adequacy or the
    MILP. Methods taking a schedule ``x`` accept a flat vector of length
    ``n_vars`` or a batch of shape (..., n_vars).
    """

    plants: tuple[Plant, ...]
    demand_mw: np.ndarray
    initial_on: np.ndarray
    label: str = ""
    meta: dict = field(default_factory=dict)
    demand_bias_mw: float = 0.0

    def __post_init__(self) -> None:
        self.plants = tuple(self.plants)
        self.demand_mw = np.asarray(self.demand_mw, dtype=float).reshape(-1)
        self.initial_on = np.asarray(self.initial_on, dtype=int).reshape(-1)
        self.demand_bias_mw = float(self.demand_bias_mw)
        if self.demand_bias_mw < 0:
            raise ValueError("demand_bias_mw must be non-negative")
        if self.initial_on.shape != (self.n_plants,):
            raise ValueError(
                f"initial_on has shape {self.initial_on.shape}, expected ({self.n_plants},)"
            )
        if not np.isin(self.initial_on, (0, 1)).all():
            raise ValueError("initial_on must be binary")
        if self.n_hours < 1:
            raise ValueError("demand_mw must cover at least one hour")

    # ---- sizes and per-plant arrays -------------------------------------

    @property
    def n_plants(self) -> int:
        return len(self.plants)

    @property
    def n_hours(self) -> int:
        return self.demand_mw.shape[0]

    @property
    def n_vars(self) -> int:
        return self.n_plants * self.n_hours

    @property
    def target_mw(self) -> np.ndarray:
        """QUBO penalty target D_t + B, shape (T,). Not a constraint."""
        return self.demand_mw + self.demand_bias_mw

    @property
    def capacity(self) -> np.ndarray:
        return np.array([p.capacity_mw for p in self.plants], dtype=float)

    @property
    def startup_cost(self) -> np.ndarray:
        return np.array([p.startup for p in self.plants], dtype=float)

    @property
    def hourly_cost(self) -> np.ndarray:
        """a_i + b_i * P_i for each plant, INR/h."""
        return np.array([p.hourly_cost for p in self.plants], dtype=float)

    # ---- variable layout (the only place that knows it) -----------------

    def var_index(self, i, t):
        """Flat index of x[i, t]. Works elementwise on integer arrays."""
        return t * self.n_plants + i

    def unflatten(self, x) -> np.ndarray:
        """(..., n_vars) -> (..., n_plants, n_hours)."""
        x = np.asarray(x)
        if x.shape[-1] != self.n_vars:
            raise ValueError(f"last axis has length {x.shape[-1]}, expected {self.n_vars}")
        return np.swapaxes(x.reshape(*x.shape[:-1], self.n_hours, self.n_plants), -1, -2)

    def flatten(self, u) -> np.ndarray:
        """(..., n_plants, n_hours) -> (..., n_vars). Inverse of :meth:`unflatten`."""
        u = np.asarray(u)
        if u.shape[-2:] != (self.n_plants, self.n_hours):
            raise ValueError(
                f"trailing shape {u.shape[-2:]}, expected ({self.n_plants}, {self.n_hours})"
            )
        return np.swapaxes(u, -1, -2).reshape(*u.shape[:-2], self.n_vars)

    # ---- schedule evaluation --------------------------------------------

    def generation(self, x) -> np.ndarray:
        """MW generated in each hour, shape (..., T)."""
        return np.einsum("i,...it->...t", self.capacity, self.unflatten(x))

    def mismatch(self, x) -> np.ndarray:
        """generation - demand, shape (..., T). Positive means surplus."""
        return self.generation(x) - self.demand_mw

    def target_mismatch(self, x) -> np.ndarray:
        """generation - (demand + bias), shape (..., T). What the QUBO penalises."""
        return self.generation(x) - self.target_mw

    def fuel_cost(self, x):
        return np.einsum("i,...it->...", self.hourly_cost, self.unflatten(x))

    def startup_count(self, x) -> np.ndarray:
        """Number of 0->1 transitions per unit, shape (..., N).

        Hour 0 is compared against ``initial_on``.
        """
        u = self.unflatten(x).astype(int)
        prev = np.concatenate(
            [np.broadcast_to(self.initial_on[:, None], u.shape[:-1] + (1,)), u[..., :-1]],
            axis=-1,
        )
        return (u * (1 - prev)).sum(axis=-1)

    def startup_expense(self, x):
        return self.startup_count(x) @ self.startup_cost

    def operating_cost(self, x):
        """Fuel + start-up cost, INR. No penalty term."""
        return self.fuel_cost(x) + self.startup_expense(x)

    def is_adequate(self, x):
        """True where generation >= the TRUE demand D_t (never D_t + B) in every hour."""
        return (self.generation(x) >= self.demand_mw).all(axis=-1)

    # ---- symmetry between identical blocks ------------------------------

    def symmetry_groups(self) -> list[list[int]]:
        """Groups (size >= 2) of interchangeable blocks: identical capacity,
        costs and hour -1 state. Swapping two of them, consistently across
        all hours, changes neither cost nor generation."""
        groups: dict[tuple, list[int]] = {}
        for i, p in enumerate(self.plants):
            key = (p.capacity_mw, p.a, p.b, p.startup, int(self.initial_on[i]))
            groups.setdefault(key, []).append(i)
        return [g for g in groups.values() if len(g) > 1]

    def canonical(self, x) -> np.ndarray:
        """Representative of x's symmetry class: within each group of
        interchangeable blocks, rows are sorted lexicographically."""
        u = np.array(self.unflatten(x), copy=True)
        for g in self.symmetry_groups():
            rows = u[g]
            u[g] = rows[np.lexsort(rows.T[::-1])]
        return self.flatten(u)

    def same_schedule(self, x, y) -> bool:
        """True if x and y are equal up to swapping interchangeable blocks."""
        return bool(np.array_equal(self.canonical(x), self.canonical(y)))


# ---- generators ---------------------------------------------------------

# Evening ramp: a bump centred late in the window, normalised so the peak
# hour reaches peak_mw exactly before jitter. The trough sits PEAK_DEPTH
# below the peak, so the profile always climbs then eases.
_RAMP_CENTRE = 0.7  # position of the peak within the window, in [0, 1]
_RAMP_WIDTH = 0.4
_PEAK_DEPTH = 0.18  # trough = (1 - depth) * peak
_JITTER_FRAC = 0.01  # std of hourly jitter as a fraction of peak


def merit_order(plants=HARYANA_STATIONS) -> np.ndarray:
    """Plant indices sorted by marginal cost b_i, cheapest first."""
    return np.argsort([p.b for p in plants], kind="stable")


def default_initial_on(plants) -> np.ndarray:
    """Hour -1 state: every block whose marginal cost b is among the two
    lowest distinct values. For both full fleets this is all of Jhajjar and
    Hisar, so the station and unit models start from the same state."""
    b = np.array([p.b for p in plants])
    cheapest = np.unique(b)[:2]
    return np.isin(b, cheapest).astype(int)


def load_range(plants) -> tuple[float, float]:
    """[LOAD_MIN_MW, LOAD_MAX_MW] scaled by the selection's capacity share."""
    share = sum(p.capacity_mw for p in plants) / FLEET_CAPACITY_MW
    return LOAD_MIN_MW * share, LOAD_MAX_MW * share


def evening_load_profile(
    peak_mw: float, n_hours: int, rng: np.random.Generator, lo=LOAD_MIN_MW, hi=LOAD_MAX_MW
) -> np.ndarray:
    """Summer-evening load: ramps up to ``peak_mw`` and eases, plus small jitter."""
    s = (np.arange(n_hours) + 0.5) / n_hours
    bump = np.exp(-(((s - _RAMP_CENTRE) / _RAMP_WIDTH) ** 2))
    bump /= bump.max()
    trough = (1.0 - _PEAK_DEPTH) * peak_mw
    demand = trough + (peak_mw - trough) * bump
    demand += rng.normal(0.0, _JITTER_FRAC * peak_mw, size=n_hours)
    return np.clip(demand, lo, hi)


def make_instance(
    peak_mw: float,
    n_hours: int = 4,
    initial_on=None,
    seed=None,
    demand_bias_mw: float = 0.0,
    plants="station",
) -> UCInstance:
    """Instance with a shaped evening load profile peaking near ``peak_mw``.

    ``plants`` is 'station', 'unit' or a tuple of :class:`Plant` (e.g. from
    :func:`select_plants`). By default :func:`default_initial_on` sets hour -1.
    """
    rng = np.random.default_rng(seed)
    plants, model = _resolve_plants(plants)
    if initial_on is None:
        initial_on = default_initial_on(plants)
    lo, hi = load_range(plants)
    demand = evening_load_profile(peak_mw, n_hours, rng, lo, hi)
    return UCInstance(
        plants=plants,
        demand_mw=demand,
        initial_on=initial_on,
        label=f"{model}_peak{peak_mw:.0f}_T{n_hours}",
        meta={"peak_mw": float(peak_mw), "seed": seed, "model": model, "generator": "make_instance"},
        demand_bias_mw=demand_bias_mw,
    )


def make_scenario_set(
    n_scenarios: int,
    n_hours: int = 4,
    seed: int = 0,
    demand_bias_mw: float = 0.0,
    plants="station",
) -> list[UCInstance]:
    """Instances with peaks drawn uniformly over the (scaled) load range.

    Different peaks call for different commitments, so this is the
    distribution a learned warm start is trained on and held out from. For
    a given seed and horizon, the full station and unit fleets get
    identical demand profiles.
    """
    plants, _ = _resolve_plants(plants)
    lo, hi = load_range(plants)
    rng = np.random.default_rng(seed)
    peaks = rng.uniform(lo, hi, size=n_scenarios)
    child_seeds = rng.integers(0, 2**31 - 1, size=n_scenarios)
    out = []
    for k, (peak, child) in enumerate(zip(peaks, child_seeds)):
        inst = make_instance(
            float(peak), n_hours=n_hours, seed=int(child), demand_bias_mw=demand_bias_mw, plants=plants
        )
        inst.label = f"s{k:03d}_{inst.label}"
        inst.meta.update(scenario=k, set_seed=seed, generator="make_scenario_set")
        out.append(inst)
    return out
