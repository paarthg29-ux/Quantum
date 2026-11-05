"""Step 1 sanity check: build one 5-plant x 4-hour instance and its QUBO and
print enough of both to check the terms by eye."""

import numpy as np

from qguc import build_qubo, make_instance, qubo_energy

PEAK_MW = 3900.0
N_HOURS = 4
SEED = 7
# Illustrative weights only; lambda tuning is a later step.
LAMBDA_DEMAND = 100.0  # INR / MW^2
LAMBDA_SWITCH = 0.0  # INR per on/off transition; off by default


def merit_order_schedule(inst):
    """Each hour, commit units in order of average full-block cost
    (a_i/P_i + b_i) until generation covers demand."""
    order = np.argsort(inst.hourly_cost / inst.capacity)
    u = np.zeros((inst.n_plants, inst.n_hours), dtype=int)
    for t, d in enumerate(inst.demand_mw):
        total = 0.0
        for i in order:
            if total >= d:
                break
            u[i, t] = 1
            total += inst.capacity[i]
    return inst.flatten(u)


def main():
    inst = make_instance(PEAK_MW, n_hours=N_HOURS, seed=SEED)
    q = build_qubo(inst, LAMBDA_DEMAND, lambda_switch=LAMBDA_SWITCH, include_startup=True)

    print(f"Instance {inst.label}: {inst.n_plants} plants x {inst.n_hours} hours = {inst.n_vars} variables")
    print(f"initial_on (hour -1): {dict(zip([p.name for p in inst.plants], inst.initial_on.tolist()))}")
    print("\nDemand profile (MW):")
    for t, d in enumerate(inst.demand_mw):
        print(f"  t={t}  {d:8.1f}")

    print("\nPlants:")
    print(f"  {'idx':<4}{'name':<13}{'P (MW)':>8}{'a+bP (INR/h)':>15}{'S (INR)':>12}")
    for i, p in enumerate(inst.plants):
        print(f"  q{i:<3}{p.name:<13}{p.capacity_mw:8.0f}{p.hourly_cost:15,.0f}{p.startup:12,.0f}")

    Q = q.Q
    n = q.n_vars
    diag = np.diag(Q)
    off = Q[np.triu_indices(n, k=1)]
    off_nz = off[off != 0]
    print(f"\nQUBO: Q shape {Q.shape}, offset {q.offset:.4e}")
    print(f"  params: {q.params}")
    print(f"  nonzeros (upper triangle incl. diag): {np.count_nonzero(np.triu(Q))} of {n * (n + 1) // 2}"
          f"  -> density {np.count_nonzero(np.triu(Q)) / (n * (n + 1) // 2):.3f}")
    print(f"  off-diagonal couplings: {off_nz.size} of {off.size}  -> density {off_nz.size / off.size:.3f}")
    print(f"  diagonal     : min {diag.min():+.3e}  max {diag.max():+.3e}  |.| in [{np.abs(diag).min():.3e}, {np.abs(diag).max():.3e}]")
    print(f"  off-diagonal : min {off_nz.min():+.3e}  max {off_nz.max():+.3e}  |.| in [{np.abs(off_nz).min():.3e}, {np.abs(off_nz).max():.3e}]")
    print(f"  entries below diagonal: {np.count_nonzero(np.tril(Q, k=-1))}")

    schedules = {
        "all-ON": np.ones(n, dtype=int),
        "all-OFF": np.zeros(n, dtype=int),
        "merit-order": merit_order_schedule(inst),
    }

    print("\nSchedules (rows = plants q0..q4, columns = hours):")
    for name, x in schedules.items():
        u = inst.unflatten(x)
        print(f"  {name}:")
        for i, p in enumerate(inst.plants):
            print(f"    q{i} {p.name:<12} {' '.join(str(v) for v in u[i])}")
        print(f"    generation  {np.array2string(inst.generation(x), precision=0)}")
        print(f"    mismatch    {np.array2string(inst.mismatch(x), precision=1)}")

    print(f"\nEnergy breakdown (INR; lambda_D={LAMBDA_DEMAND}, lambda_switch={LAMBDA_SWITCH:.1e}):")
    hdr = f"  {'schedule':<12}{'fuel':>14}{'startup':>14}{'demand pen':>16}{'switch pen':>14}{'sum':>16}{'QUBO E(x)':>16}{'rel err':>10}{'adequate':>10}"
    print(hdr)
    for name, x in schedules.items():
        u = inst.unflatten(x)
        prev = np.concatenate([inst.initial_on[:, None], u[:, :-1]], axis=1)
        fuel = inst.fuel_cost(x)
        startup = inst.startup_expense(x)
        demand_pen = LAMBDA_DEMAND * np.sum(inst.target_mismatch(x) ** 2)
        switch_pen = LAMBDA_SWITCH * np.sum((u - prev) ** 2)
        total = fuel + startup + demand_pen + switch_pen
        e = qubo_energy(q, x)
        rel = abs(e - total) / abs(total)
        print(f"  {name:<12}{fuel:14,.0f}{startup:14,.0f}{demand_pen:16,.0f}{switch_pen:14,.0f}"
              f"{total:16,.0f}{e:16,.0f}{rel:10.1e}{str(bool(inst.is_adequate(x))):>10}")


if __name__ == "__main__":
    main()
