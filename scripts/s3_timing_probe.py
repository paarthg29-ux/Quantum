"""Step 3: measure plain-QAOA simulation cost on our Ising Hamiltonian and
extrapolate the cost of the full benchmark.

Circuit: |+>^n, then p layers of exp(-i gamma H_C) (qml.qaoa.cost_layer)
and the standard X mixer (qml.qaoa.mixer_layer), measuring <H_C>. H_C is
the normalised Ising Hamiltonian (all Z and ZZ terms) of a unit-model
instance at T = 2 at the frozen operating point; n = 16, 18, 20, 22 use
8, 9, 10, 11 units.

Gradient methods
----------------
* default.qubit + adjoint is NOT usable here: in PennyLane 0.45.1 its
  adjoint pass applies the observable through its DENSE 2^n x 2^n matrix
  (Sum.matrix(), and SparseHamiltonian is densified too), i.e. 64 GiB at
  n = 16. So on default.qubit the cheapest exact gradient is backprop,
  whose memory grows as ~2 x (number of gates) x 2^n x 16 bytes; calls
  estimated above MEM_CAP_GB are skipped and reported as infeasible.
* lightning.qubit + adjoint applies H_C term by term in C++, with memory
  of a few state vectors. Its gradient is checked against default.qubit
  backprop before timing.

Each (simulator, n) runs in its own subprocess so peak memory is clean.
BLAS/OpenMP threads are pinned to 1: every number is per core, and the
benchmark extrapolation divides core-hours across physical cores.
"""

import json
import os
import subprocess
import sys
import time

import numpy as np

N_HOURS = 2
SIZES = {  # n_qubits -> unit names dropped from the 11-unit fleet
    22: [],
    20: ["PTPS-6"],
    18: ["PTPS-6", "DCRTPP-2"],
    16: ["PTPS-6", "DCRTPP-2", "PTPS-8"],
}
P_VALUES = (1, 2, 3, 5)
CALL_CAP_S = 900.0  # do not launch a single call predicted (linearly in p) to exceed this
MEM_CAP_GB = 9.0  # of 15.6 GB on the measurement machine

# Benchmark to extrapolate to.
ARMS, SCENARIOS, SEEDS, ITERS = 5, 10, 20, 300

CONFIGS = [  # (device, diff_method, sizes)
    ("default.qubit", "backprop", (16, 18, 20, 22)),
    ("lightning.qubit", "adjoint", (16, 18, 20, 22)),
]
THREAD_ENV = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")


def peak_rss_mb():
    import psutil

    mi = psutil.Process().memory_info()
    return getattr(mi, "peak_wset", mi.rss) / 2**20


def backprop_gb(n, n_terms, p):
    """Estimated extra memory of a default.qubit backprop gradient: autograd
    keeps one 2^n complex128 state per gate (factor 2 measured at n = 16)."""
    gates = n + p * (n_terms + n)
    return 2.0 * gates * 16 * 2**n / 2**30


def build_hamiltonian(n):
    import pennylane as qml

    from qguc import make_scenario_set, select_plants
    from qguc.config import FROZEN
    from qguc.ising import normalise, qubo_to_ising
    from qguc.qubo import build_qubo

    drop = set(SIZES[n])
    plants = tuple(p for p in select_plants("unit") if p.name not in drop)
    inst = make_scenario_set(1, n_hours=N_HOURS, seed=0, plants=plants,
                             demand_bias_mw=FROZEN.demand_bias_mw)[0]
    assert inst.n_vars == n
    m = normalise(qubo_to_ising(build_qubo(inst, FROZEN.lambda_demand)))
    coeffs, ops = [], []
    for i in range(n):
        if m.h[i] != 0:
            coeffs.append(float(m.h[i]))
            ops.append(qml.Z(i))
    for i, j in zip(*np.nonzero(m.J)):
        coeffs.append(float(m.J[i, j]))
        ops.append(qml.Z(int(i)) @ qml.Z(int(j)))
    return qml.Hamiltonian(coeffs, ops)


def make_circuit(dev, H, n, p, diff):
    import pennylane as qml

    mixer = qml.qaoa.x_mixer(range(n))

    @qml.qnode(dev, diff_method=diff, interface="autograd")
    def circuit(params):
        for w in range(n):
            qml.Hadamard(w)
        for k in range(p):
            qml.qaoa.cost_layer(params[0, k], H)
            qml.qaoa.mixer_layer(params[1, k], mixer)
        return qml.expval(H)

    return circuit


def check_gradients(n=16, p=2):
    """lightning adjoint vs default.qubit backprop on the same circuit."""
    import pennylane as qml
    from pennylane import numpy as pnp

    H = build_hamiltonian(n)
    x = pnp.array(np.random.default_rng(1).uniform(0, 1, (2, p)), requires_grad=True)
    ref = qml.grad(make_circuit(qml.device("default.qubit", wires=n), H, n, p, "backprop"))(x)
    adj = qml.grad(make_circuit(qml.device("lightning.qubit", wires=n), H, n, p, "adjoint"))(x)
    print(json.dumps({"n": n, "p": p, "max_abs_diff": float(np.abs(ref - adj).max()),
                      "max_abs_grad": float(np.abs(ref).max())}))


def worker(device, diff, n):
    import pennylane as qml
    from pennylane import numpy as pnp

    H = build_hamiltonian(n)
    n_terms = len(H.operands)
    dev = qml.device(device, wires=n)
    rng = np.random.default_rng(0)
    make_circuit(dev, H, n, 1, diff)(rng.uniform(0, 1, (2, 1)))  # warm-up
    base_mb = peak_rss_mb()
    out = {"device": device, "diff": diff, "n": n, "n_terms": n_terms, "baseline_mb": base_mb, "p": {}}

    def timed(fn, arg, estimate):
        if estimate is not None and estimate > CALL_CAP_S:
            return None, f"~{estimate:.0f} s est."
        ts = []
        while True:
            t0 = time.perf_counter()
            fn(arg)
            ts.append(time.perf_counter() - t0)
            if len(ts) >= (3 if ts[0] < 2 else 2 if ts[0] < 20 else 1):
                return float(np.median(ts)), None

    last = {}
    for p in P_VALUES:
        circuit = make_circuit(dev, H, n, p, diff)
        plain = rng.uniform(0, 1, (2, p))
        rec = {}
        est = last["energy"] * p / last["p"] if "energy" in last else None
        rec["energy_s"], rec["energy_note"] = timed(circuit, plain, est)
        if diff == "backprop" and backprop_gb(n, n_terms, p) > MEM_CAP_GB:
            rec["grad_s"], rec["grad_note"] = None, f"OOM: ~{backprop_gb(n, n_terms, p):.0f} GB"
        else:
            est = last["grad"] * p / last["gp"] if "grad" in last else None
            rec["grad_s"], rec["grad_note"] = timed(qml.grad(circuit), pnp.array(plain, requires_grad=True), est)
        if rec["energy_s"] is not None:
            last.update(energy=rec["energy_s"], p=p)
        if rec["grad_s"] is not None:
            last.update(grad=rec["grad_s"], gp=p)
        rec["peak_mb"] = peak_rss_mb()
        out["p"][p] = rec
    print(json.dumps(out))


def run_sub(*args):
    res = subprocess.run([sys.executable, os.path.abspath(__file__), *args],
                         env=dict(os.environ, **THREAD_ENV), capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  {' '.join(args)} FAILED:\n{res.stderr[-2000:]}", flush=True)
        return None
    return json.loads(res.stdout.strip().splitlines()[-1])


def cell(v, note):
    return f"{v:10.3f}" if v is not None else f"{note:>10}"


def main():
    import pennylane as qml
    import psutil

    from qguc.config import FROZEN

    phys, logical = psutil.cpu_count(logical=False), os.cpu_count()
    ram = psutil.virtual_memory().total / 2**30
    print(f"PennyLane {qml.__version__}; {phys} physical / {logical} logical cores, {ram:.1f} GB RAM; "
          f"1 thread per process")
    print(f"H_C: unit model T={N_HOURS} at the frozen point (lambda_D={FROZEN.lambda_demand:.4g}, "
          f"bias={FROZEN.demand_bias_mw:g} MW), normalised")

    chk = run_sub("--check")
    if chk:
        print(f"gradient check n={chk['n']} p={chk['p']}: lightning adjoint vs default.qubit backprop, "
              f"max |diff| = {chk['max_abs_diff']:.2e} (max |grad| = {chk['max_abs_grad']:.2e})")

    results = []
    for device, diff, sizes in CONFIGS:
        for n in sizes:
            t0 = time.perf_counter()
            print(f"running {device} / {diff} / n={n} ...", end=" ", flush=True)
            r = run_sub("--worker", device, diff, str(n))
            if r is not None:
                results.append(r)
                print(f"{time.perf_counter() - t0:.0f} s", flush=True)

    print(f"\n=== measured wall-clock per call (s, median; 'est.' = not run, linear-in-p estimate above "
          f"the {CALL_CAP_S:.0f} s cap; 'OOM' = estimated memory above {MEM_CAP_GB:g} GB) ===")
    print(f"  {'device / diff':<26} {'n':>3} {'terms':>5} {'p':>2} {'energy s':>10} {'grad s':>10} {'peak MB':>8}")
    for r in results:
        for p, rec in r["p"].items():
            print(f"  {r['device'] + ' / ' + r['diff']:<26} {r['n']:3d} {r['n_terms']:5d} {p:>2} "
                  f"{cell(rec['energy_s'], rec['energy_note'])} {cell(rec['grad_s'], rec['grad_note'])} "
                  f"{rec['peak_mb']:8.0f}")

    runs = ARMS * SCENARIOS * SEEDS
    print(f"\n=== benchmark extrapolation: {ARMS} arms x {SCENARIOS} scenarios x {SEEDS} seeds x "
          f"p in {list(P_VALUES)} x {ITERS} iterations ===")
    print(f"  {runs:,} runs per p, {runs * ITERS * len(P_VALUES):,} iterations in total. One iteration = one "
          f"gradient call (gradient-based, e.g. Adam) or one energy call (gradient-free, e.g. COBYLA).")
    print(f"  'wall' = core-hours / {phys}: {phys} single-thread runs in parallel with perfect scaling "
          f"(optimistic; memory must also fit).")
    print(f"  {'device / diff':<26} {'n':>3} | {'grad-based core-h':>17} {'wall h':>8} | "
          f"{'grad-free core-h':>16} {'wall h':>8}")
    for r in results:
        recs = r["p"].values()
        e_ok = all(x["energy_s"] is not None for x in recs)
        g_ok = all(x["grad_s"] is not None for x in recs)
        eh = runs * ITERS * sum(x["energy_s"] for x in recs) / 3600 if e_ok else None
        gh = runs * ITERS * sum(x["grad_s"] for x in recs) / 3600 if g_ok else None
        g_txt = f"{gh:17,.0f} {gh / phys:8,.1f}" if gh is not None else f"{'not measurable (see above)':>26}"
        e_txt = f"{eh:16,.0f} {eh / phys:8,.1f}" if eh is not None else f"{'not measurable':>25}"
        print(f"  {r['device'] + ' / ' + r['diff']:<26} {r['n']:3d} | {g_txt} | {e_txt}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif len(sys.argv) > 1 and sys.argv[1] == "--check":
        check_gradients()
    else:
        main()
