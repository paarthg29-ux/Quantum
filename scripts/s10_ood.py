"""G5: the learned warm start OUT OF DISTRIBUTION (L2, 16 qubits, frozen point).

C-milp is the standard L2 model (models/gnn_milp_L2.pt), trained on peaks
U(2500, 4200) MW with the rising evening profile, the full 8-block fleet;
nothing is retrained. Test sets, 20 scenarios each (fresh seeds):

  in-dist      peaks U(2500, 4200), standard profile, full fleet (baseline)
  high         peaks U(4200, 4400): above the training range (up to near the
               4442 MW capacity; demand + bias then exceeds capacity, so the
               QUBO pushes towards all-on)
  low          peaks U(1800, 2500): below the training range
  out-<block>  forced outage: one block unavailable (removed, i.e. its
               commitment fixed to 0) -> the GRAPH changes; the GCN runs on the
               reduced graph (WarmStartModel.predict_graph). Blocks: IGSTPP-1
               (cheapest per MW), Yamunanagar (mid-merit), Faridabad (most
               expensive). Peaks U(2500, 4200) redrawn until demand <= remaining
               capacity - 50 MW
  reversed     the two hours swapped: demand FALLS (training profiles rise)
  flat         both hours at the peak

Arms C-milp, D (LP at the QUBO target; it has no training distribution to be
out of), A (cold), eps = 0.1, p = 1, seeds 0-2. Paired per scenario.

    python scripts/s10_ood.py          # -> results/s10_ood.jsonl, results/s10_ood.txt
    python scripts/s10_ood.py --report
"""

import dataclasses
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
SETS = ("in-dist", "high", "low", "out-IGSTPP-1", "out-Yamunanagar", "out-Faridabad", "reversed", "flat")
N_SCEN, SEEDS, EPS, P = 20, (0, 1, 2), 0.1, 1
ARMS = ("C-milp", "D", "A")
WORKERS = 4
STORE = ROOT / "results" / "s10_ood.jsonl"
REPORT = ROOT / "results" / "s10_ood.txt"
THREAD_ENV = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
_CACHE = {}


def instance(set_name, k):
    from qguc import make_instance

    rng = np.random.default_rng([9500, SETS.index(set_name), k])
    while True:
        inst = make_instance(float(rng.uniform(2500.0, 4200.0)), n_hours=2, seed=int(rng.integers(2**31 - 1)),
                             refinement_level=2)
        if set_name in ("high", "low"):
            # the profile generator clips demand to the training load range, so draw an in-range profile and
            # RESCALE it to an out-of-range peak (shape kept): high U(4200, 4400), low U(1800, 2500) MW
            target = float(rng.uniform(*((4200.0, 4400.0) if set_name == "high" else (1800.0, 2500.0))))
            inst = dataclasses.replace(inst, demand_mw=inst.demand_mw * (target / inst.demand_mw.max()))
        if set_name.startswith("out-"):
            keep = np.array([p.name != set_name[4:] for p in inst.plants])
            assert not keep.all(), set_name
            inst = dataclasses.replace(inst, plants=tuple(p for p, kk in zip(inst.plants, keep) if kk),
                                       initial_on=inst.initial_on[keep])
            if inst.demand_mw.max() > inst.capacity.sum() - 50:
                continue
        elif set_name == "reversed":
            inst = dataclasses.replace(inst, demand_mw=inst.demand_mw[::-1].copy())
        elif set_name == "flat":
            inst = dataclasses.replace(inst, demand_mw=np.full(inst.n_hours, inst.demand_mw.max()))
        if inst.demand_mw.max() <= inst.capacity.sum():
            inst.meta.update(ood_set=set_name, scenario=k)
            return inst


def problem(set_name, k):
    from qguc.qaoa import make_problem

    if (set_name, k) not in _CACHE:
        _CACHE.clear()
        pb = make_problem(instance(set_name, k))
        pb.table
        _CACHE[(set_name, k)] = pb
    return _CACHE[(set_name, k)]


def learned_arm():
    import torch

    from qguc.arms import Arm, CVector
    from qguc.gnn import MODEL_DIR, WarmStartModel

    torch.set_num_threads(1)
    model = WarmStartModel.load(MODEL_DIR / "gnn_milp_L2.pt")

    def source(pb, seed, eps):
        t0 = time.perf_counter()
        c = pb.instance.flatten(model.predict_graph(pb.instance))
        return CVector(c, time.perf_counter() - t0, dict(gnn="milp-L2"))

    return Arm("C-milp", "standard L2 GCN (MILP labels), applied out of distribution", source)


def job(args):
    set_name, k, seed = args
    os.environ.update(THREAD_ENV)
    from qguc.arms import run_arm

    pb = problem(set_name, k)
    arms = {"C-milp": learned_arm(), "D": "D", "A": "A"}
    out = []
    for a in ARMS:
        r = run_arm(arms[a], pb, P, seed, eps=EPS)
        out.append(dict(set=set_name, scenario=k, seed=seed, arm=a, n=pb.n, demand=[float(d) for d in pb.instance.demand_mw],
                        capacity=float(pb.instance.capacity.sum()), n_ground=len(pb.ev.ground),
                        n_pinned=r.extra["n_pinned"], n_pinned_wrong=r.extra["n_pinned_wrong"],
                        overlap0=r.overlap0, g_final=r.g_final, n_evals=r.n_evals,
                        **{f"{kk}_K{K}": v for K, m in r.sampling.items() for kk, v in m.items()}))
    return out


def load():
    if not STORE.exists():
        return []
    return [json.loads(l) for l in STORE.read_text().splitlines() if l.strip()]


def run():
    done = {(r["set"], r["scenario"], r["seed"]) for r in load() if r["arm"] == ARMS[-1]}
    todo = [(s, k, sd) for s in SETS for k in range(N_SCEN) for sd in SEEDS if (s, k, sd) not in done]
    print(f"{len(todo)} jobs to do ({len(done)} done); {WORKERS} workers", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(WORKERS) as pool, STORE.open("a") as f:
        futs = {pool.submit(job, j): j for j in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                recs = fut.result()
            except Exception as exc:
                print(f"  FAILED {futs[fut]}: {exc!r}", flush=True)
                continue
            for rec in recs:
                f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 60 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}, {time.perf_counter() - t0:.0f} s", flush=True)


def report():
    from scipy import stats

    R = load()
    L = []
    W = lambda s="": L.append(s)  # noqa: E731

    def scen(rows, key):
        out = {}
        for r in rows:
            out.setdefault(r["scenario"], []).append(r[key])
        return {s: float(np.mean(v)) for s, v in out.items()}

    def ci(v):
        v = np.asarray(list(v), float)
        h = stats.t.ppf(0.975, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size) if v.std() > 0 else 0.0
        return f"{v.mean():.3f} [{v.mean() - h:.3f}, {v.mean() + h:.3f}]"

    W("G5: C-milp (standard L2 GCN, not retrained) OUT OF DISTRIBUTION vs the LP (D) and cold start (A); L2, frozen "
      f"point, p = 1, eps = 0.1; {N_SCEN} scenarios x {len(SEEDS)} seeds per set; scenario means, 95% t CI over scenarios.")
    W("The LP has no training distribution: its numbers move only because the instances differ.")
    W("")
    W(f"  {'set':<16} {'n':>3} | {'C wrong':>7} {'D wrong':>7} | {'C P(b1024)':>22} {'D P(b1024)':>22} {'A P(b1024)':>10} | "
      f"{'C - D, paired':>36}")
    for s in SETS:
        rows = [r for r in R if r["set"] == s]
        if not rows:
            continue
        C = [r for r in rows if r["arm"] == "C-milp"]
        D = [r for r in rows if r["arm"] == "D"]
        A = [r for r in rows if r["arm"] == "A"]
        pc, pd = scen(C, "p_ground_K1024"), scen(D, "p_ground_K1024")
        common = sorted(set(pc) & set(pd))
        d = np.array([pc[k] - pd[k] for k in common])
        h = stats.t.ppf(0.975, d.size - 1) * d.std(ddof=1) / np.sqrt(d.size) if d.std() > 0 else 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pw = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        W(f"  {s:<16} {rows[0]['n']:3d} | {np.mean([r['n_pinned_wrong'] for r in C]):7.2f} "
          f"{np.mean([r['n_pinned_wrong'] for r in D]):7.2f} | {ci(pc.values()):>22} {ci(pd.values()):>22} "
          f"{np.mean(list(scen(A, 'p_ground_K1024').values())):10.3f} | {d.mean():+.3f} [{d.mean() - h:+.3f}, "
          f"{d.mean() + h:+.3f}] W p={pw:.1e} ({(d > 1e-12).sum()}/{(np.abs(d) <= 1e-12).sum()}/{(d < -1e-12).sum()})")
    W("")
    W("C-milp's change from in-distribution (difference of set means; unpaired, different scenarios):")
    base = [r for r in R if r["set"] == "in-dist" and r["arm"] == "C-milp"]
    for s in SETS[1:]:
        C = [r for r in R if r["set"] == s and r["arm"] == "C-milp"]
        if not C or not base:
            continue
        x, y = list(scen(C, "p_ground_K1024").values()), list(scen(base, "p_ground_K1024").values())
        t = stats.ttest_ind(x, y, equal_var=False)
        W(f"  {s:<16}: P(b1024) {np.mean(x) - np.mean(y):+.3f} (Welch p = {t.pvalue:.2g}); wrong pins "
          f"{np.mean([r['n_pinned_wrong'] for r in C]) - np.mean([r['n_pinned_wrong'] for r in base]):+.2f}")
    REPORT.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    if "--report" not in sys.argv:
        run()
    report()
