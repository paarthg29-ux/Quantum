"""Phase 3 scenario selection: 20 STRUCTURALLY DIVERSE scenarios at L4.

Pool: the 300 i.i.d. L4 census draws (scripts/s9_census.py, seed 9300).
Signature of a draw: (exact QUBO ground state(s), LP pin pattern), the same
definition as the L2 "GL" groups. Selection:

  1. one scenario per distinct signature (so every selected scenario is
     its own structural group: effective n = number selected);
  2. signatures chosen greedily by farthest-point sampling: start from the
     most frequent signature, then repeatedly add the signature whose
     minimum distance to those already chosen is largest (ties: the more
     frequent one). Distance = Hamming distance between ground states
     (closest pair if degenerate) + Hamming distance between LP pin
     strings over {0, 1, .};
  3. the representative of a signature is its member with the median peak.

Each selected scenario carries WEIGHT = (pool members with its signature) /
300, the share of the demand distribution it stands for. Results can then be
reported per structure (unweighted, n = 20) and reweighted to the demand
distribution. The selected set over-represents rare structures by design,
so its fraction of multiply-wrong-pinned scenarios is NOT a population
estimate -- the census is.

    python scripts/s9_select.py   # -> results/s9_selected.json
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
N_SELECT = 20
OUT = ROOT / "results" / "s9_selected.json"


def ham(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def main():
    from s9_census import CENSUS_SEED, N, load

    pool = [r for r in load() if r["level"] == 4]
    assert len(pool) == N, f"census incomplete at L4: {len(pool)}/{N}"
    sigs = {}
    for r in pool:
        sigs.setdefault((tuple(r["ground"]), r["lp_pins"]), []).append(r)
    keys = list(sigs)

    def dist(a, b):
        return min(ham(g, h) for g in a[0] for h in b[0]) + ham(a[1], b[1])

    D = np.array([[dist(a, b) for b in keys] for a in keys])
    size = np.array([len(sigs[k]) for k in keys])
    chosen = [int(np.argmax(size))]
    while len(chosen) < min(N_SELECT, len(keys)):
        mind = D[:, chosen].min(axis=1).astype(float)
        mind[chosen] = -1
        best = max(range(len(keys)), key=lambda i: (mind[i], size[i]))
        chosen.append(best)
    sel = []
    for i in chosen:
        members = sorted(sigs[keys[i]], key=lambda r: r["peak"])
        rep = members[len(members) // 2]
        sel.append(dict(scenario=rep["scenario"], peak=rep["peak"], n_wrong=rep["n_wrong"], n_pinned=rep["n_pinned"],
                        n_ground=rep["n_ground"], ground=list(keys[i][0]), lp_pins=keys[i][1],
                        weight=len(members) / N, n_members=len(members),
                        min_dist_to_others=int(min(D[i, j] for j in chosen if j != i))))
    wrong = np.array([s["n_wrong"] for s in sel])
    w = np.array([s["weight"] for s in sel])
    pool_w = np.array([r["n_wrong"] for r in pool])
    info = dict(census_seed=CENSUS_SEED, level=4, pool=N, distinct_signatures=len(keys),
                distinct_ground_states=len({g for k in keys for g in k[0]}), n_selected=len(sel),
                selected_population_weight=float(w.sum()),
                selected_frac_wrong_ge2=float((wrong >= 2).mean()),
                pool_frac_wrong_ge2=float((pool_w >= 2).mean()), selected=sel)
    OUT.write_text(json.dumps(info, indent=1))
    print(f"L4 pool: {N} draws, {len(keys)} distinct signatures, {info['distinct_ground_states']} distinct ground "
          f"states. Selected {len(sel)} (one per signature), covering {w.sum():.1%} of the demand distribution.")
    print(f"wrong pins in the selection: 0/1/2/3+ = {[(wrong == k).sum() for k in (0, 1, 2)] + [(wrong >= 3).sum()]}; "
          f"fraction >= 2: selected {np.mean(wrong >= 2):.2f} vs pool {np.mean(pool_w >= 2):.3f} (the selection is "
          "NOT a population estimate)")
    print(f"{'scen':>4} {'peak':>6} {'wrong':>5} {'#gnd':>4} {'weight':>6} {'mind':>4}  lp pins")
    for s in sel:
        print(f"{s['scenario']:4d} {s['peak']:6.0f} {s['n_wrong']:5d} {s['n_ground']:4d} {s['weight']:6.3f} "
              f"{s['min_dist_to_others']:4d}  {s['lp_pins']}")


if __name__ == "__main__":
    main()
