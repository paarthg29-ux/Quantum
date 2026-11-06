"""Stability of the best regularisation eps, and the cost of choosing it wrongly
(Phase 1 data: p = 1, 30 held-out scenarios x 3 seeds, 17 structural groups).

Metric throughout: P(best of 1024 = ground), averaged over seeds per
scenario, then over scenarios within each GL group (ground state + LP pin
pattern); groups are the unit (effective n = 17). eps = 0.5 is the cold
limit, not a warm start, and is excluded from selection.

1. Selection stability. (a) Group bootstrap: resample the 17 groups with
   replacement 2000 times; on each resample pick each arm's best eps
   (highest mean P; ties -> smaller cost gap). Report how often each eps is
   picked, the modal frequency, and the SD of the picked eps. (b) 500 random
   2-fold splits of the groups: pick on one half, evaluate on the other;
   report the picked-eps distribution and the held-out P.
2. Instance dependence. Each group's OWN best eps; its spread per arm.
3. Cost of a wrong eps. Regret(eps) = P(arm at its pooled best eps) -
   P(arm at eps), per group, mean with 95% t CI over groups; and the regret
   of the single pooled-best eps versus each group's own best (what a
   per-instance oracle would add).

    python scripts/s8_eps_stability.py   # -> results/s8_eps_stability.txt
"""

import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "s8_eps_stability.txt"
B_BOOT, N_SPLITS = 2000, 500


def main():
    import s8_analysis as a
    from s8_eps_main import tasks

    keys = tasks(1)
    D = a.Data([r for r in a.load() if (r["arm"], r["eps"], r["scenario"], r["seed"], r["p"]) in keys])
    gm = a.structural_groups()["GL"]
    G = sorted(set(gm.values()))
    EPS = [e for e in a.EPS if e < 0.5]
    arms = a.SWEEP_ARMS
    # P[arm][e_index, group], gap likewise (tie-break)
    P, GAP = {}, {}
    for arm in arms:
        P[arm] = np.array([[a.to_groups(D.scen(arm, e, 1, "p_ground_K1024", a.SWEEP_SEEDS), gm)[g] for g in G]
                           for e in EPS])
        GAP[arm] = np.array([[a.to_groups(D.scen(arm, e, 1, "gap_repaired_K1024", a.SWEEP_SEEDS), gm)[g] for g in G]
                             for e in EPS])

    def pick(arm, cols):
        score = [(P[arm][i, cols].mean(), -GAP[arm][i, cols].mean()) for i in range(len(EPS))]
        return max(range(len(EPS)), key=lambda i: score[i])

    lines = []
    out = lambda s="": lines.append(s)  # noqa: E731
    out(f"EPS STABILITY (Phase 1, p = 1; unit = {len(G)} GL structural groups; metric P(best of 1024 = ground); "
        f"candidate eps {EPS})")
    rng = np.random.default_rng(0)
    allg = np.arange(len(G))

    out("\n1a. GROUP BOOTSTRAP: how often each eps is selected as best (2000 resamples of the 17 groups)")
    out(f"  {'arm':>8} | " + " ".join(f"{e:>6}" for e in EPS) + " | modal freq | SD of picked eps | pooled pick")
    boot_idx = rng.integers(0, len(G), (B_BOOT, len(G)))
    for arm in arms:
        picks = np.array([EPS[pick(arm, b)] for b in boot_idx])
        freq = [np.mean(picks == e) for e in EPS]
        out(f"  {arm:>8} | " + " ".join(f"{f:6.3f}" for f in freq)
            + f" | {max(freq):10.3f} | {picks.std():16.3f} | {EPS[pick(arm, allg)]}")

    out(f"\n1b. RANDOM 2-FOLD GROUP SPLITS ({N_SPLITS}): eps picked on one half, P evaluated on the other half")
    out(f"  {'arm':>8} | picked-eps freq " + " ".join(f"{e:>6}" for e in EPS)
        + " | SD picked | held-out P (mean over splits) | held-out regret vs that half's own best")
    splits = []
    for _ in range(N_SPLITS):  # one shared set of splits: arms are compared on identical held-out halves
        perm = rng.permutation(len(G))
        A_, B_ = perm[: len(G) // 2], perm[len(G) // 2:]
        splits += [(A_, B_), (B_, A_)]
    held = {}
    for arm in arms:
        picks, hp, hr = [], [], []
        for tr, te in splits:
            i = pick(arm, tr)
            picks.append(EPS[i])
            hp.append(P[arm][i, te].mean())
            hr.append(P[arm][pick(arm, te), te].mean() - P[arm][i, te].mean())
        picks = np.array(picks)
        held[arm] = np.array(hp)
        out(f"  {arm:>8} | {'':16}" + " ".join(f"{np.mean(picks == e):6.3f}" for e in EPS)
            + f" | {picks.std():9.3f} | {np.mean(hp):.3f} [{np.percentile(hp, 2.5):.3f}, {np.percentile(hp, 97.5):.3f}]"
            + f" | {np.mean(hr):.3f}")
    for x, y in (("D", "C-oracle"), ("D", "C-milp"), ("D", "F")):
        d = held[x] - held[y]
        out(f"  held-out P, {x} - {y} on the same splits: mean {d.mean():+.3f}, 95% of splits in "
            f"[{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}], {x} worse in {np.mean(d < 0):.1%} of splits")

    out("\n2. INSTANCE DEPENDENCE: each group's own best eps (ties -> smaller cost gap)")
    out(f"  {'arm':>8} | " + " ".join(f"{e:>6}" for e in EPS) + " | distinct | SD over groups")
    own = {}
    for arm in arms:
        own[arm] = np.array([EPS[pick(arm, np.array([g]))] for g in range(len(G))])
        out(f"  {arm:>8} | " + " ".join(f"{np.mean(own[arm] == e):6.3f}" for e in EPS)
            + f" | {len(set(own[arm])):8d} | {own[arm].std():.3f}")
    out("  note: a group whose P is 1.000 at several eps has a tie broken by cost gap; 'distinct' counts such groups "
        "too, so the SD is the cleaner summary")
    flat = {arm: np.mean(np.ptp(P[arm], axis=0) < 1e-3) for arm in arms}
    out("  fraction of groups where P is the same (to 1e-3) at every eps < 0.5 (eps irrelevant there): "
        + ", ".join(f"{arm} {v:.2f}" for arm, v in flat.items()))

    out("\n3. COST OF A WRONG EPS: regret(eps) = P(at pooled best eps) - P(at eps), per group; mean [95% t CI over "
        "17 groups]")
    out(f"  {'arm':>8} best | " + " ".join(f"{e:>20}" for e in EPS))
    for arm in arms:
        ib = pick(arm, allg)
        cells = []
        for i in range(len(EPS)):
            d = P[arm][ib] - P[arm][i]
            h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)) if d.std() > 0 else 0.0
            cells.append(f"{d.mean():+.3f} [{d.mean() - h:+.3f},{d.mean() + h:+.3f}]")
        out(f"  {arm:>8} {EPS[ib]:4} | " + " ".join(f"{c:>20}" for c in cells))
    out("  worst-case regret over eps in [0.05, 0.35] (excludes eps = 0, which zeroes D's overlap): "
        + ", ".join(f"{arm} {max(P[arm][pick(arm, allg)].mean() - P[arm][i].mean() for i in range(1, len(EPS))):.3f}"
                    for arm in arms))
    out("\n  per-instance oracle: P(each group at its own best eps) - P(at the pooled best eps), mean [95% CI]")
    for arm in arms:
        ib = pick(arm, allg)
        d = P[arm].max(axis=0) - P[arm][ib]
        h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)) if d.std() > 0 else 0.0
        out(f"  {arm:>8}: {d.mean():+.3f} [{d.mean() - h:+.3f}, {d.mean() + h:+.3f}]  (groups gaining > 0.01: "
            f"{int((d > 0.01).sum())}/{len(d)})")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
