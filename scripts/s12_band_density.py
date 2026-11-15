"""Is band density the condition under which band share beats overlap?

Hypothesis (stated before this script was run): band share adds to overlap
when the low-energy band is DENSE (many near-optimal states, as in unit
commitment) and adds nothing when it is SPARSE (the integer cut spectrum of
3-regular MaxCut). If the hypothesis is right, a per-instance measure of band
density must predict, across instances of BOTH problem classes, how much the
band improves the prediction of sampling. A difference between the two
classes alone does not test it (two clusters correlate with anything), so the
decisive numbers are the correlations WITHIN each dataset.

Per instance, from the exact spectrum (band = E - E_min <= 5% of the spectral
range, the paper's definition, unchanged):
  levels      distinct energy levels in the band above the ground level
  density     log10(states in the band / ground states)      [primary]
and from the initial states of that instance's runs:
  off_share   mean over runs of 1 - overlap0 / m_e05: the share of the band's
              initial mass that is NOT on the optimum (the mass distribution)

Outcome per instance: gain = MSE(overlap-only fit) - MSE(overlap + m_e05 fit)
over that instance's runs, with both fits made once per dataset exactly as
reported (UC 16 qubits phase 1, UC 22 qubits, MaxCut n = 12, 16, 20; logistic
least squares on runs with overlap0 > 0). gain > 0: the band helps there.

No new QAOA runs.

    python scripts/s12_band_density.py   # -> results/s12_band_density.txt, .json
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "s12_band_density.txt"
CACHE = ROOT / "results" / "s12_band_density_spectra.json"
Y = "p_ground_K1024"


def spectrum_stats(E) -> dict:
    E = np.asarray(E, float)
    rng = E.max() - E.min()
    band = E[E - E.min() <= 0.05 * rng]
    q = np.round((band - E.min()) / rng, 9)
    n_ground = int((q == 0).sum())
    return dict(n_ground=n_ground, n_band=int(len(band)), levels=int(len(np.unique(q)) - 1),
                density=float(np.log10(len(band) / n_ground)))


def spectra():
    """{dataset/instance: spectrum stats}, cached (the 22-qubit spectra take ~10 s each)."""
    import s9_l4_arms
    import s11_maxcut as X
    from s9_census import scenario as census_scenario

    from qguc import make_scenario_set
    from qguc.qaoa import make_problem

    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    def put(key, make_E):
        if key not in cache:
            cache[key] = spectrum_stats(make_E())
            CACHE.write_text(json.dumps(cache))

    insts = make_scenario_set(30, n_hours=2, seed=9100, refinement_level=2)
    for s in range(30):
        put(f"UC-16/{s}", lambda: make_problem(insts[s]).ev.E)
    for s in (x["scenario"] for x in s9_l4_arms.selected()):
        put(f"UC-22/{s}", lambda: make_problem(census_scenario(4, s)).ev.E)
        print(f"  spectrum UC-22/{s}", flush=True)
    for n in X.SIZES:
        for i in range(X.N_INST):
            def mk():
                E = X.get_problem(n, i).ev.E
                X._PROBLEMS.clear()
                return E
            put(f"MaxCut-{n}/{i}", mk)
    return cache


def datasets():
    """{name: list of runs with overlap0 > 0, each with 'inst', overlap0, m_e05, P}."""
    import s8_analysis
    import s8_eps_main
    import s9_l4_arms
    import s11_maxcut as X

    out = {}
    f8 = {tuple(json.loads(k)): v for k, v in
          json.loads((ROOT / "results" / "s8_mass_features.json").read_text()).items()}
    ph1 = s8_eps_main.tasks(1)
    rs = [r for r in s8_analysis.load() if (r["arm"], r["eps"], r["scenario"], r["seed"], r["p"]) in ph1]
    out["UC-16"] = [dict(inst=r["scenario"], overlap0=r["overlap0"], y=r[Y],
                         m_e05=f8[(r["arm"], r["eps"], r["scenario"], -1)]["m_e05"]) for r in rs]
    f9 = {tuple(json.loads(k)): v for k, v in
          json.loads((ROOT / "results" / "s9_mass_features.json").read_text()).items()}
    out["UC-22"] = [dict(inst=r["scenario"], overlap0=r["overlap0"], y=r[Y],
                         m_e05=f9[(r["arm"], r["eps"], r["scenario"])]["m_e05"]) for r in s9_l4_arms.load()]
    mc = X.load()
    for n in X.SIZES:
        out[f"MaxCut-{n}"] = [dict(inst=r["inst"], overlap0=r["overlap0"], y=r[Y], m_e05=r["m_e05"])
                              for r in mc if r["n"] == n]
    return {k: [r for r in v if r["overlap0"] > 0] for k, v in out.items()}


def rho(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan"), float("nan")
    r = stats.spearmanr(x, y)
    return float(r.statistic), float(r.pvalue)


def main():
    import s8_mass as M

    spec = spectra()
    lines = ["Band density and where the band improves on overlap. Per instance: density = log10(states within 5% of "
             "the spectral range / ground states); levels = distinct energy levels in that band above the ground "
             "level; off_share = mean share of the band's initial mass not on the optimum; gain = MSE(overlap fit) - "
             "MSE(overlap + m_e05 fit) over the instance's runs (dataset-level fits)."]
    rows = []
    store = dict(datasets={}, tests={})
    for name, rs in datasets().items():
        y = np.array([r["y"] for r in rs])
        ov = np.log10([r["overlap0"] for r in rs])
        me = np.log10([r["m_e05"] for r in rs])
        inst = np.array([r["inst"] for r in rs])
        e_ov = (M.sigmoid(np.column_stack([np.ones(len(y)), ov]) @ M.fit(np.column_stack([np.ones(len(y)), ov]), y)[0]) - y) ** 2
        Xb = np.column_stack([np.ones(len(y)), ov, me])
        e_bd = (M.sigmoid(Xb @ M.fit(Xb, y)[0]) - y) ** 2
        tss = float(((y - y.mean()) ** 2).sum())
        off = 1.0 - 10.0 ** (ov - me)
        mine = []
        for i in np.unique(inst):
            m = inst == i
            s = spec[f"{name}/{int(i)}"]
            mine.append(dict(dataset=name, inst=int(i), density=s["density"], levels=s["levels"], n_band=s["n_band"],
                             n_ground=s["n_ground"], off_share=float(off[m].mean()),
                             gain=float(e_ov[m].mean() - e_bd[m].mean()), mse_ov=float(e_ov[m].mean()),
                             p_spread=float(y[m].std())))
        rows += mine
        d = np.array([r["density"] for r in mine])
        lv = np.array([r["levels"] for r in mine])
        g = np.array([r["gain"] for r in mine])
        store["datasets"][name] = dict(
            n_inst=len(mine), r2_ov=1 - float(e_ov.sum()) / tss, r2_band=1 - float(e_bd.sum()) / tss,
            density_median=float(np.median(d)), levels_median=float(np.median(lv)),
            band_states_median=float(np.median([r["n_band"] for r in mine])),
            off_share_mean=float(np.mean([r["off_share"] for r in mine])), gain_mean=float(g.mean()),
            n_gain_pos=int((g > 1e-4).sum()))
        o = store["datasets"][name]
        lines.append(f"\n{name}: {len(mine)} instances, {len(y)} runs | R^2 overlap {o['r2_ov']:.3f} -> + band "
                     f"{o['r2_band']:.3f} | band: median {o['band_states_median']:.0f} states, {o['levels_median']:.0f} "
                     f"levels above ground (range {lv.min()}-{lv.max()}), density median {o['density_median']:.2f} "
                     f"(range {d.min():.2f}-{d.max():.2f}) | off-optimum share of band mass {o['off_share_mean']:.3f} | "
                     f"mean gain {o['gain_mean']:+.4f}, band helps (gain > 1e-4) in {o['n_gain_pos']}/{len(mine)}")
        for lab, x in (("density", d), ("levels", lv), ("off_share", [r["off_share"] for r in mine])):
            r, p = rho(x, g)
            store["tests"][f"{name}/{lab}"] = dict(rho=r, p=p, n=len(mine))
            lines.append(f"    within {name}: Spearman(gain, {lab}) = {r:+.3f} (p = {p:.2g}, n = {len(mine)})")

    lines.append("\nPOOLED over all instances of both classes (confounded with the class difference; read with the "
                 "within-dataset rows above):")
    g = np.array([r["gain"] for r in rows])
    ds = np.array([r["dataset"] for r in rows])
    for lab in ("density", "levels", "off_share"):
        x = np.array([r[lab] for r in rows], float)
        r, p = rho(x, g)
        # class-adjusted: ranks taken within each dataset, then pooled (removes any between-dataset difference)
        xr, gr = np.empty(len(x)), np.empty(len(x))
        for k in np.unique(ds):
            m = ds == k
            xr[m], gr[m] = stats.rankdata(x[m]) / m.sum(), stats.rankdata(g[m]) / m.sum()
        ra, pa = rho(xr, gr)
        store["tests"][f"pooled/{lab}"] = dict(rho=r, p=p, n=len(rows), rho_within=ra, p_within=pa)
        lines.append(f"    Spearman(gain, {lab}) = {r:+.3f} (p = {p:.2g}, n = {len(rows)}); with ranks taken within "
                     f"each dataset: {ra:+.3f} (p = {pa:.2g})")
    lines.append("  by density tercile (pooled): " + " | ".join(
        f"density {lo:.2f}-{hi:.2f}: mean gain {g[m].mean():+.4f}, band helps in {(g[m] > 1e-4).sum()}/{m.sum()}"
        for lo, hi, m in ((lo, hi, (dd >= lo) & (dd <= hi)) for dd in [np.array([r['density'] for r in rows])]
                          for lo, hi in zip(np.quantile(dd, [0, 1 / 3, 2 / 3]), np.quantile(dd, [1 / 3, 2 / 3, 1])))))
    store["instances"] = rows
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUT.with_suffix(".json").write_text(json.dumps(store, indent=1, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
