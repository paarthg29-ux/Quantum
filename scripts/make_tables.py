"""Every number in the paper, from the result files. No number is typed by hand.

Reads the result stores under results/ (run stores, census stores, model
metadata and the structured outputs of the analysis scripts) and writes

    paper/numbers.tex     one \\newcommand per in-text number
    paper/tables/*.tex    the body (tabular) of every table; captions live in
                          the .tex sources and may use the macros

Statistics are recomputed here from the stores, deterministically (fixed
seeds), with the same definitions as the analysis scripts:
scenario-level = values averaged over seeds within a scenario, then a paired
t-interval over scenarios; group-level = the same with one value per
structural group (s8_analysis.structural_groups). If a result file changes,
re-run this script and the paper updates.

    python scripts/make_tables.py
"""

import contextlib
import csv
import io
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PAPER = ROOT / "paper"
TABLES = PAPER / "tables"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

MACROS: dict[str, str] = {}
LOW_THRESHOLD = 0.9  # "an arm falls below P_K = LOW_THRESHOLD in ... scenarios" (descriptive count in Results)
REGRET_RANGE = 0.25  # the eps range [0, REGRET_RANGE] over which the learned arms' maximum regret is quoted


# ---- formatting -------------------------------------------------------------------


def unsign_zero(s: str) -> str:
    """'+0.000' / '-0.000' (a value that rounds to zero) is printed unsigned as '0.000'."""
    return re.sub(r"[+-](0\.0+)(?![\d])", lambda m: m.group(1), s)


def math_signs(cell: str) -> str:
    """Signed numbers in text mode ('-0.100') get a real minus sign: wrap them in \\ensuremath. Numbers already in
    math (inside a CI '[...]' or after '\\,') are left alone."""
    return re.sub(r"(?<![\[,\\\w{$])([+-]\d+\.\d+)", lambda m: "\\ensuremath{" + m.group(1) + "}", cell)


def mac(name: str, value: str):
    value = unsign_zero(value)
    if not re.fullmatch(r"[A-Za-z]+", name):
        raise ValueError(f"macro name must be letters only: {name!r}")
    if name in MACROS:
        raise ValueError(f"macro defined twice: {name}")
    MACROS[name] = value


def f3(x):
    return f"{x:.3f}"


def f2(x):
    return f"{x:.2f}"


def s3(x):
    """Signed, math mode (a real minus sign)."""
    return rf"\ensuremath{{{x:+.3f}}}"


def s2(x):
    return rf"\ensuremath{{{x:+.2f}}}"


def ci(lo, hi, f=lambda v: f"{v:+.3f}"):
    return rf"\ensuremath{{[{f(lo)},\,{f(hi)}]}}"


def ciu(lo, hi, nd=3):
    return rf"\ensuremath{{[{lo:.{nd}f},\,{hi:.{nd}f}]}}"


def sci(x, nd=1):
    m, e = f"{x:.{nd}e}".split("e")
    return rf"\ensuremath{{{m}\times 10^{{{int(e)}}}}}"


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}"


def tci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    m = v.mean()
    if v.size < 2 or v.std() == 0:
        return m, m, m
    h = stats.t.ppf(0.975, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size)
    return m, m - h, m + h


def paired(x: dict, y: dict):
    common = sorted(s for s in x if s in y and x[s] is not None and y[s] is not None)
    d = np.array([x[s] - y[s] for s in common], float)
    m, lo, hi = tci(d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pw = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return dict(mean=m, lo=lo, hi=hi, n=d.size, win=int((d > 1e-12).sum()), tie=int((np.abs(d) <= 1e-12).sum()),
                loss=int((d < -1e-12).sum()), p=pw, d=d)


def table(name: str, colspec: str, header: list[str], rows: list[list[str]], midrules=()):
    lines = [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for i, r in enumerate(rows):
        if i in midrules:
            lines.append(r"\midrule")
        lines.append(" & ".join(math_signs(unsign_zero(c)) for c in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


ARM_TEX = {"A": "A (cold)", "B": "B (GCN, X mixer)", "C-milp": "C-milp (GCN)", "C-oracle": "C-oracle (GCN)",
           "D": "D (LP)", "E": "E (random)", "F": "F (merit order)"}


# ---- data ---------------------------------------------------------------------------


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def load_s8():
    import s8_analysis as A8
    from s8_eps_main import tasks

    recs = quiet(A8.load)
    k1, k2 = tasks(1), tasks(2)
    key = lambda r: (r["arm"], r["eps"], r["scenario"], r["seed"], r["p"])  # noqa: E731
    D1 = A8.Data([r for r in recs if key(r) in k1])
    D2 = A8.Data([r for r in recs if key(r) in k2])
    groups = A8.structural_groups()
    return A8, D1, D2, groups


# ---- sections ------------------------------------------------------------------------


def methods_numbers(A8, D2, groups):
    from qguc.config import FROZEN
    from qguc.instance import FLEET_CAPACITY_MW, LOAD_MAX_MW, LOAD_MIN_MW, REFINEMENT_LEVELS, refinement_fleet
    from qguc.qaoa import OptimiserSettings
    from qguc.sampling import KS
    from qguc.warmstart import DEFAULT_EPS
    from s8_eps_main import EPS, MAIN_SEEDS, N_SCEN, SCEN_SEED, SWEEP_SEEDS

    mac("FleetCap", f"{FLEET_CAPACITY_MW:.0f}")
    mac("LoadMin", f"{LOAD_MIN_MW:.0f}")
    mac("LoadMax", f"{LOAD_MAX_MW:.0f}")
    mac("LambdaD", f"{FROZEN.lambda_demand:g}")
    mac("BiasB", f"{FROZEN.demand_bias_mw:g}")
    mac("LambdaSwitch", f"{FROZEN.lambda_switch:g}")
    mac("EpsDefault", f"{DEFAULT_EPS:g}")
    s = OptimiserSettings()
    mac("RhoBeg", f"{s.rhobeg:g}")
    mac("RhoEnd", rf"\ensuremath{{10^{{{int(np.log10(s.rhoend))}}}}}")
    mac("CapPerParam", f"{s.cap_per_param}")
    mac("Patience", f"{s.patience}")
    mac("MinDelta", rf"\ensuremath{{10^{{{int(np.log10(s.min_delta))}}}}}")
    mac("KList", ", ".join(str(k) for k in KS))
    mac("KMax", f"{max(KS)}")
    mac("LowThreshold", f"{LOW_THRESHOLD:g}")
    mac("EpsSecond", f"{sorted(EPS)[1]:g}")
    mac("RegretRange", f"{REGRET_RANGE:g}")
    import s10_robust as RB
    import s10_ood as OD
    import s10_sa_compiled as SC
    import s9_l4_arms as L4

    mac("ShotsMin", f"{min(RB.SHOTS)}")
    mac("NSeedsRobust", f"{len(RB.SEEDS)}")
    mac("NSeedsOOD", f"{len(OD.SEEDS)}")
    mac("NSeedsLfour", f"{len(L4.SEEDS)}")
    mac("SAReplicas", f"{SC.R}")
    lad = sorted(SC.LADDER)
    mac("SABudgetMid", rf"\ensuremath{{10^{{{int(np.log10(lad[1]))}}}}}")
    mac("SABudgetHigh", rf"\ensuremath{{10^{{{int(np.log10(lad[2]))}}}}}")
    mac("NScen", f"{N_SCEN}")
    mac("ScenSeed", f"{SCEN_SEED}")
    mac("NSeedsMain", f"{len(MAIN_SEEDS)}")
    mac("NSeedsSweep", f"{len(SWEEP_SEEDS)}")
    mac("EpsGrid", ", ".join(f"{e:g}" for e in EPS))
    mac("EpsMax", f"{max(EPS):g}")
    from s8_eps_main import MAIN_EPS

    mac("EpsMain", f"{MAIN_EPS:g}")
    mac("NGroupsGL", f"{len(set(groups['GL'].values()))}")
    mac("NGroupsG", f"{len(set(groups['G'].values()))}")
    blocks = [len(refinement_fleet(k)) for k in REFINEMENT_LEVELS]
    mac("LadderBlocks", ", ".join(str(b) for b in blocks))
    mac("LadderQubits", ", ".join(str(2 * b) for b in blocks))

    # calibration (s2/s4b log lines: the selected cell, its test check, the no-plateau reference)
    txt = (RES / "s4b_recalibrate_L2.txt").read_text()
    head = re.search(r"selection (\d+) scenarios \(seed (\d+)\), test (\d+) \(seed (\d+)\); lambda_D (\d+) points "
                     r"in \[([\d.]+), ([\d.]+)\]; biases \[([^\]]+)\]", txt)
    mac("CalSel", head.group(1))
    mac("CalSelSeed", head.group(2))
    mac("CalTest", head.group(3))
    mac("CalTestSeed", head.group(4))
    mac("CalNLambda", head.group(5))
    mac("CalLambdaMin", head.group(6))
    mac("CalLambdaMax", head.group(7))
    mac("CalNBias", f"{len(head.group(8).split(','))}")
    mac("CalBiasMax", head.group(8).split(",")[-1].strip())
    sel = re.search(r"SELECTED \[frozen rule, plateau=1\]: .*?adequacy (\d+)%, gap mean ([\d.]+)%, worst ([\d.]+)%, "
                    r"=MILP (\d+)%, dynIsing ([\d.e+]+)\n\s+TEST .*?adequacy (\d+)%, gap mean ([\d.]+)%, worst ([\d.]+)%",
                    txt)
    mac("CalSelGapMean", sel.group(2))
    mac("CalSelGapWorst", sel.group(3))
    mac("CalSelMatch", sel.group(4))
    mac("CalDynRange", sci(float(sel.group(5)), 1))
    mac("CalTestAdeq", sel.group(6))
    mac("CalTestGapMean", sel.group(7))
    mac("CalTestGapWorst", sel.group(8))
    ref = re.search(r"SELECTED \[reference, no plateau, plateau=0\]: lambda_D = ([\d.]+), bias = (\d+) MW -> "
                    r"adequacy \d+%, gap mean ([\d.]+)%, worst ([\d.]+)%.*?\n\s+TEST .*?gap mean ([\d.]+)%", txt)
    mac("CalNoPlateauLambda", f"{float(ref.group(1)):.1f}")
    mac("CalNoPlateauBias", ref.group(2))
    mac("CalNoPlateauGapMean", ref.group(3))
    # the first calibration (on L4) and the re-verification that triggered the re-calibration
    from qguc.config import SUPERSEDED

    mac("FirstBias", f"{SUPERSEDED[0].demand_bias_mw:g}")
    with open(RES / "s4_ladder_before_recal.csv") as f:
        before = {r["level"]: r for r in csv.DictReader(f)}
    mac("FirstPointLtwoAdeq", f"{100 * float(before['L2']['adequacy']):.0f}")

    # encoding-gap ladder table
    with open(RES / "s4_ladder.csv") as f:
        lad = {r["level"]: r for r in csv.DictReader(f)}
    with open(RES / "s7_gap_ladder.csv") as f:
        own = {r["level"]: r for r in csv.DictReader(f)}
    rows = []
    for k in REFINEMENT_LEVELS:
        L, O = lad[f"L{k}"], own[f"L{k}"]
        rows.append([f"L{k}", L["blocks"], L["qubits"], pct(float(L["adequacy"]), 0),
                     f2(100 * float(L["gap_mean"])), f2(100 * float(L["gap_worst"])), pct(float(L["match"]), 0),
                     sci(float(L["dyn_worst"]), 1),
                     f"{float(O['own_lambda']):.3g}", f"{float(O['own_bias']):.0f}", f2(float(O["own_mean"])),
                     f2(float(O["own_worst"]))])
    table("ladder", "lrrrrrrrrrrr",
          ["Level", "Blocks", "$n$", r"Adeq.\,\%", r"Gap\,\%", r"Worst\,\%", r"$=$MILP\,\%", r"Dyn.\ range",
           r"$\lambda_D$", r"$B$", r"Gap\,\%", r"Worst\,\%"], rows)
    gaps = [100 * float(lad[f"L{k}"]["gap_mean"]) for k in REFINEMENT_LEVELS]
    mac("GapFrozenMin", f2(min(gaps)))
    mac("GapFrozenMax", f2(max(gaps)))
    owng = [float(own[f"L{k}"]["own_mean"]) for k in REFINEMENT_LEVELS]
    mac("GapOwnMin", f2(min(owng)))
    mac("GapOwnMax", f2(max(owng)))
    mac("GapFrozenLfour", f2(gaps[-1]))
    mac("GapOwnLfour", f2(owng[-1]))
    mac("MatchLfour", pct(float(lad["L4"]["match"]), 0))
    mac("OwnBiasLfour", f"{float(own['L4']['own_bias']):.0f}")
    mac("OwnLambdaLfour", f"{float(own['L4']['own_lambda']):.3g}")
    dyns = [float(lad[f"L{k}"]["dyn_worst"]) for k in REFINEMENT_LEVELS]
    mac("DynRangeMin", sci(min(dyns), 1))
    mac("DynRangeMax", sci(max(dyns), 1))

    # configuration constants quoted in Methods (read from the code / logs, not typed)
    import inspect

    import qguc.instance as I
    from qguc.arms import PIN_TAU, c_merit_order
    from qguc.fastsim import QAOASimulator
    from qguc.gnn import assert_load_conditioned

    mac("TroughPct", f"{100 * (1 - I._PEAK_DEPTH):.0f}")
    mac("JitterPct", f"{100 * I._JITTER_FRAC:.0f}")
    mac("PinTau", f"{PIN_TAU:g}")
    mac("PinTauHigh", f"{1 - PIN_TAU:g}")
    mp = inspect.signature(c_merit_order).parameters
    mac("MeritOn", f"{mp['on'].default:g}")
    mac("MeritOff", f"{mp['off'].default:g}")
    mac("MixerGroup", f"{inspect.signature(QAOASimulator.__init__).parameters['group'].default}")
    lp = inspect.signature(assert_load_conditioned).parameters
    mac("LoadCondMin", f"{lp['min_abs_diff'].default:g}")
    mac("LoadCondFrac", f"{lp['min_capacity_fraction'].default:g}")
    import s7a_train_gnn as S7

    mac("NHeldOutGCN", f"{S7.HELD[0]}")
    src = inspect.getsource(S7.probe_pair)
    lo_hi = [float(v) for v in re.findall(r"mk\(([\d.]+)\)", src)]
    mac("LoadProbeLow", f"{lo_hi[0]:.0f}")
    mac("LoadProbeHigh", f"{lo_hi[1]:.0f}")
    lt = (RES / "s4_ladder.txt").read_text()
    mac("NLadderScen", re.search(r"; (\d+) scenarios \(seed (\d+)\)", lt).group(1))
    mac("LadderSeed", re.search(r"; (\d+) scenarios \(seed (\d+)\)", lt).group(2))
    mac("ReverifyThreshold", re.search(r"Re-verification rule \(L2 adequacy >= (\d+)%\)", lt).group(1))
    mac("NLtwo", f"{2 * len(refinement_fleet(2))}")
    mac("NLfour", f"{2 * len(refinement_fleet(4))}")
    from s8_eps_main import N_HOURS

    mac("NHoursMain", f"{N_HOURS}")
    import s8_mass as M8

    fsrc = inspect.getsource(M8.features)
    w5 = float(re.search(r'out\["m_e05"\] = float\(ps\[Es - ev.e_min <= ([\d.]+) \* rng\]', fsrc).group(1))
    w1 = float(re.search(r'out\["m_e01"\] = float\(ps\[Es - ev.e_min <= ([\d.]+) \* rng\]', fsrc).group(1))
    mac("BandFivePct", f"{100 * w5:.0f}")
    mac("BandOnePct", f"{100 * w1:.0f}")

    # raw (unnormalised) coefficient scale of the Ising model at L2
    from qguc import make_scenario_set
    from qguc.qaoa import make_problem

    pb = make_problem(make_scenario_set(1, n_hours=2, seed=SCEN_SEED, refinement_level=2)[0])
    mac("IsingScaleLtwo", sci(pb.ev.ising.scale, 1))


def validation_numbers():
    v = json.loads((RES / "s10_simulator_validation.json").read_text())
    mac("ValCases", f"{v['n_cases']}")
    mac("ValNMin", f"{min(c['n'] for c in v['cases'])}")
    mac("ValNMax", f"{max(c['n'] for c in v['cases'])}")
    mac("ValPMax", f"{max(c['p'] for c in v['cases'])}")
    mac("ValWorst", sci(v["worst"], 1))
    mac("ValUCQubits", f"{v['uc_instance']['n']}")
    mac("ValUCWorst", sci(max(max(c["state"], c["energy"]) for c in v["uc_instance"]["circuits"]), 1))
    mac("ValMCMaxZ", f2(max(m["max_z"] for m in v["monte_carlo"])))
    bugs = [b for b in v["injected_bugs"] if "correct" not in b["bug"] and b["bug"] != "none (correct simulator)"]
    mac("ValBugMinState", f2(min(b["state"] for b in bugs)))
    mac("ValBugN", f"{len(bugs)}")
    mac("ValBugQubits", f"{v['injected_n']}")
    mac("ValBugP", f"{v['injected_p']}")
    mac("ValCorrectState", sci(next(b["state"] for b in v["injected_bugs"] if b["bug"].startswith("none")), 0))
    rows = [[b["bug"].replace("_", " "), sci(b["state"], 1), sci(b["energy"], 1)] for b in v["injected_bugs"]]
    table("validation_bugs", "lrr", ["Simulator variant", "State error", "Energy error"], rows)
    t = (RES / "s3b_fastsim_timing.txt").read_text()
    m = re.search(r"L4\s+22\s+[\d.]+s \|\s+([\d.]+)\s+([\d.]+) \|", t)
    mac("SimEvalLfourCsixfour", f"{float(m.group(2)):.2f}")


def gcn_numbers():
    import torch

    from qguc.gnn import FEATURES, MODEL_DIR

    meta = torch.load(MODEL_DIR / "gnn_milp_L2.pt", weights_only=False)["meta"]
    mac("GCNHidden", f"{meta['hidden']}")
    mac("GCNLr", rf"\ensuremath{{10^{{{int(np.log10(meta['lr']))}}}}}")
    mac("GCNWd", rf"\ensuremath{{10^{{{int(np.log10(meta['weight_decay']))}}}}}")
    mac("GCNNFeatures", f"{len(FEATURES)}")
    mac("GCNNTrain", f"{meta['n_train']}")
    mac("GCNNVal", f"{meta['n_val']}")
    mac("GCNLoadDiffLtwo", f2(meta["load_check"]["mean_abs_diff"]))
    m4 = torch.load(MODEL_DIR / "gnn_milp_L4.pt", weights_only=False)["meta"]
    mac("GCNLoadDiffLfour", f2(m4["load_check"]["mean_abs_diff"]))
    from qguc.gnn import train  # noqa: F401  (signature defaults: epochs, patience)
    import inspect

    sig = inspect.signature(train)
    mac("GCNEpochs", f"{sig.parameters['epochs'].default}")
    mac("GCNPatience", f"{sig.parameters['patience'].default}")
    for fname, tag in (("s7a_train_gnn.txt", "Ltwo"), ("s9_train_gnn_L4.txt", "Lfour")):
        t = (RES / fname).read_text()
        for variant in ("milp", "oracle"):
            m = re.search(rf"C-{variant}: labels in ([\d.]+) s =+\ntrained in ([\d.]+) s", t)
            if m:
                V = "Milp" if variant == "milp" else "Oracle"
                mac(f"GCNLabels{V}{tag}", f"{float(m.group(1)):.1f}")
                mac(f"GCNTrain{V}{tag}", f"{float(m.group(2)):.0f}")
        m = re.search(r"C-milp.*?features \+ forward (\d+) us", t, re.S)
        mac(f"GCNInferUs{tag}", m.group(1))
    t = (RES / "s9_gnn_vs_constant.txt").read_text()
    for lv, tag in ((2, "Ltwo"), (4, "Lfour")):
        m = re.search(rf"L{lv}\s+\d+ \|\s+([\d.]+)\s+([\d.]+) \|\s+([\d.]+)\s+([\d.]+) \|\s+([\d.]+)\s+([\d.]+)", t)
        mac(f"GCNMae{tag}", f3(float(m.group(1))))
        mac(f"GCNMaeConst{tag}", f3(float(m.group(2))))
        mac(f"GCNBce{tag}", f3(float(m.group(3))))
        mac(f"GCNBceConst{tag}", f3(float(m.group(4))))
        mac(f"GCNAcc{tag}", f3(float(m.group(5))))
        mac(f"GCNAccMaj{tag}", f3(float(m.group(6))))


def arms_section(A8, D1, D2, groups):
    """Claim 1: main table at eps = 0.1 (Phase 2) and the best-eps table (Phase 1)."""
    E, S5, S3 = 0.1, A8.MAIN_SEEDS, A8.SWEEP_SEEDS
    gm = groups["GL"]
    arms = ("C-oracle", "C-milp", "B", "D", "F", "A", "E")
    rows = []
    for a in arms:
        pg = {p: list(D2.scen(a, E, p, "p_ground_K1024", S5).values()) for p in (1, 2, 3)}
        m, lo, hi = tci(pg[1])
        rows.append([ARM_TEX[a], f"{m:.3f} {ciu(lo, hi)}", f3(np.mean(pg[2])), f3(np.mean(pg[3])),
                     f"{np.mean(list(D2.scen(a, E, 1, 'p_ground_K1', S5).values())):.4f}",
                     f3(np.mean(list(D2.scen(a, E, 1, 'g_final', S5).values()))),
                     f2(100 * np.mean(list(D2.scen(a, E, 1, 'gap_repaired_K1024', S5).values()))),
                     f2(np.mean(list(D2.scen(a, E, 1, 'n_pinned_wrong', S5).values())))])
        tag = a.replace("-", "").replace("milp", "milp").title().replace("Coracle", "Coracle")
        tag = {"C-oracle": "Coracle", "C-milp": "Cmilp"}.get(a, a)
        mac(f"Pgnd{tag}", f3(m))
        mac(f"PgndCI{tag}", ciu(lo, hi))
        mac(f"PgndPthree{tag}", f3(np.mean(pg[3])))
        mac(f"Gfinal{tag}", f3(np.mean(list(D2.scen(a, E, 1, "g_final", S5).values()))))
        mac(f"Gap{tag}", f2(100 * np.mean(list(D2.scen(a, E, 1, "gap_repaired_K1024", S5).values()))))
    table("arms", "lcccccccc".replace("cccccccc", "ccccccc"),
          ["Arm", r"$P_{\KMax}$, $p{=}1$ [95\% CI]", r"$p{=}2$", r"$p{=}3$", r"$P_{1}$", r"$g$",
           r"Gap\,\%", "Wrong"], rows)
    n_low = {a: sum(v < LOW_THRESHOLD for v in D2.scen(a, E, 1, "p_ground_K1024", S5).values()) for a in ("C-milp", "C-oracle", "D")}
    mac("NLowD", f"{n_low['D']}")
    mac("NLowCmilp", f"{n_low['C-milp']}")
    mac("NLowCoracle", f"{n_low['C-oracle']}")

    def pair_macros(tag, a, b, key="p_ground_K1024", D=D2, ea=E, eb=E, seeds=S5, scale=1.0):
        x, y = D.scen(a, ea, 1, key, seeds), D.scen(b, eb, 1, key, seeds)
        if scale != 1.0:
            x = {k: scale * v for k, v in x.items()}
            y = {k: scale * v for k, v in y.items()}
        r = paired(x, y)
        mac(f"{tag}", s3(r["mean"]))
        mac(f"{tag}CI", ci(r["lo"], r["hi"]))
        mac(f"{tag}W", f"{r['win']}")
        mac(f"{tag}L", f"{r['loss']}")
        mac(f"{tag}N", f"{r['n']}")
        for gname, gl in (("GL", groups["GL"]), ("G", groups["G"])):
            rg = paired(A8.to_groups(x, gl), A8.to_groups(y, gl))
            mac(f"{tag}{gname}", s3(rg["mean"]))
            mac(f"{tag}{gname}CI", ci(rg["lo"], rg["hi"]))
        return r

    pair_macros("CmilpMinusD", "C-milp", "D")
    pair_macros("CoracleMinusD", "C-oracle", "D")
    pair_macros("CmilpMinusF", "C-milp", "F")
    pair_macros("CmilpMinusA", "C-milp", "A")
    pair_macros("DMinusA", "D", "A")
    pair_macros("FMinusA", "F", "A")
    pair_macros("EMinusA", "E", "A")
    pair_macros("CoracleMinusCmilp", "C-oracle", "C-milp")
    pair_macros("BMinusCmilp", "B", "C-milp")
    pair_macros("GapCmilpMinusD", "C-milp", "D", key="gap_repaired_K1024", scale=100.0)
    gflip = pair_macros("GfinalCoracleMinusD", "C-oracle", "D", key="g_final")
    pair_macros("GfinalBMinusCmilp", "B", "C-milp", key="g_final")
    pair_macros("PoneBMinusCmilp", "B", "C-milp", key="p_ground_K1")

    # best eps (Phase 1, p = 1, in-sample pick, eps = 0.5 excluded)
    best = {a: A8.best_eps(D1, a, 1) for a in A8.SWEEP_ARMS}
    rows = []
    for a in ("C-oracle", "C-milp", "B", "D", "F"):
        pb = list(D1.scen(a, best[a], 1, "p_ground_K1024", S3).values())
        m, lo, hi = tci(pb)
        p01 = np.mean(list(D1.scen(a, 0.1, 1, "p_ground_K1024", S3).values()))
        p0 = np.mean(list(D1.scen(a, 0.0, 1, "p_ground_K1024", S3).values()))
        p5 = np.mean(list(D1.scen(a, 0.5, 1, "p_ground_K1024", S3).values()))
        gb = np.mean(list(D1.scen(a, best[a], 1, "g_final", S3).values()))
        g01 = np.mean(list(D1.scen(a, 0.1, 1, "g_final", S3).values()))
        rows.append([ARM_TEX[a], f"{best[a]:g}", f"{m:.3f} {ciu(lo, hi)}", f3(p01), f3(p0), f3(p5), f3(gb), f3(g01)])
        tag = {"C-oracle": "Coracle", "C-milp": "Cmilp"}.get(a, a)
        mac(f"BestEps{tag}", f"{best[a]:g}")
        mac(f"PgndBest{tag}", f3(m))
        mac(f"PgndBestCI{tag}", ciu(lo, hi))
        mac(f"PgndEpsZero{tag}", f3(p0))
    pa = list(D1.scen("A", 0.1, 1, "p_ground_K1024", S3).values())
    rows.append([ARM_TEX["A"], "--", f"{np.mean(pa):.3f}", "--", "--", "--", "--", "--"])
    table("besteps", "lccccccc", ["Arm", r"$\varepsilon^\star$", r"$P_{\KMax}$ at $\varepsilon^\star$ [95\% CI]",
                                  r"$\varepsilon{=}0.1$", r"$\varepsilon{=}0$", r"$\varepsilon{=}0.5$",
                                  r"$g$ at $\varepsilon^\star$", r"$g$ at $0.1$"], rows, midrules=(5,))

    # decisive at best eps, scenario and group level
    dec = []
    for a, b, tag in (("D", "C-oracle", "BestDMinusCoracle"), ("D", "C-milp", "BestDMinusCmilp"),
                      ("D", "F", "BestDMinusF"), ("C-oracle", "C-milp", "BestCoracleMinusCmilp"),
                      ("B", "C-milp", "BestBMinusCmilp")):
        r = pair_macros(tag, a, b, D=D1, ea=best[a], eb=best[b], seeds=S3)
        rg = paired(A8.to_groups(D1.scen(a, best[a], 1, "p_ground_K1024", S3), groups["GL"]),
                    A8.to_groups(D1.scen(b, best[b], 1, "p_ground_K1024", S3), groups["GL"]))
        rg2 = paired(A8.to_groups(D1.scen(a, best[a], 1, "p_ground_K1024", S3), groups["G"]),
                     A8.to_groups(D1.scen(b, best[b], 1, "p_ground_K1024", S3), groups["G"]))
        dec.append([f"{a} $-$ {b}", f"{r['mean']:+.3f} {ci(r['lo'], r['hi'])}", f"{r['win']}/{r['tie']}/{r['loss']}",
                    f"{rg['mean']:+.3f} {ci(rg['lo'], rg['hi'])}", f"{rg2['mean']:+.3f} {ci(rg2['lo'], rg2['hi'])}"])
    gbest = pair_macros("GfinalBestDMinusCoracle", "D", "C-oracle", key="g_final", D=D1, ea=best["D"],
                        eb=best["C-oracle"], seeds=S3)
    mac("GfinalFlipN", f"{gflip['n']}")
    return best, dec


def cv_and_regret(A8, D1, groups, best, dec):
    """Claims 2 and 5: repeated group-fold CV (s8_eps_stability procedure) and eps regret."""
    gm = groups["GL"]
    G = sorted(set(gm.values()))
    EPS = [e for e in A8.EPS if e < 0.5]
    arms = A8.SWEEP_ARMS
    P, GAP = {}, {}
    for a in arms:
        P[a] = np.array([[A8.to_groups(D1.scen(a, e, 1, "p_ground_K1024", A8.SWEEP_SEEDS), gm)[g] for g in G] for e in EPS])
        GAP[a] = np.array([[A8.to_groups(D1.scen(a, e, 1, "gap_repaired_K1024", A8.SWEEP_SEEDS), gm)[g] for g in G]
                           for e in EPS])

    def pick(a, cols):
        sc = [(P[a][i, cols].mean(), -GAP[a][i, cols].mean()) for i in range(len(EPS))]
        return max(range(len(EPS)), key=lambda i: sc[i])

    rng = np.random.default_rng(0)
    boot_idx = rng.integers(0, len(G), (2000, len(G)))
    freq = {}
    for a in arms:
        picks = np.array([EPS[pick(a, b)] for b in boot_idx])
        freq[a] = {e: float(np.mean(picks == e)) for e in EPS}
    mac("BootPickD", pct(freq["D"][best["D"]], 0))
    mac("BootPickCoracle", pct(freq["C-oracle"][best["C-oracle"]], 0))
    mac("BootPickCmilp", pct(freq["C-milp"][best["C-milp"]], 0))
    mac("NBoot", "2000")
    splits = []
    for _ in range(500):
        perm = rng.permutation(len(G))
        A_, B_ = perm[: len(G) // 2], perm[len(G) // 2:]
        splits += [(A_, B_), (B_, A_)]
    mac("NSplits", f"{len(splits)}")
    held = {}
    for a in arms:
        held[a] = np.array([P[a][pick(a, tr), te].mean() for tr, te in splits])
    cvrows = []
    for a, b, tag in (("D", "C-oracle", "CVDMinusCoracle"), ("D", "C-milp", "CVDMinusCmilp"), ("D", "F", "CVDMinusF")):
        d = held[a] - held[b]
        mac(tag, s3(d.mean()))
        mac(f"{tag}Range", ci(np.percentile(d, 2.5), np.percentile(d, 97.5)))
        mac(f"{tag}Worse", pct(np.mean(d < 0), 0))
        cvrows.append([f"{a} $-$ {b}", f"{d.mean():+.3f}", ci(np.percentile(d, 2.5), np.percentile(d, 97.5)),
                       pct(np.mean(d < 0), 0)])
    table("decisive", "lcccc", [r"Pair (each at its $\varepsilon^\star$)", r"Scenarios ($n{=}\NScen$) [95\% CI]",
                                 "W/T/L", r"GL groups ($n{=}\NGroupsGL$)", r"G groups ($n{=}\NGroupsG$)"], dec)
    table("cv", "lccc", ["Pair", "Mean", r"95\% of splits", r"\% splits $<0$"], cvrows)

    # regret relative to the pooled best eps, per group
    rows = []
    allg = np.arange(len(G))
    for a in ("C-oracle", "C-milp", "B", "D", "F"):
        ib = pick(a, allg)
        cells = []
        for i in range(len(EPS)):
            d = P[a][ib] - P[a][i]
            m, lo, hi = tci(d)
            cells.append(f"{m:.3f}" if i != ib else r"\textbf{0}")
        rows.append([ARM_TEX[a]] + cells)
        tag = {"C-oracle": "Coracle", "C-milp": "Cmilp"}.get(a, a)
        within = [i for i, e in enumerate(EPS) if e <= REGRET_RANGE]
        mac(f"RegretMax{tag}", f3(max(P[a][ib].mean() - P[a][i].mean() for i in within)))
        mac(f"RegretEpsZero{tag}", f3(P[a][ib].mean() - P[a][0].mean()))
        mac(f"RegretEpsFive{tag}", f3(P[a][ib].mean() - P[a][1].mean()))
    table("regret", "l" + "c" * len(EPS), ["Arm"] + [rf"$\varepsilon{{=}}{e:g}$" for e in EPS], rows)
    m, lo, hi = tci(P["D"][pick("D", allg)] - P["D"][1])
    mac("RegretEpsFiveDCI", ciu(lo, hi))

    # MERGED eps table (replaces besteps / decisive / cv / regret in the paper): per arm, its best eps, P and g there,
    # g at the main eps, regret at eps = 0 and the second grid value, and the LP's deficit to it (scenario, group, CV)
    rows = []
    S3 = A8.SWEEP_SEEDS
    for a in ("C-oracle", "C-milp", "B", "F", "D"):
        ib = pick(a, allg)
        pb = list(D1.scen(a, best[a], 1, "p_ground_K1024", S3).values())
        m, lo, hi = tci(pb)
        gb = np.mean(list(D1.scen(a, best[a], 1, "g_final", S3).values()))
        g01 = np.mean(list(D1.scen(a, 0.1, 1, "g_final", S3).values()))
        cells = [ARM_TEX[a], f"{best[a]:g}", f"{m:.3f} {ciu(lo, hi)}", f3(gb), f3(g01),
                 f3(P[a][ib].mean() - P[a][0].mean()), f3(P[a][ib].mean() - P[a][1].mean())]
        if a == "D":
            cells += ["--", "--", "--"]
        else:
            xd = D1.scen("D", best["D"], 1, "p_ground_K1024", S3)
            xa = D1.scen(a, best[a], 1, "p_ground_K1024", S3)
            r = paired(xd, xa)
            rg = paired(A8.to_groups(xd, gm), A8.to_groups(xa, gm))
            d = held["D"] - held[a]
            cells += [f"{r['mean']:+.3f} {ci(r['lo'], r['hi'])}", f"{rg['mean']:+.3f} {ci(rg['lo'], rg['hi'])}",
                      f"{d.mean():+.3f} ({pct(np.mean(d < 0), 0)}\\%)"]
        rows.append(cells)
    table("epsmerged", "lcccccccccc",
          ["Arm", r"$\varepsilon^\star$", r"$P_{\KMax}$ at $\varepsilon^\star$ [95\% CI]", r"$g$ at $\varepsilon^\star$",
           r"$g$ at $\EpsMain$", r"Regret $\varepsilon{=}0$", r"Regret $\varepsilon{=}\EpsSecond$",
           r"D $-$ arm, scenarios", r"D $-$ arm, GL groups", r"D $-$ arm, group CV"], rows, midrules=(4,))
    return P, EPS


def dose_section(A8, D2, groups):
    """Claim 3: P(b1024) by the LP's wrong-pin count, L2 and L4."""
    E, S5 = 0.1, A8.MAIN_SEEDS
    wrong = D2.scen("D", E, 1, "n_pinned_wrong", S5)
    pc, pd = D2.scen("C-milp", E, 1, "p_ground_K1024", S5), D2.scen("D", E, 1, "p_ground_K1024", S5)
    rows = []

    def add(label, keys, c, d, tag):
        if not keys:
            return
        diff = np.array([c[k] - d[k] for k in keys])
        m, lo, hi = tci(diff)
        rows.append([label, f"{len(keys)}", f3(np.mean([c[k] for k in keys])), f3(np.mean([d[k] for k in keys])),
                     f"{m:+.3f} " + (ci(lo, hi) if len(keys) > 1 else "")])
        mac(f"Dose{tag}", s3(m))
        mac(f"Dose{tag}N", f"{len(keys)}")

    bins = (("0--1", lambda w: w <= 1, "Low"), ("2", lambda w: w == 2, "Two"), (r"$\geq$3", lambda w: w >= 3, "Three"))
    for lab, f, t in bins:
        add(rf"L2 ($n{{=}}16$), {lab}", [k for k in pc if f(wrong[k])], pc, pd, f"Ltwo{t}")
    for tag, path in (("Frz", "s9_runs.jsonl"), ("Own", "s9_runs_ownpoint.jsonl")):
        R = jsonl(RES / path)

        def sc(arm, eps, key):
            out = {}
            for r in R:
                if r["arm"] == arm and r["eps"] == eps and r["p"] == 1:
                    out.setdefault(r["scenario"], []).append(r[key])
            return {s: float(np.mean(v)) for s, v in out.items()}

        w4 = sc("D", 0.1, "n_pinned_wrong")
        c4, d4 = sc("C-milp", 0.1, "p_ground_K1024"), sc("D", 0.1, "p_ground_K1024")
        name = "L4 frozen" if tag == "Frz" else "L4 own point"
        for lab, f, t in bins:
            add(rf"{name} ($n{{=}}22$), {lab}", [k for k in c4 if f(w4[k])], c4, d4, f"Lfour{tag}{t}")
    table("dose", "lcccc", [r"System, LP wrong pins", "Inst.", "C-milp", "D (LP)", r"C-milp $-$ D [95\% CI]"], rows,
          midrules=(3, 6))


def bandshare_section():
    """Claim 4."""
    j1 = json.loads((RES / "s8_mass_phase1.json").read_text())["1"]
    j2 = json.loads((RES / "s8_mass_phase2.json").read_text())
    j4 = json.loads((RES / "s9_mass.json").read_text())
    ov, band, share = j1["baseline: log10 overlap0"], j1["ov + m_e05"], j1["share: log10(ov / m_e05)"]
    mac("BSRtwoOverlap", f3(ov["r2"]))
    mac("BSRtwoBand", f3(band["r2"]))
    mac("BSRtwoShare", f3(share["r2"]))
    mac("BSSpreadOverlap", f3(ov["spread"]))
    mac("BSSpreadBand", f3(band["spread"]))
    mac("BSSpreadShare", f3(share["spread"]))
    mac("BSdRtwoOverlap", f3(ov["dr2"]))
    mac("BSdRtwoBand", f3(band["dr2"]))
    mac("BSCoefOverlap", f2(band["coef"][1]))
    mac("BSCoefBand", s2(band["coef"][2]))
    mac("BSExponent", f2(-band["coef"][2] / band["coef"][1]))
    ho_ov, ho_band = j1["heldout overlap only"], j1["heldout overlap + m_e05"]
    mac("BSHeldOutOverlapMin", f3(min(ho_ov["learned_bias"].values())))
    mac("BSHeldOutOverlapMax", f3(max(ho_ov["learned_bias"].values())))
    mac("BSHeldOutBandMax", f3(max(abs(v) for v in ho_band["learned_bias"].values())))
    mac("BSLogoOverlap", f3(ho_ov["logo_cv_r2"]))
    mac("BSLogoBand", f3(ho_band["logo_cv_r2"]))
    for p, tag in (("1", "One"), ("2", "Two"), ("3", "Three")):
        mac(f"BSPhaseTwoOverlapP{tag}", f3(j2[p]["baseline: log10 overlap0"]["r2"]))
        mac(f"BSPhaseTwoBandP{tag}", f3(j2[p]["ov + m_e05"]["r2"]))
        mac(f"BSPhaseTwoSpreadOverlapP{tag}", f3(j2[p]["baseline: log10 overlap0"]["spread"]))
        mac(f"BSPhaseTwoSpreadBandP{tag}", f3(j2[p]["ov + m_e05"]["spread"]))
    r4 = j4["refit"]
    mac("BSLfourOverlap", f3(r4["log10 overlap"]["r2"]))
    mac("BSLfourBand", f3(r4["log10 overlap + log10 m_e05"]["r2"]))
    mac("BSLfourShare", f3(r4["band share log10(ov / m_e05)"]["r2"]))
    mac("BSLfourSpreadOverlap", f3(r4["log10 overlap"]["spread"]))
    mac("BSLfourSpreadBand", f3(r4["log10 overlap + log10 m_e05"]["spread"]))
    tr = j4["transfer"]
    mac("BSTransferOverlap", f3(tr["baseline: log10 overlap0"]["r2"]))
    band_all = [band["r2"]] + [j2[p]["ov + m_e05"]["r2"] for p in ("1", "2", "3")] + \
               [r4["log10 overlap + log10 m_e05"]["r2"]]
    mac("BSBandRangeMin", f2(min(band_all)))
    mac("BSBandRangeMax", f2(max(band_all)))
    mac("BSTransferBand", f3(tr["ov + m_e05"]["r2"]))
    warm = [v for a, v in tr["ov + m_e05"]["resid"].items() if a != "A"]
    mac("BSTransferWarmMax", f3(max(abs(x) for x in warm)))
    mac("BSTransferColdResid", s3(tr["ov + m_e05"]["resid"]["A"]))

    def row(lab, d):
        return [lab, f3(d["r2"]), f3(d["spread"]), f"{d['dr2']:+.3f}"]

    rows = [row(r"$\log_{10}$ overlap", ov),
            row(r"band share, plain ratio $\log_{10}(\mathrm{ov}/m_5)$", share),
            row(r"overlap $+$ top-200 mass", j1["ov + m_top200"]),
            row(r"overlap $+$ entropy", j1["ov + entropy"]),
            row(r"overlap $+$ adequate mass", j1["ov + m_adeq"]),
            row(r"overlap $+$ $m_1$ (1\% band)", j1["ov + m_e01"]),
            row(r"\textbf{overlap $+$ $m_5$ (5\% band)}", band)]
    table("bandshare", "lccc", ["Predictors (logistic)", r"$R^2$", r"Arm resid.\ spread", r"$\Delta R^2$ arm int."], rows)
    # cold start: band mass vs P across instances (features cache)
    feats = {tuple(json.loads(k)): v for k, v in json.loads((RES / "s8_mass_features.json").read_text()).items()}
    import s8_analysis as A8
    from s8_eps_main import tasks

    k1 = tasks(1)
    recs = [r for r in quiet(A8.load) if (r["arm"], r["eps"], r["scenario"], r["seed"], r["p"]) in k1
            and r["arm"] == "A"]
    y = [r["p_ground_K1024"] for r in recs]
    x5 = [np.log10(feats[("A", r["eps"], r["scenario"], -1)]["m_e05"]) for r in recs]
    xo = [np.log10(r["overlap0"]) for r in recs]
    mac("BSColdSpearmanBand", s2(stats.spearmanr(x5, y).statistic))
    mac("BSColdSpearmanOverlap", s2(stats.spearmanr(xo, y).statistic))


def size_section():
    """Claim 6: wrong pins along granularity (T = 2) and along the horizon; LP vs the learned start."""
    rng = np.random.default_rng(0)
    H = jsonl(RES / "s9_horizon_census.jsonl")
    Gc = jsonl(RES / "s9_census_ownpoint.jsonl")
    HL = jsonl(RES / "s10_horizon_learned.jsonl")
    cells = {}
    for r in H:
        cells.setdefault((r["level"], r["T"]), {})[r["scenario"]] = r["n_wrong"]
    for r in Gc:
        if r["level"] in (3, 4):
            cells.setdefault((r["level"], 2), {})[r["scenario"]] = r["n_wrong"]
    nvars = {(r["level"], r["T"]): r["n"] for r in H}
    nvars.update({(r["level"], 2): r["n"] for r in Gc if r["level"] in (3, 4)})
    learned = {}
    for r in HL:
        learned.setdefault((r["level"], r["T"]), {})[r["scenario"]] = r["C_wrong"]

    def slope(keys, source):
        common = sorted(set.intersection(*[set(source[k]) for k in keys]))
        n = np.array([nvars[k] for k in keys], float)
        M = np.array([[source[k][s] for k in keys] for s in common], float)
        f = lambda A: (np.polyfit(n, A.mean(0), 1)[0], np.polyfit(n, (A >= 2).mean(0), 1)[0])  # noqa: E731
        s = f(M)
        B = np.array([f(M[rng.integers(0, len(common), len(common))]) for _ in range(3000)])
        return s, np.percentile(B, [2.5, 97.5], axis=0)

    rows = []
    for (lv, T) in sorted(cells, key=lambda k: (k[1] != 2 or k[0] > 2, k[0], k[1])):
        pass
    order = [(lv, 2) for lv in range(5)] + [(0, 1), (0, 3), (0, 4), (1, 1), (1, 3), (2, 1)]
    for k in order:
        w = np.array(list(cells[k].values()))
        lw = np.array(list(learned[k].values())) if k in learned else None
        rows.append([f"L{k[0]}", f"{k[1]}", f"{nvars[k]}", f2(w.mean()), f3(np.mean(w >= 2)), f2(w.mean() / k[1]),
                     f2(lw.mean()) if lw is not None else "--", f3(np.mean(lw >= 2)) if lw is not None else "--"])
    table("size", "cccccccc", ["Level", "$T$", "$n$", "LP wrong", r"LP $P(\geq 2)$", "LP per h", "GCN wrong",
                               r"GCN $P(\geq 2)$"], rows, midrules=(5,))
    sg, cg = slope([(lv, 2) for lv in range(5)], cells)
    mac("SlopeGranLP", s3(sg[0]))
    mac("SlopeGranLPCI", ci(cg[0][0], cg[1][0]))
    mac("SlopeGranLPfrac", s3(sg[1]))
    mac("SlopeGranLPfracCI", ci(cg[0][1], cg[1][1]))
    for lv, Tm, tag in ((0, 4, "Lzero"), (1, 3, "Lone"), (2, 2, "Ltwo")):
        keys = [(lv, T) for T in range(1, Tm + 1)]
        s, c = slope(keys, cells)
        mac(f"SlopeHorLP{tag}", s3(s[0]))
        mac(f"SlopeHorLP{tag}CI", ci(c[0][0], c[1][0]))
        mac(f"SlopeHorLPfrac{tag}", s3(s[1]))
        s, c = slope(keys, learned)
        mac(f"SlopeHorGCN{tag}", s3(s[0]))
        mac(f"SlopeHorGCN{tag}CI", ci(c[0][0], c[1][0]))
        mac(f"SlopeHorGCNfrac{tag}", s3(s[1]))
        mac(f"SlopeHorGCNfrac{tag}CI", ci(c[0][1], c[1][1]))
        perh = [np.mean(list(cells[(lv, T)].values())) / T for T in range(1, Tm + 1)]
        mac(f"PerHourLP{tag}", ", ".join(f2(x) for x in perh))
        perhl = [np.mean(list(learned[(lv, T)].values())) / T for T in range(1, Tm + 1)]
        mac(f"PerHourGCN{tag}", ", ".join(f2(x) for x in perhl))
        for T in (1, Tm):
            common = sorted(set(cells[(lv, T)]) & set(learned[(lv, T)]))
            d = np.array([learned[(lv, T)][s] - cells[(lv, T)][s] for s in common], float)
            m, lo, hi = tci(d)
            tt = {1: "One", 2: "Two", 3: "Three", 4: "Four"}[T]
            mac(f"GapGCNLP{tag}T{tt}", s2(m))
            mac(f"GapGCNLP{tag}T{tt}CI", ci(lo, hi, lambda v: f"{v:+.2f}"))
    w = {k: np.array(list(cells[k].values())) for k in cells}
    mac("FracTwoLzeroTOne", f3(np.mean(w[(0, 1)] >= 2)))
    mac("FracTwoLzeroTFour", f3(np.mean(w[(0, 4)] >= 2)))
    mac("FracTwoGranLzero", f3(np.mean(w[(0, 2)] >= 2)))
    mac("FracTwoGranLfour", f3(np.mean(w[(4, 2)] >= 2)))
    mac("MeanWrongGranLzero", f2(w[(0, 2)].mean()))
    mac("MeanWrongGranLfour", f2(w[(4, 2)].mean()))
    mac("PerHourGranLzero", f2(w[(0, 2)].mean() / 2))
    mac("PerHourGranLfour", f2(w[(4, 2)].mean() / 2))
    mac("NCensus", f"{len(w[(0, 1)])}")
    ratios = []
    for lv, Tm in ((0, 4), (1, 3), (2, 2)):
        keys = [(lv, T) for T in range(1, Tm + 1)]
        n = np.array([nvars[k] for k in keys], float)
        a = np.polyfit(n, [np.mean(list(cells[k].values())) for k in keys], 1)[0]
        b = np.polyfit(n, [np.mean(list(learned[k].values())) for k in keys], 1)[0]
        ratios.append(a / b)
    mac("SlopeRatioMin", f"{min(ratios):.1f}")
    mac("SlopeRatioMax", f"{max(ratios):.1f}")
    # per-block concentration at L4 (frozen census): Spearman of wrong rate with capacity
    t = (RES / "s9_census_ownpoint.txt").read_text()
    m = re.search(r"L4 \(n = 22.*?Spearman\(wrong rate, capacity\) = ([+-][\d.]+); Spearman\(wrong rate, merit rank\) = "
                  r"([+-][\d.]+)", t, re.S)
    mac("ConcCapLfour", s2(float(m.group(1))))
    mac("ConcRankLfour", s2(float(m.group(2))))
    trf = {}
    for r in HL:
        trf.setdefault((r["level"], r["T"]), []).append(r["C_T2_wrong"])
    mac("TransferTTwoLzeroTFour", f2(np.mean(trf[(0, 4)])))
    mac("GCNLzeroTFour", f2(np.mean(list(learned[(0, 4)].values()))))
    mac("LPLzeroTFour", f2(w[(0, 4)].mean()))


def lfour_section():
    """Claim 2 at 22 qubits: frozen and own operating point."""
    sel = json.loads((RES / "s9_selected.json").read_text())
    weights = {s["scenario"]: s["weight"] for s in sel["selected"]}
    mac("LfourNStruct", f"{sel['n_selected']}")
    mac("LfourCoverage", pct(sel["selected_population_weight"], 0))
    mac("LfourPoolSignatures", f"{sel['distinct_signatures']}")
    mac("LfourPool", f"{sel['pool']}")
    mac("LfourSelFracTwo", pct(sel["selected_frac_wrong_ge2"], 0))
    mac("LfourPoolFracTwo", pct(sel["pool_frac_wrong_ge2"], 0))
    combos = (("A", 0.1), ("C-milp", 0.1), ("C-milp", 0.15), ("D", 0.1), ("D", 0.25), ("F", 0.1), ("F", 0.25))
    rng = np.random.default_rng(4)
    data = {}
    for tag, path in (("Frz", "s9_runs.jsonl"), ("Own", "s9_runs_ownpoint.jsonl")):
        R = jsonl(RES / path)
        d = {}
        for r in R:
            d.setdefault((r["arm"], r["eps"]), {}).setdefault(r["scenario"], []).append(r["p_ground_K1024"])
        data[tag] = {k: {s: float(np.mean(v)) for s, v in vv.items()} for k, vv in d.items()}
    S = sorted(weights)
    w = np.array([weights[s] for s in S])
    w = w / w.sum()

    def rew(vals):
        v = np.array(vals)
        idx = rng.integers(0, v.size, (5000, v.size))
        bs = (v[idx] * w[idx]).sum(1) / w[idx].sum(1)
        return float((v * w).sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    rows = []
    for a, e in combos:
        cells = [rf"{ARM_TEX[a]}, $\varepsilon{{=}}{e:g}$" if a != "A" else ARM_TEX["A"]]
        for tag in ("Frz", "Own"):
            v = [data[tag][(a, e)][s] for s in S]
            m, lo, hi = tci(v)
            rm, rlo, rhi = rew(v)
            cells += [f"{m:.3f} {ciu(lo, hi)}", f"{rm:.3f}"]
        rows.append(cells)
    table("lfour", "lcccc", ["Arm", r"Frozen: $P_{\KMax}$ [95\% CI]", "Reweighted", r"Own point: $P_{\KMax}$ [95\% CI]",
                             "Reweighted"], rows)
    for tag in ("Frz", "Own"):
        for (a, ea), (b, eb), t in ((("C-milp", 0.1), ("D", 0.1), "Ten"), (("C-milp", 0.15), ("D", 0.25), "Best")):
            r = paired(data[tag][(a, ea)], data[tag][(b, eb)])
            mac(f"Lfour{tag}CmilpMinusD{t}", s3(r["mean"]))
            mac(f"Lfour{tag}CmilpMinusD{t}CI", ci(r["lo"], r["hi"]))
            mac(f"Lfour{tag}CmilpMinusD{t}W", f"{r['win']}")
            rm, rlo, rhi = rew([data[tag][(a, ea)][s] - data[tag][(b, eb)][s] for s in S])
            mac(f"Lfour{tag}CmilpMinusD{t}Rw", s3(rm))
            mac(f"Lfour{tag}CmilpMinusD{t}RwCI", ci(rlo, rhi))
        for a, e, t in (("C-milp", 0.1, "Cmilp"), ("D", 0.1, "D"), ("A", 0.1, "A")):
            mac(f"Lfour{tag}Pgnd{t}", f3(np.mean([data[tag][(a, e)][s] for s in S])))
    for (a, ea), (b, eb), t in ((("C-milp", 0.1), ("D", 0.1), "Ten"), (("C-milp", 0.15), ("D", 0.25), "Best")):
        dd = np.array([(data["Own"][(a, ea)][s] - data["Own"][(b, eb)][s]) - (data["Frz"][(a, ea)][s] -
                                                                              data["Frz"][(b, eb)][s]) for s in S])
        m, lo, hi = tci(dd)
        mac(f"LfourChange{t}", s3(m))
        mac(f"LfourChange{t}CI", ci(lo, hi))
    for tag, path in (("Frz", "s9_runs.jsonl"), ("Own", "s9_runs_ownpoint.jsonl")):
        R = jsonl(RES / path)
        g = {}
        for r in R:
            g.setdefault((r["arm"], r["eps"]), {}).setdefault(r["scenario"], []).append(100 * r["gap_repaired_K1024"])
        g = {k: {s: float(np.mean(v)) for s, v in vv.items()} for k, vv in g.items()}
        r = paired(g[("C-milp", 0.1)], g[("D", 0.1)])
        mac(f"Lfour{tag}GapCmilpMinusD", s2(r["mean"]))
        mac(f"Lfour{tag}GapCmilpMinusDCI", ci(r["lo"], r["hi"], lambda v: f"{v:+.2f}"))
    R = jsonl(RES / "s9_runs_ownpoint.jsonl")
    wc = np.mean([r["n_pinned_wrong"] for r in R if r["arm"] == "C-milp" and r["eps"] == 0.1])
    wd = np.mean([r["n_pinned_wrong"] for r in R if r["arm"] == "D" and r["eps"] == 0.1])
    mac("LfourOwnWrongCmilp", f2(wc))
    mac("LfourOwnWrongD", f2(wd))


def robust_section(A8, D2):
    """Claim 7: finite shots and Nelder-Mead (s10_runs.jsonl), paired with the exact-COBYLA Phase-2 runs."""
    R = jsonl(RES / "s10_runs.jsonl")
    seeds = (0, 1, 2)

    def sc(arm, method, shots):
        if method == "exact":
            return {s: v for s, v in D2.scen(arm, 0.1, 1, "p_ground_K1024", seeds).items()}
        out = {}
        for r in R:
            if r["arm"] == arm and r["method"] == method and r["shots"] == shots:
                out.setdefault(r["scenario"], []).append(r["p_ground_K1024"])
        return {s: float(np.mean(v)) for s, v in out.items()}

    def ev(arm, method, shots):
        if method == "exact":
            return np.mean([r["n_evals"] for r in D2.runs(arm, 0.1, 1, seeds)])
        return np.mean([r["n_evals"] for r in R if r["arm"] == arm and r["method"] == method and r["shots"] == shots])

    views = (("exact", 0, r"exact $\langle H\rangle$, COBYLA"), ("COBYLA", 8192, "8192 shots"),
             ("COBYLA", 1024, "1024 shots"), ("COBYLA", 128, "128 shots"),
             ("Nelder-Mead", 0, r"exact $\langle H\rangle$, Nelder--Mead"))
    rows = []
    for m, sh, lab in views:
        p = {a: sc(a, m, sh) for a in ("C-milp", "D", "F", "A")}
        r = paired(p["C-milp"], p["D"])
        rows.append([lab] + [f3(np.mean(list(p[a].values()))) for a in ("C-milp", "D", "F", "A")]
                    + [f"{r['mean']:+.3f} {ci(r['lo'], r['hi'])}", f"{ev('C-milp', m, sh):.0f}"])
    table("robust", "lcccccc", ["Optimiser view", "C-milp", "D (LP)", "F", "A", r"C-milp $-$ D [95\% CI]",
                                "Evals (C-milp)"], rows)
    for a, t in (("C-milp", "Cmilp"), ("D", "D"), ("F", "F"), ("A", "A")):
        r = paired(sc(a, "COBYLA", 128), sc(a, "exact", 0))
        mac(f"ShotLoss{t}", s3(r["mean"]))
        mac(f"ShotLoss{t}CI", ci(r["lo"], r["hi"]))
    adv = {s: v - sc("D", "COBYLA", 128)[s] for s, v in sc("C-milp", "COBYLA", 128).items()}
    adv0 = {s: v - sc("D", "exact", 0)[s] for s, v in sc("C-milp", "exact", 0).items()}
    r = paired(adv, adv0)
    mac("ShotAdvChange", s3(r["mean"]))
    mac("ShotAdvChangeCI", ci(r["lo"], r["hi"]))
    for b, t in (("A", "A"), ("F", "F")):
        x = paired(sc("C-milp", "COBYLA", 128), sc(b, "COBYLA", 128))
        x0 = paired(sc("C-milp", "exact", 0), sc(b, "exact", 0))
        mac(f"ShotCmilpMinus{t}Exact", s3(x0["mean"]))
        mac(f"ShotCmilpMinus{t}OneTwoEight", s3(x["mean"]))
    nm = paired(sc("C-milp", "Nelder-Mead", 0), sc("D", "Nelder-Mead", 0))
    ex = paired(sc("C-milp", "exact", 0), sc("D", "exact", 0))
    mac("NMCmilpMinusD", s3(nm["mean"]))
    mac("NMCmilpMinusDCI", ci(nm["lo"], nm["hi"]))
    mac("ExactThreeSeedCmilpMinusD", s3(ex["mean"]))
    shot_evals = [ev(a, "COBYLA", sh) for a in ("C-milp", "D", "F", "A") for sh in (128, 1024, 8192)]
    exact_evals = [ev(a, "exact", 0) for a in ("C-milp", "D", "F", "A")]
    mac("ShotEvalsMin", f"{min(shot_evals):.0f}")
    mac("ShotEvalsMax", f"{max(shot_evals):.0f}")
    mac("ExactEvalsMin", f"{min(exact_evals):.0f}")
    mac("ExactEvalsMax", f"{max(exact_evals):.0f}")
    nmax = max(abs(paired(sc(a, "Nelder-Mead", 0), sc(b, "Nelder-Mead", 0))["mean"]
                   - paired(sc(a, "exact", 0), sc(b, "exact", 0))["mean"])
               for a, b in (("C-milp", "D"), ("C-milp", "F"), ("C-milp", "A")))
    mac("NMMaxPairDiff", f3(nmax))


def ood_section():
    """Claim 8 (negative part)."""
    R = jsonl(RES / "s10_ood.jsonl")
    import s10_ood as O

    def sc(rows, key):
        out = {}
        for r in rows:
            out.setdefault(r["scenario"], []).append(r[key])
        return {s: float(np.mean(v)) for s, v in out.items()}

    names = {"in-dist": "in distribution", "high": r"peaks above range", "low": r"peaks below range",
             "out-IGSTPP-1": "outage: IGSTPP-1", "out-Yamunanagar": "outage: Yamunanagar",
             "out-Faridabad": "outage: Faridabad", "reversed": "profile reversed", "flat": "profile flat"}
    rows = []
    for s in O.SETS:
        rs = [r for r in R if r["set"] == s]
        C = [r for r in rs if r["arm"] == "C-milp"]
        D = [r for r in rs if r["arm"] == "D"]
        A = [r for r in rs if r["arm"] == "A"]
        pc, pd = sc(C, "p_ground_K1024"), sc(D, "p_ground_K1024")
        r = paired(pc, pd)
        rows.append([names[s], f"{rs[0]['n']}", f2(np.mean([x["n_pinned_wrong"] for x in C])),
                     f2(np.mean([x["n_pinned_wrong"] for x in D])), f3(np.mean(list(pc.values()))),
                     f3(np.mean(list(pd.values()))), f3(np.mean(list(sc(A, "p_ground_K1024").values()))),
                     f"{r['mean']:+.3f} {ci(r['lo'], r['hi'])}"])
        tag = {"in-dist": "In", "high": "High", "low": "Low", "out-IGSTPP-1": "OutIgstpp",
               "out-Yamunanagar": "OutYamuna", "out-Faridabad": "OutFarid", "reversed": "Rev", "flat": "Flat"}[s]
        mac(f"OOD{tag}C", f3(np.mean(list(pc.values()))))
        mac(f"OOD{tag}D", f3(np.mean(list(pd.values()))))
        mac(f"OOD{tag}Diff", s3(r["mean"]))
        mac(f"OOD{tag}DiffCI", ci(r["lo"], r["hi"]))
        mac(f"OOD{tag}DiffL", f"{r['loss']}")
        mac(f"OOD{tag}WrongC", f2(np.mean([x["n_pinned_wrong"] for x in C])))
        mac(f"OOD{tag}WrongD", f2(np.mean([x["n_pinned_wrong"] for x in D])))
    table("ood", "lccccccc", ["Test set", "$n$", "Wrong C", "Wrong D", "C-milp", "D (LP)", "A",
                              r"C-milp $-$ D [95\% CI]"], rows, midrules=(1, 3, 6))
    mac("OODNScen", f"{O.N_SCEN}")
    outs = [np.mean(list(sc([r for r in R if r["set"] == s and r["arm"] == "C-milp"], "p_ground_K1024").values()))
            for s in ("out-IGSTPP-1", "out-Yamunanagar", "out-Faridabad")]
    mac("OODOutMin", f3(min(outs)))
    mac("OODOutMax", f3(max(outs)))


def classical_section():
    """Claim 10: compiled SA ladder vs QAOA (s10_sa_compiled.jsonl); tabu from s9_classical.jsonl."""
    R = jsonl(RES / "s10_sa_compiled.jsonl")
    T9 = jsonl(RES / "s9_classical.jsonl")
    rows = []
    for level, tag in ((2, "Ltwo"), (4, "Lfour")):
        rs = [r for r in R if r["level"] == level]
        n = rs[0]["n"]
        for arm, lab in (("C-milp", "QAOA C-milp, best of 1024"), ("D", "QAOA D (LP), best of 1024"),
                         ("A", "QAOA A (cold), best of 1024")):
            v = [r["qaoa"][arm] for r in rs]
            m, lo, hi = tci([x[0] for x in v])
            rows.append([f"L{level} ($n{{=}}{n}$)", lab, f"{m:.3f} {ciu(lo, hi)}", f2(100 * np.mean([x[1] for x in v])),
                         f"{np.mean([x[3] for x in v]):.2f} s"])
            if arm == "C-milp":
                mac(f"QAOATime{tag}", f"{np.mean([x[3] for x in v]):.1f}")
                mac(f"QAOAPgnd{tag}", f3(m))
        keys = list(rs[0]["ladder"])
        for i, k in enumerate(keys):
            v = [list(r["ladder"].values())[i] for r in rs]
            m, lo, hi = tci([x["p_ground"] for x in v])
            budget = np.mean([int(list(r["ladder"])[i]) for r in rs])
            lab = (f"SA, $= $ QAOA evaluations ({budget:.0f} flips)" if i == 0
                   else f"SA, $10^{{{int(np.log10(int(k)))}}}$ flips")
            ms = 1e3 * np.mean([x["run_s"] for x in v])
            rows.append(["", lab, f"{m:.3f} {ciu(lo, hi)}", f2(100 * np.mean([x["gap"] for x in v])),
                         f"{ms:.2f} ms" if ms < 10 else f"{ms:.0f} ms"])
            if i > 0:
                e = int(np.log10(int(k)))
                word = {4: "Four", 5: "Five", 6: "Six"}[e]
                mac(f"SAPgnd{tag}E{word}", f3(m))
                mac(f"SAms{tag}E{word}", f"{ms:.1f}")
        t9 = [r for r in T9 if r["level"] == level]
        m, lo, hi = tci([r["tabu_wall"]["p_ground"] for r in t9])
        rows.append(["", r"tabu (restarts), $=$ QAOA wall time", f"{m:.3f} {ciu(lo, hi)}",
                     f2(100 * np.mean([r["tabu_wall"]["gap"] for r in t9])), "--"])
        mac(f"TabuPgnd{tag}", f3(m))
        mac(f"SAPgndEval{tag}", f3(np.mean([list(r["ladder"].values())[0]["p_ground"] for r in rs])))
        mac(f"SAFlipsPerSec{tag}", f"{np.mean([r['flips_per_s'] for r in rs]) / 1e6:.0f}")
        mac(f"QAOAWallBuysFlips{tag}", sci(np.mean([r["wall_buys_flips"] for r in rs]), 1))
        mac(f"NClassical{tag}", f"{len(rs)}")
    table("classical", "llccc", ["System", "Method", r"$P(\text{ground})$ [95\% CI]", r"Gap\,\%", "Time per run"],
          rows, midrules=(8,))


def generalisation_section():
    """MaxCut generalisation test, ranking against depth-0 rules, eps decision regret, band density
    (scripts/s11_*.py, s12_band_density.py). Introduction, Discussion and Related Work quote these."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import s11_maxcut as X
    from s11_maxcut_analysis import by_instance

    mc = json.loads((RES / "s11_maxcut.json").read_text())
    names = {12: "Twelve", 16: "Sixteen", 20: "Twenty"}
    recs = X.load()
    mac("MCRuns", f"{len(recs):,}".replace(",", "{,}"))
    mac("MCGraphs", str(X.N_INST))
    for n, w in names.items():
        mac(f"MCSize{w}", str(n))
        fits = mc["sizes"][f"n = {n}"]["fits"]
        mac(f"MCOv{w}", f2(fits["log10 overlap"]["r2"]))
        mac(f"MCBand{w}", f2(fits["ov + m_e05"]["r2"]))
        mac(f"MCTransfer{w}", s2(mc["transfer"][f"n = {n}"]["ov + m_e05"]["r2"]))
        cold = [r for r in recs if r["n"] == n and r["arm"] == "A"]
        mac(f"MCCold{w}", f3(by_instance(cold).mean()))
    mac("MCGWTwenty", f3(by_instance([r for r in recs if r["n"] == 20 and r["arm"] == "GW" and r["eps"] == 0.1]).mean()))
    for d, w in ((2, "Two"), (3, "Three"), (4, "Four")):
        v = [mc["dose"][f"{n}/0.05/{d}"]["mean"] for n in names]
        mac(f"MCDose{w}Min", f2(min(v)))
        mac(f"MCDose{w}Max", f2(max(v)))
    mac("MCDoseEps", "0.05")
    gw = [r for r in recs if r["arm"] == "GW" and r["eps"] == 0.1]
    mac("MCGWPinsLow", f2(by_instance([r for r in gw if r["n_pinned_wrong"] <= 2]).min()))
    mac("MCGWPinsFour", f2(by_instance([r for r in gw if r["n_pinned_wrong"] == 4]).mean()))
    mac("MCGWPinsFive", f2(by_instance([r for r in gw if r["n_pinned_wrong"] == 5]).mean()))

    rk = json.loads((RES / "s11_ranking.json").read_text())

    def top(key, rule):
        o = rk[key]
        v = o["rules"][rule]["top1"]
        return (f"{v:.0f}" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"), str(o["n"])

    for tag, key in (("UCSixteen", "UC-16 (a) sources at eps = 0.1"), ("UCTwentyTwo", "UC-22 (a) sources at eps = 0.1"),
                     ("UCJoint", "UC-22 (c) all (source, eps) configurations jointly"),
                     ("MCTwenty", "MaxCut-20 all non-oracle sources (a) sources at eps = 0.1"),
                     ("MCTwelve", "MaxCut-12 all non-oracle sources (a) sources at eps = 0.1")):
        mac(f"RK{tag}N", top(key, "overlap")[1])
        for rule, w in (("overlap", "Ov"), ("bandshare", "Band"), ("ratio0", "Ratio")):
            mac(f"RK{tag}{w}", top(key, rule)[0])

    er = json.loads((RES / "s11_eps_regret.json").read_text())
    full = er["UC-16 full sweep"]
    cells = {("Sixteen", a): full[a] for a in ("D", "F", "C-milp", "C-oracle")}
    cells.update({("TwentyTwo", a): er[f"UC-22 {a}"][a] for a in ("D", "F", "C-milp")})
    for (size, a), c in cells.items():
        if a in ("D", "F"):
            mac(f"ER{a}Ov{size}", f3(c["overlap"]["mean"]))
            mac(f"ER{a}OvWorst{size}", f3(c["overlap"]["worst"]))
    heur = [c for (s, a), c in cells.items() if a in ("D", "F")]
    mac("EROvWorst", f2(max(c["overlap"]["worst"] for c in heur)))
    mac("ERBandMeanMax", f3(max(c["bandshare"]["mean"] for c in heur)))
    mac("ERBandWorstMax", f3(max(c["bandshare"]["worst"] for c in heur)))
    learned = [c for (s, a), c in cells.items() if a.startswith("C-")]
    mac("ERLearnedMax", f3(max(c[k]["mean"] for c in learned for k in ("overlap", "bandshare", "share"))))

    rows = []
    for n in names:
        fits = mc["sizes"][f"n = {n}"]["fits"]
        d = [mc["dose"][f"{n}/0.05/{k}"]["mean"] for k in range(5)]
        key = f"MaxCut-{n} all non-oracle sources (a) sources at eps = 0.1"
        rows.append([str(n), f3(by_instance([r for r in recs if r["n"] == n and r["arm"] == "A"]).mean()),
                     f2(min(d[:3])), f2(d[3]), f2(d[4]), f2(fits["log10 overlap"]["r2"]),
                     f2(fits["ov + m_e05"]["r2"]), f"{mc['transfer'][f'n = {n}']['ov + m_e05']['r2']:+.2f}",
                     "/".join(top(key, "overlap")), "/".join(top(key, "bandshare"))])
    table("maxcut", "c|c|ccc|ccc|cc",
          ["$n$", "Cold", r"$\leq$2", "3", "4", "Ov.", "+Band", "UC curve", "Ov.", "Band"], rows)

    bd = json.loads((RES / "s12_band_density.json").read_text())["tests"]
    mac("BDPooled", s2(bd["pooled/density"]["rho"]))
    mac("BDWithin", s2(bd["pooled/density"]["rho_within"]))
    mac("BDInstances", str(bd["pooled/density"]["n"]))


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    A8, D1, D2, groups = load_s8()
    methods_numbers(A8, D2, groups)
    validation_numbers()
    gcn_numbers()
    best, dec = arms_section(A8, D1, D2, groups)
    cv_and_regret(A8, D1, groups, best, dec)
    dose_section(A8, D2, groups)
    bandshare_section()
    size_section()
    lfour_section()
    robust_section(A8, D2)
    ood_section()
    classical_section()
    generalisation_section()
    import subprocess

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    lines = ["% AUTO-GENERATED by scripts/make_tables.py -- do not edit by hand.",
             f"% {len(MACROS)} macros; source commit {sha} (plus any uncommitted result files).", ""]
    lines += [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(MACROS.items())]
    (PAPER / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote paper/numbers.tex ({len(MACROS)} macros) and {len(list(TABLES.glob('*.tex')))} tables")


if __name__ == "__main__":
    main()
