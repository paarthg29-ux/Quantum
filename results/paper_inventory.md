# Paper inventory: what we can support, where it comes from, and what is missing

Status at 2026-11-06, branch `qguc-pipeline`. **Committed** = the run data and the analysis output are in git (commit given). **Uncommitted** = the file exists but is not in git.

Statistical convention for every comparison below:
- Values are averaged over seeds within a scenario, then paired across scenarios.
- Where scenarios share a structure, the unit is the **structural group**: 17 groups by (ground state, LP pin pattern), 13 by ground state alone. At L4 every selected scenario is its own structure (n = 20).
- For choosing ε, the primary numbers are **group-fold cross-validation repeated over 1,000 random group splits**. The single 2-fold scenario split leaked, because near-duplicate scenarios landed on both sides.

## Tables

| # | Table | Claim it supports | Source file (script) | Run committed? |
|---|---|---|---|---|
| T1 | Problem and encoding: fleet, ladder L0–L4 (5/6/8/9/11 blocks × 2 h = 10–22 qubits, 4,442 MW at every level), frozen operating point λ = 100, B = 250 MW | The QUBO is a faithful, calibrated encoding of Haryana unit commitment | `s4_ladder.txt`, `s4b_recalibrate_L2.txt`, `s2_model_comparison.txt` (`s4_ladder.py`, `s2_calibrate.py`) | yes (80fc040, 50f7960) |
| T2 | Encoding gap per level, at the frozen point and at each level's own calibrated point | The encoding costs 2.5–6.1% at the frozen point and 1.8–4.9% at each level's own point | `s7_gap_figure.txt`, `s7_gap_ladder.csv` | yes (f006eeb) |
| T3 | Simulator timing; PennyLane comparison | The exact NumPy statevector simulator is 11–16× faster than `lightning.qubit` at 22 qubits, which is what makes the benchmark feasible | `s3b_fastsim_timing.txt`, `s3_timing_probe.txt` | yes (917a70f, 3eff343) |
| T4 | GCN warm-start model: held-out MAE and BCE against a constant predictor (thresholded accuracy misleads for symmetrised labels), load-conditioning check, inference time (L2: C-milp and C-oracle; L4: C-milp) | The learned warm start is load-conditioned, beats a constant predictor at both sizes (L4 MAE 0.122 vs 0.178), and costs about 0.1 ms per instance | `s7a_train_gnn.txt`, `s9_train_gnn_L4.txt`, `s9_gnn_vs_constant.txt` | yes (428dec6, a6789e7) |
| T5 | **Main table, ε = 0.1**: 7 arms × 30 held-out scenarios × 5 seeds, p = 1/2/3. P(best of 1024 is ground), single-shot P(ground), best-of-1024 cost gap, final g, evaluations to g, wrong pins | Learned arms sample the ground state far better than LP (D), merit order (F), cold (A) or random (E) starts. Depth barely helps warm arms. D has the best final g at ε = 0.1 | `s8_phase2_analysis.txt` (`s8_analysis.py --phase 2`) | yes (7feab61) |
| T6 | Per-scenario detail of T5 (p = 1, 3) | D's deficit is confined to 8 of 30 scenarios (4 structural groups) where the LP pins 2–3 bits wrong | `s8_phase2_analysis.txt` | yes (7feab61) |
| T7 | **ε sweep, p = 1**: 5 warm arms × 7 ε, each arm at its own best ε, the decisive paired comparisons and the ε = 0.5 sanity checks | At its best ε, D does not match the learned arms on ground-state sampling. Primary numbers: D − C-oracle −0.094 and D − C-milp −0.070 over 1,000 random group splits, with D worse in 100% and 96% of splits | `s8_phase1_analysis.txt`, `s8_eps_stability.txt` (`s8_analysis.py --phase 1`, `s8_eps_stability.py`) | yes (7feab61, a6789e7) |
| T8 | **Statistics at group level** for every headline comparison (n = 30 scenarios / 17 / 13 groups) | Says which claims survive with honest n. "D at its best ε loses to the learned arms" survives under group-fold CV. Against C-milp at D's in-sample best ε it is borderline (t-CI touches 0) | `s8_phase1_analysis.txt`, `s8_phase2_analysis.txt` | yes (7feab61) |
| T9 | **ε regret**: P lost at a fixed ε relative to the arm's own best, per group, with CIs | **The LP needs its regularisation tuned; the learned starts do not.** Regret for D is 0.122 at ε = 0.05 and 0.917 at ε = 0; for C-milp and C-oracle it is 0.010 / 0.000 at ε = 0.05, and at most 0.018 / 0.005 anywhere in ε ∈ [0, 0.25] | `s8_eps_stability.txt` (section 3) | yes (a6789e7) |
| T10 | **Mechanism, band share**: logistic fits of P(best of 1024 is ground) on overlap, top-k mass, band mass, adequate mass and entropy, with per-arm intercept tests at p = 1, 2, 3 | Overlap alone gives R² 0.55 with a per-arm residual. Adding band mass gives R² 0.95–0.97 and removes the per-arm residual. Spreading mass over near-ground states hurts | `s8_mass_phase1.txt`, `s8_mass_phase2.txt` (`s8_mass.py`) | yes (a6789e7) |
| T11 | Mechanism checks: held-out arms (fit on the non-learned runs, predict the learned arms: bias 0.09–0.11 with overlap alone, ≤ 0.014 with band mass) and leave-one-group-out CV (R² 0.50–0.64 → 0.95–0.97), at p = 1, 2, 3 | The band-share result is predictive, not curve-fitted | `s8_mass_phase1.txt`, `s8_mass_phase2.txt` ("HELD-OUT CHECKS") | yes (a6789e7) |
| T12 | **Wrong-pin census along granularity** L0–L4 at T = 2 (300 paired demand draws per level): distribution 0/1/2/3+, mean, trend per qubit, McNemar L2→L4, per-block concentration; at the frozen point **and** at each level's own point | P(2 or more wrong LP pins) grows with granularity: 0.24 → 0.52 at own points (+0.020 per variable [0.014, 0.025]; L2→L4 McNemar p = 0.001); frozen point 0.28 → 0.54. Wrong pins fall on small, expensive marginal blocks (Spearman with capacity −0.89 at L4) | `s9_census.txt`, `s9_census_ownpoint.txt` (`s9_census.py [--point own]`) | yes (a6789e7) |
| T13 | **Wrong-pin census along the horizon** at fixed granularity: L0 × T = 1–4, L1 × T = 1–3, L2 × T = 1–2, every cell at its own calibrated point | **Wrong pins grow with problem size generally, not only with granularity.** At fixed blocks they are extensive: wrong pins per hour are flat (L0 0.52/0.55/0.54/0.52 for T = 1–4), so the total grows linearly. P(2 or more wrong) for L0 rises 0.07 → 0.57. Per added variable the horizon raises mean wrong pins by +0.10 [0.09, 0.11], against +0.05 [0.03, 0.07] for granularity. Granularity additionally raises the per-hour rate (0.55 at L0 → 0.96 at L4), non-monotonically | `s9_horizon.txt`, `s9_horizon_points.csv`, `s9_horizon_census.jsonl` (`s9_horizon.py`) | **uncommitted** |
| T14 | **L4 arms (22 qubits), frozen point**: A, C-milp, D, F at ε = 0.1 and at each arm's Phase-1 best ε, on 20 structurally diverse scenarios × 3 seeds, p = 1; per structure (n = 20) and reweighted to the demand distribution | The learned advantage holds at 22 qubits: C-milp − D +0.208 [+0.037, +0.379] at ε = 0.1, +0.177 [+0.039, +0.316] at best ε. It is concentrated where the LP pins ≥ 2 bits wrong. Absolute levels fall: C-milp 0.70, D 0.49, A 0.15 | `s9_analysis.txt`, `s9_runs.jsonl`, `s9_selected.json` | yes (a6789e7) |
| T15 | **L4 arms at L4's own operating point** (λ = 100, B = 150 MW), same 20 structures, paired with T14 | The L4 result does not depend on the L2-calibrated encoding. C-milp − D +0.231 [+0.079, +0.384] (ahead in 14 of 20; reweighted +0.175 [+0.020, +0.404]); at best ε +0.205 [+0.071, +0.339]. Change from the frozen point +0.023 [−0.220, +0.267]. C-milp's wrong pins fall to 0.05 per structure (D 2.75). Best-of-1024 cost gap falls to 2.3–2.5% for warm arms (6.6–7.1% at the frozen point) | `s9_analysis_ownpoint.txt`, `s9_runs_ownpoint.jsonl` (`s9_l4_arms.py --point own`, `s9_analysis.py --point own`) | **uncommitted** |
| T16 | **Band share at L4, and transfer from L2** | Overlap + band mass explains L4 sampling (R² 0.98, per-arm residual range 0.025). The L2 curve applied unchanged to L4 gives R² 0.91 (overlap alone: 0.60) | `s9_mass.txt` (`s9_mass.py`) | yes (a6789e7) |
| T17 | **Classical samplers on the identical QUBO**: SA and tabu at L2 (30 scenarios) and L4 (20 structures), with budgets matched to QAOA C-milp by (a) objective evaluations and (b) wall time | **Wall time matched (our simulator):** SA 0.972 vs QAOA C-milp 0.990 at 16 qubits; SA 0.929 and tabu 0.948 vs 0.697 at 22 qubits. Classical samplers match QAOA at 16 qubits and beat it at 22. **Evaluations matched:** QAOA C-milp wins (0.990 vs SA 0.278 at L2; 0.697 vs 0.068 at L4), but this counting is generous to QAOA (one expectation estimate = one evaluation). The MILP solves the operational problem in about 25 ms (T1) | `s9_classical.txt`, `s9_classical.jsonl` (`s9_classical.py`) | **uncommitted** |

## Figures

| # | Figure | Claim it supports | File (script) | Committed? |
|---|---|---|---|---|
| F1 | Encoding-gap ladder, two panels | Encoding cost per level, frozen point vs own point | `fig_encoding_gap_ladder.png/.pdf` (`s7_gap_figure.py`) | yes (f006eeb) |
| F2 | **Band share, two panels**: (a) P vs log overlap, arms separate (R² 0.55, per-arm residual spread 0.16); (b) P vs band-share index, arms collapse (R² 0.95, spread 0.01) | The mechanism | `fig_bandshare.png/.pdf` (`s8_fig_bandshare.py`) | yes (a6789e7) |
| F3 | Overlap vs sampling per arm, small multiples | Superseded by F2(a); supplement at most | `fig_overlap_vs_sampling.png/.pdf` | yes (7feab61) |

**Figures we can support but have not drawn:**
- **ε regret curves:** P(best of 1024 is ground) against ε per arm, with group CIs. D's steep curve against the learned arms' flat one, i.e. T7 and T9 as one picture. Highest-value missing figure.
- **Size vs granularity:** mean wrong pins and P(2 or more) against n, with the horizon lines (L0/L1/L2 × T) and the granularity line (T = 2, L0–L4) on one axis, from T12 and T13. Second-highest value: it carries the forward-looking claim.
- **Main-table dot plot:** arms × p, with group CIs, from T5.
- **L4 per-structure matrix:** C-milp vs D per structure, coloured by D's wrong pins, at both operating points, from T14 and T15.
- **Classical vs QAOA:** P(ground) against budget for SA, tabu and QAOA, at L2 and L4, from T17.

## Claims ledger: status after proper statistics

| Claim | Status |
|---|---|
| Learned warm starts (C-milp, C-oracle) sample the ground state far better than A, E and F | **Holds** at every level of aggregation (C − F +0.31 to +0.33 at group level) |
| At its best ε, D still loses to the learned arms on P(best of 1024 is ground) | **Holds** under repeated group-fold CV. **Borderline** at group level against C-milp when D is given its in-sample best ε |
| D's deficit comes from scenarios where the LP pins 2 or more bits wrong | **Holds** at L2 (8 of 30 scenarios, 4 groups) and at L4 at both operating points. Where the LP pins ≤ 1 bit wrong, D matches or slightly beats C-milp |
| D matches the learned arms on best-of-1024 cost | **Holds** (no significant difference) at L2 and L4: D's misses land on near-optimal adequate schedules |
| C-oracle beats C-milp | **Not supported**: tie at group level |
| B (warm state, X mixer) matches C-milp | **On sampling, yes** (tie). **On final g and single-shot P(ground), no**: B is worse in 17 of 17 groups |
| Overlap alone predicts sampling | **No**: R² 0.55 with significant per-arm residuals |
| Band share explains sampling across arms, ε, depth and size | **Holds**: R² 0.95–0.98 at L2 and L4; held-out arms within 0.015; the L2 curve transfers to L4 (R² 0.91) |
| The LP needs its regularisation tuned; the learned starts do not | **Holds** (T9): regret 0.122 at ε = 0.05 and 0.917 at ε = 0 for D, against at most 0.018 (C-milp) and 0.005 (C-oracle) anywhere in ε ∈ [0, 0.25] |
| ~~The LP's best ε is unstable~~ | **Dropped.** D picks 0.25 in 91% of bootstraps. Replaced by the regret claim above |
| Wrong LP pins grow with problem size | **Holds generally** (T13): at fixed granularity they are extensive in the number of hours, and per added variable the horizon raises them about twice as fast as granularity does. Granularity adds a further per-hour increase. Shown for one fleet |
| The learned advantage survives at 22 qubits | **Holds** at p = 1 on 20 diverse structures, per structure and reweighted, at the frozen point and at L4's own point. The absolute success rate falls for every arm |
| Final-g ranking: D best at ε = 0.1, worst among warm arms at its own best ε | **Holds** (30 of 30 scenarios both ways at L2; 20 of 20 at L4). It is the ε trade-off; state it explicitly so T5 and T7 don't look contradictory |
| QAOA is competitive with classical solvers on these instances | **No.** The MILP solves the operational problem exactly in milliseconds. At matched wall time SA matches QAOA at 16 qubits and beats it at 22. Position: the contribution is what determines warm-start quality within QAOA, not quantum advantage |

## Superseded or do not use

- `s6_*`, `s6b_*` (pilot) and `s7_arms_full.*` (10 scenarios, scenario-level statistics only) are superseded by T5–T8. Cite s6b only for the history of the LP-target fix.
- The p = 5 and p = 3 sweep records inside `s8_runs.jsonl` fall outside both phases; the analysis ignores them.
- The scenario-fold CV numbers in early Phase 1 output are leaky. Use group-fold.
- The first tabu run (no restarts) cycled and did not improve with budget. `s9_classical.jsonl` holds the restarted version.
- Untracked `fig1_lambda_sweep.png`, `fig2_gnn.png`, `gnn_params.npz`, `gnn_report.json`, `lambda_*.json` are from the old `src/quc` prototype. They are not this pipeline; do not use them.

## Gaps

### Closed in this round

| # | Gap | How it was closed |
|---|---|---|
| G1 | L4 QAOA runs used the L2-calibrated operating point | T15: re-run at L4's own point; the advantage holds (+0.231, change +0.023 [−0.220, +0.267]) |
| G2 | "Size" was only granularity of one fleet | T13: horizon census at fixed granularity. Wrong pins grow with size generally (extensive in hours). Still one fleet (see G16) |
| G4 | No classical sampler on the same QUBO | T17: SA and tabu, evaluation- and wall-matched, at L2 and L4 |
| G8 | L4 analysis script | `s9_analysis.py` |
| G9 | Band-share held-out checks not in a results file | T11 |
| G10 | L4 GCN accuracy below the majority baseline | `s9_gnn_vs_constant.txt`; T4 must report MAE/BCE against the constant predictor |

### Still open (most important first)

| # | Gap | Why a reviewer asks | Cost to close |
|---|---|---|---|
| G3 | **Noise-free, exact sampling only.** No shot noise, no hardware noise | Does the advantage survive realistic noise? | A depolarising-channel or trajectory simulation at L2: days. Alternatively argue that best-of-K is exact and noise is out of scope |
| G16 | **The learned model's wrong pins have not been measured along the horizon.** The forward-looking claim needs the LP's wrong pins to grow with size (shown) **and** the learned start's to stay low. That is shown only at T = 2 (L4: C-milp 0.40 / 0.05 vs D 2.90 / 2.75) | If C-milp's wrong pins also grow extensively with T, the gap need not widen | Train C-milp at L0/L1 for T = 3, 4 and count its wrong pins on the horizon census: about 30 min, no QAOA |
| G5 | **Generalisation of the learned model.** Training and test come from the same demand generator | What about out-of-distribution profiles, unit outages or other fleets? | Shifted-demand and outage test sets at L2, QAOA p = 1: about 1 h |
| G14 | **L4 covers 34% of the demand distribution** (the 20 selected signatures). The selection over-represents multiply-wrong structures (75% vs 54% in the pool) | The population-level L4 effect is extrapolated from the census fraction × per-structure gaps | A random (non-diverse) L4 sample of 20: about 80 min |
| G17 | **Classical timing is for a pure-numpy SA** (25,000–40,000 flips/s). A compiled SA would be much faster; not measured | "Milliseconds" for SA is not yet a measured claim (the MILP's milliseconds are) | Time a compiled SA (numba): about 20 min |
| G6 | **Band share needs the full spectrum,** so it explains the result but cannot select a warm start at scale | Is there a computable proxy? | Test whether LP wrong-pin count or GNN confidence predicts band share: about 30 min of analysis |
| G11 | **One optimiser (COBYLA) and one initial-parameter distribution** | Is the ranking an artefact of the optimiser? | Repeat T5 at p = 1 with SPSA or Nelder–Mead on a subset: about 30 min |
| G15 | **L4 at p = 1 only.** Absolute success is low for every arm | Does depth recover it at L4? | p = 2 at L4: about 5.5 h (deferred by decision) |
| G12 | **The simulator's correctness evidence lives only in tests** (PennyLane oracle at 1e-10, Monte Carlo check of best-of-K) | A reviewer wants a validation table | Export the oracle comparison to a results file: about 10 min |
| G13 | **Wall-clock accounting** omits GCN training cost (labels plus 8 s training) | Fair comparison against LP | One line in T5 from the T4 logs |
| G7 | **C-oracle is not scalable** (labels need brute force); dropped at L4 | Fine as an upper bound; say so | writing only |
| G18 | **Classical-sampler tuning is light and disclosed.** SA's end temperature was chosen on 4 L2 scenarios; tabu's tenure and restart on 3 | A reviewer may ask whether the baselines are tuned fairly | Both choices favour the classical samplers; disclose in methods |
