# Paper inventory: what we can support, where it comes from, and what is missing

Status at 2026-11-06, branch `qguc-pipeline`. **Committed** = the run data and the analysis output are in git (commit given). **Uncommitted** = the file exists but is not in git. **Pending** = the run is in progress.

Statistical convention for every comparison below:
- Values are averaged over seeds within a scenario, then paired across scenarios.
- Where scenarios share a structure, the unit is the **structural group**: 17 groups by (ground state, LP pin pattern), 13 by ground state alone.
- For choosing ε, the primary numbers are **group-fold cross-validation repeated over 1,000 random group splits**. The single 2-fold scenario split leaked, because near-duplicate scenarios landed on both sides.

## Tables

| # | Table | Claim it supports | Source file (script) | Run committed? |
|---|---|---|---|---|
| T1 | Problem and encoding: fleet, ladder L0–L4 (5/6/8/9/11 blocks × 2 h = 10–22 qubits, 4,442 MW at every level), frozen operating point λ = 100, B = 250 MW | The QUBO is a faithful, calibrated encoding of Haryana unit commitment | `s4_ladder.txt`, `s4b_recalibrate_L2.txt`, `s2_model_comparison.txt` (`s4_ladder.py`, `s2_calibrate.py`) | yes (80fc040, 50f7960) |
| T2 | Encoding gap per level, at the frozen point and at each level's own calibrated point | The encoding costs 2.5–6.1% at the frozen point and 1.8–4.9% at each level's own point | `s7_gap_figure.txt`, `s7_gap_ladder.csv` | yes (f006eeb) |
| T3 | Simulator timing; PennyLane comparison | The exact NumPy statevector simulator is 11–16× faster than `lightning.qubit` at 22 qubits, which is what makes the benchmark feasible | `s3b_fastsim_timing.txt`, `s3_timing_probe.txt` | yes (917a70f, 3eff343) |
| T4 | GCN warm-start model: held-out MAE and BCE against a constant predictor (thresholded accuracy is misleading for symmetrised labels), load-conditioning check, inference time (L2: C-milp and C-oracle; L4: C-milp) | The learned warm start is load-conditioned, beats a constant predictor at both sizes, and costs about 0.1 ms per instance | `s7a_train_gnn.txt` (L2); `s9_train_gnn_L4.txt` (L4); `s9_gnn_vs_constant.txt` | L2 yes (428dec6); **L4 uncommitted** |
| T5 | **Main table, ε = 0.1**: 7 arms × 30 held-out scenarios × 5 seeds, p = 1/2/3. P(best of 1024 is ground), single-shot P(ground), best-of-1024 cost gap, final g, evaluations to g, wrong pins | Learned arms sample the ground state far better than LP (D), merit order (F), cold (A) or random (E) starts. Depth barely helps warm arms. D has the best final g at ε = 0.1 | `s8_phase2_analysis.txt` (`s8_analysis.py --phase 2`) | yes (7feab61) |
| T6 | Per-scenario detail of T5 (p = 1, 3) | D's deficit is confined to 8 of 30 scenarios (4 structural groups) where the LP pins 2–3 bits wrong | `s8_phase2_analysis.txt` | yes (7feab61) |
| T7 | **ε sweep, p = 1**: 5 warm arms × 7 ε, with each arm at its own best ε, the decisive paired comparisons and the ε = 0.5 sanity checks | At its best ε, D does not match the learned arms on ground-state sampling. Primary numbers: D − C-oracle −0.094 and D − C-milp −0.070 over 1,000 group splits, with D worse in 100% and 96% of splits | `s8_phase1_analysis.txt`, `s8_eps_stability.txt` (`s8_analysis.py --phase 1`, `s8_eps_stability.py`) | analysis yes (7feab61); **stability file uncommitted** |
| T8 | **Statistics at group level** for every headline comparison (n = 30 scenarios / 17 / 13 groups) | Says which claims survive with honest n. "D at its best ε loses to the learned arms" survives under group-fold CV. Against C-milp at D's in-sample best ε it is borderline (t-CI touches 0) | `s8_phase1_analysis.txt`, `s8_phase2_analysis.txt` | yes (7feab61) |
| T9 | **ε robustness**: bootstrap selection frequencies, random-split held-out P, regret of a wrong ε, per-group oracle | The learned arms are insensitive to ε: C-oracle regret ≤ 0.005 for ε ≤ 0.25. D is sensitive: regret 0.122 at ε = 0.05 and 0.917 at ε = 0. *Not* supported: "D's chosen ε is unstable"; it picks 0.25 in 91% of bootstraps | `s8_eps_stability.txt` | **uncommitted** |
| T10 | **Mechanism, band share**: logistic fits of P(best of 1024 is ground) on overlap, top-k mass, band mass, adequate mass and entropy, with per-arm intercept tests at p = 1, 2, 3 | Overlap alone gives R² 0.55 with a per-arm residual. Adding band mass gives R² 0.95–0.97 and removes the per-arm residual. Spreading mass over near-ground states hurts | `s8_mass_phase1.txt`, `s8_mass_phase2.txt` (`s8_mass.py`) | yes (7feab61) |
| T11 | Mechanism checks: held-out arms (fit on the non-learned runs, predict the learned arms: bias 0.09–0.11 with overlap alone, ≤ 0.014 with band mass) and leave-one-group-out CV (R² 0.50–0.64 → 0.95–0.97), at p = 1 (Phase 1) and p = 1, 2, 3 (Phase 2) | The band-share result is predictive, not curve-fitted | `s8_mass_phase1.txt`, `s8_mass_phase2.txt` ("HELD-OUT CHECKS"; `s8_mass.py`) | **uncommitted** (re-run with the checks added) |
| T12 | **Wrong-pin census L0–L4** (300 paired demand draws per level): distribution 0/1/2/3+, mean wrong pins, trend per qubit, McNemar L2→L4, per-block concentration; at the frozen point **and** at each level's own calibrated point | The fraction of draws with 2 or more wrong LP pins grows with size: frozen point 0.28 → 0.54 (+0.022 per qubit [0.016, 0.028]); own points 0.24 → 0.52 (+0.020 [0.014, 0.025]; L2→L4 McNemar p = 0.001). The mean grows 1.19 → 2.06 (own points 1.09 → 1.92; not monotone, L3 dips). Wrong pins fall on small, expensive marginal blocks | `s9_census.txt`, `s9_census_ownpoint.txt`, + `.jsonl` (`s9_census.py [--point own]`) | **uncommitted** |
| T13 | **L4 arms (22 qubits)**: A, C-milp, D, F at ε = 0.1 and at each arm's Phase-1 best ε, on 20 structurally diverse scenarios × 3 seeds, p = 1; per structure (n = 20) and reweighted to the demand distribution | The learned advantage holds at 22 qubits: C-milp − D +0.208 [+0.037, +0.379] at ε = 0.1 and +0.177 [+0.039, +0.316] at Phase-1 best ε; reweighted +0.17 / +0.20. It is concentrated where the LP pins ≥ 2 bits wrong (+0.26 to +0.31); where it pins ≤ 1, D is slightly better (n = 5). Absolute levels fall: C-milp 0.70, D 0.49, A 0.15 per structure | `s9_analysis.txt`, `s9_runs.jsonl`, `s9_selected.json` (`s9_select.py`, `s9_l4_arms.py`, `s9_analysis.py`) | **uncommitted** |
| T14 | **Band share at L4, and transfer from L2** | Overlap + band mass explains L4 sampling (R² 0.98, per-arm residual range 0.025). The L2 curve applied unchanged to L4 gives R² 0.91 (overlap alone: 0.60) | `s9_mass.txt` (`s9_mass.py`) | **uncommitted** |

## Figures

| # | Figure | Claim it supports | File (script) | Committed? |
|---|---|---|---|---|
| F1 | Encoding-gap ladder, two panels | Encoding cost per level, frozen point vs own point | `fig_encoding_gap_ladder.png/.pdf` (`s7_gap_figure.py`) | yes (f006eeb) |
| F2 | **Band share, two panels**: (a) P vs log overlap, arms separate (R² 0.55, per-arm residual spread 0.16); (b) P vs band-share index, arms collapse (R² 0.95, spread 0.01) | The mechanism | `fig_bandshare.png/.pdf` (`s8_fig_bandshare.py`) | **uncommitted** |
| F3 | Overlap vs sampling per arm, small multiples | Superseded by F2(a); keep for the supplement at most | `fig_overlap_vs_sampling.png/.pdf` | yes (7feab61) |

**Figures we can support but have not drawn:**
- **ε-sweep curves:** P(best of 1024 is ground) against ε per arm, with group CIs. This shows D's steep curve against the learned arms' flat one, i.e. T7 and T9 as one picture. Highest-value missing figure.
- **Census scaling:** P(wrong pins ≥ 2) and the mean against n, with the 0/1/2/3+ distribution stacked, from T12.
- **Main-table dot plot:** arms × p, with group CIs, from T5.
- **Per-scenario matrix:** D vs C-oracle per scenario, coloured by wrong pins, from T6. It shows the deficit is concentrated.

## Claims ledger: status after proper statistics

| Claim | Status |
|---|---|
| Learned warm starts (C-milp, C-oracle) sample the ground state far better than A, E and F | **Holds** at every level of aggregation (C − F +0.31 to +0.33 at group level) |
| At its best ε, D still loses to the learned arms on P(best of 1024 is ground) | **Holds** under repeated group-fold CV. **Borderline** at group level against C-milp when D is given its in-sample best ε |
| D's deficit comes from scenarios where the LP pins 2 or more bits wrong | **Holds**: 8 of 30 scenarios, 4 groups. D matches the learned arms where it pins 1 bit wrong |
| D matches the learned arms on best-of-1024 cost | **Holds** (no significant difference): D's misses land on near-optimal adequate schedules |
| C-oracle beats C-milp | **Not supported**: tie at group level |
| B (warm state, X mixer) matches C-milp | **On sampling, yes** (tie). **On final g and single-shot P(ground), no**: B is worse in 17 of 17 groups |
| Overlap alone predicts sampling | **No**: R² 0.55 with significant per-arm residuals |
| Band share explains sampling across arms, ε and depth | **Holds**: R² 0.95–0.97, held-out arms within 0.015 |
| The LP's best ε is unstable | **Not supported** (D picks 0.25 in 91% of bootstraps). Replace with: "the LP is ε-*sensitive*, the learned arms are not" |
| Multiply-wrong LP pins become more common with size | **Holds** at the frozen point (L2 → L4 McNemar p ≈ 1e-4) and at each level's own calibrated point (p = 0.001). "Size" here is the granularity of one fleet; see G2 |
| The learned advantage survives at 22 qubits | **Holds** at p = 1 on 20 diverse structures, per structure and reweighted. The absolute success rate falls for every arm |
| Band share is the mechanism at scale | **Holds** at L4, and the L2 curve transfers to L4 without refitting (R² 0.91) |
| Final-g ranking: D best at ε = 0.1, worst among warm arms at its own best ε | **Holds** (30 of 30 scenarios both ways). It is the ε trade-off; state it explicitly so T5 and T7 don't look contradictory |

## Superseded or do not use

- `s6_*`, `s6b_*` (pilot) and `s7_arms_full.*` (10 scenarios, scenario-level statistics only) are superseded by T5–T8. Cite s6b only for the history of the LP-target fix.
- The p = 5 and p = 3 sweep records inside `s8_runs.jsonl` fall outside both phases; the analysis ignores them.
- The scenario-fold CV numbers in early Phase 1 output are leaky. Use group-fold.
- Untracked `fig1_lambda_sweep.png`, `fig2_gnn.png`, `gnn_params.npz`, `gnn_report.json`, `lambda_*.json` are from the old `src/quc` prototype. They are not this pipeline; do not use them.

## Gaps a reviewer would ask about (most important first)

| # | Gap | Why a reviewer asks | Cost to close |
|---|---|---|---|
| G1 | **The L4 QAOA runs (T13) use the L2-calibrated operating point.** At L4 the frozen point has a 6.1% mean encoding gap, and its ground state equals the MILP schedule in only 4% of scenarios; L4's own point (B = 150 MW) has 1.85%. *Closed for the census*: at each level's own point the trend holds (+0.020 per qubit [0.014, 0.025]) | A reviewer can still ask whether T13 depends on the mis-calibrated L4 encoding | Re-run T13 at L4's own point: about 80 min |
| G2 | **"Size" here means granularity of one fleet.** The ladder splits stations into units, and it is precisely the added small marginal units that attract the wrong pins. Horizon length (T = 2 only) and other fleets are untested | Does the effect grow with system size in general, or only when more small marginal units are added? | Census at T = 3, 4 (L2/L3, cheap) and on a second synthetic fleet: about 1 h |
| G3 | **Noise-free, exact sampling only.** No shot noise in training, no hardware noise | Does the advantage survive realistic noise? | A depolarising-channel simulation at L2 needs density matrices or trajectories: days of work. Alternatively argue that best-of-K is exact and noise is out of scope |
| G4 | **No classical sampler on the same QUBO** (simulated annealing or tabu with a matched sample budget) | Why use QAOA at all? The MILP solves in milliseconds | SA on the QUBO at L2/L4 with P(ground) at K = 1024: about 1 h |
| G5 | **Generalisation of the learned model.** Training and test come from the same demand generator | What about out-of-distribution profiles, unit outages or other fleets? | Shifted-demand and outage test sets at L2, QAOA p = 1: about 1 h |
| G6 | **Band share needs the full spectrum,** so it explains the result but cannot select a warm start at scale | Is there a computable proxy? | Test whether LP wrong-pin count or GNN confidence predicts band share: about 30 min of analysis |
| G7 | **C-oracle is not scalable** (labels need brute force); dropped at L4 | Fine as an upper bound, but say so | none, writing only |
| G8 | ~~L4 analysis script~~ | **Closed**: `s9_analysis.py` | — |
| G14 | **L4 covers 34% of the demand distribution** (the 20 selected signatures), and the reweighted estimate applies only to that share. The selection over-represents multiply-wrong structures (75% vs 54% in the pool) | The population effect at L4 is extrapolated from the census fraction × per-structure gaps | More structures, or a random (non-diverse) L4 sample of 20: about 80 min each |
| G15 | **L4 at p = 1 only.** Absolute success is low for every arm (C-milp 0.70) | Does depth recover it at L4, where it did nothing at L2? | p = 2 at L4: about 5.5 h |
| G9 | ~~Held-out-arm and group-CV checks for band share not in a results file~~ | **Closed**: now in `s8_mass.py` output (T11) | — |
| G10 | **The L4 GCN's thresholded accuracy is below the majority baseline** (0.866 vs 0.888), though its MAE and BCE beat a constant predictor (0.122 vs 0.178, 0.241 vs 0.319). The cause is the 0.5 labels of interchangeable units | A reviewer reading T4 will see the accuracy number first | **Closed in data**: `s9_gnn_vs_constant.txt` (`s9_gnn_vs_constant.py`). T4 must report MAE/BCE against the constant baseline, not accuracy alone |
| G11 | **One optimiser (COBYLA) and one initial-parameter distribution** | Is the ranking an artefact of the optimiser? | Repeat T5 at p = 1 with SPSA or Nelder–Mead on a subset: about 30 min |
| G12 | **The simulator's correctness evidence lives only in tests** (`tests/test_fastsim.py`: PennyLane oracle at 1e-10, Monte Carlo check of best-of-K) | A reviewer wants a validation table | Export the oracle comparison to a results file: about 10 min |
| G13 | **Wall-clock accounting** omits GNN training cost (labels plus 8 s training) | Fair comparison against LP | Training and label time are in the T4 logs; add one line to T5 |
