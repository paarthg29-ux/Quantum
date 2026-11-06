# Paper inventory: what we can support, where it comes from, and what is missing

Status at 2026-11-06, branch `qguc-pipeline`, after the last experimental round. **Committed** = the run data and the analysis output are in git (commit given). **Uncommitted** = the file exists but is not in git.

**Framing (locked): no quantum-advantage claim anywhere.** A classical sampler solves these instances in milliseconds (T19). The contribution is what determines warm-start quality within QAOA.

Statistical convention for every comparison below:
- Values are averaged over seeds within a scenario, then paired across scenarios.
- Where scenarios share a structure, the unit is the **structural group**: 17 groups by (ground state, LP pin pattern), 13 by ground state alone. At L4 every selected scenario is its own structure (n = 20).
- For choosing ε, the primary numbers are **group-fold cross-validation repeated over 1,000 random group splits**.

## Claims skeleton (what the paper says, with the strongest number for each)

The paper studies warm-started QAOA for Haryana unit commitment, encoded as a calibrated QUBO on 10–22 qubits and simulated exactly; it makes no quantum-advantage claim. Ten claims, each with its strongest number:

1. **A learned warm start makes QAOA sample the exact ground state far more reliably than the alternatives.** At 16 qubits, p = 1, a GCN trained on MILP schedules reaches P(best of 1024 samples is the ground state) = 0.990. The LP relaxation reaches 0.816, merit order 0.656 and the cold start 0.549 (T5).
2. **The advantage over the LP survives tuning the LP's regularisation.** With ε chosen by group-fold cross-validation, the LP still trails C-milp by 0.070, and is worse in 96% of 1,000 random splits (T7). It also survives the move to 22 qubits at that level's own operating point: C-milp − D = +0.231 [+0.079, +0.384] over 20 structurally distinct instances (T15).
3. **The deficit has a precise cause.** The LP pins marginal units to the wrong value. Where it pins 2 or more bits wrong, C-milp leads by +0.26 to +0.50 at 22 qubits (both operating points, both ε). Where it pins at most one, the LP is as good or better (T14, T15).
4. **Initial overlap alone does not explain sampling; band share does.** Band share is the ground state's share of the initial low-energy band. Overlap gives R² 0.55 with arm-specific residuals. Adding band mass gives R² 0.95–0.98 at 16 and 22 qubits, and predicts held-out arms within 0.015 (T10, T11, T16). Spreading initial mass over near-ground states hurts. Band share is a diagnostic, not a predictor: it needs the full spectrum.
5. **The LP needs its regularisation tuned; the learned start does not.** A wrong ε costs the LP up to 0.917 (0.122 at ε = 0.05). It costs the learned starts at most 0.018 (T9).
6. **The LP's errors grow with problem size, the learned start's grow much more slowly.** Along the horizon at fixed granularity, each added variable adds 0.09–0.10 wrongly pinned bits for the LP, against 0.014–0.042 for the learned start (2–7× slower, depending on granularity). The gap widens from 0.44 to 1.73 wrong pins between T = 1 and T = 4 (T13, T18).
7. **The advantage is robust to the optimiser's view.** With 128-shot expectation estimates the learned start loses only 0.021 in P(best of 1024 is ground), against 0.106 for the cold start and 0.142 for merit order; the C-milp − LP gap is unchanged (+0.02, not significant) (T21). A second optimiser (Nelder–Mead) reproduces the comparison within 0.002 (T22).
8. **The learned start generalises across graph changes, but not across demand regimes.** Under a forced outage (one block removed, so a different graph) it keeps 0.95–0.99, at or above the LP. Below the training demand range it falls under the LP (0.723 vs 0.861). With a changed hourly profile it falls to LP level (T20).
9. **On cost, all warm starts are equivalent.** The best-of-1024 operating-cost gap of the LP and the learned arms does not differ significantly (T5, T14): the LP's misses are near-optimal adequate schedules.
10. **No quantum advantage.** Compiled simulated annealing on the identical QUBO reaches P(ground) 0.996 in 4.5 ms at 16 qubits and 0.994 in 34 ms at 22 qubits, against QAOA's 0.990 in 1.4 s and 0.697 in 47 s of simulation (T19).

## Tables

| # | Table | Claim it supports | Source file (script) | Run committed? |
|---|---|---|---|---|
| T1 | Problem and encoding: fleet, ladder L0–L4 (5/6/8/9/11 blocks × 2 h = 10–22 qubits, 4,442 MW at every level), frozen operating point λ = 100, B = 250 MW | The QUBO is a faithful, calibrated encoding of Haryana unit commitment; the MILP solves the operational problem in about 25 ms | `s4_ladder.txt`, `s4b_recalibrate_L2.txt`, `s2_model_comparison.txt` | yes (80fc040, 50f7960) |
| T2 | Encoding gap per level, at the frozen point and at each level's own point | The encoding costs 2.5–6.1% at the frozen point and 1.8–4.9% at each level's own point | `s7_gap_figure.txt`, `s7_gap_ladder.csv` | yes (f006eeb) |
| T3 | Simulator timing; PennyLane comparison | The exact NumPy statevector simulator is 11–16× faster than `lightning.qubit` at 22 qubits | `s3b_fastsim_timing.txt`, `s3_timing_probe.txt` | yes (917a70f, 3eff343) |
| T4 | GCN warm start: held-out MAE and BCE against a constant predictor, load conditioning, **training and inference cost** | Beats a constant predictor at L2 and L4 (L4 MAE 0.122 vs 0.178). **Cost (G13):** L2 C-milp: MILP labels 9.7 s + training 54 s; L2 C-oracle: brute-force labels 15 s + 42 s; L4 C-milp: 6.5 s + 8 s; inference 0.04–0.3 ms per instance; LP solve about 1.6 ms | `s7a_train_gnn.txt`, `s9_train_gnn_L4.txt`, `s9_gnn_vs_constant.txt` | yes (428dec6, a6789e7) |
| T5 | **Main table, ε = 0.1**: 7 arms × 30 held-out scenarios × 5 seeds, p = 1/2/3 | Claim 1 (and 9) | `s8_phase2_analysis.txt` | yes (7feab61) |
| T6 | Per-scenario detail of T5 | D's deficit is confined to 8 of 30 scenarios where the LP pins 2–3 bits wrong | `s8_phase2_analysis.txt` | yes (7feab61) |
| T7 | **ε sweep, p = 1**, best ε, decisive comparisons, ε = 0.5 sanity checks, repeated group-fold CV | Claim 2: D − C-oracle −0.094 and D − C-milp −0.070 over 1,000 group splits (D worse in 100% / 96%) | `s8_phase1_analysis.txt`, `s8_eps_stability.txt` | yes (7feab61, a6789e7) |
| T8 | Statistics at group level (n = 30 / 17 / 13) | Which claims survive with honest n | `s8_phase1_analysis.txt`, `s8_phase2_analysis.txt` | yes (7feab61) |
| T9 | **ε regret** | Claim 5: D 0.122 at ε = 0.05, 0.917 at ε = 0; C-milp / C-oracle ≤ 0.018 / 0.005 over ε ∈ [0, 0.25] | `s8_eps_stability.txt` (section 3) | yes (a6789e7) |
| T10 | **Band share** at L2, p = 1, 2, 3 | Claim 4: R² 0.55 → 0.95–0.97; per-arm residual removed; spreading hurts | `s8_mass_phase1.txt`, `s8_mass_phase2.txt` | yes (a6789e7) |
| T11 | Band-share held-out checks | Held-out arms within 0.015; leave-one-group-out R² 0.50–0.64 → 0.95–0.97 | `s8_mass_phase*.txt` ("HELD-OUT CHECKS") | yes (a6789e7) |
| T12 | Wrong-pin census along granularity (L0–L4, T = 2), frozen and own points; per-block concentration | Wrong pins grow with granularity and fall on small, expensive marginal blocks | `s9_census.txt`, `s9_census_ownpoint.txt` | yes (a6789e7) |
| T13 | Wrong-pin census along the horizon at fixed granularity (LP) | Claim 6, first half: the LP's wrong pins are extensive in the horizon (per hour flat; +0.10 per variable) | `s9_horizon.txt` | yes (63ef26d) |
| T14 | L4 arms at the frozen point | C-milp − D +0.208 [+0.037, +0.379] | `s9_analysis.txt` | yes (a6789e7) |
| T15 | L4 arms at L4's own point, paired with T14 | Claim 2 at 22 qubits: +0.231 [+0.079, +0.384]; change from the frozen point +0.023 [−0.220, +0.267] | `s9_analysis_ownpoint.txt` | yes (63ef26d) |
| T16 | Band share at L4; L2 curve transferred to L4 | Claim 4 at scale: R² 0.98; transfer R² 0.91 | `s9_mass.txt` | yes (a6789e7) |
| T17 | Classical samplers, numpy implementation (SA, tabu with restarts), evaluation- and wall-matched | Superseded for timing by T19; kept for tabu and the evaluation-matched comparison (QAOA C-milp wins when one expectation estimate counts as one evaluation) | `s9_classical.txt` | yes (63ef26d) |
| T18 | **Learned start's wrong pins along the horizon (G16)**: C-milp trained at each horizon, and the T = 2 model transferred, on the T13 draws | Claim 6, second half: C-milp +0.014 / +0.031 / +0.042 wrong pins per variable (L0/L1/L2) vs LP +0.103 / +0.104 / +0.091. P(2 or more) slope +0.001–0.009 vs +0.028–0.035. Paired gap widens 0.44 → 1.73 (L0, T = 1 → 4). The models hit the 6,000-epoch cap while improving, which is conservative | `s10_horizon_learned.txt` (`s10_horizon_learned.py`; models in `models/horizon/`) | **uncommitted** |
| T19 | **Compiled SA (numba, G17)**: budget ladder with measured time per run | Claim 10: 0.996 in 4.5 ms (L2, 10⁵ flips), 0.994 in 34 ms (L4, 10⁶ flips); 22–30 M flips/s per chain; QAOA's wall time buys 3 × 10⁷ / 1.4 × 10⁹ flips | `s10_sa_compiled.txt` (`s10_sa_compiled.py`) | **uncommitted** |
| T20 | **Out of distribution (G5)**: shifted demand (high/low), forced outages (3 blocks), reversed and flat profiles; C-milp vs D vs A | Claim 8. Outages: C 0.947–0.986, ≥ D. Low demand: C 0.723 vs D 0.861, wrong pins 2.85 vs 1.45. Flat: 0.793 vs 0.790; reversed 0.893 vs 0.839. High demand: trivial (all arms ≈ 1.000) | `s10_ood.txt` (`s10_ood.py`) | **uncommitted** |
| T21 | **Finite-shot optimisation (G3 scoped)**: 128 / 1024 / 8192 shots, exact evaluation | Claim 7. Loss at 128 shots: C-milp −0.021, D −0.041, A −0.106, F −0.142. C-milp − D advantage change +0.02 (not significant). Under shot noise COBYLA stops after 21–34 evaluations (72–183 exact) | `s10_robust.txt` (`s10_robust.py`) | **uncommitted** |
| T22 | **Second optimiser (G11)**: Nelder–Mead, exact expectations | Claim 7: C-milp − D +0.172 (COBYLA +0.174); all pairs within 0.01 | `s10_robust.txt` | **uncommitted** |
| T23 | **Simulator validation (G12)** against PennyLane default.qubit; best-of-K vs Monte Carlo | Worst error over 45 cases 1.5 × 10⁻¹⁴; a real UC instance to 2.2 × 10⁻¹⁴; Monte Carlo within 2.5 standard errors | `s10_simulator_validation.txt` (`s10_validation.py`) | **uncommitted** |

## Figures

| # | Figure | Claim | File | Committed? |
|---|---|---|---|---|
| F1 | Encoding-gap ladder | T2 | `fig_encoding_gap_ladder.png/.pdf` | yes (f006eeb) |
| F2 | **Band share, two panels** (arms separate on overlap, collapse on band share) | Claim 4 | `fig_bandshare.png/.pdf` | yes (a6789e7) |
| F3 | Overlap vs sampling, small multiples | supplement only | `fig_overlap_vs_sampling.png/.pdf` | yes (7feab61) |

**To draw while writing** (data in hand, no new runs):
1. **ε regret curves** (claim 5): P against ε per arm, with group CIs.
2. **Size scaling** (claim 6): mean wrong pins against n, LP vs C-milp, along the horizon (T13, T18), with the granularity line (T12).
3. **Main-table dot plot** (claim 1): arms × p, with group CIs.
4. **L4 per-structure matrix** (claims 2–3): C-milp vs D, coloured by D's wrong pins.
5. **Classical vs QAOA** (claim 10): P(ground) against measured time, compiled SA ladder vs QAOA arms, at L2 and L4.
6. **Robustness panel** (claims 7–8): P loss under shots per arm, and OOD per set.

## Claims ledger

| Claim | Status |
|---|---|
| Learned warm starts sample the ground state far better than A, E and F | **Holds** at every level of aggregation |
| At its best ε, D still loses to the learned arms | **Holds** under repeated group-fold CV. **Borderline** at group level against C-milp when D gets its in-sample best ε |
| D's deficit comes from scenarios where the LP pins 2 or more bits wrong | **Holds** at L2 and at L4 (both operating points) |
| D matches the learned arms on best-of-1024 cost | **Holds** |
| C-oracle beats C-milp | **Not supported** (tie). **Writing (G7):** present C-oracle as the labels upper bound. Its labels need the exact QUBO ground state (brute force: 15 s for 350 instances at 16 qubits, about 8 min at 22), so it does not scale, and it buys nothing over MILP labels |
| B matches C-milp | On sampling yes (tie); on final g and single-shot P(ground) no |
| Overlap alone predicts sampling | **No** |
| Band share explains sampling | **Holds** (R² 0.95–0.98, held-out arms within 0.015, transfers L2 → L4). Diagnostic, not predictive |
| The LP needs ε tuned; learned starts do not | **Holds** |
| ~~The LP's best ε is unstable~~ | **Dropped**; replaced by the regret claim |
| The LP's wrong pins grow with problem size | **Holds** generally (horizon and granularity), one fleet |
| The learned start's wrong pins stay low with size | **Narrowed to "grow 2–7× more slowly"** (T18). They do grow |
| The learned advantage survives at 22 qubits | **Holds** at both operating points, p = 1 |
| The learned advantage is robust to shot noise | **Against cold and heuristic starts: grows.** Against the LP: unchanged (not significant). Regime caveat: COBYLA stops early under noise |
| The learned start generalises out of distribution | **Partly**: graph changes (outages) yes; lower demand no (worse than LP); new profile shapes fall to LP level |
| QAOA is competitive with classical solvers | **No**: compiled SA 0.99+ in 4.5–34 ms |
| Final-g ranking flips with ε | **Holds** (30/30 at L2, 20/20 at L4); state explicitly |

## Methods notes for the writing (G18 and others)

- **Disclosed tuning of the classical baselines (G18), both choices in the classical samplers' favour.** SA's end temperature (5th percentile of positive |ΔE| / ln 100) was chosen among three options on 4 L2 scenarios. Tabu's tenure (n/2) and restart rule (after 5n non-improving steps) were chosen among six options on the 3 hardest L2 scenarios. Without restarts tabu cycles and does not improve with budget.
- **The learned starts were never tuned:** default hyperparameters at every level and horizon. Several models hit the epoch cap, which is conservative for them.
- **Finite-shot runs:** the patience stop and the choice of reported parameters use the noisy estimates; every metric is exact at the chosen parameters.
- **Bugs found and fixed during the study, each with a regression test:**
  - **ε = 0.5 rejected:** the regulariser refused the cold-limit value, which crashed the sweep.
  - **complex64 normalisation check:** a correct warm state was rejected at 20 or more qubits, because the check summed in single precision.
  - **LP infeasible above capacity:** the LP at the QUBO target was infeasible when demand + bias exceeded capacity; the target is now capped.
  - None of these changed a published number; each would have crashed rather than mislead.

## Superseded or do not use

- `s6_*`, `s6b_*` (pilot) and `s7_arms_full.*` (10 scenarios) are superseded by T5–T8.
- The p = 5 and p = 3 sweep records inside `s8_runs.jsonl` fall outside both phases.
- Scenario-fold CV numbers in early Phase 1 output are leaky; use group-fold.
- T17's numpy SA timings are superseded by T19. The first tabu run (no restarts) was replaced.
- The untracked `fig1_lambda_sweep.png`, `fig2_gnn.png`, `gnn_params.npz`, `gnn_report.json` and `lambda_*.json` are from the old `src/quc` prototype. Do not use them.

## Gaps

### Closed

G1 (L4 own point), G2 (size vs granularity), G4 (classical samplers), G5 (out of distribution, partly negative), G7 (C-oracle as labels upper bound, writing note above), G8, G9, G10, G11 (Nelder–Mead), G12 (validation table), G13 (training cost in T4), G16 (learned start along the horizon), G17 (compiled SA), G18 (tuning disclosure above).

### Deferred, to be stated as limitations

| # | Limitation | How to state it |
|---|---|---|
| G3 | Noise-free circuits. Only finite-shot expectation estimates are modelled (T21); gate and decoherence noise need density matrices (4^n) and are out of scope | Limitations section |
| G14 | L4 covers 34% of the demand distribution (20 selected signatures), mitigated by reweighting; the selection over-represents multiply-wrong structures (75% vs 54%) | Report per-structure and reweighted, and say so |
| G15 | L4 at p = 1 only | Depth did nothing for warm arms at L2 (0.990 → 0.991, p = 1 → 3) |
| G6 | No computable proxy for band share: it needs the full spectrum, so it is a diagnostic of why warm starts work, not a way to choose one | State explicitly in the mechanism section |
| — | One fleet (Haryana) | Scaling shown along two axes of one system |
| — | Out-of-distribution weaknesses of the learned start: lower demand and new profile shapes (T20) | Results and limitations; suggests training on wider demand and shape ranges |
| — | Finite-shot regime: under shot noise COBYLA stops after 21–34 evaluations, so T21 describes a lightly optimised regime | State with T21 |
