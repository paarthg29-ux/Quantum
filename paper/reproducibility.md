# Reproducibility

Every number and figure in the paper is generated from the result files in `results/` by two scripts. No number in the `.tex` sources is typed by hand, and `scripts/lint_paper.py` enforces that.

## Provenance

| What | Commit |
|---|---|
| All run stores (`results/*.jsonl`) and models (`models/`) | `9afe77d` (last experimental round); earlier stores in `7feab61`, `a6789e7`, `63ef26d` |
| Paper generation layer: `paper/`, `scripts/make_tables.py`, `make_figures.py`, `lint_paper.py`, and three analysis scripts extended to also write JSON (`s8_mass.py` band-share-alone model, `s9_mass.py`, `s10_validation.py` with the injected-bug checks), with their regenerated outputs | `d86f60c` |

`paper/numbers.tex` records the commit it was generated from in its header.

## Repository layout

```
qguc/                 the library
  instance.py         fleet data, cost model, demand profile, refinement ladder (L0-L4)
  qubo.py             QUBO construction (fuel, demand penalty with bias, start-up; switching penalty off)
  ising.py            QUBO -> Ising, normalisation
  exact.py            brute force (QUBO ground states), MILP and LP relaxation (HiGHS)
  calibrate.py        operating-point sweep and the plateau selection rule
  config.py           the FROZEN operating point and its history
  fastsim.py          diagonal-cost statevector QAOA simulator (pure NumPy)
  warmstart.py        regularisation, RY product state, Egger warm-start mixer
  arms.py             the seven arms (sources of c), wrong-pin counting
  gnn.py              the GCN warm-start model (plain PyTorch)
  qaoa.py             optimisation loop, stopping rule, finite-shot option, metrics
  sampling.py         exact best-of-K sampling metrics
  repair.py, metrics.py
scripts/              experiments (s2-s10), analyses, paper generation
tests/                unit tests, including the PennyLane oracle for the simulator
models/               trained GCNs (gnn_milp_L2.pt, gnn_oracle_L2.pt, gnn_milp_L4.pt, horizon/*.pt)
results/              run stores (.jsonl), analysis outputs (.txt/.json), figures
paper/                main.tex, sections/, tables/ (generated), figures/ (generated), numbers.tex (generated)
```

## Environment

Python 3.14.2 (Windows 11, 10 physical cores, 15.6 GB RAM). Package versions used for the results:

| package | version |
|---|---|
| numpy | 2.5.3 |
| scipy | 1.18.1 (HiGHS via `scipy.optimize`) |
| torch | 2.14.0 (CPU, one thread) |
| pennylane | 0.45.1 (validation oracle only) |
| numba | 0.68.1 (compiled SA baseline only) |
| matplotlib | 3.11.2 |
| psutil | 7.2.2 |
| pytest | 9.2.0 |

```
python -m venv .venv
.venv/Scripts/python -m pip install -e . torch pennylane numba matplotlib psutil pytest
.venv/Scripts/python -m pytest -q          # 212 tests
```

All experiments set `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` and parallelise over processes.

## Frozen operating point

λ_D = 100, demand bias B = 250 MW, λ_switch = 0 (minimum up/down time NOT enforced), calibrated on L2 with T = 2 by the plateau rule (`qguc/config.py`, `results/s4b_recalibrate_L2.txt`). It was frozen before any QAOA experiment. The robustness checks at each level's own calibrated point use `results/s7_gap_ladder.csv` (ladder, T = 2) and `results/s9_horizon_points.csv` (horizon cells).

## Seeds

| Use | Seed |
|---|---|
| Calibration selection / test sets | 0 / 2026 |
| Encoding-gap ladder (100 scenarios) | 4242 |
| GCN training / validation / held-out accuracy | 8001 / 8002 / 9002 |
| Pilot and first full run (superseded) | 7001 / 9001 |
| **Main experiments** (30 held-out scenarios, L2) | **9100** |
| Wrong-pin census and the L4 pool (300 draws per cell) | 9300 |
| Out-of-distribution test sets | 9500 (+ set index, scenario) |
| QAOA initial parameters | `default_rng([seed, p])`, seeds 0-4; identical for every arm |
| Random arm E | `default_rng([seed, scenario, set_seed, 0xE])` |
| Finite-shot estimates | `default_rng([seed, p, shots])` |
| Statistics (bootstraps, CV splits) | fixed in `scripts/make_tables.py` (0 and 4) |

## Regenerating everything from scratch

Each run script writes a resumable store and skips finished runs; delete a store to recompute it. The measured wall times are for the machine above.

| Step | Command | Output | Time |
|---|---|---|---|
| Calibration | `scripts/s2_calibrate.py L2`, `scripts/s4_ladder.py`, `scripts/s7_gap_figure.py` | `s4b_recalibrate_L2.txt`, `s4_ladder.*`, `s7_gap_ladder.csv` | minutes |
| Simulator timing | `scripts/s3b_fastsim_timing.py` | `s3b_fastsim_timing.txt` | ~10 min |
| GCN training | `scripts/s7a_train_gnn.py`; `scripts/s7a_train_gnn.py --level 4 --variants milp` | `models/*.pt` | ~2 min |
| ε sweep (Phase 1) | `scripts/s8_eps_main.py --phase 1` | `s8_runs.jsonl` | ~10 min |
| Main table (Phase 2) | `scripts/s8_eps_main.py --phase 2` | `s8_runs.jsonl` | ~15 min |
| Analyses | `scripts/s8_analysis.py --phase 1`, `--phase 2`; `scripts/s8_mass.py --phase 1`, `--phase 2`; `scripts/s8_eps_stability.py` | `s8_*` | ~10 min |
| Census (granularity) | `scripts/s9_census.py`; `scripts/s9_census.py --point own` | `s9_census*.jsonl` | ~10 min |
| L4 selection and arms | `scripts/s9_select.py`; `scripts/s9_l4_arms.py`; `scripts/s9_l4_arms.py --point own` | `s9_selected.json`, `s9_runs*.jsonl` | ~80 min each |
| L4 analyses | `scripts/s9_analysis.py` (and `--point own`); `scripts/s9_mass.py`; `scripts/s9_gnn_vs_constant.py` | `s9_*` | ~10 min |
| Census (horizon) | `scripts/s9_horizon.py`; `scripts/s10_horizon_learned.py` | `s9_horizon*`, `s10_horizon_learned*` | ~10 min |
| Classical baselines | `scripts/s9_classical.py` then `--tabu-only`; `scripts/s10_sa_compiled.py` | `s9_classical*`, `s10_sa_compiled*` | ~45 min |
| Robustness, out of distribution | `scripts/s10_robust.py`; `scripts/s10_ood.py` | `s10_runs.jsonl`, `s10_ood*` | ~30 min |
| Simulator validation | `scripts/s10_validation.py` | `s10_simulator_validation.*` | ~2 min |
| **Paper** | `scripts/make_tables.py`; `scripts/make_figures.py`; `scripts/lint_paper.py` | `paper/numbers.tex`, `paper/tables/`, `paper/figures/` | ~3 min |

Then compile `paper/main.tex` with pdflatex (IEEEtran, conference option). No LaTeX toolchain was available on the machine above, so the sources were checked by `lint_paper.py` (macros defined, inputs and figures present, references labelled, no hand-typed numbers), not by compilation.

## Known caveats of the stores

- `results/s8_runs.jsonl` contains 1,868 records written twice by two concurrent runners. They are identical except the timing fields, and every loader keeps one record per (arm, ε, scenario, seed, p).
- Timing fields (`wall_s`) and the numpy SA's wall-matched budgets depend on the machine and its load. The compiled SA's budgets are fixed (10⁴, 10⁵, 10⁶ flips plus the evaluation-matched budget); only its measured times vary.
- Everything else is deterministic given the seeds above.
