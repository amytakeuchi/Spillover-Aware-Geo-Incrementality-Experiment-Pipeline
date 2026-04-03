# Spillover Aware Geo Incrementality Measurement — End-to-End Causal Inference Pipeline

> A production-style geo experiment framework implementing five causal estimators,
> a spillover-aware validity gate, and a decision-grade business reporting layer.
> Built to demonstrate measurement engineering at the level expected of a senior
> DS role in a large-scale ads or growth organization.

---

## What this project does

This project measures the **true incremental lift** of a paid social campaign across
60 geographic markets. It does not stop at a single model. It runs five estimators in
parallel, applies method-specific trust gates, detects and quantifies spillover
contamination, triangulates across valid methods, and translates the result into
an auditable business recommendation with full uncertainty propagation.

The pipeline answers three questions in sequence:

1. **Is the experimental design valid?** (balance, parallel trends, placebo tests)
2. **Which estimators can be trusted, and why?** (spillover flags, RMSPE gates, LOO stability)
3. **Was the campaign worth the spend? (Business Question)** (iROAS with CI, break-even probability, gated recommendation)

---

## Results summary

| Estimator | Lift estimate | vs. Ground truth | Valid? | Reason if invalid |
|---|---|---|---|---|
| DiD | −49,607 | −154% | ❌ | Spillover inverts estimate |
| TBR / GBR | 76,636 | −16.3% | ❌ | Spillover contaminates controls |
| CUPED | **82,519** | −9.9% | ✅ | Primary — only valid CI-bearing method |
| CUPAC | 73,881 | −19.3% | ❌ | Spillover contaminates controls |
| Synth (SCM) | **87,714** | −4.2% | ✅ | RMSPE/lift=0.023, LOO rel-std=0.015 |
| Bayes hier | **93,575** | +2.2% | ✅ | Most accurate; no CI (corroborating only) |

**Consensus range (3 valid methods): 82.5K – 93.6K incremental units**

**Primary estimator: CUPED** — selected for having the only valid bootstrap CI,
enabling iROAS bounds and break-even probability computation.

| Metric | Value |
|---|---|
| Incremental lift (CUPED) | 82,519 units · CI [36,712 — 135,163] |
| Total ad spend | $945,700 |
| iROAS | 0.09x · CI [0.04x — 0.14x] |
| P(iROAS > 1) | 0.0% |
| **Recommendation** | **PAUSE ⚠️** |

<p align="center">
  <img src="image/final_reccomendation.png" width="800" title="Final Recommendation">
</p>
<p align="center"><i>Figure: Final Recommendation Summary</i></p>
---

## Why five estimators

Each estimator makes different assumptions. Running them in parallel and requiring
consensus is the production pattern at measurement-mature organizations because no
single method is robust to all failure modes simultaneously.

| Estimator | Key assumption | Failure mode |
|---|---|---|
| **DiD** | Parallel trends + clean controls | Breaks under spillover or pretrend violation |
| **TBR / GBR** | Controls are valid counterfactuals | Biased when control pool is contaminated |
| **CUPED** | Pre-period treated units predict post | Spillover-immune; wide CI under high noise |
| **CUPAC** | Auxiliary model improves variance reduction | Same spillover exposure as TBR |
| **SCM** | Donor-weighted synthetic counterfactual | Fails if donor scaling is ignored (see below) |
| **Bayes hier** | Partial pooling across markets | Requires MCMC CI for production use |

The multi-model framework surfaces something no single estimator would:
DiD and TBR fail in opposite directions (DiD inverts the sign; TBR attenuates),
while CUPED, SCM, and Bayes converge. That convergence *is* the finding.

---

## The SCM debugging story

The initial SCM implementation produced a lift estimate of **1,714,415** — an
**18× overestimate** of the ground truth of 91,540. **Debugging write-up:** [Building a Production-Grade Geo Incrementality System: How Synthetic Control Failed by 18× — and What Fixed It →](https://medium.com/@a.takeuchi121/building-a-production-grade-geo-incrementality-system-how-synthetic-control-failed-by-18-and-bd497ebefa08)

**Root cause:** Raw unscaled donor series were passed directly to the constrained
weight optimizer. The donor pool aggregate was 3–5× larger in absolute magnitude
than the treated aggregate. The simplex constraint (weights sum to 1) forced the
optimizer toward uniform initialization (all weights ≈ 0.017) rather than finding a
genuine fit. The resulting synthetic control tracked donor volume, not treated trend.

**The fix — three changes:**

1. **Per-donor column scaling** (root cause): divide each donor column by its
   own pre-period mean; divide `yT_pre` by its mean. Optimizer works on
   dimensionless residuals. Restore units after fitting. Column means computed
   from pre-period only — never post-period (information leakage).

2. **`top_k_donors` pruning**: refit after dropping near-zero weight donors.
   Reduces noise, improves RMSPE.

3. **`rmspe_lift_ratio` validity gate**: `prefit_rmspe / |lift_hat_total|`.
   Scale-free threshold: <0.10 good · 0.10–0.25 marginal · >0.25 do not report.

**Before and after:**

| Metric | Pre-fix | Post-fix |
|---|---|---|
| Lift estimate | 1,714,415 | 87,714 |
| Pre-fit RMSPE | 51,332 | 1,998 |
| RMSPE reduction | — | **96%** |
| LOO rel-std | 0.002 (stably wrong) | 0.015 |
| Valid | ❌ | ✅ |

The LOO stability of 0.002 on the broken model is the critical diagnostic lesson:
**a stably wrong model still passes consistency checks.** RMSPE validates the fit
itself. LOO only validates consistency of a fit you have already verified.

---

## Pipeline architecture

```
00_experimental_design.ipynb
    ├── Simulate geo panel (SimConfig: 60 markets, 90-day pre, 30-day post)
    ├── Stratified rerandomization assignment (ε=0.10, max_tries=3000)
    ├── Apply nonlinear spend → lift → spillover effects
    ├── Design validity: Love Plot (SMD=0.074 < 0.10 ✅)
    ├── Causal validity: AA test, placebo-in-time, placebo-in-space
    ├── Parallel trends: aggregate + individual small multiples + regression test
    │       pretrend p=0.67 → no violation ✅
    └── Spillover diagnostics: buffer zone vs guardrail excess

01_causal_estimation.ipynb
    ├── Run 5 estimators (DiD, TBR, CUPED, CUPAC, SCM, Bayes hier)
    ├── Block bootstrap CIs (300 reps, market-level blocks)
    ├── Trust metric gates (per-estimator validity rules)
    ├── Estimator summary table (lift + CI + trust metrics + valid flag)
    ├── SCM LOO donor sensitivity (loo_rel_std=0.015, range_rel=0.069 → STABLE)
    └── Spillover sensitivity: all controls vs guardrail-only refit
            SCM shift: −4.3% · TBR shift: +5.4% → contamination manageable

02_business_reporting.ipynb
    ├── Rule-based primary estimator selection (validity + CI availability)
    ├── compute_experiment_totals() → spend from data, never hardcoded
    ├── reporting_metrics_from_lift() → iROAS, ROI, CPIU with CI propagation
    ├── Break-even probability P(iROAS > 1) via Normal(lift, bootstrap SE)
    ├── Sensitivity table: AOV × gross margin → iROAS heatmap
    └── Decision table: Recommendation + risk flag register (HTML)
```

---

## Repository structure

```
incrementality_geo_experiment/
├── README.md
├── requirements.txt
├── pyproject.toml
├── data/                              # generated by notebook 00
│   ├── simulated_geo_panel.csv
│   ├── simulated_wide_prepost.csv
│   └── simulated_agg_timeseries.csv
├── notebooks/
│   ├── 00_experimental_design.ipynb
│   ├── 01_causal_estimation.ipynb
│   └── 02_business_reporting.ipynb
├── scripts/
│   └── run_quick_check.py             # CLI entrypoint (no UI)
└── src/
    └── geo_experiment/
        ├── config.py                  # SimConfig dataclass + defaults
        ├── simulate.py                # simulate_geo_panel()
        ├── design.py                  # stratified assignment + rerandomization + SMD
        ├── effects.py                 # spend → nonlinear lift → spillover → sales
        ├── features.py                # wide table builder + panel features
        ├── estimators/
        │   ├── did.py                 # DiD pre/post
        │   ├── tbr.py                 # regression adjustment on controls
        │   ├── cuped.py               # CUPED variance reduction
        │   ├── cupac.py               # CUPAC residualization (aux prediction)
        │   ├── synth.py               # constrained SCM + per-donor scaling + RMSPE gate
        │   └── bayes_hier.py          # empirical Bayes partial pooling
        ├── diagnostics/
        │   ├── sanity.py              # basic data checks
        │   ├── balance.py             # SMD + group summaries + Love Plot
        │   ├── pretrend.py            # parallel trends plots + regression test
        │   ├── spillover.py           # buffer vs guardrail contamination
        │   ├── placebo.py             # placebo-in-time + placebo-in-space + AA test
        │   └── inference.py           # block bootstrap CI
        ├── reporting.py               # quick check report table
        └── reporting_metrics.py       # iROAS, ROI, CPIU from lift estimates
```

---

## Key engineering decisions

**Spend is always read from data.** `compute_experiment_totals()` reads
`paid_social_spend` directly from the panel. Hardcoding spend produces
an artificially inflated iROAS and a wrong recommendation. The difference between
`TOTAL_SPEND = 50_000` and `compute_experiment_totals(df)` is the difference
between a 1.65x iROAS and a 0.09x iROAS — between SCALE and PAUSE. This was a
deliberate decision to demonstrate production discipline over notebook convenience.

**Uncertainty propagates end-to-end.** CI endpoints are transformed through
`reporting_metrics_from_lift()` independently. iROAS CI bounds are not
approximated from a point estimate — they are the exact bounds corresponding
to the lift CI endpoints. CPA bounds are deliberately flipped (higher lift =
lower cost per unit).

**Estimator selection is rule-based and auditable.** The primary estimator
is chosen by a ranked filter: validity gate first, CI availability second,
CI width third. The selection logic prints explicit exclusion reasons for
every rejected method. No judgment calls, no hardcoded names.

**Trust gates are method-specific, not generic.** DiD gates on pretrend and
spillover. SCM gates on `rmspe_lift_ratio` and `w_max`. CUPED has no spillover
gate because it uses pre-period treated units as its own covariate, bypassing
the contaminated control pool entirely. Each gate reflects the actual failure
mode of its method — not a one-size-fits-all p-value threshold.

**Block bootstrap resamples at market level.** Day-level bootstrap underestimates
variance in geo experiments because market-level shocks are correlated across
time. 300 replicates with market-level blocks gives CIs that reflect the true
unit of randomization.

**LOO stability ≠ fit validity — explicitly.** The broken SCM had
`loo_rel_std = 0.002`, more stable than the fixed model at 0.015. This
distinction is documented in the code, the notebook, and this README because
it is the single most common misuse of LOO diagnostics in practice.

---

## Design validity results

| Check | Result | Threshold | Pass? |
|---|---|---|---|
| SMD (historical sales) | 0.074 | < 0.10 | ✅ |
| Pretrend regression p-value | 0.67 | > 0.05 | ✅ |
| AA test lift | −51K (CI crosses zero) | CI ∋ 0 | ✅ |
| Placebo-in-time (28-day shift) | −68K (CI crosses zero) | CI ∋ 0 | ✅ |
| Placebo-in-space | +230K (wide CI, n=5 treated) | Expected under small n | ✅ |
| Spillover flag | True (Buffer +0.75, Guardrail +0.50) | Detected and quantified | ⚠️ |

Spillover is real but spatially bounded and manageable. A 4–6% lift shift when
switching to guardrail-only controls is within normal range for geo experiments.
A 20%+ shift would require pool redesign.

---


Notebooks are designed to be run top-to-bottom. Each notebook saves its outputs
to `data/` so downstream notebooks can run independently without re-executing
the full simulation.

---

## Simulation parameters

| Parameter | Value | Why it matters |
|---|---|---|
| Markets | 60 (21 treated, 39 control) | Enough for block bootstrap stability |
| Pre-period | 90 days | Sufficient for SCM donor weight convergence |
| Post-period | 30 days | Standard campaign flight window |
| Spillover fraction | 0.18 | 18% of lift bleeds to buffer zone neighbors |
| Nonlinear response | lift_a=450, lift_b=3500 | Diminishing returns — stresses DiD assumption |
| Treatment fraction | 35% | Stratified rerandomization with ε=0.10 balance |

The nonlinear response curve and 18% spillover fraction are not arbitrary — they
are the parameters that cause DiD and TBR to fail while CUPED, SCM, and Bayes
remain valid. The simulation is designed to stress-test the estimator suite,
not to make all methods look good.

---

## What to look for

- **The SCM debugging section** (notebook 01, section 5.4) demonstrates end-to-end
numerical debugging: symptom identification → root cause isolation → fix →
validation. The fix is three lines of code. The diagnosis required understanding
the geometry of constrained optimization under scale mismatch and the difference
between optimizer collapse and optimizer failure.

- **The estimator validity framework** demonstrates that knowing *when not to trust
a model* is more valuable than knowing how to run one. DiD produces a negative
estimate on a positive ground truth. The framework catches it automatically via
a spillover flag that was set before any model was run.

- **The business reporting layer** (notebook 02) demonstrates the bridge between
statistical output and decision-grade reporting that is typically absent from
academic portfolios: uncertainty propagation into financial metrics, rule-based
recommendation logic, and an explicit risk flag register that updates automatically
when data changes.

**The spend data lineage** is a single design choice that separates production
thinking from notebook thinking. The two-line difference between `TOTAL_SPEND = 50_000`
and `compute_experiment_totals(df)` changes the recommendation from SCALE to PAUSE
and changes the iROAS from 1.65x to 0.09x. That is the whole point.
