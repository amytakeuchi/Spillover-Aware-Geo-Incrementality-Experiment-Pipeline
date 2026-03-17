'''
**What this does:**

Implements **experiment design / randomization**:
- stratified assignment by market size (Small/Medium/Large)
- excludes guardrail markets from treatment
- rerandomizes until treated/control are balanced on `historical_sales_pre` (SMD <= threshold)
- broadcasts market-level assignment back to panel

**Why it exists:**
by the randomization; it's ensuring the covariate balance (like real geo tests).”

**Outputs:**

- updated `df` with `is_test` and `treatment_level`
- a balance diagnostics dict (SMD values)
'''
import numpy as np
import pandas as pd
# Calculates the Standardized Mean Difference (SMD) to measure covariate balance between groups.
# It divides the difference in means by the pooled standard deviation (Cohen's d).
# Used here to determine if the Treated and Control markets are statistically "similar" 
# regarding their historical sales volume.
def standardized_mean_diff(x_t: np.ndarray, x_c: np.ndarray) -> float:
    mt, mc = np.mean(x_t), np.mean(x_c)
    st, sc = np.std(x_t, ddof=1), np.std(x_c, ddof=1)
    pooled = np.sqrt((st**2 + sc**2) / 2.0)
    return 0.0 if pooled == 0 else (mt - mc) / pooled

# Maps market-level randomization decisions back to the original granular panel/unit-level DataFrame.
# It handles the cleanup of existing assignment columns and ensures that 'Guardrail' markets 
# or those not explicitly selected for treatment default to a Control status (0).
def broadcast_market_assignment(df: pd.DataFrame, mkt_assign: pd.DataFrame) -> pd.DataFrame:
    out = df.drop(columns=["is_test", "treatment_level"], errors="ignore").copy()
    out = out.merge(mkt_assign[["market_id", "is_test"]], on="market_id", how="left")
    out["is_test"] = out["is_test"].fillna(0).astype(int)
    out["treatment_level"] = out["is_test"].astype(float)
    return out

# Performs a constrained randomization (re-randomization) to ensure high-quality experimental groups.
# 1. Filters out 'Guardrail' markets to prevent contamination/spillover.
# 2. Stratifies by market size to ensure the treatment is representative of the whole population.
# 3. Iteratively generates assignments (up to max_tries) until the SMD for historical sales 
#     is below the 'eps' threshold for both the overall population and within each size stratum.
# 4. Returns the best-balanced assignment found if the threshold isn't met within the try limit.
def assign_treatment_stratified_rerand(
    df: pd.DataFrame,
    test_frac: float = 0.35,
    eps: float = 0.10,
    max_tries: int = 3000,
) -> tuple[pd.DataFrame, dict]:
    mkt = (
        df[["market_id", "market_size_group", "spillover_risk_group", "historical_sales_pre"]]
        .drop_duplicates().reset_index(drop=True)
    )
    mkt["eligible"] = (mkt["spillover_risk_group"] != "Control Guardrail").astype(int)
    strata = mkt["market_size_group"].unique().tolist()

    def propose() -> np.ndarray:
        is_test = np.zeros(len(mkt), dtype=int)
        for s in strata:
            idx = mkt.index[(mkt["market_size_group"] == s) & (mkt["eligible"] == 1)].to_numpy()
            n = len(idx)
            if n < 2:
                continue
            n_t = int(np.round(test_frac * n))
            n_t = max(1, min(n_t, n-1))
            chosen = np.random.choice(idx, size=n_t, replace=False)
            is_test[chosen] = 1
        is_test[mkt.index[mkt["eligible"] == 0]] = 0
        return is_test

    best_score, best_is_test, best_diag = np.inf, None, {}

    for _ in range(max_tries):
        is_test = propose()
        x = mkt["historical_sales_pre"].values
        smd_overall = abs(standardized_mean_diff(x[is_test == 1], x[is_test == 0]))

        smd_by = {}
        ok = smd_overall <= eps
        for s in strata:
            mask = (mkt["market_size_group"] == s)
            xs, ts = x[mask], is_test[mask]
            if (ts == 1).sum() < 2 or (ts == 0).sum() < 2:
                continue
            smd_s = abs(standardized_mean_diff(xs[ts == 1], xs[ts == 0]))
            smd_by[s] = smd_s
            ok = ok and (smd_s <= eps)

        score = smd_overall + sum(max(0, v - eps) for v in smd_by.values())
        if score < best_score:
            best_score, best_is_test = score, is_test.copy()
            best_diag = {"smd_overall": smd_overall, **{f"smd_{k}": v for k, v in smd_by.items()}}

        if ok:
            out = mkt.copy()
            out["is_test"] = is_test
            return broadcast_market_assignment(df, out), best_diag

    out = mkt.copy()
    out["is_test"] = best_is_test if best_is_test is not None else 0
    return broadcast_market_assignment(df, out), best_diag
