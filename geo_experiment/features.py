'''**What this does:**
Builds the tables your estimators want:

1. `build_agg_timeseries(df)`
    Creates daily aggregated series for plotting (test vs control):
    
- sales, spend, ground-truth lift, spillover
1. `build_wide_market_prepost(df)`
    Creates a market-level table (`wide`) with:
- `sales_pre`, `sales_post`
- `macro_pre`, `macro_post`
- `hist`, `size`, `spill`, `is_test`
- `gt_lift_post`, `gt_spill_post` (for simulation evaluation only)

**Why this exists:**
Most estimators are easier to run on a clean “wide” dataset.
'''
import numpy as np
import pandas as pd

# Aggregates granular daily data into a global time-series view split by test assignment.
# This view is primarily used for "Counterfactual" plotting, allowing for a side-by-side 
# visual comparison of total Sales, Spend, and Ground Truth effects over the 
# entire duration of the experiment.
def build_agg_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["date", "is_post", "is_test"], as_index=False)
          .agg(
              sales=("sales", "sum"),
              spend=("paid_social_spend", "sum"),
              gt_lift=("ground_truth_lift", "sum"),
              gt_spill=("ground_truth_spillover", "sum"),
          )
    )

# Summarizes the experiment into a single row per market, pivoting the time dimension 
# into distinct "Pre" and "Post" period columns.
# 1. Daily Aggregation: Collapses unit-level data into market-level daily totals.
# 2. Period Splitting: Separates the timeline to calculate metrics like 'sales_pre' 
#    and 'sales_post' independently.
# 3. Covariate Join: Merges market-level metadata (size, risk group, macro-trends) 
#    to create a feature set for regression-based incrementality models.
# 4. Ground Truth Alignment: Attaches the hidden simulation truths (lift and spillover) 
#    specifically for the post-period to enable final model validation and bias checking.
def build_wide_market_prepost(df: pd.DataFrame) -> pd.DataFrame:
    # daily market aggregates
    mkt_daily = (
        df.groupby(["market_id", "date"], as_index=False)
          .agg(
              sales=("sales", "sum"),
              sales_baseline=("sales_baseline", "sum"),
              macro=("macro_covariate_1", "mean"),
              is_post=("is_post", "first"),
              is_test=("is_test", "first"),
              hist=("historical_sales_pre", "first"),
              size=("market_size_group", "first"),
              spill=("spillover_risk_group", "first"),
          )
    )

    def sum_if(mask):
        return lambda x: x[mask.loc[x.index].values].sum()

    # safer explicit splits
    pre = mkt_daily[mkt_daily["is_post"] == 0]
    post = mkt_daily[mkt_daily["is_post"] == 1]

    pre_sum = pre.groupby("market_id")["sales"].sum().rename("sales_pre")
    post_sum = post.groupby("market_id")["sales"].sum().rename("sales_post")
    macro_pre = pre.groupby("market_id")["macro"].mean().rename("macro_pre")
    macro_post = post.groupby("market_id")["macro"].mean().rename("macro_post")

    base = (
        df[["market_id", "is_test", "spillover_risk_group", "market_size_group", "historical_sales_pre"]]
        .drop_duplicates()
        .rename(columns={
            "spillover_risk_group": "spill",
            "market_size_group": "size",
            "historical_sales_pre": "hist"
        })
        .set_index("market_id")
    )

    wide = base.join([pre_sum, post_sum, macro_pre, macro_post]).reset_index()

    gt = (
        df[df["is_post"] == 1]
        .groupby("market_id", as_index=False)
        .agg(
            gt_lift_post=("ground_truth_lift", "sum"),
            gt_spill_post=("ground_truth_spillover", "sum"),
        )
    )
    wide = wide.merge(gt, on="market_id", how="left")
    return wide