import pandas as pd

def spillover_contamination_table(df: pd.DataFrame) -> pd.DataFrame:
    # controls only, post only
    ctrl = df[(df["is_test"] == 0) & (df["is_post"] == 1)].copy()
    ctrl["excess_over_baseline"] = ctrl["sales"] - ctrl["sales_baseline"]

    out = (
        ctrl.groupby("spillover_risk_group", as_index=False)
            .agg(
                n_markets=("market_id", "nunique"),
                total_excess=("excess_over_baseline", "sum"),
                mean_excess=("excess_over_baseline", "mean"),
                gt_spill=("ground_truth_spillover", "sum"),
            )
            .sort_values("total_excess", ascending=False)
    )
    return out