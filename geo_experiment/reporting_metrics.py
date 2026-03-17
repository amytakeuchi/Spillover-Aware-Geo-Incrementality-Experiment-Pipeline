'''
**What this does:**
Transforms raw statistical "lift" estimates into business-facing KPIs like iROAS (incremental Return on Ad Spend), ROI, and Incremental CPA. It takes the abstract unit lift calculated by an estimator and applies financial variables (spend, margin, unit value) to determine the actual dollar impact of the experiment.

**Key Components:**
- `compute_experiment_totals`: Aggregates total spend and sales for treatment and control groups during the "post" period to provide the baseline for financial calculations.
- `reporting_metrics_from_lift`: The core engine that converts "lift units" into:
    - iROAS: Revenue generated per dollar spent ($Incremental Revenue / Spend$).
    - ROI: Net profit after accounting for gross margins and fixed costs.
    - CPA/CPIU: The efficiency cost of acquiring one incremental unit or conversion.

**Why this exists:**
While a statistical model might tell you "lift is 500 units," a marketing stakeholder needs to know if the campaign was profitable. This code bridge the gap between "statistical significance" and "economic viability."
'''

# 1) Core reporting metrics (lift, incremental revenue, iROAS, incremental CPA)
import numpy as np
import pandas as pd

def compute_experiment_totals(df: pd.DataFrame) ->dict:
    post_treat = (df["is_post"]==1) & (df["is_test"]==1)
    post_ctrl  = (df["is_post"]==1) & (df["is_test"]==0)

    return {
    "spend_treated_post":float(df.loc[post_treat,"paid_social_spend"].sum()),
    "sales_treated_post":float(df.loc[post_treat,"sales"].sum()),
    "sales_ctrl_post":float(df.loc[post_ctrl,"sales"].sum()),
    "n_treated_markets":int(df.loc[df["is_test"]==1,"market_id"].nunique()),
    "n_ctrl_markets":int(df.loc[df["is_test"]==0,"market_id"].nunique()),
    }

def reporting_metrics_from_lift(
    lift_hat_total: float,
    spend_total: float,
    unit_value: float = 1.0,
    gross_margin: float = 1.0,
    fixed_cost: float = 0.0,
    incremental_conversions: float | None = None,
) -> dict:
    
    inc_lift = float(lift_hat_total)
    inc_revenue = float(lift_hat_total * unit_value)
    inc_profit = float(inc_revenue * gross_margin)

    iROAS = np.nan if spend_total <= 0 else inc_revenue / spend_total
    ROI = np.nan if spend_total <= 0 else (inc_profit - spend_total - fixed_cost) / spend_total

    if incremental_conversions is None:
        # Added spaces around <= and else to fix the SyntaxWarning
        cpiu = np.nan if inc_lift <= 0 else spend_total / inc_lift
        cpa_inc = np.nan
    else:
        cpa_inc = np.nan if incremental_conversions <= 0 else spend_total / incremental_conversions
        cpiu = np.nan

    # INDENTED THIS BLOCK
    return {
        "incremental_lift_units": inc_lift,
        "incremental_revenue": inc_revenue,
        "incremental_profit_gross": inc_profit,
        "spend_total": float(spend_total),
        "iROAS": float(iROAS),
        "ROI": float(ROI),
        "cost_per_incremental_unit": float(cpiu),
        "CPA_incremental": float(cpa_inc),
    }

