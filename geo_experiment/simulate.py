'''
**What this does:**
Builds the **baseline world** (the “no-treatment universe”):
- assigns markets to *size groups and spillover risk groups*
- creates daily dates and market×date panel
- simulates baseline sales with trend + weekly seasonality + macro covariate + AR noise
- creates a “historical pre sales” feature used for balance/design
- initializes placeholders (`is_test`, spend, lift, spillover, final sales)

**Why this exists:**
Separates “data generation” from “treatment assignment” and “effect injection.”

**Output:**
A panel dataframe `df` with:
- `sales_baseline` filled
- treatment/spend/lift placeholders set to 0
'''

import numpy as np
import pandas as pd
from .config import SimConfig

def _assign_market_groups(n_markets: int) -> pd.DataFrame:
    market_ids = np.arange(1, n_markets + 1)
    sizes = np.random.choice(["Small", "Medium", "Large"], size=n_markets, p=[0.4, 0.4, 0.2])
    spill_groups = np.random.choice(
        ["Core Market", "Buffer Zone", "Control Guardrail"],
        size=n_markets,
        p=[0.55, 0.25, 0.20],
    )
    return pd.DataFrame({"market_id": market_ids, "market_size_group": sizes, "spillover_risk_group": spill_groups})

def simulate_geo_panel(cfg: SimConfig, start_date: str = "2025-01-01") -> pd.DataFrame:
    markets = _assign_market_groups(cfg.n_markets)

    start = pd.Timestamp(start_date)
    dates = pd.date_range(start, periods=cfg.n_days, freq="D")
    df = markets.merge(pd.DataFrame({"date": dates}), how="cross")

    df["day_index"] = (df["date"] - df["date"].min()).dt.days
    df["is_post"] = (df["day_index"] >= cfg.pre_days).astype(int)
    df["period"] = np.where(df["is_post"] == 1, "Treatment-Period", "Pre-Period")

    macro_base = 1.0 + 0.03 * np.sin(2 * np.pi * df["day_index"] / 30.0)
    market_macro_shift = df["market_id"].map({m: np.random.normal(0, 0.02) for m in df["market_id"].unique()})
    df["macro_covariate_1"] = macro_base + market_macro_shift + np.random.normal(0, 0.02, size=len(df))

    dow = df["date"].dt.dayofweek.values
    weekly = 1.0 + cfg.weekly_seasonality_amp * np.sin(2 * np.pi * dow / 7.0)
    trend = 1.0 + cfg.trend_per_day * df["day_index"].values

    size_mult = df["market_size_group"].map(cfg.market_size_multipliers).astype(float).values
    market_level = df["market_id"].map(
        {m: np.random.lognormal(mean=8.2, sigma=0.25) for m in df["market_id"].unique()}
    ).values

    eps = np.zeros(len(df))
    for m in df["market_id"].unique():
        idx = df.index[df["market_id"] == m].to_numpy()
        e = np.zeros(len(idx))
        for t in range(len(idx)):
            innov = np.random.normal(0, cfg.shock_sd)
            e[t] = cfg.ar1_rho * (e[t-1] if t > 0 else 0.0) + innov
        eps[idx] = e
    noise_mult = np.exp(eps)

    base_sales = (
        market_level * size_mult * weekly * trend
        * (1.0 + 0.15*(df["macro_covariate_1"].values - 1.0))
        * noise_mult
    )
    df["sales_baseline"] = base_sales

    pre_mask = df["is_post"] == 0
    hist = (
        df.loc[pre_mask].groupby("market_id")["sales_baseline"].mean()
        .rename("historical_sales_pre").reset_index()
    )
    hist["historical_sales_pre"] *= np.random.normal(1.0, 0.05, size=len(hist))
    df = df.merge(hist, on="market_id", how="left")

    # placeholders
    df["is_test"] = 0
    df["treatment_level"] = 0.0
    df["paid_social_spend"] = 0.0
    df["ground_truth_lift"] = 0.0
    df["ground_truth_spillover"] = 0.0
    df["sales"] = df["sales_baseline"]

    return df