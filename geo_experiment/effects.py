'''**What it does:**
Implements the **treatment reality**:
- generates paid social spend in treated markets during post only
- applies the true nonlinear response curve to generate `ground_truth_lift`
- spreads a fraction of lift to neighboring markets (spillover) but **only if the recipient is Buffer Zone**
- produces final observed `sales = baseline + lift + spillover + measurement noise`

**Why it exists:**
This is the “physics engine” of the simulation: it defines the truth.

**Outputs:**
`df` now contains:
- `paid_social_spend`
- `ground_truth_lift`
- `ground_truth_spillover`
- `sales` (observed)
'''
import numpy as np
import pandas as pd
from .config import SimConfig

# Defines the geographic topology of the simulation by creating a synthetic "ring" of markets.
# Each market is linked to its nearest neighbors based on its ID position.
# This structure provides a deterministic way to model proximity, which is essential 
# for simulating spatial spillover and contamination effects between test and control areas.
def build_ring_neighbors(market_ids: np.ndarray, k: int = 2) -> dict[int, list[int]]:
    mids = np.array(sorted(market_ids))
    n = len(mids)
    pos = {m: i for i, m in enumerate(mids)}
    neigh = {}
    for m in mids:
        i = pos[m]
        ns = []
        for step in range(1, k + 1):
            ns.append(mids[(i + step) % n])
            ns.append(mids[(i - step) % n])
        neigh[m] = ns
    return neigh

# This function serves as the "Ground Truth" engine for the simulation. 
# It transforms experimental assignments into financial reality by calculating 
# noisy ad spend, translating that spend into incremental sales via a non-linear 
# saturation curve, and modeling geographic contamination (spillover) before 
# adding final measurement error to the observed sales data.
def apply_spend_and_effects(df: pd.DataFrame, cfg: SimConfig, neighbors_k: int = 2) -> pd.DataFrame:
    out = df.copy()

    mkt = out[["market_id", "market_size_group", "spillover_risk_group"]].drop_duplicates().copy()
    mkt["size_mult"] = mkt["market_size_group"].map(cfg.market_size_multipliers).astype(float)
    neigh = build_ring_neighbors(mkt["market_id"].values, k=neighbors_k)

    out = out.merge(mkt[["market_id", "size_mult"]], on="market_id", how="left")

    post_mask = out["is_post"].eq(1)
    treat_mask = out["is_test"].eq(1) & post_mask

    # Acts as the "Physics Engine" of the simulation, translating experimental design into observed sales.
    # 1. Spend Generation: Calculates paid social spend for treated markets in the 'post' period, 
    #    adjusted by market size multipliers and lognormal noise to mimic real-world budget variance.
    spend_noise = np.random.lognormal(mean=0.0, sigma=cfg.spend_noise_sd, size=len(out))
    base = cfg.base_daily_spend * out["size_mult"].values
    out["paid_social_spend"] = 0.0
    out.loc[treat_mask, "paid_social_spend"] = base[treat_mask] * spend_noise[treat_mask]

    # 2. Response Curve: Applies a Diminishing Returns (Exponential) function to spend to derive 
    #    the ground truth lift, ensuring the simulation respects non-linear marketing saturation.
    out["ground_truth_lift"] = 0.0
    spend = out["paid_social_spend"].values
    treat_level = out["treatment_level"].values
    lift = cfg.lift_a * (1.0 - np.exp(-spend / cfg.lift_b)) * treat_level
    out.loc[treat_mask, "ground_truth_lift"] = lift[treat_mask]

    # 3. Spillover Logic: Leaks a fraction of the generated lift into neighboring "Buffer Zone" 
    #    markets. This simulates the reality where ad signals cross market boundaries.
    out["ground_truth_spillover"] = 0.0
    for date, g in out.loc[post_mask].groupby("date", sort=False):
        treated_today = g.loc[g["is_test"] == 1, ["market_id", "ground_truth_lift"]]
        if treated_today.empty:
            continue

        lift_by_market = dict(zip(treated_today["market_id"], treated_today["ground_truth_lift"]))
        idx_today = g.index.to_numpy()
        pos_in_today = {mid: j for j, mid in enumerate(g["market_id"].values)}
        spill_add = np.zeros(len(idx_today))

        for tm, lval in lift_by_market.items():
            leak = cfg.spillover_frac * lval
            nbs = neigh.get(tm, [])
            if not nbs:
                continue
            for nb in nbs:
                j = pos_in_today.get(nb, None)
                if j is None:
                    continue
                # recipients are Buffer Zone only (treated or control)
                if g.iloc[j]["spillover_risk_group"] == "Buffer Zone":
                    spill_add[j] += leak / len(nbs)

        out.loc[idx_today, "ground_truth_spillover"] = spill_add

    # 4. Final Observation: Combines baseline sales, lift, and spillover, then applies 
    #    multiplicative measurement noise to produce the final "Sales" column used for analysis.
    meas_noise = np.random.normal(0, 0.02, size=len(out))
    out["sales"] = (out["sales_baseline"] + out["ground_truth_lift"] + out["ground_truth_spillover"]) * (1.0 + meas_noise)

    return out