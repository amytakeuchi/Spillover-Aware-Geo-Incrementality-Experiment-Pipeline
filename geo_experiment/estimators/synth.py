"""
Synthetic Control (aggregate SCM) utilities.

This module provides:
- aggregate_synth_control: fit weights on pre-period and estimate total post lift
- synth_leave_one_out_sensitivity: donor leave-one-out sensitivity (trust metric)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# -----------------------------
# Helpers: constrained weights
# -----------------------------
def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """
    Enforces the "Physical Reality" of the weights.
    What it does: It takes a vector v of raw numbers and "projects" them so they satisfy two rules:
    every weight must be non-negative (w >= 0) and the total must sum to 1 (sum(w) = 1).

    Why it matters: In real-world marketing, we don't want a "Synthetic Chicago" that is made of
    -50% of New York and 150% of LA.

    Project vector v onto the probability simplex:
        w >= 0, sum(w) = 1
    """
    v = v.astype(float)
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho = np.nonzero(u - cssv / (np.arange(n) + 1) > 0)[0]
    if len(rho) == 0:
        return np.ones(n) / n
    rho = int(rho[-1])
    theta = float(cssv[rho] / (rho + 1.0))
    w = np.maximum(v - theta, 0.0)
    s = float(w.sum())
    return (np.ones(n) / n) if s <= 0 else (w / s)


def constrained_weights_pg(
    y: np.ndarray,
    X: np.ndarray,
    lr: float = 5e-7,
    n_iter: int = 12000,
) -> np.ndarray:
    """
    The "Engine" that finds the best match.
    What it does: It uses Projected Gradient Descent to solve an optimization problem.
    It iteratively adjusts the weights of the donor cities (X) until their combined
    trend line matches the treated city's trend (y) as closely as possible during the pre-campaign period.

    Why it matters: This avoids standard OLS regression, which can "overfit" and produce
    nonsensical negative weights. It finds the "best fit" within the constraints defined by the simplex.

    Projected-gradient solution for:
        min_w || y - X w ||^2  s.t. w in simplex

    Args:
      y: (T,) treated aggregate series in PRE — should be scaled before passing in
      X: (T, K) donor matrix in PRE — should be column-scaled before passing in
      lr: learning rate
      n_iter: number of iterations

    Returns:
      w: (K,) donor weights on simplex
    """
    y = y.astype(float)
    X = X.astype(float)
    K = int(X.shape[1])
    w = np.ones(K) / K

    for _ in range(int(n_iter)):
        r = y - (X @ w)
        grad = -2.0 * (X.T @ r)
        w = project_to_simplex(w - lr * grad)

    return w


def _safe_series_by_date(
    df: pd.DataFrame,
    market_id: int,
    date_index: pd.Index,
    outcome_col: str,
) -> np.ndarray:
    """
    Build an aligned daily series for a market_id, reindexed to date_index.
    Missing dates are filled with 0.
    """
    s = (
        df.loc[df["market_id"] == market_id]
        .groupby("date")[outcome_col]
        .sum()
        .reindex(date_index)
        .fillna(0.0)
    )
    return s.values.astype(float)


# -----------------------------------------
# Aggregate Synthetic Control (SCM) estimator
# -----------------------------------------
def aggregate_synth_control(
    df: pd.DataFrame,
    donor_markets: list[int] | None = None,
    constrained: bool = True,
    l2: float = 1e-3,
    outcome_col: str = "sales",
    top_k_donors: int | None = None,
) -> dict:
    """
    Aggregate SCM with per-donor scaling to fix scale-mismatch bias.

    Key changes vs original:
      1. Per-donor column scaling: each donor series is normalised to its own
         pre-period mean before weight fitting, then the scale is restored when
         applying weights to the post-period. This prevents large-pool donors
         from dominating the simplex optimisation and producing an inflated
         counterfactual.
      2. Treated series scaling: yT_pre is also normalised to unit mean so the
         optimiser operates on dimensionless residuals, making the learning rate
         and convergence independent of the outcome's absolute magnitude.
      3. RMSPE/lift ratio: returned as `rmspe_lift_ratio` — the primary
         trust gate. Values above 0.25 should be flagged; above 0.50 the
         estimate should not be reported as primary.
      4. Optional top_k_donors pruning: after fitting, drop low-weight donors
         (weight ≈ 0) and refit with only the top-k. This reduces noise from
         donors that contribute nothing but dilute the optimisation signal.

    Required df columns:
      - date
      - market_id
      - is_test   (1 = treated markets, 0 = controls)
      - is_post   (1 = post, 0 = pre)
      - outcome_col (default "sales")

    Args:
      df:            long panel DataFrame
      donor_markets: list of market_ids to use as donors. If None, uses all
                     controls found in the PRE period.
      constrained:   if True, solve weights on simplex (w>=0, sum=1).
                     If False, use ridge/OLS (may produce negative weights).
      l2:            ridge regularisation for unconstrained weights.
      outcome_col:   outcome column name.
      top_k_donors:  if set, refit using only the top-k donors by weight after
                     the initial fit. Useful when you have many donors and want
                     to remove near-zero contributors. Ignored if None.

    Returns:
      dict with:
        - lift_hat_total      : total post-period lift estimate
        - prefit_rmspe        : RMSPE of synthetic vs treated in the PRE period
                                (same units as outcome_col)
        - rmspe_lift_ratio    : prefit_rmspe / |lift_hat_total|
                                  < 0.10  → good
                                  0.10–0.25 → marginal
                                  0.25–0.50 → poor
                                  > 0.50  → do not report as primary estimate
        - n_donors            : number of donors used in the final fit
        - w_min, w_max, w_sum : weight diagnostics
        - weights             : DataFrame(market_id, w) sorted descending
        - col_means           : per-donor pre-period means used for scaling
                                (store these if you want to apply the same
                                 scaling to a held-out period)
        - yT_pre, yS_pre      : Series for pre-period plot / debug
        - yT_post, yS_post    : Series for post-period plot / debug
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    for col in ("date", "market_id", "is_test", "is_post"):
        if col not in df.columns:
            raise ValueError(f"df must include column '{col}'")
    if outcome_col not in df.columns:
        raise ValueError(f"df must include outcome column '{outcome_col}'")

    pre  = df[df["is_post"] == 0].copy()
    post = df[df["is_post"] == 1].copy()

    # ------------------------------------------------------------------
    # Donor pool
    # ------------------------------------------------------------------
    if donor_markets is None:
        donor_markets = sorted(
            pre.loc[pre["is_test"] == 0, "market_id"].unique().tolist()
        )
    donor_markets = [int(m) for m in donor_markets]
    if len(donor_markets) == 0:
        raise ValueError("donor_markets is empty; cannot fit synth control")

    # ------------------------------------------------------------------
    # Treated aggregate — PRE
    # ------------------------------------------------------------------
    yT_pre = (
        pre.loc[pre["is_test"] == 1]
        .groupby("date")[outcome_col]
        .sum()
        .sort_index()
    )
    if yT_pre.empty:
        raise ValueError(
            "No treated observations found in PRE period (is_test==1 & is_post==0)."
        )
    pre_dates = yT_pre.index

    # ------------------------------------------------------------------
    # Donor matrix — PRE
    # ------------------------------------------------------------------
    X_pre = np.column_stack([
        _safe_series_by_date(pre, m, pre_dates, outcome_col)
        for m in donor_markets
    ])  # shape: (T_pre, K)

    # ------------------------------------------------------------------
    # FIX 1: Per-donor column scaling
    #
    # Each donor column is divided by its own pre-period mean so every
    # column has mean ≈ 1. This prevents donors with large absolute sales
    # from dominating the gradient and compressing weights for smaller
    # donors toward zero (which then gets amplified when weights are
    # applied back to unscaled post-period series).
    #
    # The treated series is also scaled to its own mean so the optimiser
    # works on dimensionless residuals — this makes lr / convergence
    # behaviour robust to the absolute magnitude of the outcome.
    # ------------------------------------------------------------------
    col_means = X_pre.mean(axis=0)                        # (K,)
    col_means = np.where(col_means == 0, 1.0, col_means)  # guard divide-by-zero

    X_pre_scaled = X_pre / col_means                      # (T_pre, K)

    yT_mean = float(yT_pre.values.mean())
    yT_mean = yT_mean if yT_mean != 0.0 else 1.0
    yT_pre_scaled = yT_pre.values / yT_mean               # (T_pre,)

    # ------------------------------------------------------------------
    # Fit weights on scaled series
    # ------------------------------------------------------------------
    if constrained:
        w = constrained_weights_pg(yT_pre_scaled, X_pre_scaled)
    else:
        XtX = X_pre_scaled.T @ X_pre_scaled
        Xty = X_pre_scaled.T @ yT_pre_scaled
        w = np.linalg.solve(XtX + l2 * np.eye(XtX.shape[0]), Xty)

    # ------------------------------------------------------------------
    # FIX 2: Optional top-k donor pruning
    #
    # After the initial fit, donors with near-zero weight contribute noise
    # but no signal. Dropping them and refitting concentrates the
    # optimisation on the donors that actually matter, typically improving
    # both RMSPE and stability.
    # ------------------------------------------------------------------
    if top_k_donors is not None and int(top_k_donors) < len(donor_markets):
        top_k_donors = int(top_k_donors)
        top_idx = np.argsort(w)[::-1][:top_k_donors]
        donor_markets = [donor_markets[i] for i in top_idx]
        X_pre_scaled  = X_pre_scaled[:, top_idx]
        col_means      = col_means[top_idx]

        if constrained:
            w = constrained_weights_pg(yT_pre_scaled, X_pre_scaled)
        else:
            XtX = X_pre_scaled.T @ X_pre_scaled
            Xty = X_pre_scaled.T @ yT_pre_scaled
            w = np.linalg.solve(XtX + l2 * np.eye(XtX.shape[0]), Xty)

    # ------------------------------------------------------------------
    # Pre-period fit quality — computed in ORIGINAL units
    #
    # Multiply scaled synthetic back by yT_mean so RMSPE is interpretable
    # in the same units as the outcome (e.g. sales dollars / units).
    # ------------------------------------------------------------------
    yS_pre_vals = (X_pre_scaled @ w) * yT_mean           # (T_pre,) in original units
    prefit_rmspe = float(
        np.sqrt(np.mean((yT_pre.values - yS_pre_vals) ** 2))
    )

    # ------------------------------------------------------------------
    # Treated aggregate — POST
    # ------------------------------------------------------------------
    yT_post = (
        post.loc[post["is_test"] == 1]
        .groupby("date")[outcome_col]
        .sum()
        .sort_index()
    )
    if yT_post.empty:
        raise ValueError(
            "No treated observations found in POST period (is_test==1 & is_post==1)."
        )
    post_dates = yT_post.index

    # ------------------------------------------------------------------
    # Donor matrix — POST, scaled with PRE col_means
    #
    # IMPORTANT: we reuse col_means from the PRE period — not the post-
    # period means — so we apply exactly the same transformation that was
    # used during fitting.
    # ------------------------------------------------------------------
    X_post = np.column_stack([
        _safe_series_by_date(post, m, post_dates, outcome_col)
        for m in donor_markets
    ])  # shape: (T_post, K)

    X_post_scaled = X_post / col_means                    # (T_post, K)

    # Counterfactual in original units
    yS_post = (X_post_scaled @ w) * yT_mean              # (T_post,)

    # ------------------------------------------------------------------
    # Total post-period lift
    # ------------------------------------------------------------------
    lift_hat_total = float((yT_post.values - yS_post).sum())

    # ------------------------------------------------------------------
    # FIX 3: RMSPE / |lift| ratio — primary trust gate
    #
    # Thresholds (add to your validity logic):
    #   < 0.10  → good
    #   0.10–0.25 → marginal, report with caution
    #   0.25–0.50 → poor, consider donor filtering
    #   > 0.50  → do not report SCM as a primary estimate
    # ------------------------------------------------------------------
    denom_lift = max(abs(lift_hat_total), 1e-9)
    rmspe_lift_ratio = float(prefit_rmspe / denom_lift)

    # ------------------------------------------------------------------
    # Weights table
    # ------------------------------------------------------------------
    weights_df = (
        pd.DataFrame({"market_id": donor_markets, "w": w})
        .sort_values("w", ascending=False)
        .reset_index(drop=True)
    )

    return {
        # --- primary output ---
        "lift_hat_total":   lift_hat_total,
        # --- trust metrics ---
        "prefit_rmspe":     prefit_rmspe,
        "rmspe_lift_ratio": rmspe_lift_ratio,   # KEY: use this as validity gate
        # --- weight diagnostics ---
        "n_donors":         int(len(donor_markets)),
        "w_min":            float(np.min(w)),
        "w_max":            float(np.max(w)),
        "w_sum":            float(np.sum(w)),
        "weights":          weights_df,
        # --- scaling artefact (store for reproducibility) ---
        "col_means":        col_means,
        "yT_mean":          yT_mean,
        # --- series for plots / debug ---
        "yT_pre":  yT_pre,
        "yS_pre":  pd.Series(yS_pre_vals, index=pre_dates,  name="yS_pre"),
        "yT_post": yT_post,
        "yS_post": pd.Series(yS_post,     index=post_dates, name="yS_post"),
    }


# -----------------------------------------
# Trust metric: Leave-one-out donor sensitivity
# -----------------------------------------
def synth_leave_one_out_sensitivity(
    df: pd.DataFrame,
    donor_markets: list[int] | None = None,
    constrained: bool = True,
    max_drop: int | None = None,
    outcome_col: str = "sales",
    top_k_donors: int | None = None,
) -> dict:
    """
    Leave-one-out donor sensitivity for aggregate SCM.
    Returns a compact table + stability metrics.

    Notes:
      - This computes SCM repeatedly, each time dropping one donor.
      - Use max_drop to cap donors (speed) if you have many controls.
      - top_k_donors is forwarded to aggregate_synth_control.

    Returns:
      dict with:
        - lift_full
        - loo_std
        - loo_range
        - loo_rel_std (recommended trust metric)
        - loo_table (DataFrame)
    """
    if donor_markets is None:
        pre = df[df["is_post"] == 0]
        donor_markets = sorted(
            pre.loc[pre["is_test"] == 0, "market_id"].unique().tolist()
        )

    donor_markets = [int(m) for m in donor_markets]

    if max_drop is not None and len(donor_markets) > int(max_drop):
        donor_markets = donor_markets[: int(max_drop)]

    full = aggregate_synth_control(
        df,
        donor_markets=donor_markets,
        constrained=constrained,
        outcome_col=outcome_col,
        top_k_donors=top_k_donors,
    )
    lift_full = float(full["lift_hat_total"])

    rows = []
    for dm in donor_markets:
        donors_loo = [d for d in donor_markets if d != dm]
        if len(donors_loo) == 0:
            continue
        out = aggregate_synth_control(
            df,
            donor_markets=donors_loo,
            constrained=constrained,
            outcome_col=outcome_col,
            top_k_donors=top_k_donors,
        )
        rows.append({
            "dropped_donor":  int(dm),
            "lift_hat_total": float(out["lift_hat_total"]),
            "prefit_rmspe":   float(out["prefit_rmspe"]),
        })

    loo    = pd.DataFrame(rows)
    lifts  = loo["lift_hat_total"].values.astype(float) if len(loo) else np.array([], dtype=float)

    loo_std   = float(np.std(lifts, ddof=1)) if len(lifts) > 1 else float("nan")
    loo_min   = float(np.min(lifts))         if len(lifts) else float("nan")
    loo_max   = float(np.max(lifts))         if len(lifts) else float("nan")
    loo_range = float(loo_max - loo_min)     if len(lifts) else float("nan")

    denom        = max(abs(lift_full), 1e-9)
    loo_rel_std  = float(loo_std / denom) if np.isfinite(loo_std) else float("nan")

    return {
        "lift_full":    lift_full,
        "loo_std":      loo_std,
        "loo_range":    loo_range,
        "loo_rel_std":  loo_rel_std,   # <-- trust metric
        "loo_table":    (
            loo.sort_values("lift_hat_total").reset_index(drop=True)
            if len(loo) else loo
        ),
    }