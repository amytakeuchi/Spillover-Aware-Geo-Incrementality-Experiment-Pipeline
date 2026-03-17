from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, List, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ..design import standardized_mean_diff
'''
**What this does:**

Checks if treated/control are comparable pre-treatment:
- SMD on `historical_sales_pre`
- treated/control size mix counts

**Why this exists:**
If balance fails, all causal estimates are suspect.
'''

def balance_smd_table(
    df: pd.DataFrame,
    smd_col: str = "historical_sales_pre",
    group_col: str = "market_size_group",
    treat_col: str = "is_test",
    market_col: str = "market_id",
    eps: float = 0.10,
) -> dict:
    """
    Returns a notebook-friendly balance summary.

    Outputs:
      - "overall": dict with treated/control counts + overall SMD
      - "size_mix": pd.DataFrame with size mix counts and shares by group
      - "smd_by_size": pd.DataFrame with SMD within each size stratum (optional but helpful)
    """
    # 1) Market-level dedupe
    cols = [market_col, treat_col, smd_col, group_col]
    mkt = df[cols].drop_duplicates().copy()

    # Basic counts
    treated = mkt[mkt[treat_col] == 1]
    control = mkt[mkt[treat_col] == 0]
    n_t = int(len(treated))
    n_c = int(len(control))

    # 2) Overall SMD (historical_sales_pre by default)
    t_vals = treated[smd_col].astype(float).values
    c_vals = control[smd_col].astype(float).values
    smd_overall = float(abs(standardized_mean_diff(t_vals, c_vals))) if (n_t > 1 and n_c > 1) else float("nan")

    # 3) Size mix table (counts + shares)
    mix = (
        mkt.groupby([treat_col, group_col], as_index=False)[market_col]
           .nunique()
           .rename(columns={market_col: "n_markets"})
    )
    mix["group"] = np.where(mix[treat_col] == 1, "Treated", "Control")

    # shares within treated/control
    mix["share_within_group"] = mix["n_markets"] / mix.groupby("group")["n_markets"].transform("sum")

    # pivot to a compact display
    size_mix_tbl = (
        mix.pivot(index=group_col, columns="group", values="n_markets")
           .fillna(0)
           .astype(int)
           .reset_index()
    )
    # add shares too (nice for “FAANG scan”)
    mix_share = (
        mix.pivot(index=group_col, columns="group", values="share_within_group")
           .fillna(0.0)
           .reset_index()
    )
    size_mix_tbl = size_mix_tbl.merge(mix_share, on=group_col, suffixes=("", "_share"))
    # columns: market_size_group, Treated, Control, Treated_share, Control_share
    if "Treated_share" not in size_mix_tbl.columns:
        size_mix_tbl["Treated_share"] = 0.0
    if "Control_share" not in size_mix_tbl.columns:
        size_mix_tbl["Control_share"] = 0.0

    # 4) SMD within size strata (optional but very useful for balance narrative)
    rows = []
    for s, g in mkt.groupby(group_col):
        t_s = g[g[treat_col] == 1][smd_col].astype(float).values
        c_s = g[g[treat_col] == 0][smd_col].astype(float).values
        if len(t_s) > 1 and len(c_s) > 1:
            smd_s = float(abs(standardized_mean_diff(t_s, c_s)))
        else:
            smd_s = float("nan")
        rows.append(
            {
                group_col: s,
                "n_treated": int((g[treat_col] == 1).sum()),
                "n_control": int((g[treat_col] == 0).sum()),
                "smd": smd_s,
                "passes_eps": bool(np.isfinite(smd_s) and (smd_s <= eps)),
            }
        )
    smd_by_size_tbl = pd.DataFrame(rows).sort_values(group_col).reset_index(drop=True)

    # 5) “Balance Table” (single-row summary for notebook)
    balance_summary_tbl = pd.DataFrame(
        [
            {
                "n_treated_markets": n_t,
                "n_control_markets": n_c,
                f"smd_{smd_col}": smd_overall,
                "passes_eps": bool(np.isfinite(smd_overall) and (smd_overall <= eps)),
                "eps_threshold": eps,
            }
        ]
    )

    return balance_summary_tbl, size_mix_tbl, smd_by_size_tbl
'''
    {
        # Keep backward-compatible keys if your notebook expects them
        "smd_hist_sales": smd_overall,
        "treated_count": n_t,
        "control_count": n_c,
        "treated_size_mix": treated[group_col].value_counts().to_dict(),
        "control_size_mix": control[group_col].value_counts().to_dict(),

        # New notebook-friendly outputs
        "balance_table": balance_summary_tbl,   # ✅ 2.2 Balance Table (overall)
        "size_mix_table": size_mix_tbl,         # ✅ size mix treated vs control (counts + shares)
        "smd_by_size": smd_by_size_tbl,         # ✅ optional but strong: SMD by stratum
    }
    '''
def _ensure_market_level_features(
    df: pd.DataFrame,
    cols: List[str],
    market_col: str,
    treat_col: str,
    post_col: str,
) -> pd.DataFrame:
    """
    Ensure we have ONE row per market with columns in `cols` and treatment assignment.
    If df is long (market-date), we aggregate pre-period mean by market for cols.

    Assumes treatment assignment is constant within market.
    """
    # already market-level?
    is_market_level = (df[market_col].nunique() == len(df)) and (post_col not in df.columns or df[post_col].nunique() == 1)

    if is_market_level and all(c in df.columns for c in cols):
        out = df[[market_col, treat_col] + cols].copy()
        return out.drop_duplicates(subset=[market_col])

    # otherwise: long panel -> aggregate PRE only
    if post_col not in df.columns:
        raise ValueError(f"`{post_col}` column not found; cannot aggregate pre-period features from long panel.")

    pre = df[df[post_col] == 0].copy()

    # keep a stable treatment flag per market
    treat_by_market = (
        df[[market_col, treat_col]]
        .drop_duplicates(subset=[market_col])
        .set_index(market_col)[treat_col]
    )

    agg = (
        pre.groupby(market_col, as_index=False)[cols]
           .mean(numeric_only=True)
    )

    agg[treat_col] = agg[market_col].map(treat_by_market).astype(int)
    return agg[[market_col, treat_col] + cols]

def love_plot_smd(
    df: pd.DataFrame,
    cols: Iterable[str],
    *,
    market_col: str = "market_id",
    treat_col: str = "is_test",
    post_col: str = "is_post",
    threshold: float = 0.10,
    use_abs: bool = True,
    sort: bool = True,
    max_features: Optional[int] = None,
    title: str = "Love Plot (Standardized Mean Differences)",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (9, 5),
    return_table: bool = True,
) -> Tuple[plt.Axes, Optional[pd.DataFrame]]:
    """
    Create a love plot for balance diagnostics using Standardized Mean Differences (SMD).

    Parameters
    ----------
    df : pd.DataFrame
        Either:
          - long panel with (market_id, date, is_post, is_test, feature cols), OR
          - market-level wide table with one row per market (recommended).
    cols : Iterable[str]
        Feature columns to compute SMD over (pre-period market-level features).
    market_col, treat_col, post_col : str
        Column names.
    threshold : float
        Common balance threshold; 0.10 is typical.
    use_abs : bool
        Plot absolute SMD (recommended for balance).
    sort : bool
        Sort features by |SMD| descending (recommended).
    max_features : Optional[int]
        If set, plot only top K worst-balanced features.
    ax : Optional[plt.Axes]
        Pass an axes to draw on; otherwise creates a new figure.
    return_table : bool
        If True, also returns a DataFrame with means and SMDs.

    Returns
    -------
    ax : matplotlib Axes
    table : Optional[pd.DataFrame]
        Columns: feature, mean_treated, mean_control, smd, abs_smd
    """
    cols = list(cols)
    if len(cols) == 0:
        raise ValueError("`cols` must contain at least one feature column.")

    # Market-level features (pre means if df is long)
    mkt = _ensure_market_level_features(df, cols, market_col, treat_col, post_col)

    treated = mkt[mkt[treat_col] == 1]
    control = mkt[mkt[treat_col] == 0]

    rows = []
    for c in cols:
        if c not in mkt.columns:
            continue
        xt = treated[c].to_numpy()
        xc = control[c].to_numpy()
        smd = standardized_mean_diff(xt.astype(float), xc.astype(float))
        rows.append({
            "feature": c,
            "mean_treated": float(np.nanmean(xt)) if len(xt) else np.nan,
            "mean_control": float(np.nanmean(xc)) if len(xc) else np.nan,
            "smd": float(smd) if np.isfinite(smd) else np.nan,
            "abs_smd": float(abs(smd)) if np.isfinite(smd) else np.nan,
        })

    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No valid features were found to compute SMDs.")

    # decide plotted values
    plot_val_col = "abs_smd" if use_abs else "smd"

    if sort:
        table = table.sort_values(plot_val_col, ascending=False, na_position="last")

    if max_features is not None:
        table = table.head(int(max_features)).copy()

    # Create axes if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    y = np.arange(len(table))
    x = table[plot_val_col].to_numpy()

    ax.scatter(x, y)
    ax.axvline(threshold, linestyle="--")
    ax.axvline(-threshold, linestyle="--") if not use_abs else None
    ax.axvline(0.0, linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(table["feature"].tolist())
    ax.invert_yaxis()  # top = worst balance if sorted
    ax.set_xlabel("|SMD|" if use_abs else "SMD")
    ax.set_title(title)

    # Optional: annotate values
    for xi, yi in zip(x, y):
        if np.isfinite(xi):
            ax.text(xi, yi, f" {xi:.3f}", va="center", fontsize=9)

    ax.grid(True, axis="x", linestyle=":")

    return ax, (table if return_table else None)