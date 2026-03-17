"""
diagnostics/pretrend.py

**What this does:**
Implements lightweight "trust gates" for the Parallel Trends assumption *before* any causal estimation.
It produces (1) an aggregate pretrend plot, (2) small-multiple plots for individual treated markets vs matched
controls, and (3) a pre-period regression slope test on the Test-Control log-gap.

**Key Components:**
- `plot_pretrend_agg`: One high-signal chart comparing Test vs Control time series (optionally log-scale),
  with an optional pre-period log-gap line to visually inspect drift.
- `match_controls_by_pre`: Pairs top-K treated markets with nearest-neighbor control markets based on
  pre-period scale (from `wide`), so individual pretrend plots are interpretable without spamming 60 charts.
- `plot_pretrend_individual`: Small multiples (<=12 panels) showing treated market vs matched control mean in pre.
- `pretrend_slope_test`: A regression-based gate: in pre only, regress the log-gap on time; slope should be ~0.

**Why this exists:**
Geo incrementality lives or dies on the counterfactual. If pre-trends drift, your estimators will confidently
estimate bias. These checks make "design validity" explicit and reviewable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm


# -------------------------
# Helpers
# -------------------------

def _to_datetime(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.datetime64):
        return s
    return pd.to_datetime(s)


def _log_transform(x: pd.Series, log: bool = True, eps: float = 1.0) -> pd.Series:
    """
    Stable log transform. Uses log(x + eps) to avoid -inf.
    eps=1 is usually fine for sales-like outcomes.
    """
    if not log:
        return x.astype(float)
    return np.log(x.astype(float) + eps)


def _infer_treatment_start(df: pd.DataFrame, date_col: str = "date", post_col: str = "is_post") -> pd.Timestamp:
    """
    Finds earliest post date. Requires `is_post` to exist.
    """
    if post_col not in df.columns:
        raise ValueError(f"post_col='{post_col}' not found in df columns.")
    d = df.copy()
    d[date_col] = _to_datetime(d[date_col])
    post_dates = d.loc[d[post_col] == 1, date_col]
    if post_dates.empty:
        raise ValueError("Could not infer treatment start: no rows with is_post==1.")
    return post_dates.min()


# -------------------------
# 4.1 Aggregate Parallel Trends Plot
# -------------------------

def plot_pretrend_agg(
    agg: pd.DataFrame,
    metric: str = "sales",
    date_col: str = "date",
    treat_col: str = "is_test",
    post_col: str = "is_post",
    log: bool = True,
    show_pre_gap: bool = True,
    eps: float = 1.0,
    title: Optional[str] = None,
) -> None:
    """
    Aggregate parallel trends check (1 plot).

    Expected `agg` format: one row per (date, is_test[, is_post]) with `metric` aggregated (sum/mean).
    If your agg table doesn't have `is_post`, we will still plot lines but won't draw a treatment-start vline.

    Plot:
    - log(metric) for Test vs Control over time
    - treatment start vertical line if available
    - optional pre-period log-gap = log(Test) - log(Control) (pre only)
    """
    a = agg.copy()
    a[date_col] = _to_datetime(a[date_col])

    required = {date_col, treat_col, metric}
    missing = required - set(a.columns)
    if missing:
        raise ValueError(f"agg is missing required columns: {missing}")

    # Pivot to Test/Control series
    piv = (
        a.groupby([date_col, treat_col], as_index=False)[metric]
         .sum()
         .pivot(index=date_col, columns=treat_col, values=metric)
         .sort_index()
    )
    # column names: 0=control, 1=test
    if 0 not in piv.columns or 1 not in piv.columns:
        raise ValueError("Expected agg to contain both is_test==0 and is_test==1 rows.")

    y_c = _log_transform(piv[0], log=log, eps=eps)
    y_t = _log_transform(piv[1], log=log, eps=eps)

    plt.figure(figsize=(11, 4))
    plt.plot(piv.index, y_c.values, label="Control")
    plt.plot(piv.index, y_t.values, label="Test")
    plt.legend()
    plt.grid(alpha=0.2)

    # Treatment start vline if possible
    if post_col in a.columns:
        try:
            t0 = _infer_treatment_start(a, date_col=date_col, post_col=post_col)
            plt.axvline(t0, linestyle="--")
        except Exception:
            pass

    ttl = title or f"Parallel Trends (Aggregate){' [log]' if log else ''}"
    plt.title(ttl)
    plt.xlabel("date")
    plt.ylabel(f"{'log ' if log else ''}{metric}")
    plt.tight_layout()
    plt.show()

    # Optional: Pre-period log-gap plot (still "1 plot max" if you keep this OFF in notebooks)
    if show_pre_gap:
        if post_col not in a.columns:
            # fallback: compute on full series (user can interpret)
            pre_mask = np.ones(len(piv.index), dtype=bool)
        else:
            pre_dates = a.loc[a[post_col] == 0, date_col].unique()
            pre_mask = piv.index.isin(pre_dates)

        gap = (y_t - y_c).loc[pre_mask]

        plt.figure(figsize=(11, 3))
        plt.plot(gap.index, gap.values, label="log(Test) - log(Control) (pre)")
        plt.axhline(0.0, linewidth=1)
        plt.grid(alpha=0.2)
        plt.title("Pre-period log-gap (should be flat ~ 0 slope)")
        plt.xlabel("date")
        plt.ylabel("log-gap")
        plt.tight_layout()
        plt.show()


# -------------------------
# 4.2 Individual Markets: matching + small multiples
# -------------------------

def match_controls_by_pre(
    wide: pd.DataFrame,
    k_treated: int = 6,
    k_controls: int = 1,
    treated_col: str = "is_test",
    market_id_col: str = "market_id",
    pre_scale_col: str = "hist",  # in your screenshot wide has 'hist' (historical_sales_pre)
    control_pool_filter: Optional[pd.Series] = None,
) -> List[Tuple[int, List[int]]]:
    """
    Select top-K treated markets by pre_scale_col and match each to k_controls nearest control markets
    by absolute distance in pre_scale_col.

    Returns:
        pairs: [(treated_market_id, [control_market_ids...]), ...]
    """
    w = wide.copy()

    required = {treated_col, market_id_col, pre_scale_col}
    missing = required - set(w.columns)
    if missing:
        raise ValueError(f"wide is missing required columns: {missing}")

    treated = w[w[treated_col] == 1].sort_values(pre_scale_col, ascending=False).head(k_treated)
    controls = w[w[treated_col] == 0].copy()

    if control_pool_filter is not None:
        controls = controls.loc[control_pool_filter.reindex(controls.index, fill_value=True)]

    if controls.empty:
        raise ValueError("No control markets available for matching.")

    pairs: List[Tuple[int, List[int]]] = []
    for _, trow in treated.iterrows():
        t_id = int(trow[market_id_col])
        t_val = float(trow[pre_scale_col])

        controls["_dist"] = (controls[pre_scale_col].astype(float) - t_val).abs()
        matched = controls.sort_values("_dist").head(k_controls)[market_id_col].astype(int).tolist()
        pairs.append((t_id, matched))

    return pairs


def plot_pretrend_individual(
    df: pd.DataFrame,
    pairs: Sequence[Tuple[int, Sequence[int]]],
    outcome_col: str = "sales",
    date_col: str = "date",
    market_id_col: str = "market_id",
    post_col: str = "is_post",
    log: bool = True,
    eps: float = 1.0,
    max_panels: int = 12,
    ncols: int = 3,
    title: str = "Parallel Trends (Individual Markets, Pre Only)",
) -> None:
    """
    Small multiples: each panel shows
      treated market series vs mean(matched controls) series (pre period only).

    - Uses `df` daily panel because we need time series.
    - Caps panels to max_panels to avoid plot spam.
    """
    d = df.copy()
    d[date_col] = _to_datetime(d[date_col])

    required = {outcome_col, date_col, market_id_col}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")

    if post_col in d.columns:
        d = d[d[post_col] == 0].copy()  # pre only

    pairs = list(pairs)[:max_panels]
    n = len(pairs)
    if n == 0:
        raise ValueError("pairs is empty.")

    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.8 * ncols, 3.0 * nrows), sharex=False, sharey=False)
    axes = np.array(axes).reshape(-1)

    for ax_i, (t_id, c_ids) in enumerate(pairs):
        ax = axes[ax_i]

        td = d[d[market_id_col] == t_id].sort_values(date_col)
        cd = d[d[market_id_col].isin(list(c_ids))].copy()

        if td.empty or cd.empty:
            ax.set_title(f"mkt {t_id} (missing data)")
            ax.axis("off")
            continue

        # Control mean per date
        c_mean = (
            cd.groupby(date_col, as_index=False)[outcome_col]
              .mean()
              .sort_values(date_col)
        )

        y_t = _log_transform(td[outcome_col], log=log, eps=eps)
        y_c = _log_transform(c_mean[outcome_col], log=log, eps=eps)

        ax.plot(td[date_col].values, y_t.values, label=f"Treated {t_id}")
        ax.plot(c_mean[date_col].values, y_c.values, label=f"Control mean {list(c_ids)}", alpha=0.9)

        ax.set_title(f"Treated {t_id} vs Control {list(c_ids)}")
        ax.grid(alpha=0.2)
        if ax_i % ncols == 0:
            ax.set_ylabel(f"{'log ' if log else ''}{outcome_col}")
        ax.tick_params(axis="x", rotation=30)

    # Turn off unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(title, y=1.02, fontsize=12)
    plt.tight_layout()
    plt.show()


# -------------------------
# 4.3 Pretrend regression slope test (pre only)
# -------------------------

@dataclass
class PretrendTestResult:
    slope: float
    se: float
    p_value: float
    ci_lo: float
    ci_hi: float
    flag_pretrend: bool
    n_days: int


def pretrend_slope_test(
    df: pd.DataFrame,
    outcome_col: str = "sales",
    date_col: str = "date",
    treat_col: str = "is_test",
    post_col: str = "is_post",
    log: bool = True,
    eps: float = 1.0,
    alpha: float = 0.05,
    slope_tol: float = 0.0,
) -> PretrendTestResult:
    """
    Regression gate (1 table / dict in notebook).

    Approach (robust + simple):
    - Pre only
    - Aggregate daily total outcome for treated and control
    - Compute log-gap: g_t = log(Y_test_t) - log(Y_ctrl_t)
    - OLS: g_t ~ 1 + t
      -> slope should be ~0 if parallel trends holds.

    Returns:
        PretrendTestResult with slope, se, pval, 95% CI and flag.
    """
    d = df.copy()
    d[date_col] = _to_datetime(d[date_col])

    required = {outcome_col, date_col, treat_col}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"df missing required columns: {missing}")

    if post_col in d.columns:
        d = d[d[post_col] == 0].copy()  # pre only

    # daily totals by group
    daily = (
        d.groupby([date_col, treat_col], as_index=False)[outcome_col]
         .sum()
         .pivot(index=date_col, columns=treat_col, values=outcome_col)
         .sort_index()
    )
    if 0 not in daily.columns or 1 not in daily.columns:
        raise ValueError("Need both treated (is_test=1) and control (is_test=0) in pre period.")

    y_c = _log_transform(daily[0], log=log, eps=eps)
    y_t = _log_transform(daily[1], log=log, eps=eps)
    gap = (y_t - y_c).dropna()

    # time index
    t = np.arange(len(gap), dtype=float)
    X = sm.add_constant(t)
    model = sm.OLS(gap.values.astype(float), X).fit()

    slope = float(model.params[1])
    se = float(model.bse[1])
    pval = float(model.pvalues[1])
    ci_lo, ci_hi = map(float, model.conf_int(alpha=0.05)[1])

    # flag if slope significantly non-zero (and optionally exceeds tolerance)
    flag = (pval < alpha) and (abs(slope) > slope_tol)

    return PretrendTestResult(
        slope=slope,
        se=se,
        p_value=pval,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        flag_pretrend=bool(flag),
        n_days=int(len(gap)),
    )