'''
**What this does:**
Implements falsification tests ("placebos") to verify that estimated lift is driven by the true treatment — not by random noise, time trends, or model bias. 
These tests simulate scenarios where no real treatment effect should exist and confirm that the estimator returns ~0 lift under those conditions.

This is a causal validity stress test layer.

**Key Components:**
- `aa_test`: 
    Randomly reassigns treatment labels and re-estimates lift. 
    Since no real intervention occurred, estimated lift should be ~0. 
    Detects estimator bias or design imbalance.

- `placebo_in_time`: 
    Artificially shifts the treatment start date earlier into the pre-period and re-estimates lift. 
    If the model reports lift before the campaign actually began, it indicates trend contamination or model misspecification.

- `placebo_in_space`: 
    Randomly assigns placebo treatment to control markets (optionally guardrail-only) and estimates lift. 
    Detects spatial bias or spillover contamination effects.

- `placebo_summary`: 
    Produces a compact summary table and a single diagnostic plot (placebo-in-time curve) for fast review.

**Why this exists:**
In real-world incrementality measurement, many experiments fail not because the estimator is weak — but because the design is flawed.

If:
- AA tests show non-zero lift,
- Pre-period placebo shifts show significant lift,
- Random control reassignment produces lift,

then the experiment cannot be trusted.

These placebo tests act as "trust gates." 
If they fail, we do not ship lift — we redesign the experiment.

This mirrors how FAANG-level Measurement / Marketing Science teams validate causal claims before presenting results to executives.
'''

from __future__ import annotations

import numpy as np
import pandas as pd

from .inference import block_bootstrap_ci

# --- helpers ---------------------------------------------------------------

def _build_wide_prepost_from_panel(
    df: pd.DataFrame,
    is_post_col: str = "is_post",
    treat_col: str = "is_test",
    market_col: str = "market_id",
    y_col: str = "sales",
) -> pd.DataFrame:
    """Minimal wide builder for placebo modules (keeps this file self-contained)."""
    mkt = df[[market_col, treat_col]].drop_duplicates().copy()

    pre = df[df[is_post_col] == 0].groupby(market_col)[y_col].sum().rename("sales_pre")
    post = df[df[is_post_col] == 1].groupby(market_col)[y_col].sum().rename("sales_post")

    wide = mkt.set_index(market_col).join([pre, post]).reset_index()
    if not df.columns.is_unique:
        dupes = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names found: {dupes}")
    return wide


def _did_lift_total_like(wide: pd.DataFrame, treat_col: str = "is_test") -> float:
    """Simple DiD on wide (pre/post sums) returning total-like lift."""
    t = wide[wide[treat_col] == 1]
    c = wide[wide[treat_col] == 0]
    if len(t) == 0 or len(c) == 0:
        return float("nan")
    did = (t["sales_post"].mean() - t["sales_pre"].mean()) - (c["sales_post"].mean() - c["sales_pre"].mean())
    return float(did * len(t))


# --- 3.1 AA Test -----------------------------------------------------------

def aa_test_preperiod(
    df: pd.DataFrame,
    seed: int = 7,
    treat_frac: float = 0.35,
    market_col: str = "market_id",
    date_col: str = "date",
    y_col: str = "sales",
    is_post_col: str = "is_post",
) -> dict:
    """
    True AA: use PRE period only, then split PRE into pseudo-pre/pseudo-post.
    Randomize treatment labels. Expect ~0 lift.
    """
    rng = np.random.default_rng(seed)
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])

    # PRE only (true no-treatment world)
    pre = d[d[is_post_col] == 0].copy()
    if pre.empty:
        raise ValueError("No pre-period rows found for AA test.")

    # Choose a pseudo boundary inside pre (median date)
    cutoff = pre[date_col].quantile(0.5)
    pre["is_post_aa"] = (pre[date_col] >= cutoff).astype(int)

    # Randomize treated markets
    markets = pre[market_col].drop_duplicates().values
    n = len(markets)
    n_t = max(1, min(int(round(treat_frac * n)), n - 1))
    treated = set(rng.choice(markets, size=n_t, replace=False).tolist())
    pre["is_test_aa"] = pre[market_col].isin(treated).astype(int)

    wide = _build_wide_prepost_from_panel(
        pre,
        is_post_col="is_post_aa",
        treat_col="is_test_aa",
        market_col=market_col,
        y_col=y_col,
    )
    lift = _did_lift_total_like(wide, treat_col="is_test_aa")

    return {"lift_hat_total": float(lift), "n_treated": int(n_t), "n_markets": int(n), "cutoff_date": cutoff}


# --- 3.2 Placebo-in-time ---------------------------------------------------

def placebo_in_time(
    df: pd.DataFrame,
    shift_days_grid: list[int] | None = None,
    seed: int = 7,
    treat_col: str = "is_test",
    market_col: str = "market_id",
    date_col: str = "date",
    y_col: str = "sales",
    use_ci: bool = True,
    n_boot: int = 200,
    alpha: float = 0.05,
) -> dict:
    """
    Placebo-in-time (PRE ONLY):
    1) Find true post_start (first is_post==1 date)
    2) Truncate data to strictly pre (date < post_start)
    3) Within pre-only, create a fake post boundary shifted earlier and estimate lift
    """
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])

    # Identify the *actual* treatment start boundary: first date with is_post==1
    post_start = d.loc[d["is_post"] == 1, date_col].min()
    if pd.isna(post_start):
        raise ValueError("df has no is_post==1 rows; cannot run placebo_in_time().")

    # TRUNCATE to true PRE ONLY so placebo cannot see real treatment outcomes
    d = d[d[date_col] < post_start].copy()
    if d.empty:
        raise ValueError("After truncating to pre-only, no rows remain for placebo-in-time.")

    if shift_days_grid is None:
        shift_days_grid = [7, 14, 21, 28]

    rows = []
    for sd in shift_days_grid:
        placebo_start = post_start - pd.Timedelta(days=int(sd))
        d2 = d.copy()
        d2["is_post_placebo"] = (d2[date_col] >= placebo_start).astype(int)

        # point estimate
        wide_p = _build_wide_prepost_from_panel(
            d2,
            is_post_col="is_post_placebo",
            treat_col=treat_col,
            market_col=market_col,
            y_col=y_col,
        )
        point = _did_lift_total_like(wide_p, treat_col=treat_col)

        if use_ci:
            def est_fn(panel: pd.DataFrame) -> float:
                w = _build_wide_prepost_from_panel(
                    panel,
                    is_post_col="is_post_placebo",
                    treat_col=treat_col,
                    market_col=market_col,
                    y_col=y_col,
                )
                return _did_lift_total_like(w, treat_col=treat_col)

            ci = block_bootstrap_ci(d2, estimator_fn=est_fn, n_boot=n_boot, alpha=alpha, seed=seed)
            rows.append(
                {
                    "shift_days": int(sd),
                    "lift_hat_total": float(ci["point"]),
                    "ci_lo": ci["ci_lo"],
                    "ci_hi": ci["ci_hi"],
                    "n_ok": int(ci["n_ok"]),
                }
            )
        else:
            rows.append({"shift_days": int(sd), "lift_hat_total": float(point)})

    tbl = pd.DataFrame(rows).sort_values("shift_days").reset_index(drop=True)
    return {"table": tbl, "plot_df": tbl}

# --- 3.3 Placebo-in-space --------------------------------------------------

def placebo_in_space(
    df: pd.DataFrame,
    seed: int = 7,
    treat_frac: float | None = None,
    restrict_to_guardrail: bool = True,
    treat_col: str = "is_test",
    market_col: str = "market_id",
    spill_col: str = "spillover_risk_group",
    y_col: str = "sales",
    use_ci: bool = True,
    n_boot: int = 200,
    alpha: float = 0.05,
) -> dict:
    """
    Treat a random set of CONTROL markets as treated (placebo).
    Expect lift ~ 0.

    If restrict_to_guardrail=True, we only sample from 'Control Guardrail' controls to avoid spillover bias.
    """
    rng = np.random.default_rng(seed)
    d = df.copy()

    # Candidate controls for placebo treatment
    ctrl_mkts = d.loc[d[treat_col] == 0, market_col].unique()
    if restrict_to_guardrail and spill_col in d.columns:
        ctrl_mkts = d.loc[(d[treat_col] == 0) & (d[spill_col] == "Control Guardrail"), market_col].unique()

    ctrl_mkts = np.array(sorted(ctrl_mkts))
    if ctrl_mkts.size < 4:
        raise ValueError("Not enough control markets available for placebo-in-space.")

    # How many placebo treated?
    if treat_frac is None:
        real_n_t = int(d.loc[d[treat_col] == 1, market_col].nunique())
        n_t = real_n_t if (1 <= real_n_t < ctrl_mkts.size) else max(
            1, min(int(round(0.35 * ctrl_mkts.size)), ctrl_mkts.size - 1)
        )
    else:
        n_t = max(1, min(int(round(treat_frac * ctrl_mkts.size)), ctrl_mkts.size - 1))

    placebo_treated = set(rng.choice(ctrl_mkts, size=n_t, replace=False).tolist())

    d2 = d.copy()
    d2["is_test_placebo"] = d2[market_col].isin(placebo_treated).astype(int)

    # Point estimate: use real post split, placebo treatment assignment
    wide_p = _build_wide_prepost_from_panel(
        d2,
        is_post_col="is_post",
        treat_col="is_test_placebo",
        market_col=market_col,
        y_col=y_col,
    )
    point = _did_lift_total_like(wide_p, treat_col="is_test_placebo")

    if use_ci:
        def est_fn(panel: pd.DataFrame) -> float:
            w = _build_wide_prepost_from_panel(
                panel,
                is_post_col="is_post",
                treat_col="is_test_placebo",
                market_col=market_col,
                y_col=y_col,
            )
            return _did_lift_total_like(w, treat_col="is_test_placebo")

        ci = block_bootstrap_ci(d2, estimator_fn=est_fn, n_boot=n_boot, alpha=alpha, seed=seed)
        return {
            "lift_hat_total": float(ci["point"]),
            "ci_lo": ci["ci_lo"],
            "ci_hi": ci["ci_hi"],
            "n_ok": int(ci["n_ok"]),
            "n_placebo_treated": int(n_t),
            "restrict_to_guardrail": bool(restrict_to_guardrail),
        }

    return {
        "lift_hat_total": float(point),
        "n_placebo_treated": int(n_t),
        "restrict_to_guardrail": bool(restrict_to_guardrail),
    }


# --- Convenience: one table + one plot df ---------------------------------

def placebo_summary(
    df: pd.DataFrame,
    wide: pd.DataFrame,
    seed: int = 7,
    shift_days_grid: list[int] | None = None,
    n_boot: int = 200,
) -> dict:
    """
    Runs:
      - AA test
      - Placebo-in-time (grid)
      - Placebo-in-space
    Returns:
      - summary_table (one small table)
      - plot_df (for the *single plot*: placebo-in-time curve)
    """
    aa = aa_test_preperiod(df, seed=seed)
    pit = placebo_in_time(df, shift_days_grid=shift_days_grid, seed=seed, n_boot=n_boot, use_ci=True)
    pis = placebo_in_space(df, seed=seed, n_boot=n_boot, use_ci=True)

    # Pick a representative placebo-in-time row for the summary (e.g., largest shift)
    pit_tbl = pit["table"].copy()
    pit_pick = pit_tbl.sort_values("shift_days", ascending=False).head(1).iloc[0].to_dict()

    summary = pd.DataFrame(
        [
            {"test": "AA (random assignment on wide)", **aa},
            {"test": f"Placebo-in-time (shift={int(pit_pick['shift_days'])}d)", **{k: pit_pick[k] for k in pit_pick if k != "shift_days"}},
            {"test": "Placebo-in-space (random controls treated)", **pis},
        ]
    )

    # Keep columns tidy
    keep = [c for c in ["test", "lift_hat_total", "ci_lo", "ci_hi", "n_ok", "n_treated", "n_markets", "n_placebo_treated", "restrict_to_guardrail"] if c in summary.columns]
    summary = summary[keep]

    return {"summary_table": summary, "plot_df": pit_tbl}