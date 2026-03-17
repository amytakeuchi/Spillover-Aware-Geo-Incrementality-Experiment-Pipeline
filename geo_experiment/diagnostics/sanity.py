from __future__ import annotations
import pandas as pd

def sanity_checks(
    df: pd.DataFrame,
    treat_days: int,
    market_col: str = "market_id",
    date_col: str = "date",
    treat_col: str = "is_test",
    post_col: str = "is_post",
    spend_col: str = "paid_social_spend",
) -> dict:
    """
    Lightweight sanity checks for the simulated / prepared geo panel.

    Returns:
      - summary: 1-row pd.DataFrame (nice for display)
      - spend_by_cell: pd.DataFrame with spend by (test/control) x (pre/post)
    """
    # 1) Duplicate market-date rows (should be 0)
    dup = int(df.duplicated([market_col, date_col]).sum())

    # 2) Post period length (days)
    post_days = int(df.loc[df[post_col] == 1, date_col].nunique())

    # 3) Spend by cell (test/control x pre/post)
    if spend_col in df.columns:
        spend_cell = (
            df.groupby([treat_col, post_col], as_index=False)[spend_col]
              .sum()
              .rename(columns={treat_col: "is_test", post_col: "is_post", spend_col: "spend"})
        )
        # Add readable labels
        spend_cell["cell"] = spend_cell.apply(
            lambda r: ("Treated" if r["is_test"] == 1 else "Control") + " / " + ("Post" if r["is_post"] == 1 else "Pre"),
            axis=1,
        )
        spend_cell = spend_cell[["cell", "is_test", "is_post", "spend"]]
    else:
        spend_cell = pd.DataFrame(columns=["cell", "is_test", "is_post", "spend"])

    # 4) Pass/fail flags (scan-friendly)
    passes_dup = (dup == 0)
    passes_post_days = (post_days == int(treat_days))

    # Treated should have spend in post; controls ideally 0 (for clean simulation)
    # (Keep it as a *check* not a hard rule; your designs may vary.)
    treated_post_spend = float(
        spend_cell.loc[(spend_cell["is_test"] == 1) & (spend_cell["is_post"] == 1), "spend"].sum()
    ) if len(spend_cell) else 0.0
    control_post_spend = float(
        spend_cell.loc[(spend_cell["is_test"] == 0) & (spend_cell["is_post"] == 1), "spend"].sum()
    ) if len(spend_cell) else 0.0

    # you can tune these rules
    passes_spend_pattern = (treated_post_spend > 0) and (control_post_spend == 0)

    summary = pd.DataFrame([{
        "dup_market_date": dup,
        "post_days": post_days,
        "post_days_expected": int(treat_days),
        "treated_post_spend": treated_post_spend,
        "control_post_spend": control_post_spend,
        "pass_dup": bool(passes_dup),
        "pass_post_days": bool(passes_post_days),
        "pass_spend_pattern": bool(passes_spend_pattern),
    }])

    return {
        "summary": summary,
        "spend_by_cell": spend_cell,
    }


def show_sanity(result: dict):
    """
    Notebook helper: pretty display for the output of sanity_checks().
    """
    from IPython.display import display  # local import so module works outside notebooks

    summary = result["summary"]
    spend_cell = result["spend_by_cell"]

    display(
        summary.style
        .format({"treated_post_spend": "{:,.0f}", "control_post_spend": "{:,.0f}"})
        .hide(axis="index")
    )

    if len(spend_cell):
        display(
            spend_cell.style
            .format({"spend": "{:,.0f}"})
            .hide(axis="index")
        )
'''
def sanity_checks(df: pd.DataFrame, treat_days: int) -> dict:
    dup = int(df.duplicated(["market_id", "date"]).sum())
    post_days = int(df[df["is_post"] == 1]["date"].nunique())
    spend_by = df.groupby(["is_test", "is_post"])["paid_social_spend"].sum().to_dict()
    return {
        "dup_market_date": dup,
        "post_days": post_days,
        "post_days_expected": treat_days,
        "spend_by_test_post": spend_by,
    }
sanity = sanity_checks(df, treat_days=cfg.treat_days)

def show_sanity(sanity: dict):
    sanity_df = pd.DataFrame([sanity])
    display(
        sanity_df.style
        .format(precision=3)
        .hide(axis="index")
    )
    '''