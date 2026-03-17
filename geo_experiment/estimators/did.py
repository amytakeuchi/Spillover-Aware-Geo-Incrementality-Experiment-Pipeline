# src/geo_experiment/estimators/did.py
import pandas as pd

def did_prepost(wide: pd.DataFrame) -> dict:
    test = wide[wide["is_test"] == 1]
    ctrl = wide[wide["is_test"] == 0]

    t_post, t_pre = test["sales_post"].mean(), test["sales_pre"].mean()
    c_post, c_pre = ctrl["sales_post"].mean(), ctrl["sales_pre"].mean()

    did_market_avg = (t_post - t_pre) - (c_post - c_pre)
    did_total_like = did_market_avg * len(test)  # market-average * #treated markets

    return {
        "did_market_avg": float(did_market_avg),
        "lift_hat_total": float(did_total_like),  # optional alias for table consistency and to make the summary table cleaner
        "did_total_like": float(did_total_like),  # keep old key if you already use it
        "n_treated": int(len(test)),
        "n_control": int(len(ctrl)),
    }