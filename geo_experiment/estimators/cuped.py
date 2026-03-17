import numpy as np
import pandas as pd

def cuped_prepost(wide: pd.DataFrame, theta_from_controls: bool = True) -> dict:
    ref = wide[wide["is_test"] == 0] if theta_from_controls else wide
    y_pre = ref["sales_pre"].values
    y_post = ref["sales_post"].values

    var_pre = np.var(y_pre, ddof=1)
    theta = 0.0 if var_pre == 0 else np.cov(y_post, y_pre, ddof=1)[0, 1] / var_pre
    mu_pre = wide["sales_pre"].mean()

    w2 = wide.copy()
    w2["post_cuped"] = w2["sales_post"] - theta * (w2["sales_pre"] - mu_pre)

    test = w2[w2["is_test"] == 1]
    ctrl = w2[w2["is_test"] == 0]

    diff = test["post_cuped"].mean() - ctrl["post_cuped"].mean()
    lift_hat_total = float(diff * len(test))

    # --- NEW: variance reduction diagnostic (measured on reference group) ---
    ref2 = w2[w2["is_test"] == 0] if theta_from_controls else w2
    var_raw = float(np.var(ref2["sales_post"].values, ddof=1))
    var_adj = float(np.var(ref2["post_cuped"].values, ddof=1))
    var_reduction_pct = float(100.0 * (1.0 - (var_adj / var_raw))) if var_raw > 0 else float("nan")

    return {
        "theta": float(theta),
        "lift_hat_total": lift_hat_total,
        "var_raw_post": var_raw,
        "var_adj_post": var_adj,
        "var_reduction_pct": var_reduction_pct,   # <-- trust metric
    }