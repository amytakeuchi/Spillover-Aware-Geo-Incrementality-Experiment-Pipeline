import numpy as np
import pandas as pd
import statsmodels.api as sm

def _to_float_df(X: pd.DataFrame) -> pd.DataFrame:
    X = X.apply(pd.to_numeric, errors="coerce")
    return X.astype(float)

def bayes_hier_lift_empirical(df: pd.DataFrame, treat_days: int) -> dict:
    d = df.copy()

    ctrl = d[d["is_test"] == 0].copy()
    ctrl["dow"] = ctrl["date"].dt.dayofweek.astype(int)
    ctrl["market_size_group"] = ctrl["market_size_group"].astype(str)

    Xc = pd.concat([
        ctrl[["sales_baseline"]].rename(columns={"sales_baseline": "base"}),
        ctrl[["macro_covariate_1", "day_index"]],
        pd.get_dummies(ctrl["dow"], prefix="dow", drop_first=True),
        pd.get_dummies(ctrl["market_size_group"], prefix="size", drop_first=True),
    ], axis=1)

    Xc = _to_float_df(Xc)
    yc = pd.to_numeric(ctrl["sales"], errors="coerce").astype(float)

    mask = np.isfinite(yc.values) & np.isfinite(Xc.values).all(axis=1)
    Xc_fit = sm.add_constant(Xc.loc[mask], has_constant="add")
    yc_fit = yc.loc[mask]

    model = sm.OLS(yc_fit, Xc_fit).fit()
    resid_sd = float(np.std(model.resid, ddof=1))

    treat_post = d[(d["is_test"] == 1) & (d["is_post"] == 1)].copy()
    treat_post["dow"] = treat_post["date"].dt.dayofweek.astype(int)
    treat_post["market_size_group"] = treat_post["market_size_group"].astype(str)

    Xt = pd.concat([
        treat_post[["sales_baseline"]].rename(columns={"sales_baseline": "base"}),
        treat_post[["macro_covariate_1", "day_index"]],
        pd.get_dummies(treat_post["dow"], prefix="dow", drop_first=True),
        pd.get_dummies(treat_post["market_size_group"], prefix="size", drop_first=True),
    ], axis=1)

    Xt = _to_float_df(Xt)
    Xt = Xt.reindex(columns=Xc.columns, fill_value=0.0)  # align
    Xt = sm.add_constant(Xt, has_constant="add")

    treat_post["yhat_cf"] = model.predict(Xt)
    treat_post["resid_cf"] = pd.to_numeric(treat_post["sales"], errors="coerce") - treat_post["yhat_cf"]

    lifts = treat_post.groupby("market_id", as_index=False).agg(lift_sum=("resid_cf", "sum"))
    lifts["se_sum"] = resid_sd * np.sqrt(treat_days)

    y = lifts["lift_sum"].values.astype(float)
    se2 = (lifts["se_sum"].values.astype(float)) ** 2

    tau2 = max(float(np.var(y, ddof=1) - np.mean(se2)), 0.0)
    tau = float(np.sqrt(tau2))

    w = 1.0 / (tau2 + se2 + 1e-12)
    mu = float(np.sum(w * y) / np.sum(w))

    # Shrinkage factors (per market)
    tau2_f = float(tau2)
    k = tau2_f / (tau2_f + se2 + 1e-12)   # elementwise
    lifts["shrinkage_k"] = k

    # Compact summary (trust metric)
    shrink_mean = float(np.mean(k)) if len(k) else float("nan")
    shrink_p10 = float(np.quantile(k, 0.10)) if len(k) else float("nan")
    shrink_p90 = float(np.quantile(k, 0.90)) if len(k) else float("nan")

    post_var = 1.0 / (1.0 / (tau2 + 1e-12) + 1.0 / (se2 + 1e-12))
    post_mean = post_var * (mu / (tau2 + 1e-12) + y / (se2 + 1e-12))

    lifts["post_mean"] = post_mean
    lifts["post_sd"] = np.sqrt(post_var)

    total_hat = float(lifts["post_mean"].sum())

    return {
    "total_lift_hat": total_hat,
    "mu": mu,
    "tau_between_sd": tau,
    "daily_resid_sd_controls": resid_sd,
    "n_controls_used": int(mask.sum()),
    "shrinkage_mean": shrink_mean,   # <-- trust metric
    "shrinkage_p10": shrink_p10,
    "shrinkage_p90": shrink_p90,
    "market_table": lifts.sort_values("post_mean", ascending=False).reset_index(drop=True),
    }
