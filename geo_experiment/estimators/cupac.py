import numpy as np
import pandas as pd
import statsmodels.api as sm

def _to_float_df(X: pd.DataFrame) -> pd.DataFrame:
    X = X.apply(pd.to_numeric, errors="coerce")
    return X.astype(float)

def one_hot_size(df_: pd.DataFrame) -> pd.DataFrame:
    s = df_["size"].astype(str)
    return pd.get_dummies(s, prefix="size", drop_first=True).astype(float)

def cupac_prepost(wide: pd.DataFrame) -> dict:
    dfc = wide[wide["is_test"] == 0].copy()
    dft = wide[wide["is_test"] == 1].copy()

    Xc = pd.concat([dfc[["sales_pre", "hist", "macro_pre"]], one_hot_size(dfc)], axis=1)
    Xt = pd.concat([dft[["sales_pre", "hist", "macro_pre"]], one_hot_size(dft)], axis=1)
    Xt = Xt.reindex(columns=Xc.columns, fill_value=0.0)

    Xc = _to_float_df(Xc)
    Xt = _to_float_df(Xt)
    yc = pd.to_numeric(dfc["sales_post"], errors="coerce").astype(float)

    mask = np.isfinite(yc.values) & np.isfinite(Xc.values).all(axis=1)
    Xc_fit = sm.add_constant(Xc.loc[mask], has_constant="add")
    yc_fit = yc.loc[mask]

    model = sm.OLS(yc_fit, Xc_fit).fit()

    yhat_c = model.predict(sm.add_constant(Xc, has_constant="add"))
    yhat_t = model.predict(sm.add_constant(Xt, has_constant="add"))

    dfc["resid"] = pd.to_numeric(dfc["sales_post"], errors="coerce") - yhat_c
    dft["resid"] = pd.to_numeric(dft["sales_post"], errors="coerce") - yhat_t

    diff = float(dft["resid"].mean() - dfc["resid"].mean())
    lift_hat_total = diff * float(len(dft))

    return {
        "lift_hat_total": float(lift_hat_total),
        "r2_controls": float(model.rsquared),
        "coef_count": int(len(model.params)),
        "n_controls_used": int(mask.sum()),
    }