import pandas as pd
import statsmodels.api as sm

def tbr_ols_controls(wide: pd.DataFrame) -> dict:
    dfc = wide[wide["is_test"] == 0].copy()
    dft = wide[wide["is_test"] == 1].copy()

    Xc = sm.add_constant(dfc[["sales_pre", "hist", "macro_pre"]], has_constant="add")
    yc = dfc["sales_post"]
    model = sm.OLS(yc, Xc).fit()

    Xt = sm.add_constant(dft[["sales_pre", "hist", "macro_pre"]], has_constant="add")
    yhat_cf = model.predict(Xt)

    lift_hat_total = float((dft["sales_post"] - yhat_cf).sum())
    return {
        "lift_hat_total": lift_hat_total,
        "r2_controls": float(model.rsquared),
        "coef": model.params.to_dict(),
    }