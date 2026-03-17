import numpy as np
import pandas as pd
from typing import Callable, Any, Optional, Union

Number = Union[int, float, np.number]


def _extract_scalar(result: Any, key: Optional[str] = None) -> float:
    """
    Convert estimator output into a single float.
    - If result is a number -> float(result)
    - If result is dict-like -> float(result[key])
    """
    if isinstance(result, (int, float, np.number)):
        return float(result)

    if isinstance(result, dict):
        if key is None:
            raise ValueError(
                "estimator_fn returned a dict, but no `value_key` was provided. "
                "Pass value_key='lift_hat_total' (or the appropriate key)."
            )
        if key not in result:
            raise KeyError(f"value_key='{key}' not found in estimator output keys={list(result.keys())}")
        return float(result[key])

    raise TypeError(f"Unsupported estimator output type: {type(result)}")


def block_bootstrap_ci(
    df: pd.DataFrame,
    estimator_fn: Callable[..., Any],
    *,
    id_col: str = "market_id",
    value_key: Optional[str] = None,
    n_boot: int = 300,
    alpha: float = 0.05,
    seed: int = 123,
    rename_id_in_bootstrap: bool = False,
    **kwargs,
) -> dict:
    """
    Block bootstrap over markets (clusters).

    Parameters
    ----------
    df : DataFrame
        Either panel df (market_id x day) OR wide df (one row per market).
        Must include `id_col`.
    estimator_fn : callable
        Function that takes a DataFrame (same schema as df) and returns either:
          - a float (lift), OR
          - a dict containing lift under `value_key`
    id_col : str
        Cluster id column (default 'market_id').
    value_key : str or None
        If estimator_fn returns dict, provide key for scalar lift. Example:
          value_key='lift_hat_total' or value_key='did_total_like'
    rename_id_in_bootstrap : bool
        If True, ensures each resampled cluster gets a unique id label.
        Usually NOT required for wide or pre/post methods; set True if an estimator
        assumes unique cluster ids and would double-count duplicates.
    """

    if id_col not in df.columns:
        raise KeyError(f"'{id_col}' not found in df columns.")

    rng = np.random.default_rng(seed)

    # 1) Point estimate
    point_raw = estimator_fn(df, **kwargs)
    point = _extract_scalar(point_raw, key=value_key)

    # 2) Pre-split blocks once for speed
    blocks = {m: g.copy() for m, g in df.groupby(id_col, sort=True)}
    markets = np.array(list(blocks.keys()))

    vals = []
    for b in range(n_boot):
        sampled = rng.choice(markets, size=len(markets), replace=True)

        if rename_id_in_bootstrap:
            parts = []
            for i, m in enumerate(sampled):
                tmp = blocks[m].copy()
                tmp[id_col] = f"{m}__{i}"  # unique id per resampled block
                parts.append(tmp)
            boot = pd.concat(parts, ignore_index=True)
        else:
            boot = pd.concat([blocks[m] for m in sampled], ignore_index=True)

        try:
            out = estimator_fn(boot, **kwargs)
            vals.append(_extract_scalar(out, key=value_key))
        except Exception:
            # keep going; report n_ok so you can detect instability
            continue

    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        lo = hi = float("nan")
    else:
        lo = float(np.quantile(vals, alpha / 2))
        hi = float(np.quantile(vals, 1 - alpha / 2))

    return {
        "point": point,
        "ci_lo": lo,
        "ci_hi": hi,
        "n_ok": int(len(vals)),
        "n_boot": int(n_boot),
        "alpha": float(alpha),
    }