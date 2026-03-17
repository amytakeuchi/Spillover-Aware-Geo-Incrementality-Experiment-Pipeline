'''
**What this does:**
Takes estimator outputs and compiles:

- `lift_hat_total`
- `error` and `pct_error` (simulation only; compares against ground truth)

**Why this exists:**
Creates a single table that makes the notebook easy to skim.
'''
import pandas as pd

def quick_check_report(gt_total: float, rows: list[dict]) -> pd.DataFrame:
    report = pd.DataFrame(rows)
    report["gt_total"] = gt_total
    report["error"] = report["lift_hat_total"] - report["gt_total"]
    report["pct_error"] = report["error"] / report["gt_total"]
    return report
