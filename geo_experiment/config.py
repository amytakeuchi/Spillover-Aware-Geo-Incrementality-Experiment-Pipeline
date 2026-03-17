'''
**What this does:**
Defines `SimConfig`, a single configuration object controlling the whole experiment:

- number of markets/days
- treatment start date split
- spend noise
- nonlinear response curve parameters
- spillover strength
- seasonality/trend/noise parameters

**Why it exists:**
to change experiment settings by editing **one place**.

**Output:**
A config object that every module uses.
'''
from dataclasses import dataclass
from typing import Dict

@dataclass
class SimConfig:
    n_markets: int = 60
    n_days: int = 120
    pre_days: int = 90
    treat_days: int = 30

    base_daily_spend: float = 2000.0
    spend_noise_sd: float = 0.15

    lift_a: float = 450.0
    lift_b: float = 3500.0

    spillover_frac: float = 0.18

    weekly_seasonality_amp: float = 0.08
    trend_per_day: float = 0.001
    shock_sd: float = 0.10
    ar1_rho: float = 0.35

    market_size_multipliers: Dict[str, float] = None

    def __post_init__(self):
        if self.market_size_multipliers is None:
            self.market_size_multipliers = {"Small": 0.7, "Medium": 1.0, "Large": 1.4}