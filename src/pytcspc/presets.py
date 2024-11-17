__all__ = [
    "MTQ2_2EXP_INIT",
    "NADH_2EXP_INIT",
]


import numpy as np

MTQ2_2EXP_INIT = {
    "shift": {"value": 0    , "err": np.nan, "min": -200 , "max":   200, "step": 1   },
    "A":     {"value": 0.995, "err": np.nan, "min": 0.700, "max": 1.000, "step": 1e-3},
    "tau1":  {"value": 3.500, "err": np.nan, "min": 0.100, "max": 9.000, "step": 1e-3},
    "tau2":  {"value": 0.2, "err": np.nan, "min": 0.010, "max": 3.000, "step": 1e-3},
    "f":     {"value": 0.405, "err": np.nan, "min": 0.010, "max": 1.000, "step": 1e-3},
}
NADH_2EXP_INIT = {
    "shift": {"value": 0    , "err": np.nan, "min": -100 , "max":   100, "step": 1   },
    "A":     {"value": 0.995, "err": np.nan, "min": 0.700, "max": 1.000, "step": 1e-3},
    "tau1":  {"value": 1.8, "err": np.nan, "min": 0.100, "max": 9.000, "step": 1e-3},
    "tau2":  {"value": 0.25, "err": np.nan, "min": 0.010, "max": 1.000, "step": 1e-3},
    "f":     {"value": 0.25, "err": np.nan, "min": 0.010, "max": 1.000, "step": 1e-3},
}