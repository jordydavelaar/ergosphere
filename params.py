# ----------------------------- #
#         params.py             #
# ----------------------------- #

from dataclasses import dataclass, field
from typing import List, Tuple
import pickle
from utils import get_logger
import os

@dataclass
class Params:
    # ── Loggger ───────────────────────────────────────────────────────────
    log_to_file:          bool =  True
    best_params_filename: str  = ""

    # ── Backtest Timeline ─────────────────────────────────────────────────
    top_n:              int = 15
    start_date:         str = "2023-11-01"
    end_date:           str = "2025-05-01"
    rebalance_interval: int = 7  # in days

    # ── Universe Inclusion Flags ─────────────────────────────────────────
    include_crypto: bool = True
    include_stock:  bool = False
    include_bonds:  bool = False
    include_cash:   bool = False

    # ── Capital & Timing ─────────────────────────────────────────────────
    initial_capital:  float = 1000.0
    time_step:        int   = 5  # used in model updates
    lookback:         int   = 1  # window for signals

    # ── Risk & Optimization Weights ──────────────────────────────────────
    risk_aversion:        float = 0.0
    regularization_param: float = 1e-6  # L2
    L1_reg:               float = 1e-2
    rebalance_param:      float = 0.05

    max_weight_shift: float = 0.5
    max_weight:       float = 1.0

    # ── Trading Friction Parameters ──────────────────────────────────────
    transaction_cost:  float = 0.001
    bid_ask_spread:    float = 0.002
    tax_rate:          float = 0.15
    management_fee:    float = 0.01
    inflation_rate:    float = 0.03
    leverage_interest: float = 0.08

    # ── Signal Horizons ─────────────────────────────────────────────────
    horizons:               List[int]   = field(default_factory=lambda: [10, 15, 20, 25])
    horizon_weights:        List[float] = field(default_factory=lambda: [1.5, 1.25, 0.75, 0.5])
    short_horizons:         List[int]   = field(default_factory=lambda: [5, 10, 20])
    short_horizon_weights:  List[float] = field(default_factory=lambda: [0.4, 0.3, 0.3])
    high_vol_zscore_threshold: float     = 1.0
    volatility_cash_boost:     bool      = True
    volatility_window:         int       = 10
    ma_slow_window:            int       = 10
    ma_fast_window:            int       = 5

    # ── Regime Parameters ────────────────────────────────────────────────
    bull_weight_lower:   float = 0.0
    bull_weight_upper:   float = 0.50
    bull_net_target:     float = 1.0
    bull_leverage_limit: float = 1.0

    allow_shorting:      bool  = False
    bear_weight_lower:   float = -0.30
    bear_weight_upper:   float = 0.05
    bear_net_target:     float = -0.30
    bear_leverage_limit: float = 1.8

    # ── Asset Class Constraints ──────────────────────────────────────────
    crypto_weight_range: Tuple[float, float] = (0.0, 1.0)
    bond_weight_range:   Tuple[float, float] = (0.0, 0.5)
    cash_weight_range:   Tuple[float, float] = (0.0, 1.0)

    # ── Bear Market Detection Settings ───────────────────────────────────
    drawdown_thresh: float = -0.15
    vol_z_thresh:    float = 1.0
    ma_below_ratio:  float = 0.25
    drawdown_window: int   = 60
    vol_window:      int   = 30
    ma_below_days:   int   = 20

    def load_overrides_from_file(self):
        import pickle
        import os
        from utils import get_logger
        logger = get_logger("Params", log_to_file=self.log_to_file)

        if not os.path.isfile(self.best_params_filename):
            logger.warning(f"Override file not found: {self.best_params_filename}. Using default Params.")
            return

        logger.info(f"Loading parameter overrides from: {self.best_params_filename}")
        try:
            with open(self.best_params_filename, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load override file: {e}. Using default Params.")
            return

        full_params = data.get("full_params", None)
        if full_params is not None and isinstance(full_params, Params):
            logger.info("Loaded full Params object from file.")
            for field in full_params.__dataclass_fields__:
                setattr(self, field, getattr(full_params, field))
        else:
            logger.warning("No full_params object found. Falling back to best_params_dict...")
            best_params = data.get("best_params_dict", {})
            for k, v in best_params.items():
                if hasattr(self, k):
                    setattr(self, k, v)
                    logger.info(f"Overriding param: {k} = {v}")
                else:
                    logger.warning(f"Ignoring unknown parameter key: {k}")