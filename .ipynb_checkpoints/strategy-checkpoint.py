# ----------------------------- #
#        strategy.py            #
# ----------------------------- #

import numpy as np
import cvxpy as cp

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from universe import AssetUniverse
from params   import Params
from data_loader import DataLoader
from utils    import get_logger
from numba    import njit, prange




class MomentumStrategy:
    def __init__(self, universe: AssetUniverse, params: Params, data_loader: DataLoader):
        self.universe = universe
        self.params = params
        self.prices = data_loader.get_prices(interval="1d")
        self.volume = data_loader.get_volume(interval="1d")
        self.signal = None
        self.logger = get_logger("strategy", log_to_file=params.log_to_file)
    def compute_volume_score(self, volume_df: pd.DataFrame, window: int = 20, clip_bounds=(0.5, 2.0)) -> pd.DataFrame:
        """
        Computes a volume score by comparing current volume to a rolling average.

        Args:
            volume_df: DataFrame of daily volume per asset.
            window: Lookback window for moving average.
            clip_bounds: Tuple to clip extreme volume ratios.

        Returns:
            DataFrame with volume score: volume / rolling mean, clipped.
        """
        rolling_avg = volume_df.rolling(window=window, min_periods=1).mean()
        volume_score = volume_df / (rolling_avg + 1e-8)  # prevent div by zero
        return volume_score.clip(*clip_bounds)

    def generate_signal(self):
        """
        This method replicates the version you provided—using an EWMA smoothing
        of h-day returns and then adding higher-order deltas (1st, 2nd, 3rd, 4th)
        before weighting them by dynamic weights based on z-score of volatility.
        """
        vol_window = self.params.volatility_window
        use_cash_boost = self.params.volatility_cash_boost

        # --- 1) Compute rolling volatility and z-score ---
        # pct_change().rolling(vol_window).std() gives a DataFrame of rolling std per asset;
        # then .mean(axis=1) collapses to a single "market" volatility time series.
        vol = self.prices.pct_change().rolling(vol_window, min_periods=1).std().mean(axis=1)
        zscore_vol = (vol - vol.mean()) / vol.std()

        # --- 2) Build two parallel lists (one element per date) of (horizon list, weight list) ---
        dynamic_horizons = []
        dynamic_weights = []

        for date in self.prices.index:
            # If today's vol‐zscore exceeds the threshold, use the "short" horizons & weights;
            # otherwise use the normal (long) ones.
            if zscore_vol.get(date, 0) > self.params.high_vol_zscore_threshold:
                dynamic_horizons.append(self.params.short_horizons)
                dynamic_weights.append(self.params.short_horizon_weights)
            else:
                dynamic_horizons.append(self.params.horizons)
                dynamic_weights.append(self.params.horizon_weights)

        # Preallocate a DataFrame to hold our final signals (one column per asset)
        total_signal = pd.DataFrame(index=self.prices.index, columns=self.prices.columns, dtype=float)

        # --- 3) Loop over each date, build the weighted sum of EWMA + deltas for each horizon ---
        for idx, date in enumerate(self.prices.index):
            daily_signal = pd.Series(0.0, index=self.prices.columns)
            horizon_list = dynamic_horizons[idx]
            weight_list = dynamic_weights[idx]
            total_weight = sum(weight_list)

            # If all weights are zero, we’ll just leave daily_signal at zero
            if total_weight == 0:
                total_signal.loc[date] = daily_signal
                continue

            # For each (h, w) pair in today’s regime:
            for h, w in zip(horizon_list, weight_list):
                if w <= 0:
                    continue

                # 3.1) Compute an EWMA of h‐day returns.
                #     We do: r_t = (price.pct_change(periods=h)).ewm(span=smoothing_span).mean()
                #     Then we turn that into a per‐day equivalent by (1+r_t)**(1/h) - 1.
                smoothing_span = max(3, int(h * 2))  # e.g. if h=15, span=30

                # r_t is a DataFrame of the same shape as self.prices:
                r_t = self.prices.pct_change(periods=h).ewm(span=smoothing_span, adjust=False).mean()

                # Convert the EWMA of h-day return back into an "equivalent daily return"
                # i.e. (1 + r_t)^(1/h) – 1
                r_t = (1 + r_t) ** (1.0 / h) - 1

                # 3.2) Compute successive deltas (differences) of r_t along the time axis:
                #       delta_r  = r_t.diff()
                #       delta2_r = delta_r.diff()
                #       delta3_r = delta2_r.diff()
                #       delta4_r = delta3_r.diff()
                delta_r  = r_t.diff()
                delta2_r = delta_r.diff()
                delta3_r = delta2_r.diff()
                delta4_r = delta3_r.diff()

                # 3.3) Build the "signal" for this horizon:
                #       signal = r_t + 1*delta_r + 0.5*delta2_r + (1/6)*delta3_r + (1/24)*delta4_r
                signal_df = r_t  #+ delta_r + 0.5 * delta2_r + (1/6.0) * delta3_r + (1/24.0) * delta4_r

                # 3.4) Shift by 1 day to avoid lookahead, fill NaN with 0 on the first day
                signal_df = signal_df.shift(1).fillna(0.0)

                # 3.5) Add to our daily_signal (for date=“date”), weighted by w
                daily_signal += w * signal_df.loc[date]

            vol_score = self.compute_volume_score(self.volume)
            daily_vol_score = vol_score.loc[date].reindex(daily_signal.index).fillna(1.0)
            total_signal.loc[date] = daily_vol_score * daily_signal / total_weight

        # --- 5) Reindex/ffill/bfill to align with self.universe, ensure no missing dates ---
        ordered_cols = [t for t in self.universe.all() if t in total_signal.columns]
        total_signal = total_signal[ordered_cols].copy()
        self.signal = total_signal.reindex(self.prices.index).ffill().bfill().infer_objects()

        self.logger.info("Momentum signal generated (EWMA + higher‐order deltas).")

    def detect_bear_market(self, today, top_coins=None, drawdown_window=None, drawdown_thresh=None, vol_window=None, vol_z_thresh=None, ma_below_days=None, ma_below_ratio=None) -> bool:
        drawdown_window = drawdown_window or self.params.drawdown_window
        drawdown_thresh = drawdown_thresh or self.params.drawdown_thresh
        vol_window = vol_window or self.params.vol_window
        vol_z_thresh = vol_z_thresh or self.params.vol_z_thresh
        ma_below_days = ma_below_days or self.params.ma_below_days
        ma_below_ratio = ma_below_ratio or self.params.ma_below_ratio

        prices = self.prices
        if top_coins is None:
            top_coins = self.universe.crypto[:4]
        top_coins = [c for c in top_coins if c in prices.columns]
        if not top_coins:
            raise ValueError("No valid tickers for bear-market basket.")

        basket_prices = prices[top_coins].loc[:today].ffill()
        eq_index = basket_prices.mean(axis=1)
        price_slice = eq_index.tail(drawdown_window)
        if len(price_slice) < vol_window:
            return False

        drawdown = (price_slice / price_slice.cummax() - 1).min()
        rets = eq_index.loc[:today].pct_change().dropna()
        if len(rets) < vol_window:
            vol_z = 0.0
        else:
            vol_roll = rets.rolling(vol_window, min_periods=1).std().dropna()
            last_vol, mean_vol, std_vol = vol_roll.iloc[-1], vol_roll.mean(), vol_roll.std()
            vol_z = 0.0 if std_vol == 0 else (last_vol - mean_vol) / std_vol

        high_vol_z = vol_z > vol_z_thresh
        ma_slow = eq_index.rolling(self.params.ma_slow_window, min_periods=1).mean()
        ma_fast = eq_index.rolling(self.params.ma_fast_window, min_periods=1).mean()
        frac_below = (eq_index.values[-ma_below_days:] < ma_slow.values[-ma_below_days:]).mean()
        below_ma = frac_below > ma_below_ratio
        death_cross = self.check_death_cross(eq_index, self.params.ma_fast_window, self.params.ma_slow_window)

        bear_score = 0.35 * (drawdown < drawdown_thresh) + 0.25 * high_vol_z + 0.25 * below_ma + 0.15 * death_cross
        return bear_score > 0.45

    def rolling_volatility(self, asset: str = "BTC-USD", window: int = 20) -> pd.Series:
        prices = self.prices[asset]
        returns = prices.pct_change()
        return returns.rolling(window, min_periods=1).std()

    def check_death_cross(self, price_slice, fast_window=20, slow_window=100) -> bool:
        ma_fast = price_slice.rolling(fast_window, min_periods=1).mean()
        ma_slow = price_slice.rolling(slow_window, min_periods=1).mean()
        if len(ma_fast.dropna()) < 2 or len(ma_slow.dropna()) < 2:
            return False
        return ma_fast.iloc[-1] < ma_slow.iloc[-1]

    def adjust_weights(self, *,
                   pred_returns: np.ndarray,
                   cov_matrix:  np.ndarray,
                   prev_weights: np.ndarray,
                   market_vol:   float,
                   weight_lower: float,
                   weight_upper: float,
                   net_target:   float,
                   lev_limit:    float,
                   regime_changed: bool,
                   curr_regime:   str):
        p = self.params
        n = len(pred_returns)
        w = cp.Variable(n)

        μ = np.asarray(pred_returns).flatten()
        Σ = np.asarray(cov_matrix)
        Σ += np.eye(n) * 1e-1
        Σ = 0.5 * (Σ + Σ.T) + 1e-9 * np.eye(n)

        vol = float(np.asarray(market_vol).ravel()[0]) if market_vol is not None else 40.0
        λ = p.risk_aversion * (vol / 40.0)
        risk_term = cp.quad_form(w, cp.psd_wrap(Σ))

        tickers = list(self.prices.columns)
        crypto_mask = np.array([t in self.universe.crypto for t in tickers], dtype=float)
        bond_mask = np.array([t in self.universe.bonds for t in tickers], dtype=float)
        cash_mask = np.array([t in self.universe.cash for t in tickers], dtype=float)

        constraints = []
        for j in range(n):
            if crypto_mask[j]:
                constraints += [w[j] >= weight_lower, w[j] <= weight_upper]
            else:
                constraints += [w[j] >= 0.0, w[j] <= p.max_weight]

        def sleeve(mask, lo, hi):
            if mask.any():
                constraints += [
                    cp.sum(cp.multiply(mask, w)) >= lo,
                    cp.sum(cp.multiply(mask, w)) <= hi
                ]

        if curr_regime == "bull":
            constraints += [cp.abs(cp.sum(w) - 1.0) <= 1e-4]
            sleeve(bond_mask, 0.0, 0.0)
        elif curr_regime == "bear":
            constraints += [cp.sum(cp.multiply(crypto_mask, w)) == p.bear_net_target]
            sleeve(bond_mask, *p.bond_weight_range)
        sleeve(cash_mask, *p.cash_weight_range)

        constraints += [cp.sum(cp.abs(w)) <= lev_limit]

        base_shift = p.max_weight_shift if not regime_changed else 10.0
        turnover_budget = n * base_shift
        turnover = cp.norm1(w - prev_weights)
        constraints += [turnover <= turnover_budget]

        if not regime_changed:
            prev_net = float(prev_weights @ crypto_mask)
            new_net = cp.sum(cp.multiply(crypto_mask, w))
            delta_sleeve = cp.abs(new_net - prev_net)
            constraints += [delta_sleeve <= p.max_weight_shift]
            constraints += [cp.norm_inf(cp.multiply(crypto_mask, w - prev_weights)) <= p.max_weight_shift]

        reb_pen = cp.sum_squares(w - prev_weights)
        objective = cp.Maximize(μ @ w - λ * risk_term - p.rebalance_param * reb_pen - 1e-3 * cp.sum_squares(w))

        prob = cp.Problem(objective, constraints)
        prob.solve( solver=cp.ECOS,
                    verbose=False, 
                    abstol=1e-6, 
                    reltol=1e-6, 
                    feastol=1e-6,
                    max_iters=50000
                )

        if w.value is None:
            return prev_weights

        return w.value