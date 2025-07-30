import pandas as pd
import numpy as np
import yfinance as yf
import sys
from numba import njit, prange
import logging

from universe import AssetUniverse
from params   import Params
from strategy import MomentumStrategy

logger = logging.getLogger(__name__)

class BaseBacktester:
    def __init__(self, params: Params):
        if not isinstance(params, Params):
            raise TypeError("params must be an instance of Params")
        self.params          = params
        self.performance_log = []

    def log_performance(self, strategy_name: str, portfolio_values: pd.Series):
        if not isinstance(portfolio_values, pd.Series) or portfolio_values.empty:
            raise ValueError("portfolio_values must be a non-empty pandas Series.")
        returns      = portfolio_values.pct_change(fill_method=None).dropna()
        if returns.empty:
            raise ValueError("Not enough data to compute performance metrics.")
        annual_ret   = returns.mean() * 365
        annual_vol   = returns.std()  * np.sqrt(365)
        sharpe       = (annual_ret - 0.0) / annual_vol if annual_vol > 0 else np.nan

        start_val    = portfolio_values.iloc[0]
        end_val      = portfolio_values.iloc[-1]
        yrs          = (portfolio_values.index[-1] - portfolio_values.index[0]).days / 365.25
        cagr         = (end_val / start_val)**(1/yrs) - 1 if yrs > 0 else np.nan

        self.performance_log.append({
            "Strategy":          strategy_name,
            "CAGR":              round(cagr, 4),
            "Sharpe Ratio":      round(sharpe, 4),
            "Annual Return":     round(annual_ret, 4),
            "Annual Volatility": round(annual_vol, 4),
        })


class BuyAndHoldBacktester(BaseBacktester):
    def __init__(self, data, params: Params):
        super().__init__(params)
        self.data   = data
        daily_prices = data.get_prices(interval="1d")
        if not isinstance(daily_prices, pd.DataFrame) or daily_prices.empty:
            raise ValueError("data.get_prices('1d') must return a non-empty DataFrame.")
        self.prices = daily_prices.sort_index()
        self.result = None

    def run(self):
        p     = self.params
        daily = self.prices.copy()
        n     = len(daily.columns)
        w0    = np.ones(n) / n
        initial_prices = daily.iloc[0].values
        shares= (w0 * p.initial_capital) / initial_prices
        vals  = (daily.values * shares).sum(axis=1)

        df = pd.DataFrame({
            "Date":            daily.index,
            "Portfolio Value": vals
        }).set_index("Date")

        self.result = df
        self.log_performance("buy_and_hold", df["Portfolio Value"])


class AdaptiveBacktester(BaseBacktester):
    def __init__(self, data, strategy: MomentumStrategy, params: Params):
        super().__init__(params)
        self.data     = data
        self.strategy = strategy

        self.prices_daily  = data.get_prices(interval="1d")
        self.prices_weekly = data.get_prices(interval="7d")
        self.prices        = self.prices_weekly  # Ensure consistency with signal frequency
        self.signal = strategy.signal  # aligned to weekly prices
        self.CVI          = data.CVI if hasattr(data, 'CVI') else None
        self.cols         = self.prices.columns.intersection(self.signal.columns)
        if self.cols.empty:
            raise ValueError("No overlapping columns between prices and signal.")

        self.result       = None
        self.weights_log  = []

    def run(self):
        p = self.params

        # Convert to numpy
        daily_prices   = self.prices_daily[self.cols].to_numpy(copy=False)
        dates          = self.prices_daily.index.to_numpy()
        signal_dates   = self.signal.index.to_numpy()
        signal_values  = self.signal[self.cols].to_numpy(copy=False)

        signal_date_to_index = {date: i for i, date in enumerate(signal_dates)}
        rebalance_mask       = np.isin(dates, signal_dates)

        cvi_values = (
            self.CVI.reindex(self.signal.index).ffill().to_numpy(copy=False)
            if self.CVI is not None else np.zeros(len(signal_values))
        )
        aligned_cvi = pd.Series(cvi_values, index=signal_dates).reindex(dates, method='ffill').to_numpy(copy=False)

        crypto_cols = np.array([j for j, t in enumerate(self.cols) if t in self.strategy.universe.crypto])

        # Precompute covariances
        n = len(self.cols)
        if p.risk_aversion != 0.0:
            precomputed_cov = {}
            for i, date in enumerate(signal_dates):
                idx = np.where(dates == date)[0]
                if len(idx) == 0:
                    continue
                idx = idx[0]
                start_idx = max(0, idx - p.time_step)
                window_data = daily_prices[start_idx:idx+1].T
                if window_data.shape[1] < 2:
                    precomputed_cov[date] = np.eye(n)
                else:
                    precomputed_cov[date] = np.cov(window_data)
        else:
            precomputed_cov = {date: np.eye(n) for date in signal_dates}

        capital = p.initial_capital
        weights = np.ones(n) / n
        prev_rebalance_weights = weights.copy()
        capital_series = []
        weights_series = []

        for i in range(1, len(dates)):
            rets = daily_prices[i] / daily_prices[i - 1] - 1
            weights = weights * (1 + rets)
            weights /= weights.sum()
            capital *= (weights @ rets) + 1

            if rebalance_mask[i]:
                date = dates[i]
                if date not in signal_date_to_index:
                    continue
                idx = signal_date_to_index[date]
                pred = signal_values[idx]
                market_vol = aligned_cvi[i]
                cov = precomputed_cov.get(date, np.eye(n))

                # Regime detection
                if p.allow_shorting:
                    is_bear = self.strategy.detect_bear_market(today=date)
                    row_crypto = pred[crypto_cols] if crypto_cols.size > 0 else np.array([])
                    crypto_all_red = (row_crypto.size > 0 and np.all(row_crypto < 0))
                    if is_bear and crypto_all_red:
                        lower, upper, net_target, lev = (
                            p.bear_weight_lower, p.bear_weight_upper, p.bear_net_target, p.bear_leverage_limit
                        )
                        regime = "bear"
                    else:
                        lower, upper, net_target, lev = (
                            p.bull_weight_lower, p.bull_weight_upper, p.bull_net_target, p.bull_leverage_limit
                        )
                        regime = "bull"
                    regime_changed = (regime != getattr(self, '_prev_regime', None))
                    self._prev_regime = regime
                else:
                    lower, upper, net_target, lev, regime_changed, regime = (
                        p.bull_weight_lower, p.bull_weight_upper, p.bull_net_target, p.bull_leverage_limit, False, "bull"
                    )

                try:
                    w_new = self.strategy.adjust_weights(
                        pred_returns=pred,
                        cov_matrix=cov,
                        prev_weights=weights,
                        market_vol=market_vol,
                        weight_lower=lower,
                        weight_upper=upper,
                        net_target=net_target,
                        lev_limit=lev,
                        regime_changed=regime_changed,
                        curr_regime=regime,
                    )
                except Exception as e:
                    logger.warning(f"adjust_weights failed on {date}: {e}, using previous weights.")
                    w_new = weights.copy()

                if not isinstance(w_new, np.ndarray) or w_new.shape != weights.shape or \
                   np.isnan(w_new).any() or np.isinf(w_new).any() or \
                   np.sum(np.abs(w_new - weights)) < 0.02:
                    w_new = weights.copy()

                turnover = np.sum(np.abs(w_new - prev_rebalance_weights))
                prev_rebalance_weights = w_new.copy()
                weights = w_new.copy()

                # Costs
                interval_fraction = p.rebalance_interval / 365
                tx_cost = p.transaction_cost * turnover * capital
                spread_cost = p.bid_ask_spread * turnover * capital
                mgmt_fee = p.management_fee * interval_fraction * capital
                profit_tax = p.tax_rate * max(0, capital - p.initial_capital) * interval_fraction
                borrow_cost = p.leverage_interest * max(0, np.sum(np.abs(w_new)) - 1) * interval_fraction * capital
                infl_cost = p.inflation_rate * interval_fraction * capital
                total_cost = tx_cost + spread_cost + mgmt_fee + profit_tax + borrow_cost + infl_cost
                capital -= total_cost

            capital_series.append(capital)
            weights_series.append(weights.copy())

        self.result = pd.DataFrame({"Date": dates[1:], "Portfolio Value": capital_series}).set_index("Date")
        self.weights_log = weights_series
        self.log_performance("adaptive", self.result["Portfolio Value"])
