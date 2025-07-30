# ----------------------------- #
#       analysis.py             #
# ----------------------------- #
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc  # or mplfinance equivalent
from typing import List, Optional

from params import Params
from utils import calculate_max_drawdown


class StrategyAnalyzer:
    def __init__(self,
                 prices: pd.DataFrame,
                 signal: pd.DataFrame,
                 bh_result: pd.Series,
                 adaptive_result: pd.Series,
                 adaptive_weights_log: List[np.ndarray],
                 params: Params):
        # Validate inputs
        if not isinstance(prices, pd.DataFrame):
            raise TypeError("prices must be a pandas DataFrame")
        if not isinstance(signal, pd.DataFrame):
            raise TypeError("signal must be a pandas DataFrame")
        if not (isinstance(bh_result, pd.Series) or (isinstance(bh_result, pd.DataFrame) and 'Portfolio Value' in bh_result.columns)):
            raise TypeError("bh_result must be a pandas Series or DataFrame with 'Portfolio Value'")
        if not (isinstance(adaptive_result, pd.Series) or (isinstance(adaptive_result, pd.DataFrame) and 'Portfolio Value' in adaptive_result.columns)):
            raise TypeError("adaptive_result must be a pandas Series or DataFrame with 'Portfolio Value'")
        if not isinstance(adaptive_weights_log, list):
            raise TypeError("adaptive_weights_log must be a list of numpy arrays")
        if not isinstance(params, Params):
            raise TypeError("params must be an instance of Params")

        self.prices = prices.copy()
        self.signal = signal.copy()
        self.buy_and_hold_result = bh_result.copy() if isinstance(bh_result, pd.Series) else bh_result[['Portfolio Value']].copy()
        self.adaptive_result = adaptive_result.copy() if isinstance(adaptive_result, pd.Series) else adaptive_result[['Portfolio Value']].copy()
        self.adaptive_weights_log = adaptive_weights_log.copy()
        self.params = params

    def plot_signal_vs_realized_return(self, ticker: str):
        if ticker not in self.signal.columns or ticker not in self.prices.columns:
            raise ValueError(f"Ticker '{ticker}' not found in signal or price data.")

        lookback = getattr(self.params, 'lookback', None)
        if lookback is None or not isinstance(lookback, int):
            raise ValueError("params.lookback must be defined as an integer")

        signal = self.signal[ticker].dropna()
        if signal.empty:
            raise ValueError(f"Signal series for {ticker} is empty.")

        future_return = self.prices[ticker].pct_change(periods=lookback).dropna()
        future_return = (1 + future_return) ** (1 / lookback) - 1

        common = signal.index.intersection(future_return.index)
        if common.empty:
            raise ValueError(f"No overlapping dates between signal and future returns for {ticker}.")

        signal = signal.loc[common]
        future_return = future_return.loc[common]

        # Time series plot
        plt.figure(figsize=(12, 4))
        plt.plot(common, signal, label="Predicted Signal", alpha=0.8)
        plt.plot(common, future_return, label="Future Return", alpha=0.5)
        plt.title(f"{ticker} — Signal vs. Realized Return")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Metrics
        if len(signal) < 2:
            print("Not enough data points to compute correlation metrics.")
            return
        pearson_corr = np.corrcoef(signal, future_return)[0, 1]
        spearman_corr, _ = spearmanr(signal, future_return)
        hit_rate = np.mean(np.sign(signal) == np.sign(future_return))

        print(f"📈 Pearson correlation:  {pearson_corr:.4f}")
        print(f"📈 Spearman correlation: {spearman_corr:.4f}")
        print(f"🎯 Directional hit rate:  {hit_rate:.2%}")

        # Decile performance
        df = pd.DataFrame({'Signal': signal, 'FutureReturn': future_return})
        try:
            df['Decile'] = pd.qcut(df['Signal'], 10, labels=False, duplicates='drop')
            decile_mean = df.groupby('Decile')['FutureReturn'].mean()
        except ValueError:
            print("Not enough variation in signal to compute deciles.")
            return

        plt.figure(figsize=(8, 4))
        sns.barplot(x=decile_mean.index, y=decile_mean.values, palette="coolwarm")
        plt.title("Average Future Return by Signal Decile")
        plt.xlabel("Signal Decile (Low → High)")
        plt.ylabel("Avg Future Return")
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()

    def plot_results(self, freq: str = "W-FRI"):
        # Determine series
        if isinstance(self.buy_and_hold_result, pd.Series):
            bh_series = self.buy_and_hold_result
        else:
            if 'Portfolio Value' not in self.buy_and_hold_result.columns:
                raise ValueError("buy_and_hold_result DataFrame must contain 'Portfolio Value' column.")
            bh_series = self.buy_and_hold_result['Portfolio Value']

        if isinstance(self.adaptive_result, pd.Series):
            adapt_series = self.adaptive_result
        else:
            if 'Portfolio Value' not in self.adaptive_result.columns:
                raise ValueError("adaptive_result DataFrame must contain 'Portfolio Value' column.")
            adapt_series = self.adaptive_result['Portfolio Value']

        # Resample to OHLC
        bh = bh_series.resample(freq).ohlc().dropna().reset_index()
        adapt = adapt_series.resample(freq).ohlc().dropna().reset_index()

        if bh.empty or adapt.empty:
            raise ValueError("Insufficient data for resampling to plot results.")

        bh['DateNum'] = bh['Date'].map(mdates.date2num)
        adapt['DateNum'] = adapt['Date'].map(mdates.date2num)

        quotes_adapt = adapt[['DateNum', 'open', 'high', 'low', 'close']].values
        quotes_bh = bh[['DateNum', 'open', 'high', 'low', 'close']].values

        bg_color = '#121212'
        grid_color = '#333333'
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), facecolor=bg_color, sharex=True)
        for ax in (ax1, ax2):
            ax.set_facecolor(bg_color)
            ax.grid(True, color=grid_color, linestyle='--', linewidth=0.7)
            for spine in ax.spines.values():
                spine.set_color('#666666')
            ax.tick_params(colors='white', labelsize=12)

        candlestick_ohlc(ax1, quotes_adapt, width=1.0, colorup='green', colordown='red', alpha=1.0)
        ax1.set_title('Adaptive Strategy (Weekly)', color='white', fontsize=18)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'${y:,.2f}'))

        candlestick_ohlc(ax2, quotes_bh, width=1.0, colorup='green', colordown='red', alpha=1.0)
        ax2.set_title('Buy & Hold Strategy (Weekly)', color='white', fontsize=18)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'${y:,.2f}'))

        ax2.xaxis_date()
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', color='white', fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_covariance_matrix(self, annotate: bool = False, figsize: tuple = (10, 8), cmap: str = 'viridis'):
        if self.prices.empty:
            raise ValueError("Price DataFrame is empty.")
        returns = self.prices.pct_change().dropna()
        if returns.empty:
            raise ValueError("Not enough data to compute returns for covariance matrix.")
        cov = returns.cov()
        plt.figure(figsize=figsize)
        sns.heatmap(cov, xticklabels=cov.columns, yticklabels=cov.index,
                    annot=annotate, fmt='.2e', cmap=cmap, linewidths=0.5)
        plt.title('Covariance Matrix of Portfolio Assets')
        plt.tight_layout()
        plt.show()

    def plot_correlation_matrix(self):
        if self.prices.empty:
            raise ValueError("Price DataFrame is empty.")
        returns = self.prices.pct_change().dropna()
        if returns.empty:
            raise ValueError("Not enough data to compute returns for correlation matrix.")
        corr = returns.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
        plt.title('Correlation Matrix of Portfolio Assets')
        plt.tight_layout()
        plt.show()

    def plot_return_and_price(self, ticker: str, return_window: int = 1):
        if ticker not in self.prices.columns:
            raise ValueError(f"No price data available for {ticker}")

        price = self.prices[ticker].dropna()
        if price.empty:
            raise ValueError(f"Price series for {ticker} is empty.")

        if return_window < 1:
            raise ValueError("return_window must be >= 1")

        returns = price.pct_change(periods=return_window).dropna()
        if returns.empty:
            raise ValueError("Not enough data to compute returns for the specified window.")
        returns.index = price.index[return_window:]

        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        axs[0].plot(returns.index, returns, label=f'{return_window}-Day Return', color='green')
        axs[0].set_title(f'{ticker} — Realized {return_window}-Day Return')
        axs[0].set_ylabel('Return')
        axs[0].grid(True)
        axs[0].legend()

        axs[1].plot(price.index, price, label='Price', color='orange')
        axs[1].set_title(f'{ticker} — Price Over Time')
        axs[1].set_ylabel('Price')
        axs[1].set_xlabel('Date')
        axs[1].grid(True)
        axs[1].legend()

        plt.tight_layout()
        plt.show()

    def compute_yoy_returns(self):
        if isinstance(self.adaptive_result, pd.DataFrame):
            if 'Portfolio Value' not in self.adaptive_result.columns:
                raise ValueError("adaptive_result DataFrame must contain 'Portfolio Value' column.")
            s = self.adaptive_result['Portfolio Value']
        else:
            if not isinstance(self.adaptive_result, pd.Series):
                raise TypeError("adaptive_result must be a pandas Series or DataFrame with 'Portfolio Value'.")
            s = self.adaptive_result

        if s.empty:
            raise ValueError("adaptive_result series is empty.")

        year_end = s.resample("A").last()
        yoy = year_end.pct_change().dropna()

        df = pd.DataFrame({"YoY Return": yoy.map(lambda x: f"{x:.2%}")})
        print("📅 Year-over-Year Returns:")
        print(df)

    def plot_weights_over_time(self, top_n: int = None):
        """
        Plot how asset weights evolve over time according to the adaptive strategy.
        If top_n is provided, only the top N assets by average weight are shown.
        """
        if not isinstance(self.adaptive_weights_log, list) or not self.adaptive_weights_log:
            raise ValueError("No adaptive_weights_log available or it is not a list.")

        # Align weight log length with adaptive_result index
        log = self.adaptive_weights_log
        if not hasattr(self.adaptive_result, "index"):
            raise ValueError("adaptive_result must have an index (e.g., a pandas Series or DataFrame).")

        # If there is an extra initial entry, drop it
        if len(log) > len(self.adaptive_result.index):
            log = log[1:]

        # Build a DataFrame: rows = timestamps, columns = tickers
        try:
            df = pd.DataFrame(log, columns=self.prices.columns, index=self.adaptive_result.index)
        except Exception as e:
            raise ValueError(f"Error constructing weights DataFrame: {e}")

        if top_n is not None:
            if not isinstance(top_n, int) or top_n < 1:
                raise ValueError("top_n must be a positive integer.")
            # Select top N tickers by average weight
            top_cols = df.mean().nlargest(top_n).index
            df = df[top_cols]

        plt.figure(figsize=(14, 6))
        df.plot(ax=plt.gca())
        plt.title("Asset Weights Over Time (Adaptive Strategy)")
        plt.xlabel("Date")
        plt.ylabel("Weight")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    def plot_asset_contributions(self, top_n: int = 15):
        """
        Plot the top N asset weight contributions based on the final adaptive allocation.
        """
        if not isinstance(self.adaptive_weights_log, list) or not self.adaptive_weights_log:
            raise ValueError("No adaptive_weights_log available or it is not a list.")

        last = self.adaptive_weights_log[-1]
        if not isinstance(last, (list, np.ndarray)):
            raise TypeError("Last entry in adaptive_weights_log must be a list or numpy array.")

        tickers = list(self.prices.columns)
        if len(last) != len(tickers):
            raise ValueError("Length of the final weight array does not match number of tickers.")

        # Pair tickers with their final weights and sort by weight descending
        contrib = sorted(zip(tickers, last), key=lambda x: x[1], reverse=True)[:top_n]
        if not contrib:
            raise ValueError("No contributions to plot (check adaptive_weights_log content).")

        top_tickers, top_weights = zip(*contrib)
        plt.figure(figsize=(10, 6))
        plt.bar(top_tickers, top_weights, color="skyblue")
        plt.ylabel("Weight")
        plt.title(f"Top {top_n} Asset Contributions (Final Allocation)")
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()
    def get_logs(self) -> pd.DataFrame:
        """
        Return the internal performance log as a pandas DataFrame.
        `self.performance_log` should be a list of dictionaries or a list of lists that can form a DataFrame.
        """
        if not hasattr(self, "performance_log"):
            raise AttributeError("No attribute 'performance_log' found.")
        try:
            return pd.DataFrame(self.performance_log)
        except Exception as e:
            raise ValueError(f"Error converting performance_log to DataFrame: {e}")
    def save_logs(self, path: str = "backtest_log.csv"):
        """
        Save the performance log DataFrame to a CSV file at the given path.
        """
        df = self.get_logs()
        try:
            df.to_csv(path, index=False)
        except Exception as e:
            raise IOError(f"Failed to save logs to '{path}': {e}")