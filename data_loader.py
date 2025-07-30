import os
import pickle
import pandas as pd
import yfinance as yf
from utils import get_logger


class DataLoader:
    def __init__(self, universe, params, cache_path: str = "cached_data.pkl"):
        self.logger = get_logger("data_loader", log_to_file=params.log_to_file)
        self.universe = universe
        self.params = params
        self.start_date = params.start_date
        self.end_date = params.end_date
        self.cache_path = cache_path

        self.prices_daily = None
        self.prices_hourly = None
        self.volume_daily = None
        self.CVI = None
        self.VIX = None

    def download_data(self, overwrite: bool = False, interval: str = "1d") -> None:
        if not overwrite and os.path.exists(self.cache_path):
            try:
                full_df = self._load_cache()
            except Exception as e:
                self.logger.warning(f"⚠️ Cache load failed: {e}. Downloading fresh data.")
                full_df = self._download_and_process()
        else:
            full_df = self._download_and_process()

        if interval == "1d":
            self.prices_daily = full_df.resample("1d").last().ffill(limit=3)
            self._compute_volatility_indices(full_df)
        elif interval == "1h":
            self.prices_hourly = full_df.resample("1h").last().ffill(limit=3)
            # CVI/VIX not used in hourly-only mode
        else:
            raise ValueError(f"Unsupported interval: {interval}")

    def _download_and_process(self) -> pd.DataFrame:
        close_df, volume_df = self._download_yahoo()

        close_df = self._process_df(close_df)
        volume_df = self._process_df(volume_df)

        self.volume_daily = volume_df  # Store for volume weighting
        self._save_cache(close_df)     # Only cache price for now

        return close_df
    
    def _compute_volatility_indices(self, full_df: pd.DataFrame):
        # --- CVI ---
        crypto_prices = full_df[self.universe.crypto].ffill() if self.universe.crypto else pd.DataFrame(index=full_df.index)
        if not crypto_prices.empty:
            vol = crypto_prices.pct_change().rolling(7, min_periods=1).std().mean(axis=1).fillna(0.0)
            baseline_vol = vol.median()
            cvi = (vol / baseline_vol) * 100
        else:
            cvi = pd.Series(0.0, index=full_df.index)
        self.CVI = cvi.resample("1d").last().ffill(limit=3)

        # --- VIX ---
        try:
            vix_data = yf.download("^VIX", start=self.start_date, end=self.end_date, auto_adjust=True, progress=False).Close
            vix_data = vix_data.ffill().reindex(full_df.index).ffill()
        except Exception as e:
            self.logger.warning(f"⚠️ VIX download failed: {e}")
            vix_data = pd.Series(20.0, index=full_df.index)
        self.VIX = vix_data.resample("1d").last().ffill(limit=3)

    def _load_cache(self) -> pd.DataFrame:
        self.logger.info("Loading data from cache...")
        with open(self.cache_path, "rb") as f:
            return self._order_and_normalize(pickle.load(f))

    def _save_cache(self, df: pd.DataFrame):
        with open(self.cache_path, "wb") as f:
            pickle.dump(df, f)

    def _download_yahoo(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        tickers = self.universe.all()
        self.logger.info(f"Downloading {len(tickers)} tickers from Yahoo Finance...")
        raw = yf.download(
            tickers,
            group_by="ticker",
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True,
            progress=False
        )

        if not isinstance(raw.columns, pd.MultiIndex):
            raise ValueError("Expected MultiIndex columns from yf.download(group_by='ticker')")

        try:
            close_df = raw.xs("Close", axis=1, level=1)
            volume_df = raw.xs("Volume", axis=1, level=1)
        except KeyError as e:
            raise ValueError(f"Missing required data: {e}")

        return close_df.dropna(how="all"), volume_df.dropna(how="all")

    def _process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df.index >= pd.Timestamp(self.start_date)]
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        df = df.tz_localize("UTC") if df.index.tz is None else df.tz_convert("US/Eastern")

        df.index = df.index.normalize()
        df = df.asfreq("D").dropna(how="any")
        return self._order_and_normalize(df)

    def _order_and_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [t for t in self.universe.all() if t in df.columns]
        return df[cols]

    def get_prices(self, interval="1d"):
        if interval == "7d":
            return self.prices_daily.resample("W").last().dropna(how="all")
        elif interval == "1d":
            return self.prices_daily
        elif interval == "1h":
            return self.prices_hourly
        else:
            raise ValueError(f"Unknown interval: {interval}")

    def get_volatility_indices(self) -> tuple:
        return self.CVI, self.VIX

    def get_volume(self, interval="1d"):
        if interval == "1d":
            return self.volume_daily
        else:
            raise NotImplementedError("Only daily volume is currently supported.")