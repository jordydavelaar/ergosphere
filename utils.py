# ----------------------------- #
#          utils.py             #
# ----------------------------- #

import requests, os
import pandas as pd
import yfinance as yf
from typing import List, Optional
import logging
from datetime import datetime

def get_logger(name: str, log_dir: str = "logs", log_to_file: bool = True):
    logger = logging.getLogger(name)
    
    if logger.hasHandlers():
        return logger  # Avoid adding handlers multiple times

    logger.setLevel(logging.INFO)
    
    # Stream (console) handler
    formatter = logging.Formatter("[%(asctime)s] %(name)s — %(levelname)s: %(message)s")

    # Optional file handler
    if log_to_file:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"{name}_{timestamp}.log")
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger

def validate_yf_tickers(tickers: List[str], min_days: int = 500, logger=None) -> List[str]:
    """
    Return only tickers with at least `min_days` of Yahoo Finance history.
    """
    valid: List[str] = []
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period="max")
            if hist.empty:
                logger.warning(f"{t} skipped — no price data found")
            elif len(hist) < min_days:
                logger.warning(f"{t} skipped — only {len(hist)} days of data (needs ≥ {min_days})")
            else:
                valid.append(t)
        except Exception as e:
            logger.warning(f"Skipping {t}: {e}")
    return valid

def get_dynamic_crypto_universe(
    top_n: int = 10,
    start_date: str = "2023-11-01"
) -> List[str]:
    """
    Fetch top-N tokens by market cap from CoinGecko and filter by Yahoo Finance coverage.
    """
    static_excluded = {
        "USDT", "USDC", "DAI", "USDS", "USDE",
        "WBTC", "WETH", "STETH", "WSTETH", "WEETH",
        "CBBTC", "BSC-USD", "LEO", "BGB", "WBT",
        "OKB", "TKX", "TAO", "BUIDL", "HYPE", "SUI"
    }

    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": top_n,
        "page": 1,
        "sparkline": False
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    raw_syms = [coin["symbol"].upper() for coin in data]

    def to_yf(sym: str) -> str:
        return sym.upper() if sym.endswith("-USD") else f"{sym.upper()}-USD"

    candidates = [to_yf(s) for s in raw_syms if s not in static_excluded]

    df = (
        yf.download(
            tickers=candidates,
            start=start_date,
            auto_adjust=True,
            progress=False
        )
        .get("Close")
    )
    df.index = df.index.normalize()

    first_obs = df.apply(lambda col: col.first_valid_index())
    cutoff = pd.to_datetime(start_date)
    dynamic_excluded = set(first_obs[first_obs > cutoff].index)

    return [t for t in candidates if t not in dynamic_excluded]

def get_stock_universe(sectors: Optional[List[str]] = None) -> List[str]:
    """
    Return static equity tickers, or filter by sectors if provided.
    """
    full_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "JNJ", "V", "JPM", "PG"]
    if sectors is None:
        return full_list
    raise NotImplementedError("Sector-based filtering not implemented.")

def get_bonds_universe(risk: str = "short") -> List[str]:
    """
    Return bond ETF tickers. Valid risk levels: 'short', 'intermediate', 'aggregate'.
    """
    mapping = {
        "short": ["SHY"],
        "intermediate": ["IEF"],
        "aggregate": ["BND"]
    }
    return mapping.get(risk, mapping["short"])

def get_cash_universe() -> List[str]:
    """
    Return tickers for cash-like ultra-short ETFs.
    """
    return ["BIL"]

def calculate_max_drawdown(prices: pd.Series) -> float:
    """
    Compute the maximum drawdown of a portfolio or asset price series.
    Returns a positive float representing the worst drop from peak to trough.
    """
    if prices.empty or not isinstance(prices, pd.Series):
        raise ValueError("prices must be a non-empty pandas Series")

    cumulative_max = prices.cummax()
    drawdowns = (prices - cumulative_max) / cumulative_max
    max_drawdown = drawdowns.min()

    return abs(max_drawdown)