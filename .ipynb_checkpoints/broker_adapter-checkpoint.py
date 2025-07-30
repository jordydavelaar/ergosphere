import ccxt
from typing import Dict, Optional
from datetime import datetime

from utils import get_logger
from params import Params
from universe import AssetUniverse
from portfolio import PortfolioState


class BrokerAdapter:
    """Adapter to interface with the Binance exchange via ccxt or dry-run simulation."""

    def __init__(self, params: Params, universe: AssetUniverse, portfolio: PortfolioState):
        self.logger = get_logger("broker_adapter", log_to_file=params.log_to_file)
        self.params = params
        self.universe = universe
        self.portfolio = portfolio

        self.exchange = self._initialize_exchange() if not params.dry_run else None
        self.symbol_map = self._initialize_symbol_map()

    def _initialize_exchange(self):
        if not self.params.api_key or not self.params.api_secret:
            self.logger.warning("API credentials are missing.")
        try:
            return ccxt.binance({
                'apiKey': self.params.api_key,
                'secret': self.params.api_secret,
                'enableRateLimit': True,
            })
        except Exception as e:
            self.logger.error(f"Failed to initialize exchange: {e}")
            raise

    def _initialize_symbol_map(self) -> Dict[str, str]:
        try:
            symbols = self.universe.all()
            return {ticker: self._standardize_symbol(ticker) for ticker in symbols}
        except Exception as e:
            self.logger.error(f"Failed to initialize symbol map: {e}")
            return {}

    def _standardize_symbol(self, ticker: str) -> str:
        if "/" in ticker:
            return ticker
        return ticker.replace("-", "/") if "-" in ticker else f"{ticker}/USDT"

    def get_price(self, ticker: str) -> float:
        if self.params.dry_run:
            return self.params.mock_prices.get(ticker, 1.0)

        symbol = self.symbol_map.get(ticker)
        if not symbol:
            self.logger.warning(f"Ticker {ticker} not in symbol_map.")
            return 0.0
        try:
            ticker_data = self.exchange.fetch_ticker(symbol)
            price = ticker_data.get('last', 0.0)
            if price == 0.0:
                self.logger.warning(f"Price for {ticker} ({symbol}) is zero.")
            return price
        except Exception as e:
            self.logger.error(f"Failed to fetch price for {ticker} ({symbol}): {e}")
            return 0.0

    def get_available_capital(self) -> float:
        if self.params.dry_run:
            return self.portfolio.get_capital()
        try:
            balance = self.exchange.fetch_balance()
            return balance.get('total', {}).get('USDT', 0.0)
        except Exception as e:
            self.logger.error(f"Failed to fetch available capital: {e}")
            return 0.0

    def get_current_positions(self) -> Dict[str, float]:
        if self.params.dry_run:
            return self.portfolio.get_weights()

        try:
            balance = self.exchange.fetch_balance()
            total_capital = self.get_available_capital()
            positions = {}

            for ticker in self.universe.all():
                symbol = self.symbol_map.get(ticker)
                if not symbol:
                    self.logger.warning(f"Ticker {ticker} not in symbol_map.")
                    positions[ticker] = 0.0
                    continue

                asset = symbol.split("/")[0]
                asset_balance = balance.get(asset, {})
                free_amount = asset_balance.get('total', 0.0)
                price = self.get_price(ticker)
                value = free_amount * price
                weight = value / total_capital if total_capital > 0 else 0.0
                positions[ticker] = weight

            return positions
        except Exception as e:
            self.logger.error(f"Failed to fetch current positions: {e}")
            return {}

    def get_cost_basis(self, ticker: str) -> float:
        cost_basis = self.params.mock_cost_basis.get(ticker)
        if cost_basis is None:
            cost_basis = self.get_price(ticker)
        return cost_basis

    def place_order(self, order: Dict) -> Optional[str]:
        symbol = self.symbol_map.get(order['ticker'])
        if not symbol:
            self.logger.warning(f"Cannot place order: unknown ticker {order['ticker']}")
            return None

        side = 'buy' if order['action'] == 'BUY' else 'sell'
        amount = order.get('units', 0)
        if amount <= 0:
            self.logger.warning(f"Skipping zero or negative size order for {order['ticker']}")
            return None

        if self.params.dry_run:
            self.logger.info(f"[DRY-RUN] Simulated {side} {amount} of {symbol}")
            fake_order = {
                "ticker": order["ticker"],
                "units": order["units"],
                "action": order["action"],
                "price": order["price"],
                "order_id": f"dry_{datetime.utcnow().timestamp()}",
                "timestamp": datetime.utcnow().isoformat()
            }
            self.portfolio.update_after_fill(fake_order)
            return fake_order["order_id"]

        try:
            result = self.exchange.create_market_order(symbol, side, amount)
            order_id = result.get("id", None)
            self.logger.info(f"Order placed: {side} {amount} of {symbol} | ID: {order_id}")

            if order_id:
                order_to_track = {
                    "ticker": order["ticker"],
                    "units": order["units"],
                    "action": order["action"],
                    "price": order["price"],
                    "order_id": order_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
                self.portfolio.add_pending_order(order_to_track)

            return order_id
        except Exception as e:
            self.logger.error(f"Order failed for {order['ticker']}: {e}")
            return None

    def check_order_status(self, order_id: str, symbol: str) -> str:
        if self.params.dry_run:
            return "closed"

        try:
            order = self.exchange.fetch_order(order_id, symbol)
            status = order.get("status", "unknown")
            self.logger.debug(f"Order {order_id} on {symbol} status: {status}")
            return status
        except Exception as e:
            self.logger.error(f"Failed to fetch status for order {order_id}: {e}")
            return "error"

    def mark_order_fulfilled(self, order: Dict):
        """Mark an order as fulfilled and update portfolio accordingly."""
        self.logger.info(f"Order filled: {order}")
        self.portfolio.update_after_fill(order)
        self.portfolio.save(self.params.portfolio_state_path)