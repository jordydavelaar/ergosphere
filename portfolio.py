import os
import json
from typing import Dict, List
from datetime import datetime

from utils import get_logger


class PortfolioState:
    def __init__(self, initial_capital: float, save_path: str = "portfolio_state.json"):
        self.logger = get_logger("portfolio_state")
        self.save_path = save_path

        if os.path.exists(self.save_path):
            self._load_state()
        else:
            self.state = {
                "capital": initial_capital,
                "positions": {},       # ticker: units
                "weights": {},         # ticker: weight
                "history": [],         # list of trade dicts
                "pending_orders": []   # list of order dicts
            }
            self._save_state()

    def _load_state(self):
        with open(self.save_path, "r") as f:
            self.state = json.load(f)
        self.logger.info(f"Loaded portfolio state from {self.save_path}")

    def _save_state(self):
        try:
            with open(self.save_path, "w") as f:
                json.dump(self.state, f, indent=2)
            self.logger.debug("Portfolio state saved.")
        except Exception as e:
            self.logger.error(f"Failed to save portfolio state: {e}")

    def log_trade(self, trade: Dict):
        self.logger.info(f"Logging trade: {trade}")
        self.state["history"].append(trade)
        self._update_positions(trade)
        self._save_state()

    def _update_positions(self, trade: Dict):
        ticker = trade["ticker"]
        units = trade["units"]
        price = trade["price"]
        action = trade["action"]

        if action == "BUY":
            self.state["capital"] -= units * price
            self.state["positions"][ticker] = self.state["positions"].get(ticker, 0) + units
        elif action == "SELL":
            self.state["capital"] += units * price
            self.state["positions"][ticker] = self.state["positions"].get(ticker, 0) - units
            if self.state["positions"][ticker] <= 0:
                del self.state["positions"][ticker]

    def update_weights(self, weights: Dict[str, float]):
        self.logger.info(f"Updating weights: {weights}")
        self.state["weights"] = weights
        self._save_state()

    def add_pending_order(self, order: Dict):
        self.logger.info(f"Adding pending order: {order}")
        self.state["pending_orders"].append(order)
        self._save_state()

    def mark_order_fulfilled(self, order_id: str):
        before = len(self.state["pending_orders"])
        self.state["pending_orders"] = [
            o for o in self.state["pending_orders"] if o.get("order_id") != order_id
        ]
        after = len(self.state["pending_orders"])
        if before != after:
            self.logger.info(f"Marked order {order_id} as fulfilled.")
        self._save_state()

    def reset(self, initial_capital: float):
        self.logger.warning("Resetting portfolio state")
        self.state = {
            "capital": initial_capital,
            "positions": {},
            "weights": {},
            "history": [],
            "pending_orders": []
        }
        self._save_state()

    def get_capital(self) -> float:
        return self.state["capital"]

    def get_positions(self) -> Dict[str, float]:
        return self.state["positions"]

    def get_weights(self) -> Dict[str, float]:
        return self.state["weights"]

    def get_pending_orders(self) -> List[Dict]:
        return self.state["pending_orders"]
    
    def update_after_fill(self, order: Dict):
        """Handle post-fill update: log trade and remove from pending."""
        self.log_trade(order)
        self.mark_order_fulfilled(order.get("order_id"))