# ----------------------------- #
#      execution_engine.py     #
# ----------------------------- #

from utils import get_logger
from params import Params
from broker_adapter import BrokerAdapter
from universe import AssetUniverse
from portfolio import PortfolioState
from typing import Dict, List

class ExecutionEngine:
    def __init__(self, params: Params, universe: AssetUniverse, portfolio: PortfolioState, broker: BrokerAdapter):
        self.logger = get_logger("execution_engine", log_to_file=params.log_to_file)
        self.params = params
        self.universe = universe
        self.broker = broker
        self.portfolio = portfolio  # now stores pending_orders

    def execute_rebalance(self, new_weights: Dict[str, float]):
        self.logger.info("Executing scheduled weekly rebalance.")
        self._rebalance_to_target(new_weights)

    def emergency_rebalance(self, bad_assets: List[str]):
        self.logger.warning(f"Emergency rebalance triggered for: {bad_assets}")
        current_positions = self.broker.get_current_positions()
        safe_assets = {k: v for k, v in current_positions.items() if k not in bad_assets}

        total_safe_weight = sum(safe_assets.values())
        if total_safe_weight == 0:
            self.logger.warning("No safe assets to rebalance into. Aborting.")
            return

        redistributed_weights = {
            ticker: weight / total_safe_weight
            for ticker, weight in safe_assets.items()
        }

        self._rebalance_to_target(redistributed_weights)

    def _rebalance_to_target(self, target_weights: Dict[str, float]):
        current_positions = self.portfolio.get_weights()
        capital = self.portfolio.get_capital()
        trades = self._compute_orders(current_positions, target_weights, capital)

        cost = self._log_and_estimate_trade_costs(trades, target_weights)

        if self.params.dry_run:
            self.logger.info(f"[DRY RUN] Orders to execute: {trades}")
        else:
            for order in trades:
                order_id = self.broker.place_order(order)
                order['order_id'] = order_id
                self.portfolio.add_pending_order(order)

        self.logger.info("Rebalance execution complete.")

    def check_pending_orders(self):
        completed = []
        for order in self.portfolio.pending_orders:
            status = self.broker.check_order_status(order['order_id'], order['ticker'])
            if status == "closed":
                self.broker.mark_order_fulfilled(order)
                completed.append(order)
        for order in completed:
            self.portfolio.pending_orders.remove(order)
        if completed:
            self.logger.info(f"Updated portfolio with {len(completed)} filled trades.")

    def _compute_orders(self, current: Dict[str, float], target: Dict[str, float], capital: float) -> List[Dict]:
        orders = []
        for ticker in self.universe.all():
            target_weight = target.get(ticker, 0.0)
            curr_weight = current.get(ticker, 0.0)
            delta_weight = target_weight - curr_weight

            if abs(delta_weight) < self.params.min_trade_delta:
                continue

            action = "BUY" if delta_weight > 0 else "SELL"
            amount = abs(delta_weight) * capital
            price = self.broker.get_price(ticker)
            if price == 0:
                self.logger.warning(f"Price for {ticker} is zero. Skipping.")
                continue
            units = amount / price
            cost_basis = self.broker.get_cost_basis(ticker) if action == "SELL" else None

            orders.append({
                "ticker": ticker,
                "action": action,
                "amount": amount,
                "price": price,
                "units": units,
                "cost_basis": cost_basis
            })

        return orders

    def _log_and_estimate_trade_costs(self, trades: List[Dict], target_weights: Dict[str, float]) -> float:
        p = self.params
        interval_fraction = p.rebalance_interval / 365
        total_cost = 0.0
        capital = self.broker.get_available_capital()

        for trade in trades:
            amount = trade["amount"]
            action = trade["action"]
            units = trade["units"]
            price = trade["price"]
            cost_basis = trade.get("cost_basis")

            slippage = amount * p.slippage_rate
            fee = amount * p.transaction_fee_rate
            mgmt_fee = capital * p.management_fee * interval_fraction
            inflation = capital * p.inflation_rate * interval_fraction
            leverage_penalty = max(0, sum(abs(w) for w in target_weights.values()) - 1) * capital * p.leverage_interest * interval_fraction

            tax = 0.0
            if action == "SELL" and cost_basis is not None:
                gain = max(0, (price - cost_basis) * units)
                tax = gain * p.tax_rate

            trade_cost = slippage + fee + mgmt_fee + inflation + leverage_penalty + tax
            total_cost += trade_cost

            self.logger.info(
                f"[{action}] {trade['ticker']}: ${amount:.2f} | "
                f"Slippage ${slippage:.2f}, Fee ${fee:.2f}, "
                f"Mgmt ${mgmt_fee:.2f}, Infl ${inflation:.2f}, "
                f"{'Tax $' + format(tax, '.2f') if tax > 0 else 'No tax'} → "
                f"Total Cost ${trade_cost:.2f}"
            )

        self.logger.info(f"Total estimated cost of rebalance: ${total_cost:.2f}")
        return total_cost