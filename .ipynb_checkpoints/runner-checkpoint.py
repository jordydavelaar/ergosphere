import datetime
import time
from data_loader import DataLoader
from utils import get_logger
from params import Params
from universe import AssetUniverse
from portfolio import PortfolioState
from broker_adapter import BrokerAdapter
from strategy import MomentumStrategy
from watchdog import RiskWatchdog, AlertLevel
from optimizer import Optimizer


class Runner:
    def __init__(self, dry_run: bool = False):
        self.params = Params()
        self.params.dry_run = dry_run
        self.universe = AssetUniverse(self.params)
        self.portfolio = PortfolioState(initial_capital=self.params.initial_capital)
        self.broker = BrokerAdapter(self.params, self.universe)
        self.strategy = MomentumStrategy(self.params)
        self.optimizer = Optimizer(self.params)
        self.logger = get_logger("runner", log_to_file=self.params.log_to_file)

        self.data_loader = DataLoader(self.universe, self.params)

        self.watchdog = RiskWatchdog(
            universe=self.universe.symbols,
            thresholds={
                "drop_pct": -0.05,
                "weight_threshold": 0.10
            },
            params=self.params,
            alert_callback=self.handle_alert
        )

        if dry_run:
            self.data_loader.download_data(interval="1h", overwrite=False)
            self.hourly_prices = self.data_loader.get_prices(interval="1h")
            self.strategy.set_prices(self.hourly_prices)
        else:
            self.data_loader.download_data(interval="1d", overwrite=False)
            self.hourly_prices = None

    def run_hourly_check(self):
        self.logger.info("Running hourly check.")

        if self.params.dry_run:
            self.logger.debug("Using hourly prices for dry run.")
        else:
            if self.hourly_prices is None:
                self.data_loader.download_data(interval="1h", overwrite=False)
                self.hourly_prices = self.data_loader.get_prices(interval="1h")

        self.check_pending_orders()

        latest_prices = self.hourly_prices.iloc[-1].to_dict()
        latest_weights = self.portfolio.get_weights()
        alert = self.watchdog.check_alert(latest_prices, latest_weights)

        if alert == AlertLevel.RED:
            self.logger.warning("Red alert triggered. Executing emergency rebalance.")
            self.optimizer.emergency_rebalance()
        elif alert == AlertLevel.BLACK:
            self.logger.critical("Black alert triggered! Execute full liquidation (TODO).")
            # TODO: self.execution_engine.liquidate_all_to_safe_assets()
        else:
            self.logger.info("Market conditions normal. No action taken.")

        self._save_portfolio_snapshot()

    def run_weekly_rebalance(self):
        self.logger.info("Weekly rebalance triggered.")
        prices = self.data_loader.get_prices(interval="1d")
        self.strategy.set_prices(prices)
        self.strategy.generate_signal()
        self.strategy.adjust_weights()
        self.optimizer.execute_rebalance()

    def check_pending_orders(self):
        pending_orders = self.portfolio.get_pending_orders()
        if not pending_orders:
            self.logger.info("No pending orders to verify.")
            return

        for order in pending_orders:
            order_id = order.get("order_id")
            symbol = self.broker.symbol_map.get(order["ticker"])
            if not order_id or not symbol:
                continue

            try:
                status = self.broker.exchange.fetch_order_status(order_id, symbol)
                if status == 'closed':
                    self.logger.info(f"Order {order_id} completed. Logging trade.")
                    self.portfolio.log_trade(order)
                    self.portfolio.mark_order_fulfilled(order_id)
                else:
                    self.logger.info(f"Order {order_id} still open: {status}")
            except Exception as e:
                self.logger.warning(f"Failed to check order {order_id}: {e}")

    def _save_portfolio_snapshot(self):
        try:
            self.portfolio._save_state()
            self.logger.info("Hourly portfolio snapshot saved.")
        except Exception as e:
            self.logger.error(f"Failed to save portfolio snapshot: {e}")

    def handle_alert(self, alert: AlertLevel):
        self.logger.warning(f"[ALERT CALLBACK] Triggered alert: {alert.name}")
        # Future: Send email, Slack, webhook, or escalate to execution engine here.

    def start_loop(self):
        self.logger.info("Starting main runner loop.")
        while True:
            now = datetime.datetime.now()
            self.run_hourly_check()

            if now.weekday() == 0 and now.hour == 9:
                self.run_weekly_rebalance()

            time.sleep(3600)

    def dry_loop(self, hours: int = 48):
        self.logger.info(f"[DRY LOOP] Starting dry-run loop for {hours} virtual hours.")
        now = datetime.datetime.now()

        for hour in range(hours):
            virtual_time = now + datetime.timedelta(hours=hour)
            self.logger.info(f"[DRY LOOP] Simulated time: {virtual_time.strftime('%Y-%m-%d %H:%M')}")
            self.run_hourly_check()

            if virtual_time.weekday() == 0 and virtual_time.hour == 9:
                self.run_weekly_rebalance()

        self.logger.info("[DRY LOOP] Completed dry-run simulation.")