from typing import Callable, Dict, List, Optional
from enum import Enum

from utils import get_logger
from params import Params

class AlertLevel(Enum):
    NONE = 0
    GREEN = 1
    ORANGE = 2
    RED = 3
    BLACK = 4

class RiskWatchdog:
    def __init__(
        self,
        universe: List[str],
        thresholds: Dict,
        params: Params,
        logger=None,
        check_interval: int = 3600,
        alert_callback: Optional[Callable[[AlertLevel], None]] = None
    ):
        self.universe = universe
        self.thresholds = thresholds
        self.check_interval = check_interval
        self.params = params
        self.logger = logger or get_logger("risk_watchdog", log_to_file=params.log_to_file)
        self.alert_callback = alert_callback

        self.prices = {ticker: [] for ticker in universe}
        self.portfolio_value_history = []
        self.current_alert_level = AlertLevel.NONE

    def check_black_alert(self) -> Optional[AlertLevel]:
        drops = []
        for ticker, history in self.prices.items():
            if len(history) >= 2:
                p1, p2 = history[-2], history[-1]
                if p1 > 0:
                    drop = (p2 - p1) / p1
                    drops.append(drop)
        if drops and sum(d <= -0.20 for d in drops) / len(drops) >= 0.60:
            return AlertLevel.BLACK
        return None

    def check_red_alert(self) -> Optional[AlertLevel]:
        recent_drops = []
        for ticker, history in self.prices.items():
            if len(history) >= 8:  # ~7-day
                p1, p2 = history[-8], history[-1]
                if p1 > 0:
                    drop = (p2 - p1) / p1
                    recent_drops.append(drop)
        if recent_drops:
            median_drop = sorted(recent_drops)[len(recent_drops) // 2]
            if median_drop < 0:
                return AlertLevel.RED
        return None

    def check_orange_alert(self) -> Optional[AlertLevel]:
        for ticker, history in self.prices.items():
            if len(history) >= 2:
                p1, p2 = history[-2], history[-1]
                if p1 > 0 and (p2 - p1) / p1 <= -0.05:
                    return AlertLevel.ORANGE
        return None

    def check_green_alert(self, weights: Dict[str, float]) -> Optional[AlertLevel]:
        for ticker, history in self.prices.items():
            if len(history) >= 1 and weights.get(ticker, 0) < 0.10:
                p0, p1 = history[0], history[-1]
                if p0 > 0 and (p1 - p0) / p0 >= 0.05:
                    return AlertLevel.GREEN
        return None

    def evaluate_alerts(self, weights: Dict[str, float]) -> AlertLevel:
        checks = [
            self.check_black_alert(),
            self.check_red_alert(),
            self.check_orange_alert(),
            self.check_green_alert(weights)
        ]
        valid_alerts = [a for a in checks if a is not None]
        return max(valid_alerts, key=lambda a: a.value) if valid_alerts else AlertLevel.NONE

    def check_alert(self, prices: Dict[str, float], weights: Dict[str, float]) -> AlertLevel:
        for ticker, price in prices.items():
            self.prices[ticker].append(price)

        portfolio_value = sum(weights.get(t, 0) * prices[t] for t in prices)
        self.portfolio_value_history.append(portfolio_value)

        alert = self.evaluate_alerts(weights)
        self.current_alert_level = alert

        if alert != AlertLevel.NONE:
            self.logger.warning(f"ALERT TRIGGERED: {alert.name}")
            if self.alert_callback:
                self.alert_callback(alert)

        return alert

    def get_current_alert_level(self) -> AlertLevel:
        return self.current_alert_level