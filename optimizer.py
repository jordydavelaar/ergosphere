# ----------------------------- #
#         optimizer.py         #
# ----------------------------- #

from execution_engine import ExecutionEngine
from strategy import MomentumStrategy
from universe import AssetUniverse
from params import Params
from utils import get_logger


class Optimizer:
    def __init__(self, params: Params):
        self.logger = get_logger("optimizer", log_to_file=params.log_to_file)
        self.params = params
        self.universe = AssetUniverse(params)
        self.strategy = MomentumStrategy(params)
        self.engine = ExecutionEngine(params, self.universe, dry_run=False)

    def execute_rebalance(self):
        self.logger.info("Running full weekly rebalance...")
        self.strategy.generate_signal()
        target_weights = self.strategy.get_target_weights()
        self.engine.execute_rebalance(target_weights)

    def emergency_rebalance(self):
        self.logger.warning("Running emergency rebalance...")
        bad_assets = self.strategy.detect_bad_assets()
        self.engine.emergency_rebalance(bad_assets)