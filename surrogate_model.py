import numpy as np
import pandas as pd
import pickle
from itertools import product
from copy import deepcopy
from tqdm import tqdm
from schwimmbad import MultiPool
from scipy.interpolate import RBFInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from strategy import MomentumStrategy
from backtester import AdaptiveBacktester
from utils import get_logger

class SurrogateModel:
    def __init__(self, params, universe, data, param_specs=None):
        self.logger = get_logger("surrogate_model", log_to_file=params.log_to_file)
        self.params = params
        self.universe = universe
        self.data = data
        self.results = []
        self.X = None
        self.Y = None

        # If no custom grid is provided, use these seven dimensions by default:
        self.param_specs = param_specs if param_specs else [
            ("max_weight_shift",    np.linspace(0.05, 1.0, 10)),
            ("bull_weight_upper",   np.linspace(0.05, 1.0, 10)),
            ("bull_net_target",     np.linspace(0.5, 1.5, 10)),
            ("bull_leverage_limit", np.linspace(1.0, 3.0, 5)),
            ("bear_weight_lower",   np.linspace(-1.0, 0.0, 5)),
            ("bear_net_target",     np.linspace(-1.0, 0.0, 5)),
            ("bear_leverage_limit", np.linspace(1.0, 3.0, 5)),
        ]

        self.fixed_params = {}   # if you want to pin certain params
        self.tasks = []
        self.bounds = []
        self.cagr_rbf = None
        self.sharpe_rbf = None
        self.gp = None

    def define_grid(self):
        """Builds self.tasks (all combinations) and self.bounds from param_specs."""
        self.logger.info("Defining parameter grid…")
        self.param_names = [name for name, _ in self.param_specs]
        grids = [grid for _, grid in self.param_specs]
        self.bounds = [(grid.min(), grid.max()) for grid in grids]
        self.tasks = list(product(*grids))
        self.logger.info(f"Total grid points: {len(self.tasks)}")

    def run_backtests(self, n_processes=8):
        """
        Runs one AdaptiveBacktester for each point in self.tasks (in parallel).
        Stores raw (CAGR, Sharpe) in self.results, then builds self.X/self.Y without any NaNs.
        """
        self.logger.info("Running backtests with parallel processing…")

        def run_adaptive(param_values):
            # 1) clone self.params and override grid dims + any fixed_params:
            p = deepcopy(self.params)
            for name, val in zip(self.param_names, param_values):
                setattr(p, name, val)
            for name, val in self.fixed_params.items():
                setattr(p, name, val)

            try:
                # 2) instantiate MomentumStrategy(correct arg order!), generate signal:
                strat = MomentumStrategy(self.universe, p, self.data)
                strat.generate_signal()

                # 3) backtest:
                bt = AdaptiveBacktester(self.data, strat, p)
                bt.run()
                perf = bt.performance_log[-1]
                if (perf["CAGR"] < -10) or np.isnan(perf["CAGR"]):
                    return (-1e6, -1e6)
                return (perf["CAGR"], perf["Sharpe Ratio"])

            except Exception as e:
                self.logger.warning(f"Infeasible at {param_values}: {e}")
                return (-1e6, -1e6)

        # Parallel map over self.tasks:
        with MultiPool(processes=n_processes) as pool:
            self.results = list(tqdm(pool.map(run_adaptive, self.tasks), total=len(self.tasks)))

        # Build X, Y arrays:
        self.X = np.array(self.tasks)
        Y_dirty = np.array(self.results)   # shape = (len(self.tasks), 2)

        # Filter out rows where we returned (-1e6, -1e6):
        mask_valid = ~((Y_dirty[:,0] == -1e6) & (Y_dirty[:,1] == -1e6))
        if not np.any(mask_valid):
            raise RuntimeError("All grid points were infeasible. Nothing to fit.")
        self.X = self.X[mask_valid]
        self.Y = Y_dirty[mask_valid]

        self.logger.info("Plotting raw performance data (before fitting)…")
        # if at least two dims, do a 3D scatter of first two dims vs CAGR
        if self.X.shape[1] >= 2:
            fig = plt.figure(figsize=(10,6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.X[:, 0], self.X[:, 1], self.Y[:, 0],
                       c=self.Y[:, 0], cmap='viridis', marker='o')
            ax.set_xlabel(self.param_names[0])
            ax.set_ylabel(self.param_names[1])
            ax.set_zlabel("CAGR")
            plt.title("Raw Backtest Results (CAGR)")
            plt.tight_layout()
            plt.show()
        else:
            # 1D case: just plot X[:,0] vs Y[:,0]
            plt.figure(figsize=(8,5))
            plt.plot(self.X[:,0], self.Y[:,0], "o-")
            plt.xlabel(self.param_names[0])
            plt.ylabel("CAGR")
            plt.title("Raw Backtest Results (CAGR)")
            plt.tight_layout()
            plt.show()

    def fit_surrogate_models(self):
        """
        Fit:
          - an RBF interpolator for CAGR and Sharpe
          - a GP on X → CAGR
        """
        self.logger.info("Fitting surrogate models (RBF + GP)…")
        # RBF on the cleaned X,Y:
        self.cagr_rbf = RBFInterpolator(self.X, self.Y[:,0], smoothing=0.1)
        self.sharpe_rbf = RBFInterpolator(self.X, self.Y[:,1], smoothing=0.1)

        # GP on X → CAGR
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(self.X.shape[1])) + WhiteKernel()
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3)
        self.gp.fit(self.X, self.Y[:,0])
        self.logger.info("Surrogate models fitted.")

    def optimize_rbf(self):
        """
        Find x* that maximizes the RBF‐predicted CAGR (i.e. minimizes negative).
        """
        self.logger.info("Optimizing RBF surrogate for max CAGR…")

        def neg_cagr(x):
            return -self.cagr_rbf(x.reshape(1,-1))[0]

        x0 = np.mean(self.X, axis=0)
        opt = minimize(neg_cagr, x0=x0, bounds=self.bounds)
        self.logger.info(f"RBF Max CAGR: {-opt.fun:.4f} at {opt.x}")
        return opt

    def predict_gp(self, n_samples=100):
        """
        Sample random points in the hypercube defined by self.bounds, then predict with GP.
        Returns (x_query, y_pred, y_std).
        """
        self.logger.info("Predicting with Gaussian Process surrogate…")
        dim = len(self.bounds)
        low  = [b[0] for b in self.bounds]
        high = [b[1] for b in self.bounds]
        x_query = np.random.uniform(low, high, size=(n_samples, dim))
        y_pred, y_std = self.gp.predict(x_query, return_std=True)
        self.logger.info(f"GP best predicted CAGR: {np.max(y_pred):.4f}")
        return x_query, y_pred, y_std

    def save(self, filename, opt_result=None):
        """
        Save the fitted surrogates + best params dictionary in a pickle.
        """
        best_params_dict = {}
        if opt_result is not None:
            best_params_dict = {
                name: val for name, val in zip(self.param_names, opt_result.x)
            }

        tuned_params = deepcopy(self.params)
        for k, v in best_params_dict.items():
            setattr(tuned_params, k, v)

        save_dict = {
            "param_names":      self.param_names,
            "X":                self.X,
            "Y":                self.Y,
            "cagr_rbf":         self.cagr_rbf,
            "sharpe_rbf":       self.sharpe_rbf,
            "gp":               self.gp,
            "full_params":      tuned_params,
            "best_params_dict": best_params_dict,
        }
        with open(filename, "wb") as f:
            pickle.dump(save_dict, f)

        self.logger.info(f"Surrogate model and best-tuned params saved to {filename}")

    def plot_slices(self, fixed_dims={}):
        """
        Plot 2D contour slices of the GP, holding all but first two dims fixed
        according to fixed_dims (dictionary {param_name: value}).
        """
        self.logger.info("Plotting 2D slices with fixed parameters (GP)…")
        if len(self.param_names) < 2:
            self.logger.warning("Need at least 2 dimensions for a 2D slice.")
            return

        slice_dims = self.param_names[:2]
        # create 200×200 grid for the first two dims:
        grid_axes = [np.linspace(*self.bounds[i], 200) for i in range(2)]
        grid1, grid2 = np.meshgrid(*grid_axes)

        # For dims ≥2, use fixed=mean(bound) unless overridden in fixed_dims:
        extra_dims = []
        for i, name in enumerate(self.param_names[2:], start=2):
            if name in fixed_dims:
                extra_dims.append(fixed_dims[name])
            else:
                extra_dims.append( np.mean(self.bounds[i]) )

        all_points = np.array([
            [v1, v2] + extra_dims
            for v1, v2 in zip(np.ravel(grid1), np.ravel(grid2))
        ])

        cagr_vals, std_vals = self.gp.predict(all_points, return_std=True)
        cagr_vals = cagr_vals.reshape(grid1.shape)
        std_vals  = std_vals.reshape(grid1.shape)

        plt.figure(figsize=(10,6))
        cp = plt.pcolormesh(grid1, grid2, cagr_vals, cmap='viridis', shading='auto')
        plt.colorbar(cp)
        plt.xlabel(slice_dims[0])
        plt.ylabel(slice_dims[1])
        plt.title(f"CAGR Contour Slice (GP)\nFixed: {fixed_dims}")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10,6))
        cp_std = plt.pcolormesh(grid1, grid2, std_vals, cmap='coolwarm', shading='auto')
        plt.colorbar(cp_std)
        plt.xlabel(slice_dims[0])
        plt.ylabel(slice_dims[1])
        plt.title(f"GP Uncertainty (Std Dev)\nFixed: {fixed_dims}")
        plt.tight_layout()
        plt.show()