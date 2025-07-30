# ergosphere

Some example run:
ergosphere.ipynb

**Finished:**

data_loader.py (download data from yfinance and set up data structures)

universe.py (data class for assets)

params.py (parameter class, either set to defaults or load from file)

strategy.py (momentum strategy builder)

analysis.py (analysis tools for plotting)

surrogate_model.py (builds a GP model on the grid to find optimal strategy parameters)

backtester.py (runs a backtest on historical data, comparing with a buy-and-hold scenario)

utils.py (log functions and some others misc functions)

**In progress:**

broker_adapter.py (interface with ccxt)

runner.py (functions that are ran either hourly or weekly)

optimizer.py (class that calls the executions for rebalancing protfolio)

watchdog.py (checks for certain market conditions that require emergency rebalance, stop-light system GREEN (nothing needed) ORANGE (partial shift of crypto to TradFin) RED (full shift to TradFin)

