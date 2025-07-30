# ----------------------------- #
#        universe.py            #
# ----------------------------- #

from dataclasses import dataclass, field
from typing import List
import yfinance as yf

from utils import (
    get_dynamic_crypto_universe,
    get_stock_universe,
    get_bonds_universe,
    get_cash_universe,
)
from params import Params  # changed for consistency

@dataclass
class AssetUniverse:
    params: Params  # changed for consistency

    # these will be populated in __post_init__
    crypto: List[str] = field(init=False)
    stock:  List[str] = field(init=False)
    bonds:  List[str] = field(init=False)
    cash:   List[str] = field(init=False)

    def __post_init__(self):
        # build each sleeve exactly once
        self.crypto = (
            get_dynamic_crypto_universe(top_n=self.params.top_n)
            if hasattr(self.params, "include_crypto") and self.params.include_crypto
            else []
        )
        self.stock = (
            get_stock_universe()
            if hasattr(self.params, "include_stock") and self.params.include_stock
            else []
        )
        self.bonds = (
            get_bonds_universe()
            if hasattr(self.params, "include_bonds") and self.params.include_bonds
            else []
        )
        self.cash = (
            get_cash_universe()
            if hasattr(self.params, "include_cash") and self.params.include_cash
            else []
        )

    def all(self) -> List[str]:
        """Returns all assets from enabled categories."""
        return self.crypto + self.stock + self.bonds + self.cash