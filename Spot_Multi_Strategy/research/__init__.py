"""Unified point-in-time research infrastructure.

The package wraps the existing alpha, risk and backtest modules.  It does not
replace the legacy discrete-trade engine or the Mini-Medallion exposure engine.
"""

from .contracts import CostModel, ResearchRunConfig, alpha_to_exposure
from .data_store import MarketDataStore, utc_us
from .registry import ResearchRegistry
from .runner import ResearchRunner

__all__ = [
    "CostModel",
    "MarketDataStore",
    "ResearchRegistry",
    "ResearchRunConfig",
    "ResearchRunner",
    "alpha_to_exposure",
    "utc_us",
]
