"""Shared contracts for alpha, costs, risk and backtest execution."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict

import pandas as pd

from alphas.base import AlphaSignal
from risk_overlay import RiskConfig


@dataclass(frozen=True)
class CostModel:
    version: str = "spot-v1"
    fee_rate: float = 0.001
    slippage_rate: float = 0.0005

    def __post_init__(self):
        if not self.version.strip():
            raise ValueError("cost model version is required")
        if not 0 <= self.fee_rate < 1:
            raise ValueError("fee_rate must be in [0, 1)")
        if not 0 <= self.slippage_rate < 1:
            raise ValueError("slippage_rate must be in [0, 1)")


@dataclass(frozen=True)
class ResearchRunConfig:
    initial_capital: float = 5000.0
    execution_delay_bars: int = 1
    allow_short: bool = False
    cost: CostModel = field(default_factory=CostModel)
    risk: RiskConfig = field(default_factory=RiskConfig)

    def __post_init__(self):
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.execution_delay_bars < 1:
            raise ValueError("execution_delay_bars must be at least one")
        if self.allow_short:
            raise ValueError(
                "the current risk_overlay is long-only; short exposure requires "
                "a separately validated execution model"
            )

    def snapshot(self) -> Dict[str, Any]:
        return asdict(self)


def alpha_to_exposure(alpha: AlphaSignal, allow_short: bool = False) -> pd.Series:
    """Convert the repository's canonical AlphaSignal to target exposure.

    Existing Mini-Medallion semantics are preserved: negative alpha means cash,
    not a short position.  The explicit flag prevents accidental use of the
    long-only simulator for long/short research.
    """
    if not isinstance(alpha, AlphaSignal):
        raise TypeError("alpha must be an alphas.base.AlphaSignal")
    signal = alpha.normalized_signal.astype(float)
    if allow_short:
        return signal.clip(-1.0, 1.0)
    return signal.clip(0.0, 1.0)
