"""Point-in-time round-trip execution costs for heterogeneous markets."""
import numpy as np
import pandas as pd


MARKET_ALIASES = {
    "stock": "equity", "stocks": "equity", "etf": "equity",
    "options": "option", "future": "futures", "forex": "fx",
    "crypto_spot": "crypto", "crypto_perpetual": "crypto", "perpetual": "crypto",
}


def _column(frame, name, allow_negative=False):
    if name not in frame:
        raise ValueError(f"Execution cost input missing {name}")
    values = pd.to_numeric(frame[name], errors="coerce")
    if not np.isfinite(values).all() or (not allow_negative and (values < 0).any()):
        raise ValueError(f"Execution cost input {name} is invalid")
    return values.astype(float)


def round_trip_cost_bps(frame: pd.DataFrame, market: str) -> pd.Series:
    """Convert point-in-time market-specific costs to comparable basis points."""
    frame = pd.DataFrame(frame)
    market = MARKET_ALIASES.get(str(market).lower(), str(market).lower())
    if market not in {"equity", "option", "futures", "fx", "crypto"}:
        raise ValueError(f"Unsupported execution-cost market {market}")
    bid, ask = _column(frame, "best_bid"), _column(frame, "best_ask")
    mid = (bid + ask) / 2
    if (bid <= 0).any() or (ask <= bid).any():
        raise ValueError("Execution costs require positive non-crossed quotes")
    spread = (ask - bid) / mid * 1e4
    financing = _column(frame, "financing_bps_horizon", allow_negative=True)

    if market == "equity":
        commission = 2 * _column(frame, "commission_per_unit_each_side") / mid * 1e4
        impact = 2 * _column(frame, "impact_bps_each_side")
    elif market in {"option", "futures"}:
        commission_per_contract = _column(frame, "commission_per_contract_each_side")
        multiplier = _column(frame, "contract_multiplier")
        tick_size = _column(frame, "tick_size")
        slippage_ticks = _column(frame, "slippage_ticks_each_side")
        if (multiplier <= 0).any() or (tick_size <= 0).any():
            raise ValueError("Contract multiplier and tick size must be positive")
        commission = 2 * commission_per_contract / (mid * multiplier) * 1e4
        impact = 2 * slippage_ticks * tick_size / mid * 1e4
    elif market == "fx":
        commission = 2 * _column(frame, "commission_per_million_each_side") / 1e6 * 1e4
        impact = 2 * _column(frame, "impact_bps_each_side")
    else:
        commission = 2 * _column(frame, "fee_bps_each_side")
        impact = 2 * _column(frame, "impact_bps_each_side")

    execution_floor = spread + commission + impact
    total = execution_floor + financing
    return pd.Series(np.maximum(total, execution_floor), index=frame.index,
                     name="round_trip_cost_bps")
