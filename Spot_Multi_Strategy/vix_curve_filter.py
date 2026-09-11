"""Causal VIX/VIX3M term-structure filter for a precomputed return sleeve."""
from io import StringIO

import pandas as pd
import requests

from cboe_options_benchmark import summarize_returns


CBOE_HISTORY_URL = (
    "https://cdn.cboe.com/api/global/us_indices/daily_prices/{}_History.csv"
)


def load_cboe_close(symbol, output_dir):
    """Download and cache an official Cboe daily index close."""
    symbol = symbol.upper()
    if symbol not in {"VIX", "VIX3M"}:
        raise ValueError("Only the preregistered VIX and VIX3M series are allowed")
    response = requests.get(CBOE_HISTORY_URL.format(symbol), timeout=30)
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text))
    required = {"DATE", "CLOSE"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Unexpected Cboe schema for {symbol}: {list(frame)}")
    frame["DATE"] = pd.to_datetime(frame["DATE"], format="%m/%d/%Y", utc=True)
    frame["CLOSE"] = pd.to_numeric(frame["CLOSE"], errors="raise")
    frame = frame.sort_values("DATE").drop_duplicates("DATE", keep="last")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / f"{symbol}_History.csv", index=False)
    return frame.set_index("DATE")["CLOSE"].rename(symbol)


def build_risk_position(vix, vix3m, target_index=None):
    """Hold risk only when the prior common close has VIX <= VIX3M."""
    curve = pd.concat(
        [vix.rename("VIX"), vix3m.rename("VIX3M")], axis=1, sort=False
    )
    if target_index is not None:
        curve = curve.reindex(target_index)
        if curve.isna().any().any():
            missing = curve.index[curve.isna().any(axis=1)]
            raise ValueError(f"Missing Cboe curve observations: {list(missing[:5])}")
    else:
        curve = curve.dropna()
    risk_on_target = (curve["VIX"] <= curve["VIX3M"]).astype(float)
    return risk_on_target.shift(1).fillna(0.0).rename("risk_position")


def run_filter(
    base_returns, vix, vix3m, leverage=1.0, transaction_cost=0.0005,
    annual_financing=0.04
):
    base_returns = base_returns.sort_index().astype(float)
    position = build_risk_position(vix, vix3m, base_returns.index)
    turnover = position.diff().abs()
    turnover.iloc[0] = abs(position.iloc[0])
    filtered_1x = position * base_returns - turnover * transaction_cost
    financing = (
        (position * leverage - 1.0).clip(lower=0.0)
        * annual_financing / 252.0
    )
    net_return = filtered_1x * leverage - financing
    simulation, yearly, metrics = summarize_returns(net_return)
    simulation["base_return"] = base_returns.values
    simulation["risk_position"] = position.values
    simulation["gross_exposure"] = (position * leverage).values
    simulation["turnover"] = (turnover * leverage).values
    simulation["filter_trading_cost"] = (
        turnover * transaction_cost * leverage
    ).values
    simulation["financing_cost"] = financing.values
    simulation["equity"] = (1.0 + net_return).cumprod().values
    metrics.update({
        "average_risk_position": float(position.mean()),
        "filter_switches": int((turnover > 0.0).sum()),
        "total_filter_cost": float(
            (turnover * transaction_cost * leverage).sum() * 10000.0
        ),
    })
    return simulation, position, yearly, metrics
