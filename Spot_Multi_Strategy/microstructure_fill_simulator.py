"""Conservative one-horizon taker fills for minute microstructure signals."""
import numpy as np
import pandas as pd

import microstructure_blackbox as blackbox


IDENTITY_COLUMNS = ("market", "venue", "symbol", "alpha_id")


def simulate_taker_fills(signals: pd.DataFrame,
                         market_frame: pd.DataFrame) -> pd.DataFrame:
    signals = pd.DataFrame(signals).copy().sort_index()
    market_frame = blackbox.validate_frame(market_frame)
    required = {*IDENTITY_COLUMNS, "cost_gate_bps", "valid_for_seconds",
                "kelly_position", "model_version"}
    if not required.issubset(signals.columns):
        raise ValueError(f"Signals missing {sorted(required - set(signals.columns))}")
    action_column = "raw_action" if "raw_action" in signals else "action"
    if action_column not in signals:
        raise ValueError("Signals need action or raw_action")
    if not signals.index.isin(market_frame.index).all():
        raise ValueError("Every signal timestamp needs an exact market quote")
    if signals[list(IDENTITY_COLUMNS)].nunique().max() != 1:
        raise ValueError("Fill simulation accepts one market/venue/symbol/alpha at a time")

    quote = market_frame[["best_bid", "best_ask"]]
    next_quote = quote.shift(-1).rename(columns={
        "best_bid": "next_bid", "best_ask": "next_ask"
    })
    next_timestamp = pd.Series(quote.index, index=quote.index).shift(-1)
    aligned = signals.join(quote).join(next_quote)
    aligned["exit_timestamp"] = next_timestamp.reindex(aligned.index)
    elapsed = (aligned["exit_timestamp"] - aligned.index.to_series()).dt.total_seconds()
    action = aligned[action_column].astype(str)
    eligible = (
        action.isin(["BUY", "SELL"])
        & aligned["next_bid"].notna() & aligned["next_ask"].notna()
        & elapsed.le(pd.to_numeric(aligned["valid_for_seconds"], errors="coerce"))
    )
    trades = aligned.loc[eligible].copy()
    if trades.empty:
        return pd.DataFrame(columns=[
            "signal_timestamp", "exit_timestamp", *IDENTITY_COLUMNS, "action",
            "entry_price", "exit_price", "gross_quote_return",
            "extra_cost_bps", "net_return", "portfolio_return", "model_version",
        ])
    trade_action = action.loc[eligible]
    buy = trade_action.eq("BUY")
    trades["entry_price"] = np.where(buy, trades["best_ask"], trades["best_bid"])
    trades["exit_price"] = np.where(buy, trades["next_bid"], trades["next_ask"])
    trades["gross_quote_return"] = np.where(
        buy,
        trades["exit_price"] / trades["entry_price"] - 1,
        trades["entry_price"] / trades["exit_price"] - 1,
    )
    mid = (trades["best_bid"] + trades["best_ask"]) / 2
    spread_bps = (trades["best_ask"] - trades["best_bid"]) / mid * 1e4
    cost_gate = pd.to_numeric(trades["cost_gate_bps"], errors="coerce")
    if not np.isfinite(cost_gate).all() or (cost_gate < spread_bps).any():
        raise ValueError("Cost gate must be finite and already include quoted spread")
    trades["extra_cost_bps"] = cost_gate - spread_bps
    trades["net_return"] = trades["gross_quote_return"] - trades["extra_cost_bps"] / 1e4
    trades["portfolio_return"] = trades["net_return"] * trades["kelly_position"].abs()
    trades["signal_timestamp"] = trades.index
    trades["action"] = trade_action
    columns = [
        "signal_timestamp", "exit_timestamp", *IDENTITY_COLUMNS, "action",
        "entry_price", "exit_price", "gross_quote_return", "extra_cost_bps",
        "net_return", "portfolio_return", "model_version",
    ]
    return trades[columns].reset_index(drop=True)
