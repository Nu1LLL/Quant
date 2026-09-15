import numpy as np
import pandas as pd
import pytest

import chan_third_point_live_oos as strategy


def raw_frame(start="2013-12-18", end="2026-09-01", seed=1):
    index = pd.bdate_range(start, end, tz="UTC")
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.01, len(index)))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({
        "raw_open": open_, "raw_high": np.maximum(open_, close) * 1.005,
        "raw_low": np.minimum(open_, close) * 0.995, "raw_close": close,
        "adjusted_close": close * 0.9,
    }, index=index)


def markets():
    return strategy.align_markets({symbol: raw_frame(seed=i + 1)
                                   for i, symbol in enumerate(strategy.SYMBOLS)})


def test_fixed_live_universe_and_coverage():
    aligned = markets()
    assert tuple(aligned) == ("SCHB", "USDU", "DBB", "BIV")
    assert strategy.coverage_audit(aligned)["passed"]


def test_missing_symbol_is_rejected():
    raw = {symbol: raw_frame(seed=i + 1) for i, symbol in enumerate(strategy.SYMBOLS)}
    raw.pop("USDU")
    with pytest.raises(ValueError):
        strategy.align_markets(raw)


def test_unregistered_leverage_is_rejected():
    with pytest.raises(ValueError):
        strategy.run_portfolio(markets(), 6.0)


def test_wrapper_uses_frozen_engine_and_fixed_equal_weights(monkeypatch):
    aligned = markets()
    index = aligned["SCHB"].index
    event = pd.DataFrame(columns=["confirmed_at", "signal", "boundary", "zg", "zd",
                                  "departure_price", "retest_price"])
    def fixed_position(frame):
        position = pd.Series(1.0, index=frame.index)
        return position, position.copy(), pd.Series(np.nan, index=frame.index), event.copy()
    monkeypatch.setattr(strategy.core, "build_position", fixed_position)
    returns, positions, _, _, _, _, _ = strategy.run_portfolio(aligned, 1.0)
    assert returns.index.equals(positions.index)
    assert np.allclose(positions.abs().sum(axis=1), 1.0)
    assert np.allclose(positions["SCHB"], 0.25)
