import numpy as np
import pandas as pd

import long_multi_asset_golden_cross_oos as strategy


def synthetic_prices(start="2007-12-17", end="2026-09-01"):
    index = pd.bdate_range(start, end, tz="UTC")
    return pd.DataFrame({symbol: 100.0 * (1.0001 + i * 0.00001) ** np.arange(len(index))
                         for i, symbol in enumerate(strategy.SYMBOLS)}, index=index)


def test_align_requires_fixed_four_symbols():
    frame = synthetic_prices()
    aligned = strategy.align_prices({symbol: frame[symbol] for symbol in strategy.SYMBOLS})
    assert list(aligned.columns) == list(strategy.SYMBOLS)


def test_align_uses_calendar_date_and_rejects_duplicate():
    frame = synthetic_prices("2020-01-01", "2020-01-10")
    raw = {symbol: frame[symbol].copy() for symbol in strategy.SYMBOLS}
    raw["MDY"].index = raw["MDY"].index + pd.Timedelta(hours=14)
    assert len(strategy.align_prices(raw)) == len(frame)
    raw["MDY"] = pd.concat([raw["MDY"], raw["MDY"].iloc[[0]]])
    try:
        strategy.align_prices(raw)
        assert False
    except ValueError:
        pass


def test_coverage_accepts_long_complete_panel():
    assert strategy.coverage_audit(synthetic_prices())["passed"]


def test_sleeve_signal_is_lagged_and_liquidated():
    prices = synthetic_prices("2020-01-01", "2022-01-01")["MDY"]
    _, detail = strategy.run_sleeve(prices, leverage=1.0)
    first_active = detail.index[detail.position.gt(0)][0]
    assert first_active > prices.index[199]
    assert detail.iloc[-1].trading_cost >= strategy.LEG_COST


def test_portfolio_is_equal_weighted():
    prices = synthetic_prices()
    combined, sleeves, _ = strategy.run_portfolio(prices, leverage=1.0)
    pd.testing.assert_series_equal(combined, sleeves.mean(axis=1).rename("net_return"))
