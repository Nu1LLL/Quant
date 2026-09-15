import numpy as np
import pandas as pd

import daa_g12_oos as strategy


def synthetic_prices(start="2006-07-03", end="2026-09-01"):
    index = pd.bdate_range(start, end, tz="UTC")
    return pd.DataFrame({
        symbol: 100.0 * (1.0001 + rank * 0.000002) ** np.arange(len(index))
        for rank, symbol in enumerate(strategy.SYMBOLS)
    }, index=index)


def test_fixed_universe_and_long_coverage():
    prices = synthetic_prices()
    aligned = strategy.align_prices({symbol: prices[symbol] for symbol in strategy.SYMBOLS})
    assert list(aligned.columns) == list(strategy.SYMBOLS)
    assert strategy.coverage_audit(aligned)["passed"]


def test_13612w_formula_is_exact():
    index = pd.date_range("2020-01-31", periods=13, freq="ME", tz="UTC")
    monthly = pd.DataFrame({"X": np.arange(100.0, 113.0)}, index=index)
    score = strategy.momentum_13612w(monthly).iloc[-1, 0]
    expected = 12 * (112 / 111 - 1) + 4 * (112 / 109 - 1)
    expected += 2 * (112 / 106 - 1) + (112 / 100 - 1)
    assert np.isclose(score, expected)


def test_canary_breadth_controls_fixed_risky_count(monkeypatch):
    dates = pd.DatetimeIndex([pd.Timestamp("2020-12-31", tz="UTC")])
    scores = pd.DataFrame(
        [{symbol: float(len(strategy.SYMBOLS) - i)
          for i, symbol in enumerate(strategy.SYMBOLS)}], index=dates,
    )
    scores.loc[:, "VWO"] = -1.0
    scores.loc[:, "AGG"] = 1.0
    scores.loc[:, "SHY"] = 99.0
    monkeypatch.setattr(strategy, "momentum_13612w", lambda _: scores)
    monkeypatch.setattr(strategy, "monthly_levels", lambda prices: prices)
    weights, _ = strategy.build_month_end_weights(synthetic_prices("2020-01-01", "2020-01-03"))
    risky_weight = weights.loc[dates[0], list(strategy.RISKY)].sum()
    assert np.isclose(risky_weight, 0.5)
    assert np.isclose(weights.loc[dates[0]].sum(), 1.0)
    assert (weights.loc[dates[0], list(strategy.RISKY)] > 0).sum() == 3


def test_month_end_signal_is_applied_one_trading_day_later():
    prices = synthetic_prices("2018-01-01", "2020-03-31")
    positions, month_end, _ = strategy.build_daily_positions(prices)
    signal_date = month_end.index[0]
    signal_location = prices.index.get_loc(signal_date)
    next_date = prices.index[signal_location + 1]
    assert positions.loc[signal_date].abs().sum() == 0.0
    assert np.isclose(positions.loc[next_date].abs().sum(), 1.0)


def test_future_price_change_cannot_change_prior_positions():
    prices = synthetic_prices("2018-01-01", "2021-12-31")
    original, _, _ = strategy.build_daily_positions(prices)
    cutoff = pd.Timestamp("2020-06-15", tz="UTC")
    changed = prices.copy()
    changed.loc[changed.index > cutoff, "SPY"] *= 7.0
    revised, _, _ = strategy.build_daily_positions(changed)
    pd.testing.assert_frame_equal(original.loc[:cutoff], revised.loc[:cutoff])


def test_backtest_charges_initial_and_final_liquidation():
    prices = synthetic_prices("2018-01-01", "2021-12-31")
    returns, positions, _, _, detail = strategy.run_backtest(prices, leverage=1.0)
    assert returns.index[0] == positions.index[0]
    assert detail.iloc[0].trading_cost > 0.0
    assert detail.iloc[-1].trading_cost >= strategy.LEG_COST - 1e-12
