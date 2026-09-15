import numpy as np
import pandas as pd

import chan_third_point_oos as strategy


def raw_frame(start="2008-05-01", end="2026-09-01", seed=1):
    index = pd.bdate_range(start, end, tz="UTC")
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.01, len(index)))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.005
    low = np.minimum(open_, close) * 0.995
    return pd.DataFrame({"raw_open": open_, "raw_high": high, "raw_low": low,
                         "raw_close": close, "adjusted_close": close * 0.9}, index=index)


def adjusted_markets():
    return strategy.align_markets({symbol: raw_frame(seed=i + 1)
                                   for i, symbol in enumerate(strategy.SYMBOLS)})


def test_ohlc_uses_same_day_adjustment_factor():
    frame = raw_frame("2020-01-01", "2020-01-10")
    adjusted = strategy.adjust_and_validate_ohlc(frame)
    assert np.allclose(adjusted["open"], frame["raw_open"] * 0.9)
    assert np.allclose(adjusted["high"], frame["raw_high"] * 0.9)


def test_fixed_four_market_coverage():
    markets = adjusted_markets()
    assert strategy.coverage_audit(markets)["passed"]
    assert tuple(markets) == strategy.SYMBOLS


def test_locked_pivot_requires_endpoint_after_fourth():
    points = [(0, -1, 8.0, 1), (2, 1, 12.0, 3),
              (4, -1, 10.0, 5), (6, 1, 11.0, 7)]
    assert strategy._locked_pivots(points[:-1]) == []
    points.append((8, -1, 10.5, 9))
    pivots = strategy._locked_pivots(points[:-1])
    assert len(pivots) == 1
    assert pivots[0]["zg"] == 11.0 and pivots[0]["zd"] == 10.0


def test_detects_third_buy_and_sell_from_confirmed_sequence():
    buy = [(0, -1, 8.0, 1), (2, 1, 12.0, 3), (4, -1, 10.0, 5),
           (6, 1, 11.0, 7), (8, -1, 10.5, 9), (10, 1, 13.0, 11),
           (12, -1, 11.5, 13)]
    buy_events = strategy.events_from_confirmed_fractals(buy)
    assert int(buy_events.iloc[-1].signal) == 1
    assert int(buy_events.iloc[-1].confirmed_at) == 13
    sell = [(0, 1, 12.0, 1), (2, -1, 8.0, 3), (4, 1, 10.0, 5),
            (6, -1, 9.0, 7), (8, 1, 9.5, 9), (10, -1, 7.0, 11),
            (12, 1, 8.5, 13)]
    sell_events = strategy.events_from_confirmed_fractals(sell)
    assert int(sell_events.iloc[-1].signal) == -1


def test_signal_is_applied_after_confirmation(monkeypatch):
    frame = strategy.adjust_and_validate_ohlc(raw_frame("2020-01-01", "2020-01-31"))
    event = pd.DataFrame([{"confirmed_at": 5, "signal": 1, "boundary": 1.0,
                           "zg": 1.0, "zd": 0.5, "departure_price": 2.0,
                           "retest_price": 1.5}])
    monkeypatch.setattr(strategy, "detect_events", lambda _: event)
    position, target, _, _ = strategy.build_position(frame)
    assert target.iloc[5] == 1.0
    assert position.iloc[5] == 0.0 and position.iloc[6] == 1.0


def test_mutating_future_prices_cannot_change_prior_positions():
    frame = strategy.adjust_and_validate_ohlc(raw_frame("2010-01-01", "2022-12-31"))
    original = strategy.build_position(frame)[0]
    cutoff = int(len(frame) * 0.8)
    changed = frame.copy()
    changed.iloc[cutoff:, :] *= 3.0
    revised = strategy.build_position(changed)[0]
    pd.testing.assert_series_equal(original.iloc[:cutoff], revised.iloc[:cutoff])


def test_portfolio_charges_final_liquidation():
    markets = adjusted_markets()
    returns, positions, _, _, _, detail = strategy.run_portfolio(markets, 1.0)
    assert returns.index.equals(positions.index)
    assert detail.iloc[-1].trading_cost >= positions.iloc[-1].abs().sum() * strategy.LEG_COST - 1e-12
