import numpy as np
import pandas as pd

import cboe_ptlt_oos as strategy


def long_levels():
    index = pd.bdate_range("2008-01-01", "2026-09-01", tz="UTC")
    return pd.DataFrame({"PTLT": 100.0 * 1.0001 ** np.arange(len(index))}, index=index)


def test_coverage_accepts_long_complete_history():
    assert strategy.coverage_audit(long_levels())["passed"]


def test_coverage_rejects_late_start():
    assert not strategy.coverage_audit(long_levels().loc["2012":])["passed"]


def test_base_returns_charge_drag_and_endpoints():
    levels = pd.DataFrame({"PTLT": [100.0, 100.0, 100.0]},
                          index=pd.bdate_range("2024-01-01", periods=3, tz="UTC"))
    returns = strategy.base_returns(levels)
    assert np.isclose(returns.sum(), -3 * 0.015 / 252.0 - 0.002)


def test_leverage_deducts_financing():
    base = pd.Series([0.01, -0.01], index=pd.date_range("2024-01-01", periods=2))
    result = strategy.apply_leverage(base, 2.0)
    pd.testing.assert_series_equal(result, base * 2.0 - 0.04 / 252.0)


def test_bankruptcy_does_not_revive():
    base = pd.Series([0.1, -0.4, 1.0], index=pd.date_range("2024-01-01", periods=3))
    assert strategy.apply_leverage(base, 3.0).tolist()[1:] == [-1.0, 0.0]
