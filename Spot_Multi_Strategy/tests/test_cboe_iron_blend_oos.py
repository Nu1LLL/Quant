import numpy as np
import pandas as pd

import cboe_iron_blend_oos as strategy


def sample_levels():
    index = pd.bdate_range("1987-01-01", "2026-09-01", tz="UTC")
    periods = len(index)
    return pd.DataFrame({"CNDR": 100.0 * 1.0002 ** np.arange(periods),
                         "BFLY": 100.0 * 1.0001 ** np.arange(periods)}, index=index)


def test_monthly_blend_charges_entry_exit_and_drag():
    levels = pd.DataFrame(100.0, index=pd.bdate_range("2024-01-01", periods=30, tz="UTC"),
                          columns=strategy.SYMBOLS)
    returns, turnover = strategy.blend_returns(levels)
    assert np.isclose(turnover.iloc[0], 1.0)
    assert np.isclose(turnover.iloc[-1], 1.0)
    assert np.isclose(returns.sum(), -0.002 - 30 * 0.02 / 252.0)


def test_weight_drift_is_rebalanced_at_month_start():
    levels = pd.DataFrame({"CNDR": [100, 110, 110], "BFLY": [100, 100, 100]},
                          index=pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01"], utc=True))
    _, turnover = strategy.blend_returns(levels, annual_implementation_drag=0.0)
    assert turnover.iloc[1] == 0.0
    assert turnover.iloc[2] > 0.0


def test_leverage_financing_and_bankruptcy():
    index = pd.date_range("2024-01-01", periods=3)
    base = pd.Series([0.01, -0.4, 0.5], index=index)
    result = strategy.apply_leverage(base, 3.0)
    assert result.iloc[0] < 0.03
    assert result.tolist()[1:] == [-1.0, 0.0]


def test_coverage_accepts_long_complete_panel():
    assert strategy.coverage_audit(sample_levels())["passed"]


def test_coverage_rejects_nonpositive_level():
    levels = sample_levels()
    levels.iloc[10, 0] = 0.0
    assert not strategy.coverage_audit(levels)["passed"]
