import numpy as np
import pandas as pd

import multicrypto_lowturnover_oos as strategy


def synthetic_panel(periods=900):
    index = pd.date_range("2020-01-01", periods=periods, freq="8h", tz="UTC")
    data = {}
    for number, symbol in enumerate(strategy.SYMBOLS, 1):
        data[f"{symbol}__spot"] = 10.0 + number + np.arange(periods) * 0.001
        data[f"{symbol}__mark"] = 10.0 + number + np.arange(periods) * 0.0012
        data[f"{symbol}__funding"] = np.full(periods, number * 0.00001)
    return pd.DataFrame(data, index=index)


def test_positions_are_lagged_and_fixed_gross():
    panel = synthetic_panel()
    sleeves, _, _, _ = strategy.sleeve_positions(panel)
    monday = panel.index[(panel.index.weekday == 0) & (panel.index.hour == 0)][-5]
    next_interval = monday + pd.Timedelta(hours=8)
    assert sleeves["funding_perp"].loc[monday].equals(
        sleeves["funding_perp"].loc[monday - pd.Timedelta(hours=8)]
    )
    assert np.isclose(sleeves["funding_perp"].loc[next_interval].abs().sum(), 0.35)
    assert np.isclose(sleeves["trend_perp"].loc[next_interval].abs().sum(), 0.35)


def test_funding_ranking_uses_two_low_and_two_high():
    panel = synthetic_panel()
    sleeves, _, _, _ = strategy.sleeve_positions(panel)
    active = sleeves["funding_perp"].abs().sum(axis=1).gt(0)
    row = sleeves["funding_perp"].loc[active].iloc[0]
    assert (row > 0).sum() == 2
    assert (row < 0).sum() == 2
    assert np.isclose(row.sum(), 0.0)


def test_capped_inverse_vol_is_normalized():
    result = strategy._capped_inverse_vol(
        pd.Series([1, 2, 3, 4, 5, 6], index=strategy.SYMBOLS),
        pd.Series([1, -1, 1, -1, 1, -1], index=strategy.SYMBOLS),
    )
    assert np.isclose(result.abs().sum(), 1.0)
    assert result.abs().max() <= 0.30 + 1e-12


def test_bankruptcy_does_not_revive():
    values = pd.Series([0.1, -1.2, 2.0], index=pd.date_range("2020-01-01", periods=3))
    result = strategy.apply_bankruptcy(values)
    assert result.tolist() == [0.1, -1.0, 0.0]


def test_coverage_rejects_short_panel():
    audit = strategy.coverage_audit(synthetic_panel())
    assert not audit["passed"]
