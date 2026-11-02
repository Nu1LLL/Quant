"""Synthetic-only checks of the frozen Fed-reserves crypto Gate."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from unittest.mock import patch

import pandas as pd
import pytest

from research import fed_reserves_crypto_metrics as metrics


@dataclass(frozen=True)
class Row:
    day: date
    equity_open: float
    equity_next_open: float
    return_net: float
    conservative_high_equity: float
    conservative_low_equity: float
    exposed: bool


@dataclass(frozen=True)
class Ledger:
    records: tuple[Row, ...]
    final_equity: float
    conservative_max_drawdown: float = 0.0
    conservative_max_drawdown_by_year: dict[int, float] = field(
        default_factory=lambda: {year: 0.0 for year in range(2021, 2026)}
    )
    fee_usdt: float = 0.0
    slippage_usdt: float = 0.0
    funding_usdt: float = 0.0
    turnover_usdt: float = 0.0
    trades: tuple[object, ...] = ()
    capacity_breaches: int = 0
    execution_breaches: int = 0
    margin_proxy_breaches: int = 0
    terminal_flat: bool = True


def _flat_ledger() -> Ledger:
    rows = []
    day = date(2021, 1, 1)
    while day < date(2026, 1, 1):
        rows.append(Row(day, 10_000.0, 10_000.0, 0.0,
                        10_000.0, 10_000.0, False))
        day += timedelta(days=1)
    return Ledger(tuple(rows), 10_000.0)


def _btc_bars() -> pd.DataFrame:
    index = pd.date_range("2020-12-01", "2026-01-01", freq="4h",
                          inclusive="left", tz="UTC", name="open_time")
    bars = pd.DataFrame({"close": 100.0}, index=index)
    terminal = pd.DataFrame({"close": [float("nan")]}, index=pd.DatetimeIndex(
        [pd.Timestamp("2026-01-01", tz="UTC")], name="open_time"
    ))
    return pd.concat((bars, terminal))


def test_fixed_five_year_gate_and_macro_specific_report():
    ledger = _flat_ledger()
    with patch.object(metrics, "block_monte_carlo", return_value={
        "paths": 10_000, "block_days": 21, "seed": 20261103,
        "passed": False,
    }) as mc:
        report = metrics.evaluate(ledger, ledger, _btc_bars())
    assert report["strategy"]["days"] == 1826
    assert report["historical_time_holdout_2024_2025"]["days"] == 731
    assert report["monthly_stability"]["complete_months"] == 60
    assert "wikimedia_affected_windows" not in report
    assert report["independent_oos"] is False
    assert report["full_gate"] is False
    assert report["numerical_gate"] is False
    assert report["benchmark_path_complete"] is True
    mc.assert_called_once()
    assert mc.call_args.kwargs == {
        "paths": 10_000, "block_days": 21, "seed": 20261103,
    }


def test_execution_and_missing_data_fail_closed():
    ledger = _flat_ledger()
    with patch.object(metrics, "block_monte_carlo", return_value={"passed": False}):
        report = metrics.evaluate(replace(ledger, execution_breaches=1), None,
                                  _btc_bars())
    assert report["gate_checks"]["minimum_order_execution"] is False
    assert report["benchmark_path_complete"] is False
    with pytest.raises(metrics.MetricsDataError, match="daily grid"):
        metrics.evaluate(replace(ledger, records=ledger.records[:-1]), None,
                         _btc_bars())
