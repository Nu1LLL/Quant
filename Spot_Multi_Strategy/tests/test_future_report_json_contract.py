"""Synthetic regression for future one-shot report serialization.

No live sources, strategy signal, or historical PnL are read by these tests.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import json
from unittest.mock import patch

import numpy as np
import pytest

from research import fed_reserves_crypto_metrics as metrics
from research.future_report_json_contract import (
    dumps_report,
    json_safe_report,
    roundtrip_report,
)
from tests.test_fed_reserves_crypto_metrics import _btc_bars, _flat_ledger


def _diagnostics(*, benchmark: bool = True) -> dict:
    ledger = _flat_ledger()
    with patch.object(metrics, "block_monte_carlo", return_value={
        "paths": 10_000,
        "block_days": 21,
        "seed": metrics.MC_SEED,
        "passed": False,
    }):
        return metrics.evaluate(
            ledger, ledger if benchmark else None, _btc_bars(),
        )


def _report_with_years(years: dict[object, object]) -> dict:
    return {"diagnostics": {
        "strategy_execution": {"conservative_max_drawdown_by_year": years},
        "benchmark_execution": None,
    }}


def test_real_metrics_shape_roundtrips_without_mutating_synthetic_input():
    diagnostics = _diagnostics()
    source = {"diagnostics": diagnostics,
              "counts": np.int64(7), "passed": np.bool_(False),
              "date": date(2025, 12, 31),
              "utc": datetime(2025, 12, 31, tzinfo=timezone.utc)}
    assert set(diagnostics["strategy_execution"]["conservative_max_drawdown_by_year"]) == set(
        range(2021, 2026)
    )

    text = dumps_report(source)
    decoded = json.loads(text)
    assert decoded == roundtrip_report(source)
    assert decoded == json_safe_report(source)
    for branch in ("strategy_execution", "benchmark_execution"):
        annual = decoded["diagnostics"][branch]["conservative_max_drawdown_by_year"]
        assert list(annual) == [str(year) for year in range(2021, 2026)]
        assert all(value == 0.0 for value in annual.values())
        assert set(diagnostics[branch]["conservative_max_drawdown_by_year"]) == set(
            range(2021, 2026)
        )
    assert decoded["counts"] == 7 and type(decoded["counts"]) is int
    assert decoded["passed"] is False
    assert decoded["date"] == "2025-12-31"
    assert decoded["utc"] == "2025-12-31T00:00:00+00:00"


def test_missing_benchmark_execution_is_valid():
    decoded = roundtrip_report({"diagnostics": _diagnostics(benchmark=False)})
    assert decoded["diagnostics"]["benchmark_execution"] is None
    assert list(decoded["diagnostics"]["strategy_execution"][
        "conservative_max_drawdown_by_year"
    ]) == [str(year) for year in range(2021, 2026)]


@pytest.mark.parametrize("years, error", [
    ({2021: 0.0}, "every year"),
    ({**{year: 0.0 for year in range(2021, 2026)}, 2026: 0.0}, "outside"),
    ({**{year: 0.0 for year in range(2021, 2026)}, "2021": 0.0}, "collision"),
    ({**{year: 0.0 for year in range(2021, 2026)}, True: 0.0}, "boolean"),
    ({**{year: 0.0 for year in range(2021, 2026)}, 2023: float("nan")}, "finite"),
    ({**{year: 0.0 for year in range(2021, 2026)}, 2023: -1.1}, "outside"),
    ({**{year: 0.0 for year in range(2021, 2026)}, 2023: True}, "numeric"),
    ({**{year: 0.0 for year in range(2021, 2026)}, 2023: Decimal("-0.1")}, "numeric"),
])
def test_annual_map_rejects_ambiguous_or_invalid_values(years, error):
    with pytest.raises((TypeError, ValueError), match=error):
        roundtrip_report(_report_with_years(years))


def test_only_the_two_known_year_maps_are_normalized():
    years = {year: -0.1 for year in range(2021, 2026)}
    payload = _report_with_years(years)
    payload["other"] = {"2021": "remain-a-string"}
    assert roundtrip_report(payload)["other"] == {"2021": "remain-a-string"}
    payload["other"] = {2021: "not-a-known-annual-map"}
    with pytest.raises(TypeError, match="keys must be strings"):
        roundtrip_report(payload)


@pytest.mark.parametrize("value, error", [
    (float("nan"), "nonfinite"),
    (np.float32("inf"), "nonfinite"),
    (Decimal("0.1"), "unsupported"),
    (np.array([1.0]), "unsupported"),
    (datetime(2025, 1, 1), "timezone-aware UTC"),
    (datetime(2025, 1, 1, tzinfo=timezone(timedelta(hours=1))), "timezone-aware UTC"),
])
def test_generic_json_contract_fails_closed(value, error):
    with pytest.raises((TypeError, ValueError), match=error):
        dumps_report({"nested": [value]})
