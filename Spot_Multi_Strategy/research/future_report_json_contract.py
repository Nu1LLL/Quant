"""Strict, side-effect-free JSON contract for future sealed research reports.

The frozen five-year metrics evaluator emits integer year keys in its annual
drawdown metadata. JSON object keys must be strings. Only those two known
report paths are normalized here; unexpected key types fail rather than being
silently stringified. This module does not read market data or write reports.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, timedelta
import json
import math
from typing import Any

import numpy as np


ANNUAL_YEARS = tuple(range(2021, 2026))
_ANNUAL_PATHS = frozenset({
    ("diagnostics", "strategy_execution", "conservative_max_drawdown_by_year"),
    ("diagnostics", "benchmark_execution", "conservative_max_drawdown_by_year"),
})


def _annual_year(key: object) -> str:
    if isinstance(key, (bool, np.bool_)):
        raise TypeError("annual drawdown year key must not be boolean")
    if isinstance(key, (int, np.integer)):
        year = int(key)
    elif isinstance(key, str) and key in {str(item) for item in ANNUAL_YEARS}:
        year = int(key)
    else:
        raise TypeError("annual drawdown year key must be a canonical 2021-2025 year")
    if year not in ANNUAL_YEARS:
        raise ValueError("annual drawdown year outside 2021-2025")
    return str(year)


def _annual_drawdowns(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError("annual drawdowns must be a mapping")
    result: dict[str, float] = {}
    for key, raw in value.items():
        year = _annual_year(key)
        if year in result:
            raise ValueError(f"annual drawdown normalized year collision: {year}")
        if isinstance(raw, (bool, np.bool_)) or not isinstance(
            raw, (int, float, np.integer, np.floating),
        ):
            raise TypeError(f"annual drawdown for {year} must be numeric")
        drawdown = float(raw)
        if not math.isfinite(drawdown):
            raise ValueError(f"annual drawdown for {year} must be finite")
        if not -1.0 <= drawdown <= 0.0:
            raise ValueError(f"annual drawdown for {year} outside [-1, 0]")
        result[year] = drawdown
    if set(result) != {str(year) for year in ANNUAL_YEARS}:
        raise ValueError("annual drawdowns must contain every year 2021-2025 exactly once")
    return {str(year): result[str(year)] for year in ANNUAL_YEARS}


def _normalize_known_paths(value: object, path: tuple[object, ...] = ()) -> object:
    if path in _ANNUAL_PATHS:
        return _annual_drawdowns(value)
    if isinstance(value, Mapping):
        return {
            key: _normalize_known_paths(item, path + (key,))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_known_paths(item, path) for item in value]
    return value


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("nonfinite numeric report value")
        return number
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("report datetime must be timezone-aware UTC")
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("report object keys must be strings")
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"unsupported report value type: {type(value).__name__}")


def json_safe_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and copy a report; normalize only known five-year annual maps."""
    if not isinstance(payload, Mapping):
        raise TypeError("report payload must be a mapping")
    result = _json_safe(_normalize_known_paths(payload))
    assert isinstance(result, dict)  # The mapping branch always returns a dict.
    return result


def dumps_report(payload: Mapping[str, Any]) -> str:
    """Serialize with no NaN and no silent scalar or object-key coercion."""
    return json.dumps(
        json_safe_report(payload), ensure_ascii=False, indent=2, allow_nan=False,
    ) + "\n"


def roundtrip_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Preflight the exact report tree before a one-shot terminal write."""
    decoded = json.loads(dumps_report(payload))
    assert isinstance(decoded, dict)
    return decoded
