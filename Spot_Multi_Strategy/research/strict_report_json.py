"""Lossless, deterministic JSON normalization for isolated research reports.

Numeric *keys* in audited diagnostics (for example calendar years) become
decimal strings. Collisions, nonfinite numbers and unrecognized types fail.
No strategy result is calculated here.
"""

from __future__ import annotations

from datetime import date, datetime
import json
import math

import numpy as np


def normalize(value: object) -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite report number")
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if isinstance(key, str):
                name = key
            elif type(key) is int or isinstance(key, np.integer):
                name = str(int(key))
            else:
                raise TypeError("report keys must be strings or integer calendar keys")
            if name in result:
                raise ValueError("report key collision after JSON normalization")
            result[name] = normalize(item)
        return result
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    raise TypeError(f"unsupported report value type: {type(value).__name__}")


def dumps(value: object) -> str:
    return json.dumps(normalize(value), ensure_ascii=False, indent=2,
                      allow_nan=False, sort_keys=True) + "\n"
