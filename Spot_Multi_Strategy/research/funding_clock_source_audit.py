"""Read-only funding-clock audit over the pinned 215 launch windows.

This checks timestamp mapping, not the sealed launch strategy's return/PnL.
The sealed result is never imported or rerun.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path

from .binance_new_perpetual_data import _funding_rows
from .crypto_blackbox.funding_clock import settled_hour_ms, unique_settlements


HERE = Path(__file__).parent
MANIFEST = HERE / "preregistrations/2026-11-08_binance_futures_launch_funding_integrity.json"
OUTPUT = HERE / "results/2026-11-08_funding_clock_source_audit.json"


def _one(row: dict) -> tuple[str, list[int], int]:
    start = datetime.fromisoformat(row["entry_utc"])
    end = datetime.fromisoformat(row["exit_utc"])
    start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    source_raw = [int(event.time.timestamp() * 1000)
                  for source in row["sources"] for event in _funding_rows(source)]
    raw_held = [stamp for stamp in source_raw if start_ms <= stamp < end_ms]
    if len(raw_held) != row["funding_events_during_hold"]:
        raise ValueError("pinned raw funding event count changed")
    relevant = [stamp for stamp in source_raw
                if (start_ms <= stamp < end_ms
                    or start_ms <= settled_hour_ms(stamp) < end_ms)]
    anchors = unique_settlements(relevant)
    changed_boundary = sum((start_ms <= stamp < end_ms) !=
                           (start_ms <= anchor < end_ms)
                           for stamp, anchor in zip(relevant, anchors))
    return row["contract_symbol"], [stamp - settled_hour_ms(stamp) for stamp in relevant], changed_boundary


def audit() -> dict:
    if OUTPUT.exists():
        raise FileExistsError("funding clock source audit already exists")
    source_bytes = MANIFEST.read_bytes()
    manifest = json.loads(source_bytes)
    if manifest.get("events") != 215:
        raise ValueError("registered source event count changed")
    result_rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = [pool.submit(_one, row) for row in manifest["rows"]]
        for future in as_completed(pending):
            result_rows.append(future.result())
    offsets = [offset for _, group, _ in result_rows for offset in group]
    if not offsets:
        raise ValueError("no funding settlement times audited")
    report = {
        "source_manifest_sha256": sha256(source_bytes).hexdigest(),
        "windows": len(result_rows),
        "funding_events": len(offsets),
        "nonzero_millisecond_offsets": sum(offset > 0 for offset in offsets),
        "affected_contracts": sum(any(offset > 0 for offset in group)
                                  for _, group, _ in result_rows),
        "max_post_hour_offset_ms": max(offsets),
        "offset_ms_counts": dict(sorted(Counter(offsets).items())),
        "window_boundary_assignment_changes": sum(changed for _, _, changed in result_rows),
        "source_only_no_strategy_return_read": True,
    }
    with OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


if __name__ == "__main__":
    result = audit()
    print(json.dumps({key: result[key] for key in
                      ("windows", "funding_events", "nonzero_millisecond_offsets",
                       "affected_contracts", "max_post_hour_offset_ms",
                       "window_boundary_assignment_changes")}, sort_keys=True))
