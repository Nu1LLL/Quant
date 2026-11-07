"""Single-shot formal result writer; sealed outcomes cannot be overwritten."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from .weekday_seasonality_data import WeekdayDataError, load_tape
from .weekday_seasonality_sim import WeekdayStrategyFailure, evaluate


RESULT = Path(__file__).parent / "results/eight_coin_weekday_seasonality_2026_11_07.json"
PREREG_COMMIT = "ac5f39b"


def _head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def run_one_shot() -> dict:
    if RESULT.exists():
        raise FileExistsError("formal weekday-seasonality outcome already exists; no rerun")
    started = datetime.now(timezone.utc).isoformat()
    head = _head()
    data_verified = False
    try:
        tape = load_tape()
        data_verified = True
        checksums = {"price_archives": len(tape.price_hashes),
                     "funding_archives": len(tape.funding_hashes)}
        outcome = evaluate(tape)
        status = ("numerically_passed_pending_prospective_oos" if outcome["gate"]["all_numeric"]
                  else "evaluated_rejected")
        result = {"experiment_id": "eight_coin_weekday_seasonality_2026_11_07",
                  "status": status, "preregistration_commit": PREREG_COMMIT,
                  "runner_commit": head, "started_utc": started,
                  "finished_utc": datetime.now(timezone.utc).isoformat(),
                  "data_verified": True, "verified_archive_counts": checksums,
                  **outcome}
    except (WeekdayDataError, WeekdayStrategyFailure, ValueError, OSError,
            UnicodeError) as exc:
        result = {"experiment_id": "eight_coin_weekday_seasonality_2026_11_07",
                  "status": "sealed_failed_run", "preregistration_commit": PREREG_COMMIT,
                  "runner_commit": head, "started_utc": started,
                  "finished_utc": datetime.now(timezone.utc).isoformat(),
                  "data_verified": data_verified, "metrics": None,
                  "gate": {"full": False}, "failure_reason": f"{type(exc).__name__}: {exc}"}
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    with RESULT.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return result


def main() -> None:
    outcome = run_one_shot()
    print(json.dumps({"report": str(RESULT), "status": outcome["status"],
                      "gate": outcome["gate"], "failure_reason": outcome.get("failure_reason")},
                     sort_keys=True))


if __name__ == "__main__":
    main()
