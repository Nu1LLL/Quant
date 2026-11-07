"""Immutable one-shot writer for the preregistered seven-alt IVOL transfer."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from .idiosyncratic_volatility_sim import IvolFailure, evaluate
from .weekday_seasonality_data import WeekdayDataError, load_tape
from .weekday_seasonality_sim import WeekdayStrategyFailure


RESULT = Path(__file__).parent / "results/seven_alt_daily_idiosyncratic_volatility_2026_11_07.json"
PREREG_COMMIT = "ac0561c"


def run_one_shot() -> dict:
    if RESULT.exists():
        raise FileExistsError("daily IVOL formal outcome already exists; no rerun")
    started = datetime.now(timezone.utc).isoformat()
    head = subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()
    data_verified = False
    try:
        tape = load_tape()
        data_verified = True
        counts = {"price_archives": len(tape.price_hashes),
                  "funding_archives": len(tape.funding_hashes)}
        outcome = evaluate(tape)
        status = ("numerically_passed_pending_prospective_oos" if outcome["gate"]["all_numeric"]
                  else "evaluated_rejected")
        result = {
            "experiment_id": "seven_alt_daily_idiosyncratic_volatility_2026_11_07",
            "status": status, "preregistration_commit": PREREG_COMMIT,
            "runner_commit": head, "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "data_verified": True, "verified_archive_counts": counts,
            **outcome,
        }
    except (WeekdayDataError, WeekdayStrategyFailure, IvolFailure,
            ValueError, OSError, UnicodeError) as exc:
        result = {
            "experiment_id": "seven_alt_daily_idiosyncratic_volatility_2026_11_07",
            "status": "sealed_failed_run", "preregistration_commit": PREREG_COMMIT,
            "runner_commit": head, "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "data_verified": data_verified, "metrics": None,
            "gate": {"full": False},
            "failure_reason": f"{type(exc).__name__}: {exc}",
        }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    with RESULT.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return result


def main() -> None:
    result = run_one_shot()
    print(json.dumps({"report": str(RESULT), "status": result["status"],
                      "gate": result["gate"],
                      "failure_reason": result.get("failure_reason")}, sort_keys=True))


if __name__ == "__main__":
    main()
