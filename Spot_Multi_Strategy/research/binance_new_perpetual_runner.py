"""Immutable one-shot writer for the preregistered launch short experiment."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from .binance_new_perpetual_data import load_tape
from .binance_new_perpetual_sim import evaluate


RESULT = Path(__file__).parent / "results/binance_new_perpetual_post_launch_short_2026_11_08.json"
PREREG_COMMIT = "d2b23e0"
AMENDMENT_COMMIT = "894d2d4"


def run_one_shot() -> dict:
    if RESULT.exists():
        raise FileExistsError("formal launch-short outcome already exists; no rerun")
    started = datetime.now(timezone.utc).isoformat()
    head = subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()
    data_verified = False
    try:
        tape = load_tape()
        data_verified = True
        outcome = evaluate(tape)
        status = ("numerically_passed_pending_independent_executable_oos"
                  if outcome["gate"]["all_numeric"] else "evaluated_rejected")
        result = {
            "experiment_id": "binance_new_perpetual_post_launch_short_2026_11_08",
            "status": status, "preregistration_commit": PREREG_COMMIT,
            "amendment_commit": AMENDMENT_COMMIT, "runner_commit": head,
            "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
            "data_verified": True, **outcome,
        }
    except Exception as exc:
        result = {
            "experiment_id": "binance_new_perpetual_post_launch_short_2026_11_08",
            "status": "sealed_failed_run", "preregistration_commit": PREREG_COMMIT,
            "amendment_commit": AMENDMENT_COMMIT, "runner_commit": head,
            "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
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
