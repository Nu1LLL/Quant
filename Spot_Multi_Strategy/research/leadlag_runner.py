"""One-shot official-archive replay of preregistered ten-coin lead-lag rule."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .basis_hourly_preflight import PreflightError
from .leadlag_data import (
    FUNDING_MANIFEST, PRICE_MANIFEST, LeadLagDataError,
    load_price_tape, verify_funding_schedule,
)
from .leadlag_metrics import evaluate_leadlag_gate
from .leadlag_sim import LeadLagSimulationError, run_leadlag_simulation


EXPERIMENT_ID = "ten_coin_delayed_leadlag_2026_11_07"
PREREGISTRATION_COMMIT = "0976483"
BOUNDARY_AMENDMENT_COMMIT = "982e707"
REPORT_PATH = Path(__file__).parent / "results" / f"{EXPERIMENT_ID}.json"


def run_one_shot() -> dict:
    funding_hashes: dict[tuple[str, str], str] = {}
    price_hashes: dict[tuple[str, str], str] = {}
    funding_events = verify_funding_schedule(audit_hashes=funding_hashes)
    tape = load_price_tape(audit_hashes=price_hashes)
    if len(funding_hashes) != 600 or len(price_hashes) != 610:
        raise LeadLagDataError("all 1,210 committed archive digests are required")
    scenarios = run_leadlag_simulation(tape)
    report = evaluate_leadlag_gate(scenarios, funding_schedule_verified=True)
    report.update({
        "experiment_id": EXPERIMENT_ID,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "boundary_amendment_commit": BOUNDARY_AMENDMENT_COMMIT,
        "manifest_sha256": {
            "price": sha256(PRICE_MANIFEST.read_bytes()).hexdigest(),
            "funding_time": sha256(FUNDING_MANIFEST.read_bytes()).hexdigest(),
        },
        "measured_archive_sha256": {
            "price": {f"{month}:{symbol}": digest
                      for (month, symbol), digest in sorted(price_hashes.items())},
            "funding_time": {f"{month}:{symbol}": digest
                             for (month, symbol), digest in sorted(funding_hashes.items())},
        },
        "validated_funding_events": funding_events,
        "funding_rate_values_read": False,
    })
    return report


def main() -> None:
    if REPORT_PATH.exists():
        raise RuntimeError("sealed report already exists; do not rerun this experiment")
    try:
        report = run_one_shot()
    except (LeadLagDataError, LeadLagSimulationError, PreflightError,
            np.linalg.LinAlgError, ValueError) as exc:
        transport_only = (isinstance(exc, PreflightError)
                          and str(exc).startswith("transport unavailable after two tries:"))
        report = {
            "experiment_id": EXPERIMENT_ID,
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "boundary_amendment_commit": BOUNDARY_AMENDMENT_COMMIT,
            "status": ("transport_interrupted_no_result" if transport_only
                       else "sealed_failed_run"),
            "failure_type": type(exc).__name__,
            "failure_reason": str(exc),
            "gate": {"full": False},
        }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp = REPORT_PATH.with_suffix(".json.tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    temp.replace(REPORT_PATH)
    print(json.dumps({"report": str(REPORT_PATH), "gate": report["gate"],
                      "metrics": report.get("metrics"),
                      "completed_pairs": report.get("completed_pairs"),
                      "failure_reason": report.get("failure_reason")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
