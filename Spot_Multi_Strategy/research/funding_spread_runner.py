"""Explicit one-shot entry point for the registered funding-spread test."""

from __future__ import annotations

from contextlib import closing
from hashlib import sha256
import json
from pathlib import Path

from .funding_spread_data import (
    FUNDING_MANIFEST, KLINE_MANIFEST, FundingSpreadDataError, iter_hours, load_funding,
)
from .funding_spread_metrics import evaluate_funding_spread_gate
from .funding_spread_sim import FundingSimulationError, run_funding_spread_simulation
from .basis_hourly_preflight import PreflightError


EXPERIMENT_ID = "three_alt_funding_spread_2026_11_07"
PREREGISTRATION_COMMIT = "991a4ae"
BOUNDARY_AMENDMENT_COMMIT = "363b296"
REPORT_PATH = Path(__file__).parent / "results" / f"{EXPERIMENT_ID}.json"


def run_one_shot() -> dict:
    funding_hashes: dict[tuple[str, str], str] = {}
    kline_hashes: dict[tuple[str, str], str] = {}
    events = load_funding(audit_hashes=funding_hashes)
    with closing(iter_hours(audit_hashes=kline_hashes)) as bars:
        scenarios = run_funding_spread_simulation(bars, events)
    if len(funding_hashes) != 183 or len(kline_hashes) != 186:
        raise ValueError("all 369 committed archive digests are required")
    report = evaluate_funding_spread_gate(scenarios)
    report.update({
        "experiment_id": EXPERIMENT_ID,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "boundary_amendment_commit": BOUNDARY_AMENDMENT_COMMIT,
        "manifest_sha256": {
            "kline": sha256(KLINE_MANIFEST.read_bytes()).hexdigest(),
            "funding": sha256(FUNDING_MANIFEST.read_bytes()).hexdigest(),
        },
        "measured_archive_sha256": {
            "kline": {f"{m}:{s}": v for (m, s), v in sorted(kline_hashes.items())},
            "funding": {f"{m}:{s}": v for (m, s), v in sorted(funding_hashes.items())},
        },
    })
    return report


def main() -> None:
    if REPORT_PATH.exists():
        raise RuntimeError("sealed report already exists; do not rerun this experiment")
    try:
        report = run_one_shot()
    except (FundingSpreadDataError, FundingSimulationError, PreflightError, ValueError) as exc:
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
