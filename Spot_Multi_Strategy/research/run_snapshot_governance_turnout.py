"""Run the preregistered governance experiment exactly once and seal outputs."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import time

from .snapshot_governance_source import load_prices, load_proposals
from .snapshot_governance_sim import run


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/mini_medallion_snapshot_governance_turnout_oos"
PREREG = ROOT / "research/preregistrations/2026-11-08_snapshot_governance_turnout_spike.md"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                               indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    if REPORT_DIR.exists():
        raise RuntimeError("formal experiment already has a sealed result directory")
    if not PREREG.is_file():
        raise RuntimeError("preregistration missing")
    REPORT_DIR.mkdir(parents=True)
    audit: dict = {"preregistration_sha256": sha256(PREREG.read_bytes()).hexdigest(),
                   "run_started_at_utc_ms": time.time_ns() // 1_000_000}
    try:
        proposals, proposal_audit = load_proposals()
        manifest = [asdict(proposal) for proposal in proposals]
        _write_json(REPORT_DIR / "PROPOSAL_MANIFEST.json", manifest)
        audit["snapshot"] = proposal_audit
        prices, terminal, price_audit = load_prices()
        audit["binance_months"] = price_audit
        audit["terminal_open_symbols"] = sorted(terminal)
        result = run(proposals, prices, terminal)
        audit["run_finished_at_utc_ms"] = time.time_ns() // 1_000_000
        _write_json(REPORT_DIR / "SOURCE_AUDIT.json", audit)
        _write_json(REPORT_DIR / "RESULT.json", result)
        print(json.dumps({"status": "sealed", "numerical_gate": result["numerical_gate"],
                          "full_gate": False, "report_dir": str(REPORT_DIR)},
                         sort_keys=True))
    except Exception as exc:
        failure = {"status": "DATA_OR_SIMULATION_FAILURE",
                   "type": type(exc).__name__, "reason": str(exc),
                   "independent_oos": False, "full_gate": False,
                   "sealed_at_utc_ms": time.time_ns() // 1_000_000}
        _write_json(REPORT_DIR / "FAILURE.json", failure)
        _write_json(REPORT_DIR / "SOURCE_AUDIT.json", audit)
        print(json.dumps({"status": "sealed_failure", "type": type(exc).__name__,
                          "report_dir": str(REPORT_DIR)}, sort_keys=True))


if __name__ == "__main__":
    main()
