"""Single formal execution and append-only seal of the trade-size test."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess

from .trade_size_pressure_data import load_tape
from .trade_size_pressure_sim import (
    TradeSizeStrategyFailure, build_schedule, evaluate,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = (ROOT / "Spot_Multi_Strategy" / "research" / "results"
          / "three_coin_trade_size_pressure_2026_11_08.json")
PREREG = "08a63cd"


def _source_root(hashes: dict[str, str]) -> str:
    return sha256("".join(f"{key}:{value}\n" for key, value in sorted(hashes.items()))
                  .encode("ascii")).hexdigest()


def run() -> Path:
    if subprocess.run(["git", "merge-base", "--is-ancestor", PREREG, "HEAD"],
                      cwd=ROOT, check=False).returncode:
        raise RuntimeError("the sole preregistration commit must precede this run")
    if OUTPUT.exists():
        raise FileExistsError("this hypothesis has already been sealed")
    started = datetime.now(timezone.utc).isoformat()
    code = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                   cwd=ROOT, text=True).strip()
    base = {
        "id": "three_coin_trade_size_pressure_2026_11_08",
        "prereg_commit": PREREG,
        "code_commit": code,
        "started_utc": started,
        "evidence_grade": "preregistered_retrospective_aggregate_trade_size_proxy",
        "independent_executable_oos": False,
    }
    try:
        tape = load_tape()
        schedule, signal_sha = build_schedule(tape)
    except ValueError as exc:
        payload = {**base, "status": "source_integrity_fail",
                   "error_type": type(exc).__name__, "error": str(exc),
                   "pnl_computed": False, "all_gate_passed": False}
    else:
        sources = {
            "five_minute_bars": len(tape.opening),
            "price_archives": len(tape.price_hashes),
            "funding_archives": len(tape.funding_hashes),
            "funding_event_bars": len(tape.funding),
            "price_pin_root_sha256": _source_root(tape.price_hashes),
            "funding_pin_root_sha256": _source_root(tape.funding_hashes),
            "signal_schedule_sha256": signal_sha,
        }
        try:
            result = evaluate(tape, schedule)
        except TradeSizeStrategyFailure as exc:
            payload = {**base, "status": "sealed_strategy_failure",
                       "error_type": type(exc).__name__, "error": str(exc),
                       "bar": exc.bar, "equity_at_failure": exc.equity,
                       "sources": sources, "all_gate_passed": False}
        else:
            payload = {**base, "status": "sealed_result", "sources": sources,
                       "result": result,
                       "all_gate_passed": bool(result["gate"]["all_passed"])}
    payload["sealed_utc"] = datetime.now(timezone.utc).isoformat()
    with OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return OUTPUT


if __name__ == "__main__":
    print(run())
