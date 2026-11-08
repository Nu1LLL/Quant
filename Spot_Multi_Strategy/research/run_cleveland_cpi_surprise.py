"""One formal CPI-nowcast surprise run; seal any source, code or Gate failure."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from research.btc_exchange_supply_source import SourceFailure
from research.cleveland_cpi_surprise_source import load_sources
from research.cleveland_cpi_surprise_sim import evaluate, simulate


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "Spot_Multi_Strategy/reports/mini_medallion_cleveland_cpi_surprise_oos"
PREREG = "799c64f"
SEALS = ("RESULT.json", "DATA_FAILURE.json", "CODE_FAILURE.json")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _preflight() -> str:
    if any((OUTPUT / name).exists() for name in SEALS):
        raise FileExistsError("Cleveland CPI surprise hypothesis already sealed")
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", PREREG, "HEAD"],
                              cwd=ROOT, check=False, capture_output=True)
    if ancestor.returncode:
        raise RuntimeError("preregistration is not an ancestor of implementation")
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("commit implementation before opening formal data")
    head = _git("rev-parse", "HEAD")
    if _git("rev-parse", "refs/remotes/origin/main") != head:
        raise RuntimeError("push implementation before opening formal data")
    return head


def _write_once(name: str, body: dict) -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / name
    with path.open("x", encoding="utf-8") as handle:
        json.dump(body, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return path


def run() -> Path:
    head = _preflight()
    base = {"experiment": "cleveland_cpi_surprise_btc_spot_2026_11_08",
            "prereg_commit": PREREG, "code_commit": head,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_grade": "preregistered_retrospective_historical_nowcast_chart",
            "independent_executable_oos": False, "real_orders_placed": False}
    try:
        events, prices, audit = load_sources()
        primary = simulate(events, prices)
        gross = simulate(events, prices, slippage=0, fee_rate=0)
        double_slip = simulate(events, prices, slippage=.001)
        delayed = simulate(events, prices, delay_hours=1)
        hold = simulate(events, prices, buy_and_hold=True)
        result = evaluate(events, primary, gross, double_slip, delayed, hold, prices)
        return _write_once("RESULT.json", {**base, "status": "sealed_formal_result",
                                           "sealed_utc": datetime.now(timezone.utc).isoformat(),
                                           "sources": audit, **result})
    except SourceFailure as exc:
        return _write_once("DATA_FAILURE.json", {**base,
                                                 "status": "sealed_source_or_execution_failure",
                                                 "sealed_utc": datetime.now(timezone.utc).isoformat(),
                                                 "reason": str(exc),
                                                 "error_type": type(exc).__name__,
                                                 "gate_passed": False})
    except Exception as exc:
        return _write_once("CODE_FAILURE.json", {**base, "status": "sealed_code_failure",
                                                 "sealed_utc": datetime.now(timezone.utc).isoformat(),
                                                 "reason": str(exc),
                                                 "error_type": type(exc).__name__,
                                                 "gate_passed": False})


if __name__ == "__main__":
    print(run())
