"""One-shot, fail-closed runner for the frozen USDT peg risk-gate study."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from research.coinbase_premium_source import SourceFailure
from research.usdt_peg_source import load_sources
from research.usdt_peg_sim import evaluate, simulate


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "Spot_Multi_Strategy/reports/mini_medallion_usdt_peg_dislocation_btc_oos"
PREREG = "cceb076"
SEALS = ("RESULT.json", "DATA_FAILURE.json", "CODE_FAILURE.json")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _preflight() -> str:
    if any((OUTPUT / name).exists() for name in SEALS):
        raise FileExistsError("USDT peg BTC historical hypothesis already sealed")
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", PREREG, "HEAD"],
                              cwd=ROOT, check=False, capture_output=True)
    if ancestor.returncode:
        raise RuntimeError("preregistration is not an ancestor of implementation")
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("commit implementation before formal source read")
    head = _git("rev-parse", "HEAD")
    if _git("rev-parse", "refs/remotes/origin/main") != head:
        raise RuntimeError("push implementation before formal source read")
    return head


def _write_once(name: str, value: dict) -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / name
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return path


def run() -> Path:
    head = _preflight()
    base = {"experiment": "usdt_peg_dislocation_btc_2026_11_09",
            "prereg_commit": PREREG, "code_commit": head,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_grade": "preregistered_retrospective_current_vintage_coinbase_usdt_peg",
            "independent_executable_oos": False, "real_orders_placed": False}
    try:
        coinbase, binance, terminal_open, sources = load_sources()
        primary = simulate(coinbase, binance, terminal_open)
        gross = simulate(coinbase, binance, terminal_open, slippage=0, fee_rate=0)
        double_slip = simulate(coinbase, binance, terminal_open, slippage=.001)
        delayed = simulate(coinbase, binance, terminal_open, latency_days=1)
        buy_hold = simulate(coinbase, binance, terminal_open, mode="buy_hold")
        cash_only = simulate(coinbase, binance, terminal_open, mode="cash")
        result = evaluate(primary, gross, double_slip, delayed, buy_hold,
                          cash_only, binance)
        return _write_once("RESULT.json", {**base, "status": "sealed_formal_result",
                                           "sealed_utc": datetime.now(timezone.utc).isoformat(),
                                           "sources": sources, **result})
    except SourceFailure as exc:
        return _write_once("DATA_FAILURE.json", {**base,
                                                 "status": "sealed_source_or_execution_failure",
                                                 "sealed_utc": datetime.now(timezone.utc).isoformat(),
                                                 "reason": str(exc),
                                                 "error_type": type(exc).__name__,
                                                 "gate_passed": False})
    except Exception as exc:
        return _write_once("CODE_FAILURE.json", {**base,
                                                 "status": "sealed_code_failure",
                                                 "sealed_utc": datetime.now(timezone.utc).isoformat(),
                                                 "reason": str(exc),
                                                 "error_type": type(exc).__name__,
                                                 "gate_passed": False})


if __name__ == "__main__":
    print(run())
