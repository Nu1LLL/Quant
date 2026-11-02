"""One-shot run of the preregistered five-coin Wikipedia attention study.

Only audited source fingerprints and compact diagnostics are persisted.  The
historical page-view counts, bars, and funding events remain in memory.  A
source defect seals DATA_FAILURE; a margin breach is a failed strategy result.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

if __package__:
    from .wiki_attention_data import ArchiveIntegrityError, load_and_audit
    from .wiki_attention_metrics import evaluate
    from .wiki_attention_sim import MarginRuinError, SimulationDataError, run_backtest
else:
    from wiki_attention_data import ArchiveIntegrityError, load_and_audit
    from wiki_attention_metrics import evaluate
    from wiki_attention_sim import MarginRuinError, SimulationDataError, run_backtest


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = (ROOT / "Spot_Multi_Strategy" / "reports" /
              "mini_medallion_wikipedia_attention_oos")
PREREG_COMMIT = "c83917d"


def _ancestor(commit: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=ROOT, check=False, capture_output=True,
    ).returncode == 0


def _current_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()


def _write_once(name: str, payload: dict) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / name
    serialized = json.dumps(payload, ensure_ascii=False, indent=2,
                            allow_nan=False, default=str) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
    return path


def _base_payload(*, started: str, code_commit: str) -> dict:
    return {
        "started_utc": started,
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": code_commit,
        "prereg_commit": PREREG_COMMIT,
        "evidence_grade": "preregistered_retrospective_historical_holdout",
        "independent_oos": False,
        "source_warnings": [
            "Today’s Wikimedia API response has no per-observation historical publication vintage.",
            "Documented partial pageview traffic loss in 2021-06-04..2022-01-27 is not excluded.",
            "The historical Binance lot sizes and maintenance-margin tiers are not fully reconstructed.",
            "Binance public ZIPs and adjacent checksums can be updated after initial release.",
        ],
    }


def run() -> Path:
    if not _ancestor(PREREG_COMMIT):
        raise RuntimeError("Wikipedia attention preregistration must precede the run")
    if (REPORT_DIR / "RESULT.json").exists() or (REPORT_DIR / "DATA_FAILURE.json").exists():
        raise FileExistsError("the one-shot Wikipedia attention study is already sealed")
    started = datetime.now(timezone.utc).isoformat()
    code_commit = _current_commit()
    try:
        bundle = load_and_audit()
    except ArchiveIntegrityError as exc:
        return _write_once("DATA_FAILURE.json", {
            **_base_payload(started=started, code_commit=code_commit),
            "status": "data-ineligible",
            "reason": str(exc),
            "error_type": type(exc).__name__,
            "pnl_computed": False,
        })

    args = (bundle.views, bundle.trade_bars, bundle.mark_bars,
            bundle.funding_events)
    try:
        primary = run_backtest(*args, slippage_rate=0.001)
    except MarginRuinError as exc:
        return _write_once("RESULT.json", {
            **_base_payload(started=started, code_commit=code_commit),
            "status": "sealed-risk-failure",
            "reason": str(exc),
            "error_type": type(exc).__name__,
            "first_breach_utc": exc.timestamp.isoformat(),
            "equity_usdt_at_breach": exc.equity,
            "stressed_equity_usdt_at_breach": exc.stressed_equity,
            "sources": bundle.manifest,
            "full_sample_metrics_computed": False,
            "all_gate_passed": False,
        })
    except SimulationDataError as exc:
        return _write_once("DATA_FAILURE.json", {
            **_base_payload(started=started, code_commit=code_commit),
            "status": "data-ineligible", "reason": str(exc),
            "error_type": type(exc).__name__, "pnl_computed": False,
            "sources": bundle.manifest,
        })

    stress_risk = None
    try:
        stress = run_backtest(*args, slippage_rate=0.0025)
    except MarginRuinError as exc:
        stress = None
        stress_risk = str(exc)
    except SimulationDataError as exc:
        return _write_once("DATA_FAILURE.json", {
            **_base_payload(started=started, code_commit=code_commit),
            "status": "data-ineligible", "reason": str(exc),
            "error_type": type(exc).__name__, "pnl_computed": False,
            "sources": bundle.manifest,
        })
    benchmark_risk = None
    try:
        benchmark = run_backtest(*args, slippage_rate=0.001, benchmark=True)
    except MarginRuinError as exc:
        benchmark = None
        benchmark_risk = str(exc)
    except SimulationDataError as exc:
        return _write_once("DATA_FAILURE.json", {
            **_base_payload(started=started, code_commit=code_commit),
            "status": "data-ineligible", "reason": str(exc),
            "error_type": type(exc).__name__, "pnl_computed": False,
            "sources": bundle.manifest,
        })

    diagnostics = evaluate(primary, benchmark, bundle.trade_bars["BTCUSDT"])
    payload = {
        **_base_payload(started=started, code_commit=code_commit),
        "status": "sealed-result",
        "sources": bundle.manifest,
        "counts": {
            "portfolio_trades": len(primary.trades),
            "exposed_days": sum(bool(row.exposed) for row in primary.records),
            "calendar_days": len(primary.records),
        },
        "account": {
            "initial_usdt": 10_000.0,
            "final_usdt": primary.final_equity,
            "fee_usdt": primary.fee_usdt,
            "slippage_usdt": primary.slippage_usdt,
            "funding_usdt": primary.funding_usdt,
            "turnover_usdt": primary.turnover_usdt,
            "stress_25bp_final_usdt": stress.final_equity if stress is not None else None,
            "stress_25bp_conservative_max_drawdown": (
                stress.conservative_max_drawdown if stress is not None else None
            ),
            "stress_25bp_fees_usdt": stress.fee_usdt if stress is not None else None,
            "stress_25bp_slippage_usdt": (
                stress.slippage_usdt if stress is not None else None
            ),
            "stress_25bp_risk_failure": stress_risk,
            "equal_weight_benchmark_final_usdt": (
                benchmark.final_equity if benchmark is not None else None
            ),
            "equal_weight_benchmark_risk_failure": benchmark_risk,
        },
        "diagnostics": diagnostics,
    }
    return _write_once("RESULT.json", payload)


if __name__ == "__main__":
    print(run())
