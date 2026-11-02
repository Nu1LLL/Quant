"""One-shot, preregistered Wikipedia-attention test with official daily ZIP repair.

This is deliberately separate from the sealed monthly-only DATA_FAILURE runner.
No raw historical source is persisted; all outcomes, including data failure, are
written once to this study's own report directory.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

if __package__:
    from .wiki_attention_daily_archive_data import ArchiveIntegrityError, load_and_audit
    from .wiki_attention_metrics import evaluate
    from .wiki_attention_sim import MarginRuinError, SimulationDataError, run_backtest
else:
    from wiki_attention_daily_archive_data import ArchiveIntegrityError, load_and_audit
    from wiki_attention_metrics import evaluate
    from wiki_attention_sim import MarginRuinError, SimulationDataError, run_backtest


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "Spot_Multi_Strategy" / "reports" / (
    "mini_medallion_wikipedia_attention_daily_archive_oos"
)
PREREG_COMMIT = "2708aa3"


def _ancestor(commit: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=ROOT, check=False, capture_output=True,
    ).returncode == 0


def _current_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()


def _committed_and_pushed() -> bool:
    """A run must be reproducible from the public, clean committed tree."""
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True,
    )
    if dirty:
        return False
    remote = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=ROOT, text=True,
    ).strip()
    return remote == _current_commit()


def _write_once(name: str, payload: dict) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / name
    serialized = json.dumps(payload, ensure_ascii=False, indent=2,
                            allow_nan=False, default=str) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
    return path


def _base_payload(started: str, code_commit: str) -> dict:
    return {
        "started_utc": started,
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": code_commit,
        "prereg_commit": PREREG_COMMIT,
        "prior_monthly_only_study": "mini_medallion_wikipedia_attention_oos/DATA_FAILURE.json",
        "evidence_grade": "preregistered_retrospective_historical_holdout",
        "independent_oos": False,
        "source_warnings": [
            "Today’s Wikimedia history has no per-observation publication vintage.",
            "Documented 2021-06-04..2022-01-27 pageview loss is retained.",
            "Binance daily and monthly public ZIPs could be revised after publication.",
            "Historical lot-size and maintenance-margin tiers are only proxies.",
            "Funding events up to 100ms after the nominal hour are mapped to that hour; execution latency is not proven.",
        ],
    }


def _data_failure(exc: Exception, *, started: str, code_commit: str,
                  sources: dict | None = None,
                  pnl_computed: bool = False) -> Path:
    payload = {
        **_base_payload(started, code_commit),
        "status": "data-ineligible", "reason": str(exc),
        "error_type": type(exc).__name__,
        "pnl_computed": pnl_computed,
        "pnl_published": False,
    }
    if sources is not None:
        payload["sources"] = sources
    return _write_once("DATA_FAILURE.json", payload)


def run() -> Path:
    if not _ancestor(PREREG_COMMIT):
        raise RuntimeError("daily-archive preregistration must precede the run")
    if any((REPORT_DIR / name).exists() for name in (
        "RESULT.json", "DATA_FAILURE.json", "CODE_FAILURE.json",
    )):
        raise FileExistsError("the one-shot daily-archive study is already sealed")
    if not _committed_and_pushed():
        raise RuntimeError("study code must be committed, pushed, and worktree clean")
    started = datetime.now(timezone.utc).isoformat()
    code_commit = _current_commit()
    try:
        return _execute_once(started=started, code_commit=code_commit)
    except Exception as exc:
        # An unexpected parser/simulator/metrics bug after source access cannot
        # silently become a retry on the same historical sample.
        return _write_once("CODE_FAILURE.json", {
            **_base_payload(started, code_commit),
            "status": "code-failure", "reason": str(exc),
            "error_type": type(exc).__name__,
            "all_gate_passed": False,
        })


def _execute_once(*, started: str, code_commit: str) -> Path:
    try:
        bundle = load_and_audit()
    except ArchiveIntegrityError as exc:
        return _data_failure(exc, started=started, code_commit=code_commit)

    args = (bundle.views, bundle.trade_bars, bundle.mark_bars,
            bundle.funding_events)
    try:
        primary = run_backtest(*args, slippage_rate=0.001)
    except MarginRuinError as exc:
        return _write_once("RESULT.json", {
            **_base_payload(started, code_commit),
            "status": "sealed-risk-failure", "reason": str(exc),
            "error_type": type(exc).__name__,
            "first_breach_utc": exc.timestamp.isoformat(),
            "equity_usdt_at_breach": exc.equity,
            "stressed_equity_usdt_at_breach": exc.stressed_equity,
            "sources": bundle.manifest,
            "full_sample_metrics_computed": False,
            "all_gate_passed": False,
        })
    except SimulationDataError as exc:
        return _data_failure(exc, started=started, code_commit=code_commit,
                             sources=bundle.manifest, pnl_computed=False)

    stress_risk = None
    try:
        stress = run_backtest(*args, slippage_rate=0.0025)
    except MarginRuinError as exc:
        stress, stress_risk = None, str(exc)
    except SimulationDataError as exc:
        return _data_failure(exc, started=started, code_commit=code_commit,
                             sources=bundle.manifest, pnl_computed=True)

    benchmark_risk = None
    try:
        benchmark = run_backtest(*args, slippage_rate=0.001, benchmark=True)
    except MarginRuinError as exc:
        benchmark, benchmark_risk = None, str(exc)
    except SimulationDataError as exc:
        return _data_failure(exc, started=started, code_commit=code_commit,
                             sources=bundle.manifest, pnl_computed=True)

    diagnostics = evaluate(primary, benchmark, bundle.trade_bars["BTCUSDT"])
    return _write_once("RESULT.json", {
        **_base_payload(started, code_commit),
        "status": "sealed-result", "sources": bundle.manifest,
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
    })


if __name__ == "__main__":
    print(run())
