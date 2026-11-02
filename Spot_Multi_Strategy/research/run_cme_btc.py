"""One-shot official-source run of the preregistered CME BTC COT study.

Preregistration: efbc257. Only compact source fingerprints and diagnostics
are persisted; raw CFTC rows and Binance bars remain in memory. A source
integrity failure seals DATA_FAILURE without a strategy result.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

if __package__:
    from .cme_btc_cot import COTDataError, fetch_cot
    from .cme_btc_metrics import MetricsDataError, evaluate_diagnostics
    from .cme_btc_sim import StudyDataError, buy_and_hold, simulate
    from .cme_btc_spot import ArchiveIntegrityError, ArchiveUnavailable, fetch_bars
else:
    from cme_btc_cot import COTDataError, fetch_cot
    from cme_btc_metrics import MetricsDataError, evaluate_diagnostics
    from cme_btc_sim import StudyDataError, buy_and_hold, simulate
    from cme_btc_spot import ArchiveIntegrityError, ArchiveUnavailable, fetch_bars


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = (ROOT / "Spot_Multi_Strategy" / "reports" /
              "mini_medallion_cme_btc_positioning_oos")
PREREG_COMMIT = "efbc257"


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


def run() -> Path:
    if not _ancestor(PREREG_COMMIT):
        raise RuntimeError("CME BTC preregistration must be an ancestor before the run")
    if (REPORT_DIR / "RESULT.json").exists() or (REPORT_DIR / "DATA_FAILURE.json").exists():
        raise FileExistsError("the one-shot CME BTC study is already sealed")
    code_commit = _current_commit()
    started = datetime.now(timezone.utc).isoformat()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            cot_future = pool.submit(fetch_cot)
            spot_future = pool.submit(fetch_bars)
            cot = cot_future.result()
            spot = spot_future.result()
        if not cot.reports or not spot.bars:
            raise COTDataError("empty audited source bundle")
    except (COTDataError, ArchiveIntegrityError, ArchiveUnavailable) as exc:
        return _write_once("DATA_FAILURE.json", {
            "status": "data-ineligible", "reason": str(exc),
            "error_type": type(exc).__name__,
            "started_utc": started,
            "sealed_utc": datetime.now(timezone.utc).isoformat(),
            "code_commit": code_commit,
            "prereg_commit": PREREG_COMMIT,
            "independent_oos": False,
            "pnl_computed": False,
        })

    # A calculation bug is not disguised as bad market data. It must be
    # repaired and re-tested without inspecting any partial PnL.
    primary = simulate(spot.bars, cot.reports)
    stress = simulate(spot.bars, cot.reports, slippage_bps=25.0)
    benchmark = buy_and_hold(spot.bars)
    diagnostics = evaluate_diagnostics(primary, spot.bars, benchmark)
    payload = {
        "status": "sealed-result",
        "started_utc": started,
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": code_commit,
        "prereg_commit": PREREG_COMMIT,
        "evidence_grade": "retrospective_hypothesis_holdout",
        "independent_oos": False,
        "source_warnings": [
            "CFTC annual ZIPs have no published adjacent checksums; local SHA-256 is a reproducibility fingerprint only.",
            "d+14 is a conservative publication-lag approximation, not verified individual historical release timestamps.",
            "Binance public archives are for non-production research under their published terms.",
        ],
        "sources": {
            "cftc_tff_years": cot.manifest,
            "binance_spot_months": [asdict(item) for item in spot.manifest],
        },
        "counts": {
            "cftc_standard_btc_reports": len(cot.reports),
            "trades": len(primary.trades),
            "buys": sum(trade.side == "buy" for trade in primary.trades),
            "sells": sum(trade.side == "sell" for trade in primary.trades),
            "exposed_days": sum(row.exposed for row in primary.records),
            "all_days": len(primary.records),
        },
        "account": {
            "initial_usdt": 10_000.0,
            "final_usdt": primary.final_equity,
            "conservative_max_drawdown": primary.conservative_max_drawdown,
            "capacity_pass": primary.capacity_pass,
            "minimum_order_execution_pass": primary.execution_pass,
            "stress_25bp_slippage_final_usdt": stress.final_equity,
            "stress_25bp_slippage_conservative_max_drawdown": stress.conservative_max_drawdown,
            "passive_same_cost_final_usdt": benchmark.final_equity,
        },
        "diagnostics": diagnostics,
    }
    return _write_once("RESULT.json", payload)


if __name__ == "__main__":
    print(run())
