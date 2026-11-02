"""One-shot official-data run of the sealed BTC difficulty recovery study.

Preregistration: a6c6685; boundary amendment: bc6efbf.  Only compact source
digests and outcome summaries are persisted.  No raw block or market data are
saved.  A source/integrity failure writes DATA_FAILURE without any PnL.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

if __package__:
    from .btc_difficulty_blocks import DifficultySourceError, fetch_events
    from .btc_difficulty_metrics import evaluate_diagnostics
    from .btc_difficulty_sim import StudyDataError, buy_and_hold, simulate
    from .btc_difficulty_spot import (
        ArchiveIntegrityError, ArchiveUnavailable, fetch_bars,
    )
else:
    from btc_difficulty_blocks import DifficultySourceError, fetch_events
    from btc_difficulty_metrics import evaluate_diagnostics
    from btc_difficulty_sim import StudyDataError, buy_and_hold, simulate
    from btc_difficulty_spot import (
        ArchiveIntegrityError, ArchiveUnavailable, fetch_bars,
    )


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = (ROOT / "Spot_Multi_Strategy" / "reports" /
              "mini_medallion_btc_difficulty_recovery_oos")
PREREG_COMMIT = "a6c6685"
AMENDMENT_COMMIT = "bc6efbf"


def _commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _ancestor(commit: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=ROOT, check=False, capture_output=True,
    ).returncode == 0


def _write_once(filename: str, payload: dict) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    target = REPORT_DIR / filename
    if target.exists():
        raise FileExistsError(f"sealed study output already exists: {target}")
    serialized = json.dumps(payload, ensure_ascii=False, indent=2,
                            allow_nan=False, default=str) + "\n"
    # Exclusive create prevents a concurrent runner from silently overwriting
    # previously observed history or a sealed failure.
    with target.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
    return target


def _manifest_rows(items: tuple) -> list[dict]:
    return [asdict(item) for item in items]


def run() -> Path:
    if not (_ancestor(PREREG_COMMIT) and _ancestor(AMENDMENT_COMMIT)):
        raise RuntimeError("preregistration/amendment commits are not ancestors")
    if (REPORT_DIR / "RESULT.json").exists() or (REPORT_DIR / "DATA_FAILURE.json").exists():
        raise FileExistsError("the one-shot difficulty study is already sealed")
    code_commit = _commit()
    started = datetime.now(timezone.utc).isoformat()
    try:
        # Two unrelated hosts can be audited concurrently.  Nothing downstream
        # sees an incomplete source bundle; no PnL is calculated until both pass.
        with ThreadPoolExecutor(max_workers=2) as pool:
            blocks_future = pool.submit(fetch_events)
            spot_future = pool.submit(fetch_bars)
            blocks = blocks_future.result()
            spot = spot_future.result()
        if not blocks.events or not spot.bars:
            raise StudyDataError("empty audited source bundle")
        primary = simulate(spot.bars, blocks.events)
        stress = simulate(spot.bars, blocks.events, slippage_bps=25.0)
        benchmark = buy_and_hold(spot.bars)
        diagnostics = evaluate_diagnostics(primary, spot.bars, benchmark)
    except (DifficultySourceError, ArchiveIntegrityError, ArchiveUnavailable,
            StudyDataError) as exc:
        payload = {
            "status": "data-ineligible", "reason": str(exc),
            "error_type": type(exc).__name__, "started_utc": started,
            "sealed_utc": datetime.now(timezone.utc).isoformat(),
            "code_commit": code_commit, "prereg_commit": PREREG_COMMIT,
            "amendment_commit": AMENDMENT_COMMIT,
            "independent_oos": False, "pnl_computed": False,
        }
        if isinstance(exc, DifficultySourceError):
            payload["failed_source_requests"] = _manifest_rows(exc.manifest)
        return _write_once("DATA_FAILURE.json", payload)

    payload = {
        "status": "sealed-result", "started_utc": started,
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": code_commit, "prereg_commit": PREREG_COMMIT,
        "amendment_commit": AMENDMENT_COMMIT,
        "evidence_grade": "preregistered_retrospective_hypothesis_holdout",
        "independent_oos": False,
        "sources": {
            "bitcoin_retargets": _manifest_rows(blocks.manifest),
            "binance_spot_months": _manifest_rows(spot.manifest),
        },
        "counts": {
            "retarget_events": len(blocks.events),
            "recovery_events": sum(int(e.recovery) for e in blocks.events),
            "eligible_recovery_events": primary.eligible_recovery_events,
            "ignored_events_while_held": primary.ignored_events_while_held,
            "trades": len(primary.trades),
            "exposed_days": sum(int(r.exposed) for r in primary.records),
            "all_days": len(primary.records),
        },
        "account": {
            "initial_usdt": 10_000.0,
            "final_usdt": primary.final_equity,
            "max_conservative_drawdown": primary.conservative_max_drawdown,
            "capacity_pass": primary.capacity_pass,
            "stress_25bp_slippage_final_usdt": stress.final_equity,
            "stress_25bp_slippage_max_conservative_drawdown": stress.conservative_max_drawdown,
            "passive_benchmark_final_usdt": benchmark.final_equity,
        },
        "diagnostics": diagnostics,
    }
    return _write_once("RESULT.json", payload)


if __name__ == "__main__":
    print(run())
