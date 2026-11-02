"""One-shot, storage-light execution of the sealed network-adoption study.

The preregistration is commit cde1897, amendments dd77ed7/ae6e677/77efcc2,
and futures-notice source manifest 337629f (updated at 77efcc2). Archives are
verified in
memory and never extracted or retained. Only metadata and compact outcomes
may be written to the ignored study report folder after execution.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any

if __package__:
    from .network_adoption_binance import (
        ArchiveIntegrityError, ArchiveUnavailable, Bar, Funding, bar_grid_gaps,
        fetch_and_parse_month,
    )
    from .network_adoption_coinmetrics import (
        NetworkMetricError, fetch_network_metrics,
    )
    from .network_adoption_sim import (
        DataIneligibleError, evaluate, run_registered_scenarios,
    )
else:
    from network_adoption_binance import (
        ArchiveIntegrityError, ArchiveUnavailable, Bar, Funding, bar_grid_gaps,
        fetch_and_parse_month,
    )
    from network_adoption_coinmetrics import (
        NetworkMetricError, fetch_network_metrics,
    )
    from network_adoption_sim import (
        DataIneligibleError, evaluate, run_registered_scenarios,
    )


UTC = timezone.utc
ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = (ROOT / "Spot_Multi_Strategy" / "reports"
              / "mini_medallion_network_adoption_oos")
SYMBOLS = (
    "BATUSDT", "CVCUSDT", "KNCUSDT", "MKRUSDT", "OMGUSDT",
    "RENUSDT", "ZRXUSDT",
)
START = datetime(2020, 11, 1, tzinfo=UTC)
TERMINAL = datetime(2026, 1, 1, tzinfo=UTC)
BTC_START = datetime(2020, 12, 1, tzinfo=UTC)
NOTICES = {
    "CVCUSDT": datetime(2022, 11, 28, 3, 30, tzinfo=UTC),
    "RENUSDT": datetime(2024, 11, 26, 8, 30, tzinfo=UTC),
    "OMGUSDT": datetime(2024, 12, 6, 3, 0, tzinfo=UTC),
    # Publication time has no explicit timezone.  Use the preregistered
    # conservative end-of-UTC-date fallback, not a guessed 09:00 UTC.
    "MKRUSDT": datetime(2025, 7, 18, 23, 59, 59, tzinfo=UTC),
}
SEALED_COMMITS = ("cde1897", "337629f", "dd77ed7", "ae6e677", "77efcc2")
SEALED_PATHS = (
    "Spot_Multi_Strategy/research/network_adoption_binance.py",
    "Spot_Multi_Strategy/research/network_adoption_coinmetrics.py",
    "Spot_Multi_Strategy/research/network_adoption_sim.py",
    "Spot_Multi_Strategy/research/run_network_adoption.py",
    "Spot_Multi_Strategy/tests/test_network_adoption_binance.py",
    "Spot_Multi_Strategy/tests/test_network_adoption_coinmetrics.py",
    "Spot_Multi_Strategy/tests/test_network_adoption_sim.py",
    "Spot_Multi_Strategy/tests/test_run_network_adoption.py",
    "Spot_Multi_Strategy/reports/mini_medallion_network_adoption_oos/PREREGISTRATION.md",
    "Spot_Multi_Strategy/reports/mini_medallion_network_adoption_oos/AMENDMENT_01_FUNDING_TIME.md",
    "Spot_Multi_Strategy/reports/mini_medallion_network_adoption_oos/AMENDMENT_02_SOURCE_INTEGRITY.md",
    "Spot_Multi_Strategy/reports/mini_medallion_network_adoption_oos/AMENDMENT_03_MKR_SOURCE_NOTICE.md",
    "Spot_Multi_Strategy/research/NETWORK_ADOPTION_ANNOUNCEMENTS.md",
)


class StudyDataError(ValueError):
    """The fixed official sources cannot support the sealed test."""


@dataclass(frozen=True)
class ArchiveBundle:
    trade4: dict[str, dict[datetime, Bar]]
    mark4: dict[str, dict[datetime, Bar]]
    mark1: dict[str, dict[datetime, Bar]]
    funding: dict[str, dict[datetime, float]]
    btc_daily_close: dict[date, float]
    manifest: dict[str, Any]


def next_4h_open_after(notice: datetime) -> datetime:
    if notice.tzinfo is None or notice.utcoffset() != timedelta(0):
        raise ValueError("notice must be timezone-aware UTC")
    floor = notice.replace(hour=(notice.hour // 4) * 4,
                           minute=0, second=0, microsecond=0)
    return floor + timedelta(hours=4)


def _months(first: datetime, last: datetime) -> tuple[str, ...]:
    if first > last:
        raise ValueError("first month after last")
    year, month = first.year, first.month
    out: list[str] = []
    while (year, month) <= (last.year, last.month):
        out.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return tuple(out)


def archive_jobs() -> tuple[tuple[str, str, str], ...]:
    """No future-survivor screen: only fixed notices truncate archive needs."""
    jobs: list[tuple[str, str, str]] = []
    for symbol in SYMBOLS:
        last = next_4h_open_after(NOTICES[symbol]) if symbol in NOTICES else TERMINAL
        for month in _months(START, last):
            for kind in ("trade4", "mark4", "mark1", "funding"):
                jobs.append((kind, symbol, month))
    for month in _months(BTC_START, datetime(2025, 12, 1, tzinfo=UTC)):
        jobs.append(("btc_spot1d", "BTCUSDT", month))
    return tuple(jobs)


def _funding_hour(raw: datetime) -> datetime:
    """Hour bucket for *source coverage*; held-event tolerance is in simulator."""
    if raw.tzinfo is None or raw.utcoffset() != timedelta(0):
        raise StudyDataError("funding timestamp is not UTC")
    return raw.replace(minute=0, second=0, microsecond=0)


def _fetch_archive_job(job: tuple[str, str, str]):
    kind, symbol, month = job
    return job, fetch_and_parse_month(kind, symbol, month)


def load_archives(*, max_workers: int = 12) -> ArchiveBundle:
    if not 1 <= max_workers <= 32:
        raise ValueError("max_workers must be between 1 and 32")
    maps: dict[str, dict[str, dict[datetime, Bar | Funding]]] = {
        kind: {symbol: {} for symbol in SYMBOLS}
        for kind in ("trade4", "mark4", "mark1", "funding")
    }
    btc_bars: dict[datetime, Bar] = {}
    records: list[dict[str, Any]] = []
    jobs = archive_jobs()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_archive_job, job): job for job in jobs}
        for future in as_completed(futures):
            job, archive = future.result()
            kind, symbol, month = job
            records.append({"kind": kind, "symbol": symbol, "month": month,
                            "sha256": archive.sha256,
                            "zip_bytes": archive.zip_bytes,
                            "rows": len(archive.rows)})
            if kind == "btc_spot1d":
                target = btc_bars
                cutoff = datetime(2025, 12, 31, tzinfo=UTC)
            else:
                target = maps[kind][symbol]
                cutoff = (next_4h_open_after(NOTICES[symbol])
                          if symbol in NOTICES else TERMINAL)
            for row in archive.rows:
                timestamp = row.calc_time if isinstance(row, Funding) else row.open_time
                if timestamp > cutoff + (timedelta(milliseconds=100) if kind == "funding" else timedelta(0)):
                    continue
                if timestamp in target:
                    raise StudyDataError(f"duplicate {kind}/{symbol} archive timestamp: {timestamp}")
                target[timestamp] = row

    funding_transitions: dict[str, int] = {}
    funding_outside_tolerance: dict[str, int] = {}
    required_start = datetime(2020, 12, 1, tzinfo=UTC)
    for symbol in SYMBOLS:
        cutoff = (next_4h_open_after(NOTICES[symbol])
                  if symbol in NOTICES else TERMINAL)
        for kind in ("trade4", "mark4", "mark1"):
            bars = maps[kind][symbol]
            if not bars:
                raise StudyDataError(f"no {kind} bars for {symbol}")
            period_hours = 1 if kind == "mark1" else 4
            in_scope = (bar for bar in bars.values()
                        if required_start <= bar.open_time <= cutoff)
            gaps = bar_grid_gaps(in_scope, required_start,
                                 cutoff + timedelta(hours=period_hours), kind=kind)
            if gaps:
                raise StudyDataError(f"{symbol}/{kind}: {len(gaps)} missing bars; first={gaps[0]}")
        # A present interval does not guarantee its funding events are complete.
        rows = sorted(maps["funding"][symbol].values(), key=lambda item: item.calc_time)
        if not rows:
            raise StudyDataError(f"no funding events for {symbol}")
        first_trade = min(maps["trade4"][symbol])
        last_logical: datetime | None = None
        last_row: Funding | None = None
        transition_count = outside_count = 0
        for row in rows:
            logical = _funding_hour(row.calc_time)
            if row.calc_time - logical > timedelta(milliseconds=100):
                # A bad raw offset is fatal only when a position is held.
                # The simulator applies that exact amendment-01 condition.
                outside_count += 1
            if last_logical is not None and logical <= last_logical:
                raise StudyDataError(f"{symbol}: duplicate/nonmonotonic funding hour")
            if last_row is not None:
                hours = (logical - last_logical) / timedelta(hours=1)
                previous_interval = float(last_row.interval_hours)
                current_interval = float(row.interval_hours)
                if previous_interval == current_interval:
                    if hours != current_interval:
                        raise StudyDataError(
                            f"{symbol}: missing/ambiguous funding event: gap={hours}h"
                        )
                else:
                    # Monthly archive metadata alone cannot prove which
                    # interval applied between these two records.  Do not
                    # infer a complete settlement history across a change.
                    raise StudyDataError(
                        f"{symbol}: ambiguous funding interval transition "
                        f"{previous_interval}h -> {current_interval}h"
                    )
            last_logical = logical
            last_row = row
        if _funding_hour(rows[0].calc_time) - first_trade > timedelta(hours=24):
            raise StudyDataError(f"{symbol}: funding starts >24h after first trade bar")
        if _funding_hour(rows[0].calc_time) > required_start + timedelta(hours=8):
            raise StudyDataError(f"{symbol}: funding starts after December warmup")
        if cutoff - _funding_hour(rows[-1].calc_time) >= timedelta(
            hours=float(rows[-1].interval_hours)
        ):
            raise StudyDataError(f"{symbol}: funding missing at/before cutoff")
        funding_transitions[symbol] = transition_count
        funding_outside_tolerance[symbol] = outside_count

    btc_gaps = bar_grid_gaps(tuple(btc_bars.values()), BTC_START,
                             datetime(2026, 1, 1, tzinfo=UTC),
                             kind="btc_spot1d")
    if btc_gaps:
        raise StudyDataError(f"BTCUSDT spot daily missing {len(btc_gaps)} days; first={btc_gaps[0]}")
    if len(btc_bars) != 1857:  # 2020-12 (31 days) + 2021–2025 (1826 days)
        raise StudyDataError(f"BTCUSDT spot daily unexpected count: {len(btc_bars)}")

    records.sort(key=lambda item: (item["kind"], item["symbol"], item["month"]))
    digest = hashlib.sha256(json.dumps(records, sort_keys=True,
                                      separators=(",", ":")).encode()).hexdigest()
    manifest = {
        "source": "https://data.binance.vision/data/",
        "preregistration_commit": "cde1897",
        "funding_time_amendment_commit": "dd77ed7",
        "source_integrity_amendment_commit": "ae6e677",
        "mkr_notice_amendment_commit": "77efcc2",
        "notice_manifest_commit": "337629f",
        "archive_count": len(records),
        "archive_metadata_sha256": digest,
        "funding_interval_transitions": funding_transitions,
        "funding_raw_offsets_over_100ms": funding_outside_tolerance,
        "archives": records,
    }
    return ArchiveBundle(
        trade4={symbol: maps["trade4"][symbol] for symbol in SYMBOLS},
        mark4={symbol: maps["mark4"][symbol] for symbol in SYMBOLS},
        mark1={symbol: maps["mark1"][symbol] for symbol in SYMBOLS},
        funding={symbol: {raw: item.rate for raw, item in maps["funding"][symbol].items()}
                 for symbol in SYMBOLS},
        btc_daily_close={stamp.date(): bar.close for stamp, bar in btc_bars.items()},
        manifest=manifest,
    )


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            raise StudyDataError("non-finite result would corrupt formal report")
        return "Infinity" if value > 0 else "-Infinity"
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False,
                               indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def _write_source_unavailable(payload: dict[str, Any]) -> None:
    """Keep each no-P&L source interruption as a tiny append-only audit file."""
    first = REPORT_DIR / "SOURCE_UNAVAILABLE.json"
    path = first
    ordinal = 2
    while path.exists():
        path = REPORT_DIR / f"SOURCE_UNAVAILABLE_{ordinal:02d}.json"
        ordinal += 1
    _write_json(path, payload)


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                   cwd=ROOT, text=True).strip()


def _assert_sealed_code() -> str:
    """Reject uncommitted research code or a HEAD lacking the preregistration."""
    head = _git_head()
    for commit in SEALED_COMMITS:
        check = subprocess.run(["git", "merge-base", "--is-ancestor", commit, head],
                               cwd=ROOT, check=False)
        if check.returncode:
            raise StudyDataError(f"sealed preregistration commit absent from HEAD: {commit}")
    for path in SEALED_PATHS:
        check = subprocess.run(["git", "cat-file", "-e", f"HEAD:{path}"],
                               cwd=ROOT, check=False, capture_output=True)
        if check.returncode:
            raise StudyDataError(f"research source is not in HEAD: {path}")
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *SEALED_PATHS],
        cwd=ROOT, text=True,
    )
    if status.strip():
        raise StudyDataError(f"research source differs from HEAD: {status.strip()}")
    return head


def _metric_manifest(adrs: dict[str, dict[date, tuple[float, datetime]]]) -> dict[str, Any]:
    """Compact reproducibility fingerprint, not proof of historical vintages."""
    digest = hashlib.sha256()
    counts: dict[str, int] = {}
    for symbol in SYMBOLS:
        points = adrs[symbol]
        counts[symbol] = len(points)
        for day, (count, completed) in sorted(points.items()):
            digest.update(f"{symbol}|{day.isoformat()}|{count:.17g}|{completed.isoformat()}\n".encode())
    return {
        "source": "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics",
        "metrics": ["AdrActCnt", "AssetEODCompletionTime"],
        "rows_by_symbol": counts,
        "canonical_value_sha256": digest.hexdigest(),
        "historical_vintages_verified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-workers", type=int, default=12,
                        help="HTTP concurrency only, not a strategy parameter")
    args = parser.parse_args()
    result_path = REPORT_DIR / "RESULT.json"
    failure_path = REPORT_DIR / "DATA_FAILURE.json"
    if result_path.exists() or failure_path.exists():
        parser.error("a formal result or data failure already exists; do not rerun this history")
    started = datetime.now(UTC).isoformat()
    code_commit = _assert_sealed_code()
    try:
        adrs = fetch_network_metrics(date(2020, 11, 1), date(2026, 1, 1))
        metric_manifest = _metric_manifest(adrs)
        bundle = load_archives(max_workers=args.max_workers)
        _write_json(REPORT_DIR / "ARCHIVE_MANIFEST.json", bundle.manifest)
        scenarios = run_registered_scenarios(
            adrs, bundle.trade4, bundle.mark4, bundle.mark1, bundle.funding,
            announcements=NOTICES,
        )
        summary: dict[str, Any] = {}
        for name, simulation in scenarios.items():
            evaluation = evaluate(simulation, bundle.btc_daily_close)
            summary[name] = {
                "metrics": evaluation.metrics,
                "stress_drawdown": evaluation.stress_drawdown,
                "months": evaluation.months,
                "folds": evaluation.folds,
                "regimes": evaluation.regimes,
                "monte_carlo": evaluation.monte_carlo,
                "numerical_gate": evaluation.numerical_gate,
                "full_gate": evaluation.full_gate,
                "costs": {"fee_usdt": simulation.fee_usdt,
                          "slippage_usdt": simulation.slippage_usdt,
                          "funding_usdt": simulation.funding_usdt,
                          "turnover_usdt": simulation.turnover_usdt},
                "execution": {"trade_count": simulation.trade_count,
                              "cash_days": simulation.cash_days,
                              "active_days": simulation.active_days,
                              "capacity_failed": simulation.capacity_failed,
                              "leverage_failed": simulation.leverage_failed,
                              "liquidated": simulation.liquidated},
            }
        payload = {
            "status": "completed_retrospective_test",
            "evidence_grade": "retrospective_source_without_verified_vintages",
            "independent_oos": False,
            "full_gate_passed": False,
            "preregistration_commit": "cde1897",
            "funding_time_amendment_commit": "dd77ed7",
            "notice_manifest_commit": "337629f",
            "source_integrity_amendment_commit": "ae6e677",
            "mkr_notice_amendment_commit": "77efcc2",
            "code_commit": code_commit,
            "started_utc": started,
            "finished_utc": datetime.now(UTC).isoformat(),
            "archive_count": bundle.manifest["archive_count"],
            "coinmetrics": metric_manifest,
            "scenarios": summary,
        }
        _write_json(result_path, payload)
        print(json.dumps(_json_safe({"result": str(result_path),
                                    "base": summary["1x_15bp"]["metrics"],
                                    "numerical_gate": summary["1x_15bp"]["numerical_gate"],
                                    "full_gate": False}), ensure_ascii=False))
        return 0
    except ArchiveUnavailable as exc:
        # A missing or temporarily unreachable official object is not a
        # strategy outcome. Keep the one-shot formal-result lock open for a
        # later attempt under the identical sealed rules.
        _write_source_unavailable({
            "status": "source_unavailable; no strategy result computed",
            "reason": str(exc), "started_utc": started,
            "observed_utc": datetime.now(UTC).isoformat(),
            "preregistration_commit": "cde1897",
            "funding_time_amendment_commit": "dd77ed7",
            "source_integrity_amendment_commit": "ae6e677",
            "mkr_notice_amendment_commit": "77efcc2",
            "code_commit": code_commit,
        })
        print(f"Official source unavailable; no strategy P&L computed: {exc}")
        return 3
    except NetworkMetricError as exc:
        if any(marker in str(exc) for marker in (
            "HTTP 429", "HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504",
            "response failed",
        )):
            _write_source_unavailable({
                "status": "source_unavailable; no strategy result computed",
                "reason": str(exc), "started_utc": started,
                "observed_utc": datetime.now(UTC).isoformat(),
                "preregistration_commit": "cde1897",
                "source_integrity_amendment_commit": "ae6e677",
                "mkr_notice_amendment_commit": "77efcc2",
                "code_commit": code_commit,
            })
            print(f"Network metric source unavailable; no strategy P&L computed: {exc}")
            return 3
        failure = exc
        _write_json(failure_path, {
            "status": "data_ineligible; no strategy result accepted",
            "reason": str(failure), "started_utc": started,
            "finished_utc": datetime.now(UTC).isoformat(),
            "preregistration_commit": "cde1897",
            "funding_time_amendment_commit": "dd77ed7",
            "source_integrity_amendment_commit": "ae6e677",
            "mkr_notice_amendment_commit": "77efcc2",
            "notice_manifest_commit": "337629f",
            "code_commit": code_commit,
        })
        print(f"Data integrity gate failed; no strategy P&L accepted: {failure}")
        return 2
    except (ArchiveIntegrityError, StudyDataError, DataIneligibleError) as exc:
        _write_json(failure_path, {
            "status": "data_ineligible; no strategy result accepted",
            "reason": str(exc),
            "preregistration_commit": "cde1897",
            "funding_time_amendment_commit": "dd77ed7",
            "source_integrity_amendment_commit": "ae6e677",
            "mkr_notice_amendment_commit": "77efcc2",
            "notice_manifest_commit": "337629f",
            "code_commit": code_commit,
            "started_utc": started,
            "finished_utc": datetime.now(UTC).isoformat(),
        })
        print(f"Data integrity gate failed; no strategy P&L accepted: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
