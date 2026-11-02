"""One-shot execution of the frozen spot aggressive-flow study.

Run only from a committed, clean checkout with::

    PYTHONPATH=Spot_Multi_Strategy python3 -m research.run_spot_flow

Official archives are checksum-verified in memory and are never saved. This
runner writes only a compact source digest and derived research diagnostics.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import pandas as pd

from .spot_flow_archive import ArchiveIntegrityError, load_and_audit
from .spot_flow_backtest import (
    BacktestIntegrityError, END, START, _run_period, run_backtest,
)


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = (ROOT / "Spot_Multi_Strategy" / "reports"
              / "mini_medallion_spot_flow_perp_oos")
PREREGISTRATION_COMMIT = "83ec829"
INTERPRETATION_COMMIT = "a9b1558"


def _git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                   allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _compact_manifest(manifest: dict) -> dict:
    canonical = json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {
        "sources": manifest["sources"],
        "symbols": manifest["symbols"],
        "start_month": manifest["start_month"],
        "end_month": manifest["end_month"],
        "archive_count": manifest["archive_count"],
        "full_canonical_manifest_sha256": sha256(canonical).hexdigest(),
        "coverage": manifest["coverage"],
        "note": "Every source ZIP was individually checked against its official CHECKSUM; raw bars and per-ZIP manifest are not published.",
    }


def _numeric_gate(metrics: dict) -> dict[str, bool]:
    values = {
        "cagr_at_least_40pct": metrics["cagr"] is not None and metrics["cagr"] >= .40,
        "sharpe_above_1_5": metrics["sharpe"] is not None and metrics["sharpe"] > 1.5,
        "intrabar_mdd_at_least_minus_20pct": (
            metrics["max_drawdown_intrabar"] is not None
            and metrics["max_drawdown_intrabar"] >= -.20),
        "daily_pf_above_1_3": (
            metrics["profit_factor_daily"] is not None
            and metrics["profit_factor_daily"] > 1.3),
    }
    return {key: bool(value) for key, value in values.items()}


def _gate(diagnostics: dict) -> dict:
    full = _numeric_gate(diagnostics["full"])
    holdout = _numeric_gate(diagnostics["holdout"])
    months = {
        "at_least_70pct_positive_holdout_months": (
            diagnostics["holdout_positive_month_fraction"] is not None
            and diagnostics["holdout_positive_month_fraction"] >= .70),
        "worst_holdout_month_at_least_minus_10pct": (
            diagnostics["holdout_worst_month"] is not None
            and diagnostics["holdout_worst_month"] >= -.10),
    }
    annual = diagnostics["annual"]
    descriptive_years = [str(year) for year in range(2022, 2026)]
    prereg_year = {
        year: bool(year in annual and annual[year]["complete_calendar_year"]
                   and annual[year]["net_return"] is not None
                   and annual[year]["net_return"] > 0
                   and annual[year]["profit_factor_daily"] is not None
                   and annual[year]["profit_factor_daily"] > 1)
        for year in descriptive_years
    }
    prereg_annual_pass = sum(prereg_year.values()) >= 4  # 80% of four whole years rounds up to four.
    partial_2026 = ("2026" in annual and annual["2026"]["net_return"] is not None
                    and annual["2026"]["net_return"] > 0)
    v2_oos_years = [year for year in ("2024", "2025") if year in annual
                    and annual[year]["complete_calendar_year"]]
    v2_oos_fold = {
        year: bool(annual[year]["net_return"] is not None
                   and annual[year]["net_return"] > 0
                   and annual[year]["profit_factor_daily"] is not None
                   and annual[year]["profit_factor_daily"] > 1
                   and annual[year]["max_drawdown_intrabar"] is not None
                   and annual[year]["max_drawdown_intrabar"] >= -.20)
        for year in v2_oos_years
    }
    regimes = diagnostics["regimes"]
    regime_reporting = all(label in regimes and regimes[label]["days"] > 0
                           for label in ("bull", "bear", "sideways"))
    mc30 = diagnostics["monte_carlo_30d_10000"]
    mc21 = diagnostics["monte_carlo_v2_21d_10000"]
    mc30_pass = (mc30["joint_three_gate_frequency"] >= .05
                 and mc30["positive_cagr_frequency"] >= .95)
    historical_numeric = (all(full.values()) and all(holdout.values())
                          and all(months.values()) and prereg_annual_pass
                          and partial_2026 and regime_reporting and mc30_pass)
    evidence = {
        "retrospective_holdout_years": round((END - pd.Timestamp("2024-01-01", tz="UTC")).days / 365.25, 3),
        "independent_oos_at_least_5y": False,
        "at_least_5_complete_oos_yearly_folds": len(v2_oos_years) >= 5,
        "prospective_independent_oos": False,
    }
    return {
        "full_history_numeric": full,
        "historical_holdout_numeric": holdout,
        "monthly_stability": {key: bool(value) for key, value in months.items()},
        "descriptive_2022_2025_years": prereg_year,
        "descriptive_annual_pass": prereg_annual_pass,
        "partial_2026_positive": bool(partial_2026),
        "oos_complete_years": v2_oos_years,
        "oos_fold_conditions": v2_oos_fold,
        "regime_reporting_complete": regime_reporting,
        "preregistered_30d_mc_pass": mc30_pass,
        "v2_21d_mc_four_metric_joint_frequency": mc21["joint_four_gate_frequency"],
        "historical_numeric_diagnostics_pass": historical_numeric,
        "evidence": evidence,
        "full_unified_gate_passed": False,
        "reason_full_gate_false": "Frozen holdout is <5 years and has only two complete OOS calendar folds; prospective independent OOS is also unavailable.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-workers", type=int, default=8,
                        help="HTTP concurrency only; not a strategy parameter")
    args = parser.parse_args()
    if any((REPORT_DIR / name).exists() for name in ("RESULT.json", "DATA_FAILURE.json")):
        parser.error("a formal outcome already exists; do not rerun this history")
    if _git_output("status", "--porcelain"):
        parser.error("commit all source/test changes before the formal data run")
    commit = _git_output("rev-parse", "HEAD")
    started = datetime.now(timezone.utc).isoformat()
    try:
        bundle = load_and_audit(max_workers=args.max_workers)
    except ArchiveIntegrityError as exc:
        _save(REPORT_DIR / "DATA_FAILURE.json", {
            "status": "data_ineligible_no_strategy_pnl_read",
            "reason": str(exc),
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "interpretation_commit": INTERPRETATION_COMMIT,
            "code_commit": commit,
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "full_unified_gate_passed": False,
        })
        print(f"Data integrity gate failed; no strategy P&L computed: {exc}")
        return 2
    audit = _compact_manifest(bundle.manifest)
    _save(REPORT_DIR / "SOURCE_AUDIT.json", audit)
    try:
        base = run_backtest(bundle)
        high_slippage = _run_period(bundle, start=START, end=END,
                                    fee_bps=5.0, slippage_bps=25.0,
                                    monte_carlo_paths=0)
    except BacktestIntegrityError as exc:
        _save(REPORT_DIR / "RESULT.json", {
            "status": "strategy_or_simulation_invalid",
            "reason": str(exc),
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "interpretation_commit": INTERPRETATION_COMMIT,
            "code_commit": commit,
            "source_audit_sha256": audit["full_canonical_manifest_sha256"],
            "full_unified_gate_passed": False,
        })
        print(f"Strategy/simulation invalid; sealed: {exc}")
        return 3
    diagnostics = base["diagnostics"]
    result = {
        "status": "completed_retro_historical_test",
        "evidence_grade": "preregistered retrospective holdout; not independent prospective OOS",
        "research_scope": "cryptocurrency and US equities only; this study is crypto",
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "interpretation_commit": INTERPRETATION_COMMIT,
        "code_commit": commit,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "source_audit_sha256": audit["full_canonical_manifest_sha256"],
        "primary_1x": diagnostics,
        "stress_25bp_slippage_1x": {
            "full": high_slippage["diagnostics"]["full"],
            "holdout": high_slippage["diagnostics"]["holdout"],
            "accounting_usd": high_slippage["diagnostics"]["accounting_usd"],
        },
        "gate": _gate(diagnostics),
        "orders_submitted": False,
    }
    _save(REPORT_DIR / "RESULT.json", result)
    print(json.dumps({
        "result": str(REPORT_DIR / "RESULT.json"),
        "holdout": diagnostics["holdout"],
        "full_unified_gate_passed": False,
    }, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
