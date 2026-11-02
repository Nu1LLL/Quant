"""Frozen Gate for the preregistered H.4.1 reserve-balance crypto study.

Pure daily-ledger statistics are reused from the already tested Wikipedia
study. This module does not reuse its signal, data loader, or Wikimedia-specific
affected-window diagnostics. Neither model nor thresholds depend on outcomes.
"""

from __future__ import annotations

from research.wiki_attention_metrics import (
    REGIMES,
    MetricsDataError,
    _annual_walk_forward,
    _btc_daily_closes,
    _checked_records,
    _ledger_metadata,
    _monthly_stability,
    _regime_report,
    _summary,
    block_monte_carlo,
)


MC_SEED = 20261103


def evaluate(ledger: object, benchmark_ledger: object,
             btc_trade_bars: object) -> dict:
    """Evaluate the unique fixed five-year retrospective experiment."""
    rows = _checked_records(ledger, "strategy")
    metadata = _ledger_metadata(ledger, "strategy")
    strategy = _summary(rows, stressed_drawdown=metadata["conservative_max_drawdown"])
    benchmark = benchmark_metadata = None
    if benchmark_ledger is not None:
        control_rows = _checked_records(benchmark_ledger, "benchmark")
        benchmark_metadata = _ledger_metadata(benchmark_ledger, "benchmark")
        benchmark = _summary(
            control_rows,
            stressed_drawdown=benchmark_metadata["conservative_max_drawdown"],
        )
    folds = _annual_walk_forward(rows, metadata["conservative_max_drawdown_by_year"])
    fold_passes = sum(item["passed"] for item in folds)
    monthly = _monthly_stability(rows)
    regimes = _regime_report(rows, _btc_daily_closes(btc_trade_bars))
    holdout_rows = [row for row in rows if row.day.year >= 2024]
    holdout = _summary(holdout_rows)
    holdout["period"] = "2024-01-01/2025-12-31"
    holdout["independent_oos"] = False
    holdout["drawdown_basis"] = (
        "daily_close_only; exact 4h path not supplied for two-year slice"
    )
    mc = block_monte_carlo(
        [float(row.return_net) for row in rows],
        paths=10_000, block_days=21, seed=MC_SEED,
    )
    checks = {
        "history_at_least_five_complete_years": len(rows) == 1826,
        "five_complete_annual_folds": len(folds) == 5,
        "cagr_at_least_40pct": strategy["cagr"] >= 0.40,
        "sharpe_above_1_5": (
            strategy["sharpe"] is not None and strategy["sharpe"] > 1.5
        ),
        "conservative_4h_drawdown_at_least_minus_20pct": (
            strategy["conservative_max_drawdown"] >= -0.20
        ),
        "daily_profit_factor_above_1_3": (
            strategy["profit_factor_daily"] is not None
            and strategy["profit_factor_daily"] > 1.3
        ),
        "annual_walk_forward_at_least_4_of_5": fold_passes >= 4,
        "monthly_stability": monthly["passed"],
        "all_three_regimes_present": all(regimes[name]["days"] > 0 for name in REGIMES),
        "monte_carlo": mc["passed"],
        "capacity": metadata["capacity_breaches"] == 0,
        "minimum_order_execution": metadata["execution_breaches"] == 0,
        "margin_proxy": metadata["margin_proxy_breaches"] == 0,
        "terminal_flat": metadata["terminal_flat"],
    }
    return {
        "strategy": strategy,
        "benchmark": benchmark,
        "strategy_execution": metadata,
        "benchmark_execution": benchmark_metadata,
        "annual_walk_forward": folds,
        "annual_walk_forward_passes": fold_passes,
        "historical_time_holdout_2024_2025": holdout,
        "monthly_stability": monthly,
        "regimes": regimes,
        "monte_carlo": mc,
        "gate_checks": checks,
        "numerical_gate": all(checks.values()),
        "benchmark_path_complete": benchmark is not None,
        "independent_oos": False,
        "full_gate": False,
        "evidence_grade": "preregistered_retrospective_historical_holdout",
    }
