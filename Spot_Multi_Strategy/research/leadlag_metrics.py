"""Frozen five-year Gate for the preregistered ten-coin lead-lag model."""

from __future__ import annotations

import math

from .btc_eth_pairs_metrics import (
    INITIAL_EQUITY, _annual_and_monthly, _day_arrays, _metrics_from_factors,
    _passes_formal, _passes_signal_quality, _regime_report,
    monte_carlo_compressed,
)
from .leadlag_sim import SCENARIOS


NAMES = tuple(item[0] for item in SCENARIOS)
MC_SEED = 20261111


def evaluate_leadlag_gate(scenarios: dict, *, funding_schedule_verified: bool) -> dict:
    if set(scenarios) != set(NAMES):
        raise ValueError("four frozen lead-lag scenarios required")
    records = {}
    factors = {}
    metrics = {}
    replay_verified = {}
    for name in NAMES:
        result = scenarios[name]
        if result.decision_opportunities != 72 * 1826 or result.monthly_fits != 60:
            raise ValueError("wrong number of fixed decisions or monthly model fits")
        if result.min_training_rows < 8000:
            raise ValueError("training sample below predeclared minimum")
        records[name], factors[name] = _day_arrays(result.daily_records,
                                                   require_fixed_oos=True)
        metrics[name] = _metrics_from_factors(factors[name])
        replay_verified[name] = math.isclose(
            metrics[name]["max_drawdown"], result.max_drawdown,
            rel_tol=1e-8, abs_tol=1e-9,
        )
        if not replay_verified[name]:
            raise ValueError(f"compressed pressure replay disagrees with five-minute marks: {name}")
        if not math.isclose(metrics[name]["ending_equity"], result.final_equity,
                            rel_tol=1e-8, abs_tol=1e-7):
            raise ValueError(f"daily and final equity disagree: {name}")

    def schedule(result):
        return [(t.entry_signal_time, t.long_symbol, t.short_symbol,
                 t.regime, t.predicted_spread, t.exit_fill_time)
                for t in result.trades]

    baseline = schedule(scenarios["base_2x"])
    if any(schedule(scenarios[name]) != baseline for name in NAMES):
        raise ValueError("cost or latency stress changed selected signals")
    for name in NAMES:
        expected_delay = 10 if name == "latency_2x" else 5
        if any((t.entry_fill_time - t.entry_signal_time).total_seconds() != expected_delay * 60
               or (t.exit_fill_time - t.entry_signal_time).total_seconds() != 15 * 60
               for t in scenarios[name].trades):
            raise ValueError("fill timing differs from frozen scenario")
    trades = list(scenarios["base_2x"].trades)
    annual, monthly, month_summary = _annual_and_monthly(records["base_2x"], trades)
    regimes, regimes_pass = _regime_report(trades)
    mc = monte_carlo_compressed(factors["base_2x"], seed=MC_SEED)
    gate = {
        "signal_quality_1x": _passes_signal_quality(metrics["base_1x"]),
        "formal_2x": _passes_formal(metrics["base_2x"]),
        "annual_walk_forward": all(item["pass"] for item in annual.values()),
        "four_market_states": regimes_pass,
        "monte_carlo": mc["joint_pass_fraction"] >= .80,
        "double_slippage_profitable": metrics["double_slippage_2x"]["ending_equity"] > INITIAL_EQUITY,
        "extra_bar_latency_profitable": metrics["latency_2x"]["ending_equity"] > INITIAL_EQUITY,
        "funding_schedule_avoided": funding_schedule_verified,
        "independent_executable_data": False,
    }
    gate["all_numeric"] = all(value for key, value in gate.items()
                              if key != "independent_executable_data")
    gate["full"] = gate["all_numeric"] and gate["independent_executable_data"]
    pnl = {}
    for name in NAMES:
        result = scenarios[name]
        executed_price = math.fsum(t.gross_price_pnl for t in result.trades)
        fees = math.fsum(t.fee_pnl for t in result.trades)
        net = result.final_equity - INITIAL_EQUITY
        if not math.isclose(net, executed_price + fees, rel_tol=1e-7, abs_tol=1e-5):
            raise ValueError(f"trade PnL does not explain final equity: {name}")
        pnl[name] = {
            "net_pnl": net,
            "executed_price_pnl_before_fees": executed_price,
            "taker_fee_pnl": fees,
            "funding_pnl": 0.0,
        }
    return {
        "metrics": metrics,
        "formal_cagr_within_40_to_60_percent_target_band": .40 <= metrics["base_2x"]["cagr"] <= .60,
        "annual_folds": annual,
        "monthly_returns": monthly,
        "monthly_summary": month_summary,
        "completed_pairs": len(trades),
        "decision_opportunities": 72 * 1826,
        "monthly_model_fits": 60,
        "minimum_training_rows": min(s.min_training_rows for s in scenarios.values()),
        "market_states": regimes,
        "monte_carlo": mc,
        "pnl_accounting": pnl,
        "compressed_replay_verified": replay_verified,
        "gate": gate,
        "evidence_grade": "internal rolling historical holdout; five-year prospective OOS and BBO/fills absent",
    }
