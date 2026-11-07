"""Frozen Gate for the 690a7f5 three-altcoin common-flow candidate."""

from __future__ import annotations

import math

from .btc_eth_pairs_metrics import (
    INITIAL_EQUITY, _annual_and_monthly, _day_arrays, _metrics_from_factors,
    _passes_formal, _passes_signal_quality, _regime_report,
    monte_carlo_compressed,
)


SCENARIOS = ("base_1x", "base_2x", "doubled_cost_2x", "no_funding_credit_2x")
MC_SEED = 20261109


def evaluate_alt_flow_gate(scenarios: dict) -> dict:
    if set(scenarios) != set(SCENARIOS):
        raise ValueError("the four frozen accounting scenarios are required")
    records = {}
    factors = {}
    metrics = {}
    replay_verified = {}
    for name in SCENARIOS:
        item = scenarios[name]
        records[name], factors[name] = _day_arrays(item.daily_records, require_fixed_oos=True)
        metrics[name] = _metrics_from_factors(factors[name])
        replay_verified[name] = math.isclose(metrics[name]["max_drawdown"],
                                             item.max_drawdown,
                                             rel_tol=1e-8, abs_tol=1e-9)
        if not replay_verified[name]:
            raise ValueError(f"minute drawdown does not match compressed replay: {name}")
        if not math.isclose(metrics[name]["ending_equity"], item.final_equity,
                            rel_tol=1e-8, abs_tol=1e-7):
            raise ValueError(f"ending equity does not match daily accounting: {name}")
    from .alt_common_flow_sim import END, INPUT_START
    expected_windows = int((END - INPUT_START).total_seconds() // 300)
    if any(scenarios[name].windows_seen != expected_windows for name in SCENARIOS):
        raise ValueError("missing five-minute windows")

    def schedule(item: object) -> list[tuple]:
        return [(trade.entry_signal_time, trade.entry_fill_time,
                 trade.exit_fill_time, trade.direction)
                for trade in item.trades]

    baseline = schedule(scenarios["base_2x"])
    if any(schedule(scenarios[name]) != baseline for name in SCENARIOS):
        raise ValueError("cost or leverage scenario changed the fixed signal schedule")
    trades = list(scenarios["base_2x"].trades)
    annual, monthly, month_summary = _annual_and_monthly(records["base_2x"], trades)
    regimes, regimes_pass = _regime_report(trades)
    mc = monte_carlo_compressed(factors["base_2x"], seed=MC_SEED)
    gate = {
        "signal_quality_1x": _passes_signal_quality(metrics["base_1x"]),
        "formal_2x": _passes_formal(metrics["base_2x"]),
        "annual_walk_forward": all(fold["pass"] for fold in annual.values()),
        "four_market_states": regimes_pass,
        "monte_carlo": mc["joint_pass_fraction"] >= 0.80,
        "doubled_nonfee_cost_profitable": metrics["doubled_cost_2x"]["ending_equity"] > INITIAL_EQUITY,
        "no_funding_credit_profitable": metrics["no_funding_credit_2x"]["ending_equity"] > INITIAL_EQUITY,
        "independent_executable_data": False,
    }
    gate["all_numeric"] = all(v for k, v in gate.items() if k != "independent_executable_data")
    gate["full"] = gate["all_numeric"] and gate["independent_executable_data"]
    costs = {
        name: {
            "entry_and_exit_fee_pnl": math.fsum(t.fee_pnl for t in scenarios[name].trades),
            "funding_pnl": math.fsum(t.funding_pnl for t in scenarios[name].trades),
        }
        for name in SCENARIOS
    }
    return {
        "metrics": metrics,
        "formal_cagr_within_40_to_60_percent_target_band": 0.40 <= metrics["base_2x"]["cagr"] <= 0.60,
        "annual_folds": annual,
        "monthly_returns": monthly,
        "monthly_summary": month_summary,
        "completed_baskets": len(trades),
        "market_states": regimes,
        "monte_carlo": mc,
        "fee_and_funding_pnl": costs,
        "compressed_replay_verified": replay_verified,
        "gate": gate,
        "evidence_grade": "internal mechanism-level historical holdout; executable BBO/fills absent",
    }
