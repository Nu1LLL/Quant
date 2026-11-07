"""Fixed five-year 01:00-anchored Gate for the funding-spread candidate."""

from __future__ import annotations

import math

from .btc_eth_pairs_metrics import (
    INITIAL_EQUITY, _annual_and_monthly, _day_arrays, _metrics_from_factors,
    _passes_formal, _passes_signal_quality, _regime_report,
    monte_carlo_compressed,
)
from .funding_spread_sim import SCENARIOS


NAMES = tuple(item[0] for item in SCENARIOS)
MC_SEED = 20261110


def evaluate_funding_spread_gate(scenarios: dict) -> dict:
    if set(scenarios) != set(NAMES):
        raise ValueError("four frozen funding-spread scenarios required")
    records = {}
    factors = {}
    metrics = {}
    replay_verified = {}
    for name in NAMES:
        scenario = scenarios[name]
        records[name], factors[name] = _day_arrays(scenario.daily_records,
                                                   require_fixed_oos=True)
        metrics[name] = _metrics_from_factors(factors[name])
        replay_verified[name] = math.isclose(
            metrics[name]["max_drawdown"], scenario.max_drawdown,
            rel_tol=1e-8, abs_tol=1e-9,
        )
        if not replay_verified[name]:
            raise ValueError(f"compressed replay disagrees with hourly drawdown: {name}")
        if not math.isclose(metrics[name]["ending_equity"], scenario.final_equity,
                            rel_tol=1e-8, abs_tol=1e-7):
            raise ValueError(f"daily accounting disagrees with final equity: {name}")
        if scenario.decision_opportunities != 59:
            raise ValueError("wrong number of fixed monthly decisions")

    def schedule(item):
        return [(t.entry_signal_time, t.entry_fill_time, t.exit_fill_time,
                 t.long_symbol, t.short_symbol) for t in item.trades]

    baseline = schedule(scenarios["base_2x"])
    if any(schedule(scenarios[name]) != baseline for name in NAMES):
        raise ValueError("leverage or cost scenario changed the signal schedule")
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
        "doubled_nonfee_cost_profitable": metrics["doubled_cost_2x"]["ending_equity"] > INITIAL_EQUITY,
        "no_funding_credit_profitable": metrics["no_funding_credit_2x"]["ending_equity"] > INITIAL_EQUITY,
        "independent_executable_data": False,
    }
    gate["all_numeric"] = all(v for k, v in gate.items() if k != "independent_executable_data")
    gate["full"] = gate["all_numeric"] and gate["independent_executable_data"]
    pnl = {}
    for name in NAMES:
        item = scenarios[name]
        fees = math.fsum(t.fee_pnl for t in item.trades)
        funding = math.fsum(t.funding_pnl for t in item.trades)
        net = item.final_equity - INITIAL_EQUITY
        pnl[name] = {"net_pnl": net, "gross_price_pnl": net - fees - funding,
                     "fee_pnl": fees, "funding_pnl": funding}
    return {
        "metrics": metrics,
        "formal_cagr_within_40_to_60_percent_target_band": .40 <= metrics["base_2x"]["cagr"] <= .60,
        "annual_folds": annual,
        "monthly_returns": monthly,
        "monthly_summary": month_summary,
        "completed_pairs": len(trades),
        "decision_opportunities": 59,
        "market_states": regimes,
        "monte_carlo": mc,
        "pnl_accounting": pnl,
        "compressed_replay_verified": replay_verified,
        "gate": gate,
        "evidence_grade": "internal funding-mechanism historical holdout; BBO/fill and prospective OOS absent",
        "daily_boundary_utc_hour": 1,
    }
