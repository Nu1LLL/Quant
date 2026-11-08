"""Frozen evaluation for BTC option IV term structure; paper-only spot account."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from research.deribit_option_taker_sim import INITIAL, Simulation, simulate
from research.eth_defi_tvl_sim import Daily, _monthly, _numbers, _numeric_pass, _regimes


def _bootstrap(rows: tuple[Daily, ...]) -> dict:
    n = len(rows)
    factor = np.array([x.end / x.start for x in rows])
    high = np.array([x.high / x.start for x in rows])
    low = np.array([x.low / x.start for x in rows])
    rng = np.random.Generator(np.random.PCG64(20261109))
    values: list[tuple[float, float | None, float, float | None]] = []
    passes = ruined = undefined = 0
    for _ in range(5000):
        starts = rng.integers(0, n, size=math.ceil(n / 21))
        idx = ((starts[:, None] + np.arange(21)) % n).ravel()[:n]
        daily = factor[idx]
        equity = INITIAL * np.cumprod(daily)
        begin = np.r_[INITIAL, equity[:-1]]
        if not np.isfinite(equity).all() or (equity <= 0).any():
            ruined += 1
            continue
        peak = np.maximum.accumulate(np.r_[INITIAL, begin * high[idx]])[1:]
        dd = float(np.max(1 - begin * low[idx] / peak))
        returns = daily - 1
        sd = float(np.std(returns, ddof=1))
        sharpe = float(np.mean(returns) / sd * math.sqrt(365)) if sd > 0 else None
        dollars = equity - begin
        losses = float(-dollars[dollars < 0].sum())
        pf = float(dollars[dollars > 0].sum() / losses) if losses > 0 else None
        cagr = float((equity[-1] / INITIAL) ** (365.25 / n) - 1)
        values.append((cagr, sharpe, dd, pf))
        if sharpe is None or pf is None:
            undefined += 1
        if _numeric_pass({"cagr": cagr, "sharpe": sharpe,
                          "max_drawdown": dd, "profit_factor_daily_dollar": pf}):
            passes += 1

    def quantiles(field: int) -> dict:
        valid = [x[field] for x in values if x[field] is not None and math.isfinite(x[field])]
        return {f"p{q:02d}": float(np.percentile(valid, q)) if valid else None
                for q in (5, 50, 95)}

    return {"paths": 5000, "joint_passes": passes, "joint_pass_rate": passes / 5000,
            "ruined": ruined, "undefined": undefined,
            "cagr": quantiles(0), "sharpe": quantiles(1),
            "max_drawdown": quantiles(2), "profit_factor": quantiles(3)}


def evaluate(primary: Simulation, gross: Simulation, double_slip: Simulation,
             late: Simulation, hold: Simulation, prices: pd.DataFrame,
             source_audit: dict) -> dict:
    full = _numbers(primary.days)
    annual = {str(y): _numbers([x for x in primary.days if x.day.year == y])
              for y in range(2021, 2026)}
    holdout = _numbers([x for x in primary.days if x.day.year in (2024, 2025)])
    months = _monthly(primary.days)
    states = _regimes(primary.days, prices)
    mc = _bootstrap(primary.days)
    buys = sum(x["side"] == "buy" for x in primary.fills)
    fills_by_year = {str(y): sum(x["day"][:4] == str(y) for x in primary.fills)
                     for y in range(2021, 2026)}
    checks = {
        "source_coverage": source_audit["deribit"]["coverage_pass"],
        "numerical_full": _numeric_pass(full),
        "min_20_buys": buys >= 20,
        "annual_walk_forward": all(v["end"] > v["start"] and v["sharpe"] is not None and
                                   v["sharpe"] > 0 and fills_by_year[y] > 0
                                   for y, v in annual.items()),
        "historical_2024_2025_holdout": _numeric_pass(holdout),
        "monthly_stability": months["months"] == 60 and months["positive"] >= 36 and
                             months["longest_nonpositive_streak"] <= 3,
        "four_regimes": all(x["days"] >= 90 and x["net_dollar_pnl"] > 0
                            for x in states.values()),
        "block_monte_carlo": mc["joint_pass_rate"] >= .8,
        "double_slippage": double_slip.terminal_cash > INITIAL,
        "one_day_latency": late.terminal_cash > INITIAL,
        "independent_executable_oos": False,
    }
    return {"full": full, "gross_no_cost": _numbers(gross.days),
            "btc_buy_and_hold": _numbers(hold.days), "holdout_2024_2025": holdout,
            "annual": annual, "monthly": months, "regimes": states,
            "monte_carlo": mc, "checks": checks,
            "all_gate_passed": all(checks.values()), "buys": buys,
            "fills_by_year": fills_by_year, "fills": primary.fills,
            "fees_usdt": primary.fees, "slippage_reference_drag_usdt": primary.slippage_drag,
            "net_final_usdt": primary.terminal_cash,
            "gross_final_usdt": gross.terminal_cash,
            "double_slippage_final_usdt": double_slip.terminal_cash,
            "one_day_latency_final_usdt": late.terminal_cash}


__all__ = ["simulate", "evaluate"]
