"""One-shot seven-day post-CPI BTC spot replay and the frozen Gate."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from research.btc_exchange_supply_source import SourceFailure
from research.eth_defi_tvl_sim import Daily, _monthly, _numbers, _numeric_pass, _regimes
from research.ww_cpi_disinflation_sim import _buy


START = pd.Timestamp("2021-01-01", tz="UTC")
END = pd.Timestamp("2026-01-01", tz="UTC")
INITIAL = 10_000.0
FEE = .001
SLIP = .0005
MC_PATHS = 5000
MC_BLOCK = 21
MC_SEED = 20261118


@dataclass(frozen=True)
class Simulation:
    days: tuple[Daily, ...]
    fills: tuple[dict, ...]
    fees: float
    reference_slippage: float
    terminal_cash: float


def _schedule(events: tuple[dict, ...], delay_hours: int,
              buy_and_hold: bool) -> tuple[dict[pd.Timestamp, str], int]:
    if delay_hours not in (0, 1):
        raise ValueError("invalid frozen latency")
    if buy_and_hold:
        return {START: "buy"}, 0
    if len(events) != 59:
        raise SourceFailure("CPI release event count changed in simulator")
    schedule: dict[pd.Timestamp, str] = {}
    entries = 0
    previous_exit = START
    for event in events:
        if event["surprise"] >= 0:
            continue
        release = pd.Timestamp(event["released_at"])
        entry = release.ceil("h") + pd.Timedelta(hours=delay_hours)
        if entry <= release:
            entry += pd.Timedelta(hours=1)
        exit_at = entry + pd.Timedelta(hours=168)
        if not START <= entry < exit_at < END or entry <= previous_exit:
            raise SourceFailure("CPI post-release seven-day windows overlap/outside sample")
        if entry in schedule or exit_at in schedule:
            raise SourceFailure("duplicate CPI trade schedule hour")
        schedule[entry], schedule[exit_at] = "buy", "sell"
        previous_exit = exit_at
        entries += 1
    return schedule, entries


def simulate(events: tuple[dict, ...], prices: pd.DataFrame, *,
             slippage: float = SLIP, fee_rate: float = FEE,
             delay_hours: int = 0, buy_and_hold: bool = False) -> Simulation:
    if slippage < 0 or fee_rate < 0:
        raise ValueError("invalid frozen trading cost")
    if not prices.index.equals(pd.date_range("2020-12-01", "2025-12-31 23:00",
                                            tz="UTC", freq="h")):
        raise SourceFailure("BTC spot hourly grid invalid in simulator")
    schedule, _ = _schedule(events, delay_hours, buy_and_hold)
    cash, qty = INITIAL, 0.0
    total_fees, total_slip = 0.0, 0.0
    prior_equity = INITIAL
    fills: list[dict] = []
    days: list[Daily] = []
    day_high = day_low = INITIAL
    day_fee = 0.0
    for ts, row in prices.loc[START:END - pd.Timedelta(hours=1)].iterrows():
        op, hi, lo, cl = (float(row[k]) for k in ("open", "high", "low", "close"))
        action = schedule.get(ts)
        if action == "buy":
            if qty or cash <= 0:
                raise SourceFailure(f"{ts}: invalid BTC buy account state")
            new_qty, notional, fee, residual = _buy(cash, op, slippage, fee_rate)
            prior = prices.loc[ts - pd.Timedelta(hours=720):ts - pd.Timedelta(hours=1),
                               "quote_volume"]
            if len(prior) != 720 or not np.isfinite(prior.to_numpy()).all():
                raise SourceFailure(f"{ts}: 720-hour quote-volume warmup absent")
            median_volume = float(prior.median())
            if median_volume <= 0 or notional > .01 * median_volume:
                raise SourceFailure(f"{ts}: 1% hourly quote-volume capacity failed")
            qty, cash = new_qty, residual
            execution = op * (1 + slippage)
            total_slip += qty * op * slippage
            side = "buy"
        elif action == "sell":
            if qty <= 0:
                raise SourceFailure(f"{ts}: invalid BTC sell account state")
            execution = op * (1 - slippage)
            notional = qty * execution
            if notional < 10:
                raise SourceFailure(f"{ts}: BTC sell below 10 USDT")
            fee = notional * fee_rate
            cash += notional - fee
            total_slip += qty * op * slippage
            new_qty = qty
            qty = 0.0
            side = "sell"
        if action:
            total_fees += fee
            day_fee += fee
            fills.append({"utc": ts.isoformat(), "side": side,
                          "reference_open": op, "execution": execution,
                          "quantity": new_qty, "notional": notional, "fee": fee})
        account_high = cash + qty * hi
        account_low = cash + qty * lo
        account_close = cash + qty * cl
        if buy_and_hold and ts == END - pd.Timedelta(hours=1):
            if qty <= 0:
                raise SourceFailure("BTC buy-and-hold closing quantity absent")
            execution = cl * (1 - slippage)
            notional = qty * execution
            fee = notional * fee_rate
            cash += notional - fee
            total_fees += fee
            day_fee += fee
            total_slip += qty * cl * slippage
            fills.append({"utc": END.isoformat(), "side": "sell",
                          "reference_close_proxy": cl, "execution": execution,
                          "quantity": qty, "notional": notional, "fee": fee})
            qty = 0.0
            account_close = cash
        if not all(math.isfinite(v) and v > 0 for v in
                   (account_high, account_low, account_close, cash + qty * op)):
            raise SourceFailure(f"{ts}: nonpositive/invalid BTC account")
        if cash < -1e-7 or qty < 0 or qty * op > cash + qty * op + 1e-7:
            raise SourceFailure(f"{ts}: spot cash/exposure violation")
        day_high = max(day_high, account_high, account_close)
        day_low = min(day_low, account_low, account_close)
        if ts.hour == 23:
            days.append(Daily(ts.date(), prior_equity, account_close,
                              day_high, day_low, int(qty > 0), day_fee))
            prior_equity = account_close
            day_high = day_low = prior_equity
            day_fee = 0.0
    if qty != 0 or len(days) != 1826 or abs(prior_equity - cash) > 1e-7:
        raise SourceFailure("BTC post-CPI account terminal reconciliation failed")
    if len(fills) != len(schedule) + int(buy_and_hold):
        raise SourceFailure("BTC post-CPI schedule/fills mismatch")
    return Simulation(tuple(days), tuple(fills), total_fees, total_slip, cash)


def _bootstrap(rows: tuple[Daily, ...]) -> dict:
    n = len(rows)
    factors = np.array([r.end / r.start for r in rows])
    high = np.array([r.high / r.start for r in rows])
    low = np.array([r.low / r.start for r in rows])
    rng = np.random.Generator(np.random.PCG64(MC_SEED))
    metrics: list[tuple[float, float | None, float, float | None]] = []
    passes = ruined = undefined = 0
    for _ in range(MC_PATHS):
        blocks = rng.integers(0, n, size=math.ceil(n / MC_BLOCK))
        idx = ((blocks[:, None] + np.arange(MC_BLOCK)) % n).ravel()[:n]
        factor = factors[idx]
        equity = INITIAL * np.cumprod(factor)
        begins = np.r_[INITIAL, equity[:-1]]
        if not np.isfinite(equity).all() or (equity <= 0).any():
            ruined += 1
            continue
        peaks = np.maximum.accumulate(np.r_[INITIAL, begins * high[idx]])[1:]
        drawdown = float(np.max(1 - begins * low[idx] / peaks))
        returns = factor - 1
        sigma = float(np.std(returns, ddof=1))
        sharpe = float(np.mean(returns) / sigma * math.sqrt(365)) if sigma > 0 else None
        pnl = equity - begins
        losses = float(-pnl[pnl < 0].sum())
        pf = float(pnl[pnl > 0].sum() / losses) if losses > 0 else None
        cagr = float((equity[-1] / INITIAL) ** (365.25 / n) - 1)
        metrics.append((cagr, sharpe, drawdown, pf))
        if sharpe is None or pf is None:
            undefined += 1
        if _numeric_pass({"cagr": cagr, "sharpe": sharpe,
                          "max_drawdown": drawdown,
                          "profit_factor_daily_dollar": pf}):
            passes += 1
    def quantile(field: int) -> dict:
        values = [x[field] for x in metrics if x[field] is not None and math.isfinite(x[field])]
        return {f"p{q:02d}": float(np.percentile(values, q)) if values else None
                for q in (5, 50, 95)}
    return {"paths": MC_PATHS, "joint_passes": passes,
            "joint_pass_rate": passes / MC_PATHS, "ruined": ruined,
            "undefined": undefined, "cagr": quantile(0),
            "sharpe": quantile(1), "max_drawdown": quantile(2),
            "profit_factor": quantile(3)}


def evaluate(events: tuple[dict, ...], primary: Simulation, gross: Simulation,
             double_slip: Simulation, delayed: Simulation, hold: Simulation,
             prices: pd.DataFrame) -> dict:
    full = _numbers(primary.days)
    annual = {str(year): _numbers([r for r in primary.days if r.day.year == year])
              for year in range(2021, 2026)}
    holdout = _numbers([r for r in primary.days if r.day.year in (2024, 2025)])
    months = _monthly(primary.days)
    daily_prices = prices.resample("1D").agg({"close": "last"})
    regimes = _regimes(primary.days, daily_prices)
    mc = _bootstrap(primary.days)
    buys = [f for f in primary.fills if f["side"] == "buy"]
    buys_by_year = {str(y): sum(f["utc"][:4] == str(y) for f in buys)
                    for y in range(2021, 2026)}
    checks = {
        "numerical_full": _numeric_pass(full),
        "minimum_20_entries": len(buys) >= 20,
        "annual_walk_forward": all(x["end"] > x["start"] and x["sharpe"] is not None
                                   and x["sharpe"] > 0 and buys_by_year[y] > 0
                                   for y, x in annual.items()),
        "historical_2024_2025_holdout": _numeric_pass(holdout),
        "monthly_stability": months["months"] == 60 and months["positive"] >= 36
                             and months["longest_nonpositive_streak"] <= 3,
        "four_regimes": all(x["days"] >= 90 and x["net_dollar_pnl"] > 0
                            for x in regimes.values()),
        "block_monte_carlo": mc["joint_pass_rate"] >= .8,
        "double_slippage": double_slip.terminal_cash > INITIAL,
        "one_hour_latency": delayed.terminal_cash > INITIAL,
        "independent_executable_oos": False,
    }
    return {"full": full, "gross_no_cost": _numbers(gross.days),
            "btc_buy_and_hold": _numbers(hold.days),
            "holdout_2024_2025": holdout, "annual": annual,
            "monthly": months, "regimes": regimes, "monte_carlo": mc,
            "checks": checks, "all_gate_passed": all(checks.values()),
            "positive_cpi_surprises": sum(e["surprise"] > 0 for e in events),
            "negative_cpi_surprises": sum(e["surprise"] < 0 for e in events),
            "equal_cpi_surprises": sum(e["surprise"] == 0 for e in events),
            "entries": len(buys), "buys_by_year": buys_by_year,
            "fills": primary.fills, "fees_usdt": primary.fees,
            "reference_slippage_usdt": primary.reference_slippage,
            "net_final_usdt": primary.terminal_cash,
            "gross_final_usdt": gross.terminal_cash,
            "double_slippage_final_usdt": double_slip.terminal_cash,
            "one_hour_latency_final_usdt": delayed.terminal_cash}
