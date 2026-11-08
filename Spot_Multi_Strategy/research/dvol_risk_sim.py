"""Frozen two-coin spot replay for the preregistered Deribit DVOL hypothesis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import math
from statistics import median

import numpy as np
import pandas as pd

from research.dvol_risk_source import (
    CURRENCIES, END, FIRST_DVOL, FIRST_PRICE, START, SYMBOLS,
)
from research.eth_defi_tvl_sim import Daily, _monthly, _numbers, _numeric_pass


INITIAL = 10_000.0
FEE = 0.0005
SLIP = 0.001
LOTS = {"BTCUSDT": 0.00001, "ETHUSDT": 0.0001}
MC_SEED = 20261128
MC_PATHS = 5000
MC_BLOCK = 21


class DvolReplayFailure(ValueError):
    pass


@dataclass(frozen=True)
class Simulation:
    days: tuple[Daily, ...]
    trades: tuple[dict, ...]
    fees: float
    slippage_drag: float
    final: float
    entries: int


def weekly_targets(dvol: dict[str, pd.DataFrame],
                   prices: dict[str, pd.DataFrame]) -> dict[date, bool]:
    expected_dvol = pd.date_range(FIRST_DVOL, END - timedelta(days=1), tz="UTC", freq="D")
    expected_price = pd.date_range(FIRST_PRICE, END - timedelta(days=1), tz="UTC", freq="D")
    if set(dvol) != set(CURRENCIES) or set(prices) != set(SYMBOLS):
        raise DvolReplayFailure("asset universe mismatch")
    for c in CURRENCIES:
        frame = dvol[c]
        if not frame.index.equals(expected_dvol) or tuple(frame.columns) != (
                "open", "high", "low", "close"):
            raise DvolReplayFailure(f"{c}: DVOL grid/columns changed")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise DvolReplayFailure(f"{c}: nonfinite DVOL")
    for s in SYMBOLS:
        frame = prices[s]
        if not frame.index.equals(expected_price) or tuple(frame.columns) != (
                "open", "high", "low", "close", "quote_volume"):
            raise DvolReplayFailure(f"{s}: spot grid/columns changed")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise DvolReplayFailure(f"{s}: nonfinite spot data")
    targets: dict[date, bool] = {}
    for ts in pd.date_range("2021-05-02", "2026-09-27", tz="UTC", freq="W-SUN"):
        checks = []
        for c in CURRENCIES:
            s = f"{c}USDT"
            closes = prices[s].loc[ts - pd.Timedelta(days=28):ts, "close"].to_numpy(dtype=float)
            if len(closes) != 29 or (closes <= 0).any():
                raise DvolReplayFailure(f"{c} {ts.date()}: missing RV input")
            returns = np.diff(np.log(closes))
            rv = float(np.std(returns, ddof=1) * math.sqrt(365) * 100)
            iv = float(dvol[c].at[ts, "close"])
            if not math.isfinite(rv) or rv <= 0 or not math.isfinite(iv) or iv <= 2:
                raise DvolReplayFailure(f"{c} {ts.date()}: invalid IV/RV units")
            checks.append(iv / rv <= 1.25)
        targets[ts.date() + timedelta(days=3)] = all(checks)
    if len(targets) != 283:
        raise DvolReplayFailure("weekly decision schedule changed")
    return targets


def _capacity(prices: dict[str, pd.DataFrame], symbol: str, day: date,
              notional: float) -> None:
    dates = pd.date_range(day - timedelta(days=30), day - timedelta(days=1), tz="UTC")
    volume = prices[symbol].loc[dates, "quote_volume"]
    if len(volume) != 30 or not np.isfinite(volume.to_numpy(dtype=float)).all():
        raise DvolReplayFailure(f"{symbol} {day}: 30-day quote-volume history absent")
    if notional < 10 or notional > 0.01 * float(median(volume)):
        raise DvolReplayFailure(f"{symbol} {day}: order min/capacity failed")


def simulate(dvol: dict[str, pd.DataFrame], prices: dict[str, pd.DataFrame],
             terminal: dict[str, float], *, slip: float = SLIP,
             fee_rate: float = FEE, latency_days: int = 0,
             benchmark: bool = False) -> Simulation:
    if slip < 0 or fee_rate < 0 or latency_days not in (0, 1):
        raise ValueError("unregistered execution scenario")
    if (set(terminal) != set(SYMBOLS) or
            any(not math.isfinite(v) or v <= 0 for v in terminal.values())):
        raise DvolReplayFailure("terminal price invalid")
    targets = weekly_targets(dvol, prices)
    cash = INITIAL
    qty = {s: 0.0 for s in SYMBOLS}
    fees = slip_drag = 0.0
    entries = 0
    trades: list[dict] = []
    days: list[Daily] = []
    prev_equity = INITIAL

    def equity(mark: dict[str, float]) -> float:
        return cash + sum(qty[s] * mark[s] for s in SYMBOLS)

    def sell(day: date, reference: dict[str, float], *, forced: bool = False) -> float:
        nonlocal cash, fees, slip_drag
        paid = 0.0
        for s in SYMBOLS:
            if qty[s] <= 0:
                raise DvolReplayFailure(f"{day}: partial basket on sale")
            ref, amount = reference[s], qty[s]
            execution = ref * (1 - slip)
            notional = amount * execution
            if notional < 10:
                raise DvolReplayFailure(f"{s} {day}: exit below minimum")
            fee = notional * fee_rate
            cash += notional - fee
            fees += fee
            paid += fee
            slip_drag += amount * ref * slip
            trades.append({"day": str(day), "side": "sell", "symbol": s,
                           "quantity": amount, "reference": ref,
                           "execution": execution, "notional": notional,
                           "fee": fee, "forced": forced})
            qty[s] = 0.0
        return paid

    def buy(day: date, reference: dict[str, float]) -> float:
        nonlocal cash, fees, slip_drag, entries
        if any(qty.values()):
            raise DvolReplayFailure(f"{day}: entry with existing position")
        budget = cash / 2
        orders = []
        for s in SYMBOLS:
            ref = reference[s]
            execution = ref * (1 + slip)
            lot = LOTS[s]
            amount = math.floor(budget / (execution * (1 + fee_rate)) / lot) * lot
            notional = amount * execution
            _capacity(prices, s, day, notional)
            orders.append((s, amount, ref, execution, notional))
        paid = 0.0
        for s, amount, ref, execution, notional in orders:
            fee = notional * fee_rate
            cash -= notional + fee
            fees += fee
            paid += fee
            slip_drag += amount * ref * slip
            qty[s] = amount
            trades.append({"day": str(day), "side": "buy", "symbol": s,
                           "quantity": amount, "reference": ref,
                           "execution": execution, "notional": notional, "fee": fee})
        if cash < -1e-7:
            raise DvolReplayFailure(f"{day}: buy exceeded cash")
        cash = max(cash, 0.0)
        entries += 1
        return paid

    for ts in pd.date_range(START, END - timedelta(days=1), tz="UTC", freq="D"):
        day = ts.date()
        opened = {s: float(prices[s].at[ts, "open"]) for s in SYMBOLS}
        highs = {s: float(prices[s].at[ts, "high"]) for s in SYMBOLS}
        lows = {s: float(prices[s].at[ts, "low"]) for s in SYMBOLS}
        closes = {s: float(prices[s].at[ts, "close"]) for s in SYMBOLS}
        if any(highs[s] < max(opened[s], closes[s], lows[s]) or
               lows[s] > min(opened[s], closes[s], highs[s]) for s in SYMBOLS):
            raise DvolReplayFailure(f"{day}: invalid spot OHLC")
        before_trade = equity(opened)
        daily_fee = 0.0
        if benchmark and day == START:
            daily_fee += buy(day, opened)
        elif not benchmark:
            decision_day = day - timedelta(days=latency_days)
            if decision_day in targets and day.weekday() == 2 + latency_days:
                target = targets[decision_day]
                invested = all(qty[s] > 0 for s in SYMBOLS)
                if any(qty[s] > 0 for s in SYMBOLS) and not invested:
                    raise DvolReplayFailure(f"{day}: partial basket")
                if target and not invested:
                    daily_fee += buy(day, opened)
                elif not target and invested:
                    daily_fee += sell(day, opened)
        post_trade = equity(opened)
        end_equity = equity(closes)
        high_equity = max(prev_equity, before_trade, post_trade, equity(highs), end_equity)
        low_equity = min(prev_equity, before_trade, post_trade, equity(lows), end_equity)
        if (not all(math.isfinite(v) and v > 0 for v in
                    (prev_equity, end_equity, high_equity, low_equity))
                or cash < 0 or cash > end_equity * (1 + 1e-9)):
            raise DvolReplayFailure(f"{day}: invalid equity/exposure")
        days.append(Daily(day, prev_equity, end_equity, high_equity, low_equity,
                          int(any(qty.values())), daily_fee))
        prev_equity = end_equity
    if any(qty.values()):
        terminal_before = equity(terminal)
        terminal_fee = sell(END, terminal, forced=True)
        last = days[-1]
        days[-1] = Daily(last.day, last.start, cash,
                         max(last.high, terminal_before, cash),
                         min(last.low, terminal_before, cash),
                         last.position, last.fee + terminal_fee)
    else:
        cash = prev_equity
    if (len(days) != 1979 or not math.isfinite(cash) or cash <= 0
            or not math.isclose(days[-1].end, cash, rel_tol=1e-10, abs_tol=1e-7)):
        raise DvolReplayFailure("terminal account did not reconcile")
    return Simulation(tuple(days), tuple(trades), fees, slip_drag, cash, entries)


def _regimes(rows: tuple[Daily, ...], prices: dict[str, pd.DataFrame]) -> dict[str, dict]:
    closes = {ts.date(): float(x) for ts, x in prices["BTCUSDT"]["close"].items()}
    out = {f"{trend}_{vol}": {"days": 0, "net_dollar_pnl": 0.0}
           for trend in ("bull", "bear") for vol in ("high", "low")}
    for row in rows:
        points = np.array([closes[row.day - timedelta(days=k)] for k in range(31, 0, -1)])
        returns = np.diff(np.log(points))
        trend = "bull" if float(returns.sum()) > 0 else "bear"
        vol = "high" if float(np.std(returns, ddof=1) * math.sqrt(365)) >= .80 else "low"
        state = out[f"{trend}_{vol}"]
        state["days"] += 1
        state["net_dollar_pnl"] += row.end - row.start
    return out


def _bootstrap(rows: tuple[Daily, ...]) -> dict:
    n = len(rows)
    factor = np.array([r.end / r.start for r in rows], dtype=float)
    high = np.array([r.high / r.start for r in rows], dtype=float)
    low = np.array([r.low / r.start for r in rows], dtype=float)
    rng = np.random.Generator(np.random.PCG64(MC_SEED))
    paths = []
    passed = ruined = undefined = 0
    for _ in range(MC_PATHS):
        starts = rng.integers(0, n, size=math.ceil(n / MC_BLOCK))
        index = ((starts[:, None] + np.arange(MC_BLOCK)) % n).ravel()[:n]
        f = factor[index]
        equity = INITIAL * np.cumprod(f)
        begins = np.r_[INITIAL, equity[:-1]]
        if not np.isfinite(equity).all() or (equity <= 0).any():
            ruined += 1
            continue
        peak = np.maximum.accumulate(np.r_[INITIAL, begins * high[index]])[1:]
        dd = float(np.max(1 - begins * low[index] / peak))
        returns = f - 1
        sigma = float(np.std(returns, ddof=1))
        sharpe = float(np.mean(returns) / sigma * math.sqrt(365)) if sigma > 0 else None
        dollars = equity - begins
        losses = float(-dollars[dollars < 0].sum())
        pf = float(dollars[dollars > 0].sum() / losses) if losses > 0 else None
        cagr = float((equity[-1] / INITIAL) ** (365.25 / n) - 1)
        paths.append((cagr, sharpe, dd, pf))
        if sharpe is None or pf is None:
            undefined += 1
        if _numeric_pass({"cagr": cagr, "sharpe": sharpe, "max_drawdown": dd,
                          "profit_factor_daily_dollar": pf}):
            passed += 1

    def quantile(i: int) -> dict:
        values = [p[i] for p in paths if p[i] is not None and math.isfinite(p[i])]
        return {f"p{q:02d}": float(np.percentile(values, q)) if values else None
                for q in (5, 50, 95)}

    return {"paths": MC_PATHS, "joint_passes": passed,
            "joint_pass_rate": passed / MC_PATHS, "ruined": ruined,
            "undefined": undefined, "cagr": quantile(0), "sharpe": quantile(1),
            "max_drawdown": quantile(2), "profit_factor": quantile(3)}


def evaluate(net: Simulation, gross: Simulation, double_slip: Simulation,
             latency: Simulation, benchmark: Simulation,
             prices: dict[str, pd.DataFrame]) -> dict:
    full = _numbers(net.days)
    annual = {str(y): _numbers([r for r in net.days if r.day.year == y])
              for y in range(2021, 2027)}
    months = _monthly(net.days)
    holdout = _numbers([r for r in net.days if r.day >= date(2025, 1, 1)])
    regimes = _regimes(net.days, prices)
    mc = _bootstrap(net.days)
    yearly_entries = {str(y): sum(t["side"] == "buy" and t["symbol"] == "BTCUSDT"
                                  and t["day"].startswith(str(y)) for t in net.trades)
                      for y in range(2021, 2027)}
    checks = {
        "numerical_full": _numeric_pass(full),
        "at_least_20_basket_entries": net.entries >= 20,
        "annual_walk_forward": all(yearly_entries[y] > 0 and stat["end"] > stat["start"]
                                   and stat["sharpe"] is not None and stat["sharpe"] > 0
                                   for y, stat in annual.items()),
        "historical_2025_2026_holdout": _numeric_pass(holdout),
        "monthly_stability": months["months"] == 65 and months["positive"] >= 39
                             and months["longest_nonpositive_streak"] <= 3,
        "four_regimes": all(v["days"] >= 90 and v["net_dollar_pnl"] > 0
                            for v in regimes.values()),
        "block_monte_carlo": mc["joint_pass_rate"] >= .80,
        "double_slippage": double_slip.final > INITIAL,
        "one_day_latency": latency.final > INITIAL,
        "independent_executable_oos": False,
    }
    return {"full": full, "buy_and_hold": _numbers(benchmark.days),
            "gross_no_cost": _numbers(gross.days), "annual": annual,
            "annual_basket_entries": yearly_entries, "holdout_2025_2026": holdout,
            "monthly": months, "regimes": regimes, "monte_carlo": mc,
            "checks": checks, "all_gate_passed": all(checks.values()),
            "inside_40_to_60_percent_target_band": .40 <= full["cagr"] <= .60,
            "risk_on_entries": net.entries, "net_final_usdt": net.final,
            "benchmark_final_usdt": benchmark.final,
            "gross_final_usdt": gross.final,
            "double_slippage_final_usdt": double_slip.final,
            "one_day_latency_final_usdt": latency.final,
            "fees_usdt": net.fees, "funding_usdt": 0.0,
            "reference_slippage_drag_usdt": net.slippage_drag,
            "trades": net.trades}
