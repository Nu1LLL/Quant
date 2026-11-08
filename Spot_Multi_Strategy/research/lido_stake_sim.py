"""Frozen spot replay of Lido staking acceleration, with strict net-account Gate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import math
from statistics import median

import numpy as np
import pandas as pd

from research.eth_defi_tvl_sim import Daily, _monthly, _numbers, _numeric_pass


INITIAL = 10_000.0
START = date(2021, 1, 1)
END = date(2026, 1, 1)
FEE = .001
SLIP = .0005
LOT = .0001
MC_SEED = 20261129
MC_BLOCK = 21
MC_PATHS = 5000


class ReplayFailure(ValueError):
    pass


@dataclass(frozen=True)
class Simulation:
    days: tuple[Daily, ...]
    trades: tuple[dict, ...]
    fees: float
    slippage_drag: float
    final: float
    entries: int


def weekly_targets(stake: dict[date, float]) -> dict[date, bool]:
    expected = [date(2020, 12, 20) + timedelta(days=7 * i) for i in range(263)]
    if list(stake) != expected:
        raise ReplayFailure("Lido Sunday order or coverage changed")
    if any(not math.isfinite(v) or v <= 0 for v in stake.values()):
        raise ReplayFailure("invalid Sunday WETH units")
    targets = {}
    for ts in pd.date_range("2021-02-14", "2025-12-28", freq="W-SUN"):
        s = ts.date()
        recent = stake[s] / stake[s - timedelta(days=28)] - 1
        prior = stake[s - timedelta(days=28)] / stake[s - timedelta(days=56)] - 1
        targets[s + timedelta(days=4)] = recent > 0 and recent > prior
    return targets


def _capacity(prices: pd.DataFrame, day: date, notional: float) -> None:
    dates = pd.date_range(day - timedelta(days=30), day - timedelta(days=1), tz="UTC")
    history = prices.loc[dates]
    if len(history) != 30:
        raise ReplayFailure(f"{day}: missing 30-day spot liquidity history")
    prior = history["volume"].to_numpy(dtype=float) * history["close"].to_numpy(dtype=float)
    if not np.isfinite(prior).all() or (prior <= 0).any():
        raise ReplayFailure(f"{day}: invalid prior spot volume proxy")
    if notional < 10 or notional > .01 * float(median(prior)):
        raise ReplayFailure(f"{day}: minimum order/capacity gate failed")


def simulate(stake: dict[date, float], prices: pd.DataFrame, terminal: float,
             *, slip: float = SLIP, fee_rate: float = FEE,
             latency_days: int = 0, benchmark: bool = False) -> Simulation:
    if slip < 0 or fee_rate < 0 or latency_days not in (0, 1):
        raise ValueError("unregistered scenario")
    expected = pd.date_range("2020-11-01", "2025-12-31", tz="UTC", freq="D")
    if not prices.index.equals(expected) or tuple(prices.columns) != (
            "open", "high", "low", "close", "volume"):
        raise ReplayFailure("spot grid or schema changed")
    if not math.isfinite(terminal) or terminal <= 0:
        raise ReplayFailure("terminal price invalid")
    targets = weekly_targets(stake)
    cash, qty = INITIAL, 0.0
    fees = slip_drag = 0.0
    entries = 0
    trades: list[dict] = []
    days: list[Daily] = []
    prev_equity = INITIAL

    def buy(day: date, price: float) -> float:
        nonlocal cash, qty, fees, slip_drag, entries
        if qty > 0:
            raise ReplayFailure("entry while already invested")
        execution = price * (1 + slip)
        amount = math.floor(cash / (execution * (1 + fee_rate)) / LOT) * LOT
        notional = amount * execution
        _capacity(prices, day, notional)
        fee = notional * fee_rate
        cash -= notional + fee
        if cash < -1e-7:
            raise ReplayFailure(f"{day}: buy exceeded cash")
        cash = max(cash, 0.0)
        qty = amount
        fees += fee
        slip_drag += amount * price * slip
        entries += 1
        trades.append({"day": str(day), "side": "buy", "quantity": amount,
                       "reference": price, "execution": execution,
                       "notional": notional, "fee": fee})
        return fee

    def sell(day: date, price: float, forced: bool = False) -> float:
        nonlocal cash, qty, fees, slip_drag
        if qty <= 0:
            raise ReplayFailure("exit while flat")
        execution = price * (1 - slip)
        notional = qty * execution
        if notional < 10:
            raise ReplayFailure(f"{day}: exit below minimum")
        fee = notional * fee_rate
        cash += notional - fee
        fees += fee
        slip_drag += qty * price * slip
        trades.append({"day": str(day), "side": "sell", "quantity": qty,
                       "reference": price, "execution": execution,
                       "notional": notional, "fee": fee, "forced": forced})
        qty = 0.0
        return fee

    for ts in pd.date_range(START, END - timedelta(days=1), tz="UTC", freq="D"):
        day = ts.date()
        row = prices.loc[ts]
        opened, high, low, close = (float(row[k]) for k in ("open", "high", "low", "close"))
        if (not all(math.isfinite(v) and v > 0 for v in (opened, high, low, close))
                or high < max(opened, low, close) or low > min(opened, high, close)):
            raise ReplayFailure(f"{day}: invalid spot bar")
        before_trade = cash + qty * opened
        daily_fee = 0.0
        if benchmark and day == START:
            daily_fee = buy(day, opened)
        elif not benchmark:
            base_day = day - timedelta(days=latency_days)
            if base_day in targets and day.weekday() == 3 + latency_days:
                desired = targets[base_day]
                if desired and qty == 0:
                    daily_fee = buy(day, opened)
                elif not desired and qty > 0:
                    daily_fee = sell(day, opened)
        post_trade = cash + qty * opened
        end_equity = cash + qty * close
        high_equity = max(prev_equity, before_trade, post_trade, cash + qty * high,
                          end_equity)
        low_equity = min(prev_equity, before_trade, post_trade, cash + qty * low,
                         end_equity)
        if (not all(math.isfinite(v) and v > 0 for v in
                    (end_equity, high_equity, low_equity)) or cash < 0
                or cash > end_equity * (1 + 1e-9)):
            raise ReplayFailure(f"{day}: invalid equity or exposure")
        days.append(Daily(day, prev_equity, end_equity, high_equity, low_equity,
                          int(qty > 0), daily_fee))
        prev_equity = end_equity
    if qty > 0:
        terminal_before = cash + qty * terminal
        terminal_fee = sell(END, terminal, True)
        last = days[-1]
        days[-1] = Daily(last.day, last.start, cash,
                         max(last.high, terminal_before, cash),
                         min(last.low, terminal_before, cash),
                         last.position, last.fee + terminal_fee)
    else:
        cash = prev_equity
    if (len(days) != 1826 or cash <= 0 or not math.isfinite(cash)
            or not math.isclose(days[-1].end, cash, rel_tol=1e-10, abs_tol=1e-7)):
        raise ReplayFailure("terminal account mismatch")
    return Simulation(tuple(days), tuple(trades), fees, slip_drag, cash, entries)


def _regimes(rows: tuple[Daily, ...], prices: pd.DataFrame) -> dict[str, dict]:
    close = {ts.date(): float(v) for ts, v in prices["close"].items()}
    out = {f"{trend}_{vol}": {"days": 0, "net_dollar_pnl": 0.0}
           for trend in ("bull", "bear") for vol in ("high", "low")}
    for row in rows:
        points = np.array([close[row.day - timedelta(days=k)] for k in range(31, 0, -1)])
        rets = np.diff(np.log(points))
        trend = "bull" if float(rets.sum()) > 0 else "bear"
        vol = "high" if float(np.std(rets, ddof=1) * math.sqrt(365)) >= .80 else "low"
        state = out[f"{trend}_{vol}"]
        state["days"] += 1
        state["net_dollar_pnl"] += row.end - row.start
    return out


def _bootstrap(rows: tuple[Daily, ...]) -> dict:
    n = len(rows)
    factor = np.array([r.end / r.start for r in rows])
    high = np.array([r.high / r.start for r in rows])
    low = np.array([r.low / r.start for r in rows])
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
        if _numeric_pass({"cagr": cagr, "sharpe": sharpe,
                          "max_drawdown": dd, "profit_factor_daily_dollar": pf}):
            passed += 1

    def quantile(i: int) -> dict:
        valid = [p[i] for p in paths if p[i] is not None and math.isfinite(p[i])]
        return {f"p{q:02d}": float(np.percentile(valid, q)) if valid else None
                for q in (5, 50, 95)}

    return {"paths": MC_PATHS, "joint_passes": passed,
            "joint_pass_rate": passed / MC_PATHS, "ruined": ruined,
            "undefined": undefined, "cagr": quantile(0), "sharpe": quantile(1),
            "max_drawdown": quantile(2), "profit_factor": quantile(3)}


def evaluate(net: Simulation, gross: Simulation, double_slip: Simulation,
             latency: Simulation, benchmark: Simulation, prices: pd.DataFrame) -> dict:
    full = _numbers(net.days)
    annual = {str(y): _numbers([r for r in net.days if r.day.year == y])
              for y in range(2021, 2026)}
    months = _monthly(net.days)
    holdout = _numbers([r for r in net.days if r.day.year in (2024, 2025)])
    regimes = _regimes(net.days, prices)
    mc = _bootstrap(net.days)
    annual_entries = {str(y): sum(t["side"] == "buy" and t["day"].startswith(str(y))
                                  for t in net.trades) for y in range(2021, 2026)}
    checks = {
        "numerical_full": _numeric_pass(full),
        "at_least_20_entries": net.entries >= 20,
        "annual_walk_forward": all(annual_entries[y] > 0 and stat["end"] > stat["start"]
                                   and stat["sharpe"] is not None and stat["sharpe"] > 0
                                   for y, stat in annual.items()),
        "historical_2024_2025_holdout": _numeric_pass(holdout),
        "monthly_stability": months["months"] == 60 and months["positive"] >= 36
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
            "annual_entries": annual_entries, "holdout_2024_2025": holdout,
            "monthly": months, "regimes": regimes, "monte_carlo": mc,
            "checks": checks, "all_gate_passed": all(checks.values()),
            "inside_40_to_60_percent_target_band": .40 <= full["cagr"] <= .60,
            "entries": net.entries, "net_final_usdt": net.final,
            "benchmark_final_usdt": benchmark.final,
            "gross_final_usdt": gross.final,
            "double_slippage_final_usdt": double_slip.final,
            "one_day_latency_final_usdt": latency.final,
            "fees_usdt": net.fees, "funding_usdt": 0.0,
            "reference_slippage_drag_usdt": net.slippage_drag,
            "trades": net.trades}
