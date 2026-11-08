"""Frozen three-coin spot regime replay from lagged stablecoin circulation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import math
from statistics import median

import numpy as np
import pandas as pd

from research.eth_defi_tvl_sim import Daily, _monthly, _numbers, _numeric_pass
from research.stablecoin_liquidity_source import FIRST, LAST, SYMBOLS, TERMINAL


INITIAL = 10_000.0
FEE = .0005
SLIP = .001
LOT = .001
MC_PATHS = 5000
MC_BLOCK = 21
MC_SEED = 20261127
START = date(2021, 1, 1)
END = date(2026, 1, 1)


class ReplayFailure(ValueError):
    pass


@dataclass(frozen=True)
class Simulation:
    days: tuple[Daily, ...]
    trades: tuple[dict, ...]
    fees: float
    slippage_drag: float
    final: float
    risk_on_entries: int


def weekly_targets(stablecoins: pd.DataFrame) -> dict[date, bool]:
    expected = pd.date_range(FIRST, LAST, tz="UTC", freq="D")
    if not stablecoins.index.equals(expected) or tuple(stablecoins.columns) != ("USDT", "USDC"):
        raise ReplayFailure("stablecoin daily grid or IDs changed")
    values = stablecoins.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ReplayFailure("invalid stablecoin circulating amount")
    total = stablecoins["USDT"] + stablecoins["USDC"]
    out = {}
    for ts in pd.date_range("2021-01-07", "2025-12-25", freq="W-THU"):
        day = ts.date()
        sunday = day - timedelta(days=4)
        out[day] = bool(total[pd.Timestamp(sunday, tz="UTC")] >
                        total[pd.Timestamp(sunday - timedelta(days=28), tz="UTC")])
    if len(out) != 260:
        raise ReplayFailure("scheduled weekly decision count changed")
    return out


def _validate_prices(prices: dict[str, pd.DataFrame], terminal: dict[str, float]) -> None:
    expected = pd.date_range(FIRST, LAST, tz="UTC", freq="D")
    if set(prices) != set(SYMBOLS) or set(terminal) != set(SYMBOLS):
        raise ReplayFailure("spot asset universe changed")
    for symbol in SYMBOLS:
        frame = prices[symbol]
        if not frame.index.equals(expected) or tuple(frame.columns) != (
                "open", "high", "low", "close", "quote_volume"):
            raise ReplayFailure(f"{symbol}: spot grid or columns changed")
        arr = frame.to_numpy(dtype=float)
        if (not np.isfinite(arr).all() or (arr[:, :4] <= 0).any()
                or (arr[:, 4] < 0).any()
                or (arr[:, 1] < np.maximum.reduce([arr[:, 0], arr[:, 2], arr[:, 3]])).any()
                or (arr[:, 2] > np.minimum.reduce([arr[:, 0], arr[:, 1], arr[:, 3]])).any()):
            raise ReplayFailure(f"{symbol}: invalid spot OHLCV")
        if not math.isfinite(terminal[symbol]) or terminal[symbol] <= 0:
            raise ReplayFailure(f"{symbol}: invalid terminal open")


def _capacity(prices: dict[str, pd.DataFrame], symbol: str, day: date,
              notional: float) -> None:
    dates = pd.date_range(day - timedelta(days=30), day - timedelta(days=1),
                          tz="UTC", freq="D")
    values = prices[symbol].loc[dates, "quote_volume"]
    if len(values) != 30 or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ReplayFailure(f"{symbol} {day}: missing 30-day capacity history")
    cap = .01 * float(median(values))
    if cap <= 0 or notional > cap:
        raise ReplayFailure(f"{symbol} {day}: 1%-of-median quote-volume cap failed")


def simulate(stablecoins: pd.DataFrame, prices: dict[str, pd.DataFrame],
             terminal: dict[str, float], *, slip: float = SLIP,
             fee_rate: float = FEE, latency_days: int = 0,
             benchmark: bool = False) -> Simulation:
    if slip < 0 or fee_rate < 0 or latency_days not in (0, 1):
        raise ValueError("invalid frozen scenario")
    targets = weekly_targets(stablecoins)
    _validate_prices(prices, terminal)
    cash = INITIAL
    qty = {symbol: 0.0 for symbol in SYMBOLS}
    fees = slippage_drag = 0.0
    trades: list[dict] = []
    days: list[Daily] = []
    prev_equity = INITIAL
    risk_on_entries = 0

    def equity(mark: dict[str, float]) -> float:
        return cash + sum(qty[s] * mark[s] for s in SYMBOLS)

    def sell_all(day: date, reference: dict[str, float], *, terminal_exit: bool = False) -> float:
        nonlocal cash, fees, slippage_drag
        paid = 0.0
        for symbol in SYMBOLS:
            amount = qty[symbol]
            if amount <= 0:
                continue
            ref = reference[symbol]
            execution = ref * (1 - slip)
            notional = amount * execution
            if notional < 10:
                raise ReplayFailure(f"{day} {symbol}: exit below 10 USDT")
            fee = notional * fee_rate
            cash += notional - fee
            fees += fee
            paid += fee
            slippage_drag += amount * ref * slip
            trades.append({"day": str(day), "side": "sell", "symbol": symbol,
                           "quantity": amount, "reference": ref, "execution": execution,
                           "notional": notional, "fee": fee,
                           "terminal": terminal_exit})
            qty[symbol] = 0.0
        return paid

    def buy_all(day: date, reference: dict[str, float]) -> float:
        nonlocal cash, fees, slippage_drag, risk_on_entries
        if any(qty[s] > 0 for s in SYMBOLS):
            raise ReplayFailure("entry while already invested")
        budget = cash / 3.0
        orders = []
        for symbol in SYMBOLS:
            ref = reference[symbol]
            execution = ref * (1 + slip)
            amount = math.floor(budget / (execution * (1 + fee_rate)) / LOT) * LOT
            notional = amount * execution
            if notional < 10:
                raise ReplayFailure(f"{day} {symbol}: entry below 10 USDT")
            _capacity(prices, symbol, day, notional)
            orders.append((symbol, amount, ref, execution, notional))
        paid = 0.0
        for symbol, amount, ref, execution, notional in orders:
            fee = notional * fee_rate
            cash -= notional + fee
            fees += fee
            paid += fee
            slippage_drag += amount * ref * slip
            qty[symbol] = amount
            trades.append({"day": str(day), "side": "buy", "symbol": symbol,
                           "quantity": amount, "reference": ref, "execution": execution,
                           "notional": notional, "fee": fee})
        if cash < -1e-7:
            raise ReplayFailure(f"{day}: spot cash debit exceeds account")
        cash = max(0.0, cash)
        risk_on_entries += 1
        return paid

    for ts in pd.date_range(START, LAST, tz="UTC", freq="D"):
        day = ts.date()
        opened = {s: float(prices[s].at[ts, "open"]) for s in SYMBOLS}
        highs = {s: float(prices[s].at[ts, "high"]) for s in SYMBOLS}
        lows = {s: float(prices[s].at[ts, "low"]) for s in SYMBOLS}
        closes = {s: float(prices[s].at[ts, "close"]) for s in SYMBOLS}
        before_trade_open = equity(opened)
        daily_fee = 0.0
        if benchmark:
            if day == START:
                daily_fee += buy_all(day, opened)
        else:
            decision_day = day - timedelta(days=latency_days)
            if decision_day in targets and day.weekday() == (3 + latency_days):
                desired = targets[decision_day]
                invested = all(qty[s] > 0 for s in SYMBOLS)
                if any(qty[s] > 0 for s in SYMBOLS) and not invested:
                    raise ReplayFailure("partial basket position")
                if desired and not invested:
                    daily_fee += buy_all(day, opened)
                elif not desired and invested:
                    daily_fee += sell_all(day, opened)
        post_trade_open = equity(opened)
        end_equity = equity(closes)
        high_equity = max(prev_equity, before_trade_open, post_trade_open,
                          equity(highs), end_equity)
        low_equity = min(prev_equity, before_trade_open, post_trade_open,
                         equity(lows), end_equity)
        if (not all(math.isfinite(x) and x > 0 for x in
                    (prev_equity, end_equity, high_equity, low_equity))):
            raise ReplayFailure(f"{day}: nonfinite or ruined spot account")
        days.append(Daily(day, prev_equity, end_equity, high_equity, low_equity,
                          int(any(qty[s] > 0 for s in SYMBOLS)), daily_fee))
        prev_equity = end_equity
    if any(qty[s] > 0 for s in SYMBOLS):
        terminal_before = equity(terminal)
        terminal_fee = sell_all(TERMINAL, terminal, terminal_exit=True)
        last = days[-1]
        days[-1] = Daily(last.day, last.start, cash,
                         max(last.high, terminal_before, cash),
                         min(last.low, terminal_before, cash), last.position,
                         last.fee + terminal_fee)
    else:
        cash = prev_equity
    if (len(days) != 1826 or not math.isfinite(cash) or cash <= 0
            or not math.isclose(days[-1].end, cash, rel_tol=1e-10, abs_tol=1e-7)):
        raise ReplayFailure("five-year terminal spot account reconciliation failed")
    return Simulation(tuple(days), tuple(trades), fees, slippage_drag, cash,
                      risk_on_entries)


def _regimes(rows: tuple[Daily, ...], prices: dict[str, pd.DataFrame]) -> dict[str, dict]:
    closes = {s: {ts.date(): float(x) for ts, x in prices[s]["close"].items()}
              for s in SYMBOLS}
    buckets = {f"{trend}_{vol}": {"days": 0, "net_dollar_pnl": 0.0}
               for trend in ("bull", "bear") for vol in ("high", "low")}
    for row in rows:
        mean_rets = []
        for age in range(31, 1, -1):
            earlier = row.day - timedelta(days=age)
            later = earlier + timedelta(days=1)
            mean_rets.append(sum(math.log(closes[s][later] / closes[s][earlier])
                                 for s in SYMBOLS) / 3.0)
        trend = "bull" if sum(mean_rets) > 0 else "bear"
        vol = "high" if float(np.std(mean_rets, ddof=1)) * math.sqrt(365) >= .80 else "low"
        bucket = buckets[f"{trend}_{vol}"]
        bucket["days"] += 1
        bucket["net_dollar_pnl"] += row.end - row.start
    return buckets


def _bootstrap(rows: tuple[Daily, ...]) -> dict:
    n = len(rows)
    factors = np.array([r.end / r.start for r in rows], dtype=float)
    high = np.array([r.high / r.start for r in rows], dtype=float)
    low = np.array([r.low / r.start for r in rows], dtype=float)
    rng = np.random.Generator(np.random.PCG64(MC_SEED))
    paths = []
    passed = ruined = undefined = 0
    for _ in range(MC_PATHS):
        starts = rng.integers(0, n, size=math.ceil(n / MC_BLOCK))
        index = ((starts[:, None] + np.arange(MC_BLOCK)) % n).ravel()[:n]
        equity = INITIAL * np.cumprod(factors[index])
        begins = np.r_[INITIAL, equity[:-1]]
        if not np.isfinite(equity).all() or (equity <= 0).any():
            ruined += 1
            continue
        peak = np.maximum.accumulate(np.r_[INITIAL, begins * high[index]])[1:]
        drawdown = float(np.max(1 - begins * low[index] / peak))
        returns = factors[index] - 1
        sigma = float(np.std(returns, ddof=1))
        sharpe = float(np.mean(returns) / sigma * math.sqrt(365)) if sigma > 0 else None
        dollars = equity - begins
        losses = float(-dollars[dollars < 0].sum())
        pf = float(dollars[dollars > 0].sum() / losses) if losses > 0 else None
        cagr = float((equity[-1] / INITIAL) ** (365.25 / n) - 1)
        paths.append((cagr, sharpe, drawdown, pf))
        if sharpe is None or pf is None:
            undefined += 1
        if _numeric_pass({"cagr": cagr, "sharpe": sharpe,
                          "max_drawdown": drawdown, "profit_factor_daily_dollar": pf}):
            passed += 1

    def quantile(field: int) -> dict:
        values = [x[field] for x in paths if x[field] is not None and math.isfinite(x[field])]
        return {f"p{q:02d}": float(np.percentile(values, q)) if values else None
                for q in (5, 50, 95)}

    return {"paths": MC_PATHS, "joint_passes": passed,
            "joint_pass_rate": passed / MC_PATHS, "ruined": ruined,
            "undefined": undefined, "cagr": quantile(0), "sharpe": quantile(1),
            "max_drawdown": quantile(2), "profit_factor": quantile(3)}


def evaluate(net: Simulation, gross: Simulation, stress_slip: Simulation,
             stress_latency: Simulation, benchmark: Simulation,
             prices: dict[str, pd.DataFrame]) -> dict:
    full = _numbers(net.days)
    base = _numbers(benchmark.days)
    annual = {str(y): _numbers([row for row in net.days if row.day.year == y])
              for y in range(2021, 2026)}
    held_years = {str(y): any(row.position for row in net.days if row.day.year == y)
                  for y in range(2021, 2026)}
    holdout = _numbers([row for row in net.days if row.day.year in (2024, 2025)])
    months = _monthly(net.days)
    regimes = _regimes(net.days, prices)
    mc = _bootstrap(net.days)
    benchmark_better = (full["cagr"] > base["cagr"] and
                        full["sharpe"] is not None and base["sharpe"] is not None and
                        full["sharpe"] > base["sharpe"])
    checks = {
        "numerical_full": _numeric_pass(full),
        "min_10_risk_on_entries": net.risk_on_entries >= 10,
        "annual_walk_forward": all(held_years[y] and item["end"] > item["start"] and
                                   item["sharpe"] is not None and item["sharpe"] > 0
                                   for y, item in annual.items()),
        "historical_2024_2025_holdout": _numeric_pass(holdout),
        "monthly_stability": months["months"] == 60 and months["positive"] >= 36 and
                             months["longest_nonpositive_streak"] <= 3,
        "four_regimes": all(item["days"] >= 90 and item["net_dollar_pnl"] > 0
                            for item in regimes.values()),
        "block_monte_carlo": mc["joint_pass_rate"] >= .80,
        "double_slippage": stress_slip.final > INITIAL,
        "one_day_latency": stress_latency.final > INITIAL,
        "beats_buy_and_hold_cagr_and_sharpe": benchmark_better,
        "independent_executable_oos": False,
    }
    return {"full": full, "buy_and_hold": base,
            "gross_no_cost": _numbers(gross.days), "holdout_2024_2025": holdout,
            "annual": annual, "monthly": months, "regimes": regimes,
            "monte_carlo": mc, "checks": checks,
            "all_gate_passed": all(checks.values()),
            "inside_40_to_60_percent_target_band": .40 <= full["cagr"] <= .60,
            "risk_on_entries": net.risk_on_entries,
            "net_final_usdt": net.final, "benchmark_final_usdt": benchmark.final,
            "gross_final_usdt": gross.final,
            "double_slippage_final_usdt": stress_slip.final,
            "one_day_latency_final_usdt": stress_latency.final,
            "fees_usdt": net.fees, "funding_usdt": 0.0,
            "reference_slippage_drag_usdt": net.slippage_drag,
            "trades": net.trades}
