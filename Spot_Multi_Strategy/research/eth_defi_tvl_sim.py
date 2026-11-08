"""Single preregistered ETH DeFi TVL signal, accounting, and frozen Gate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import math

import numpy as np
import pandas as pd

from research.eth_defi_tvl_source import SourceFailure


START = date(2021, 1, 1)
END = date(2026, 1, 1)
INITIAL = 10000.0
FEE = 0.001
SLIP = 0.0005
MC_PATHS = 5000
MC_BLOCK = 21
MC_SEED = 20261120


@dataclass(frozen=True)
class Daily:
    day: date
    start: float
    end: float
    high: float
    low: float
    position: int
    fee: float


@dataclass(frozen=True)
class Simulation:
    days: tuple[Daily, ...]
    trades: tuple[dict, ...]
    fees: float
    slippage_drag: float
    terminal_cash: float


def _target(day: date, tvl: dict[date, float], closes: dict[date, float]) -> int:
    """Thursday's predeclared target, using only the preceding Sunday."""
    sunday = day - timedelta(days=4)
    old = sunday - timedelta(days=28)
    if sunday.weekday() != 6:
        raise SourceFailure("target decision is not tied to Sunday")
    try:
        x_now = tvl[sunday] / closes[sunday]
        x_old = tvl[old] / closes[old]
    except KeyError as exc:
        raise SourceFailure(f"missing completed signal day {exc}") from exc
    if not (math.isfinite(x_now) and math.isfinite(x_old) and x_now > 0 and x_old > 0):
        raise SourceFailure("invalid Sunday TVL/ETH ratio")
    return int(x_now > x_old)


def simulate(tvl: dict[date, float], prices: pd.DataFrame, terminal_open: float,
             *, slippage: float = SLIP, latency_days: int = 0,
             fee_rate: float = FEE) -> Simulation:
    if latency_days not in (0, 1) or slippage < 0 or fee_rate < 0:
        raise ValueError("invalid frozen scenario")
    if not math.isfinite(terminal_open) or terminal_open <= 0:
        raise SourceFailure("terminal open invalid")
    expected = pd.date_range(date(2020, 11, 1), date(2025, 12, 31), tz="UTC", freq="D")
    if not prices.index.equals(expected):
        raise SourceFailure("spot daily grid invalid in simulator")
    closes = {ts.date(): float(value) for ts, value in prices["close"].items()}
    cash, qty = INITIAL, 0.0
    fees, slip_drag = 0.0, 0.0
    days: list[Daily] = []
    trades: list[dict] = []
    prev_equity = INITIAL
    for ts in pd.date_range(START, END - timedelta(days=1), tz="UTC", freq="D"):
        day = ts.date()
        row = prices.loc[ts]
        daily_fee = 0.0
        trade_day = (day.weekday() == (3 + latency_days))
        if trade_day and day - timedelta(days=latency_days) >= START:
            base_thursday = day - timedelta(days=latency_days)
            target = _target(base_thursday, tvl, closes)
            if target != int(qty > 0):
                ref = float(row["open"])
                if qty > 0:
                    execution = ref * (1 - slippage)
                    notional = qty * execution
                    fee = notional * fee_rate
                    cash += notional - fee
                    slip_drag += qty * ref * slippage
                    trades.append({"day": str(day), "side": "sell", "reference": ref,
                                   "execution": execution, "quantity": qty,
                                   "notional": notional, "fee": fee})
                    qty = 0.0
                else:
                    execution = ref * (1 + slippage)
                    qty = cash / (execution * (1 + fee_rate))
                    notional = qty * execution
                    fee = notional * fee_rate
                    cash -= notional + fee
                    if abs(cash) < 1e-8:
                        cash = 0.0
                    slip_drag += qty * ref * slippage
                    trades.append({"day": str(day), "side": "buy", "reference": ref,
                                   "execution": execution, "quantity": qty,
                                   "notional": notional, "fee": fee})
                    if notional < 10:
                        raise SourceFailure(f"{day}: below 10 USDT order proxy")
                fees += fee
                daily_fee += fee
        end_equity = cash + qty * float(row["close"])
        high_equity = max(prev_equity, cash + qty * float(row["high"]), end_equity)
        low_equity = min(prev_equity, cash + qty * float(row["low"]), end_equity)
        if not all(math.isfinite(x) and x > 0 for x in
                   (prev_equity, end_equity, high_equity, low_equity)):
            raise SourceFailure(f"{day}: invalid/ruined account equity")
        days.append(Daily(day, prev_equity, end_equity, high_equity, low_equity,
                          int(qty > 0), daily_fee))
        prev_equity = end_equity
    if qty > 0:
        execution = terminal_open * (1 - slippage)
        notional = qty * execution
        fee = notional * fee_rate
        cash += notional - fee
        slip_drag += qty * terminal_open * slippage
        fees += fee
        trades.append({"day": str(END), "side": "sell", "reference": terminal_open,
                       "execution": execution, "quantity": qty,
                       "notional": notional, "fee": fee, "forced_terminal": True})
        last = days[-1]
        days[-1] = Daily(last.day, last.start, cash, max(last.high, cash),
                         min(last.low, cash), last.position, last.fee + fee)
    else:
        cash = prev_equity
    if cash <= 0 or not math.isfinite(cash):
        raise SourceFailure("terminal account nonpositive")
    if len(days) != 1826 or abs(days[-1].end - cash) > 1e-8:
        raise SourceFailure("daily factor/terminal cash reconciliation failed")
    return Simulation(tuple(days), tuple(trades), fees, slip_drag, cash)


def _numbers(rows: tuple[Daily, ...] | list[Daily]) -> dict:
    n = len(rows)
    if not n:
        raise SourceFailure("empty metrics interval")
    starts = np.array([x.start for x in rows], dtype=float)
    ends = np.array([x.end for x in rows], dtype=float)
    returns = ends / starts - 1
    dollars = ends - starts
    sigma = float(np.std(returns, ddof=1)) if n > 1 else 0.0
    sharpe = float(np.mean(returns) / sigma * math.sqrt(365)) if sigma > 0 else None
    losses = float(-dollars[dollars < 0].sum())
    pf = float(dollars[dollars > 0].sum() / losses) if losses > 0 else None
    peak = float(starts[0])
    dd = 0.0
    for item in rows:
        peak = max(peak, item.high)
        dd = max(dd, 1 - item.low / peak)
    cagr = float((ends[-1] / starts[0]) ** (365.25 / n) - 1)
    return {"days": n, "start": float(starts[0]), "end": float(ends[-1]),
            "cagr": cagr, "sharpe": sharpe, "max_drawdown": dd,
            "profit_factor_daily_dollar": pf, "total_pnl": float(dollars.sum()),
            "positive_days": int((dollars > 0).sum())}


def _numeric_pass(stats: dict) -> bool:
    return (stats["cagr"] >= .40 and stats["sharpe"] is not None and
            stats["sharpe"] > 1.5 and stats["max_drawdown"] <= .20 and
            stats["profit_factor_daily_dollar"] is not None and
            stats["profit_factor_daily_dollar"] > 1.3)


def _annual(rows: tuple[Daily, ...]) -> dict[str, dict]:
    return {str(year): _numbers([r for r in rows if r.day.year == year])
            for year in range(2021, 2026)}


def _monthly(rows: tuple[Daily, ...]) -> dict:
    groups: dict[str, list[Daily]] = {}
    for row in rows:
        groups.setdefault(row.day.strftime("%Y-%m"), []).append(row)
    out = {month: group[-1].end / group[0].start - 1 for month, group in groups.items()}
    longest = streak = 0
    for value in out.values():
        streak = streak + 1 if value <= 0 else 0
        longest = max(longest, streak)
    return {"returns": out, "positive": sum(v > 0 for v in out.values()),
            "months": len(out), "longest_nonpositive_streak": longest,
            "worst": min(out.values())}


def _regimes(rows: tuple[Daily, ...], prices: pd.DataFrame) -> dict[str, dict]:
    closes = {ts.date(): float(x) for ts, x in prices["close"].items()}
    buckets: dict[str, list[Daily]] = {f"{trend}_{vol}": []
                                      for trend in ("bull", "bear")
                                      for vol in ("high", "low")}
    for row in rows:
        points = np.array([closes[row.day - timedelta(days=k)] for k in range(31, 0, -1)])
        rets = np.diff(np.log(points))
        trend = "bull" if rets.sum() > 0 else "bear"
        vol = "high" if np.std(rets, ddof=1) * math.sqrt(365) >= .80 else "low"
        buckets[f"{trend}_{vol}"].append(row)
    return {name: {"days": len(items), "net_dollar_pnl": sum(x.end - x.start for x in items)}
            for name, items in buckets.items()}


def _bootstrap(rows: tuple[Daily, ...]) -> dict:
    n = len(rows)
    factor = np.array([r.end / r.start for r in rows])
    high = np.array([r.high / r.start for r in rows])
    low = np.array([r.low / r.start for r in rows])
    rng = np.random.Generator(np.random.PCG64(MC_SEED))
    all_metrics: list[tuple[float, float | None, float, float | None]] = []
    passed = ruined = undefined = 0
    for _ in range(MC_PATHS):
        starts = rng.integers(0, n, size=math.ceil(n / MC_BLOCK))
        indices = ((starts[:, None] + np.arange(MC_BLOCK)) % n).ravel()[:n]
        f = factor[indices]
        equity = INITIAL * np.cumprod(f)
        begin = np.r_[INITIAL, equity[:-1]]
        if not np.isfinite(equity).all() or (equity <= 0).any():
            ruined += 1
            continue
        highs = begin * high[indices]
        lows = begin * low[indices]
        peak = np.maximum.accumulate(np.r_[INITIAL, highs])[1:]
        dd = float(np.max(1 - lows / peak))
        ret = f - 1
        sigma = float(np.std(ret, ddof=1))
        sharpe = float(np.mean(ret) / sigma * math.sqrt(365)) if sigma > 0 else None
        pnl = equity - begin
        neg = float(-pnl[pnl < 0].sum())
        pf = float(pnl[pnl > 0].sum() / neg) if neg > 0 else None
        cagr = float((equity[-1] / INITIAL) ** (365.25 / n) - 1)
        all_metrics.append((cagr, sharpe, dd, pf))
        if sharpe is None or pf is None:
            undefined += 1
        if _numeric_pass({"cagr": cagr, "sharpe": sharpe,
                          "max_drawdown": dd, "profit_factor_daily_dollar": pf}):
            passed += 1
    def quantile(idx: int) -> dict:
        values = [m[idx] for m in all_metrics if m[idx] is not None and math.isfinite(m[idx])]
        return {"p05": float(np.percentile(values, 5)) if values else None,
                "p50": float(np.percentile(values, 50)) if values else None,
                "p95": float(np.percentile(values, 95)) if values else None}
    return {"paths": MC_PATHS, "joint_passes": passed, "joint_pass_rate": passed / MC_PATHS,
            "ruined": ruined, "undefined": undefined,
            "cagr": quantile(0), "sharpe": quantile(1),
            "max_drawdown": quantile(2), "profit_factor": quantile(3)}


def evaluate(sim: Simulation, gross: Simulation, stress_slip: Simulation,
             stress_latency: Simulation, prices: pd.DataFrame) -> dict:
    full = _numbers(sim.days)
    years = _annual(sim.days)
    holdout = _numbers([r for r in sim.days if r.day.year in (2024, 2025)])
    months = _monthly(sim.days)
    regimes = _regimes(sim.days, prices)
    mc = _bootstrap(sim.days)
    buys = sum(x["side"] == "buy" for x in sim.trades)
    checks = {
        "numerical_full": _numeric_pass(full),
        "min_20_buys": buys >= 20,
        "annual_walk_forward": all(v["end"] > v["start"] and v["sharpe"] is not None
                                   and v["sharpe"] > 0 for v in years.values()),
        "historical_2024_2025_holdout": _numeric_pass(holdout),
        "monthly_stability": months["positive"] >= 36 and months["longest_nonpositive_streak"] <= 3,
        "four_regimes": all(v["days"] >= 90 and v["net_dollar_pnl"] > 0 for v in regimes.values()),
        "block_monte_carlo": mc["joint_pass_rate"] >= .80,
        "double_slippage": stress_slip.terminal_cash > INITIAL,
        "one_day_latency": stress_latency.terminal_cash > INITIAL,
        "independent_executable_oos": False,
    }
    return {"full": full, "gross_no_cost": _numbers(gross.days),
            "holdout_2024_2025": holdout, "annual": years, "months": months,
            "regimes": regimes, "monte_carlo": mc, "checks": checks,
            "all_gate_passed": all(checks.values()), "buys": buys,
            "fills": len(sim.trades), "fees_usdt": sim.fees,
            "slippage_reference_drag_usdt": sim.slippage_drag,
            "net_final_usdt": sim.terminal_cash,
            "gross_final_usdt": gross.terminal_cash,
            "double_slippage_final_usdt": stress_slip.terminal_cash,
            "one_day_latency_final_usdt": stress_latency.terminal_cash,
            "trades": sim.trades}
