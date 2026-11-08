"""Frozen corrected Coinbase-premium signal and spot BTC accounting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import math

import numpy as np
import pandas as pd

from research.coinbase_premium_source import SourceFailure
from research.eth_defi_tvl_sim import Daily, _monthly, _numbers, _numeric_pass, _regimes


START = date(2021, 10, 1)
END = date(2026, 10, 1)
INITIAL = 10000.0
FEE = .001
SLIP = .0005
LOT = .00001
MC_PATHS = 5000
MC_BLOCK = 21
MC_SEED = 20261121


@dataclass(frozen=True)
class Simulation:
    days: tuple[Daily, ...]
    fills: tuple[dict, ...]
    fees: float
    slippage_reference_drag: float
    terminal_cash: float


def premium_series(coinbase_btc: pd.DataFrame, coinbase_usdt: pd.DataFrame,
                   binance: pd.DataFrame) -> dict[date, float]:
    expected = pd.date_range(date(2021, 9, 1), date(2026, 9, 30), tz="UTC", freq="D")
    if not coinbase_btc.index.equals(expected) or not coinbase_usdt.index.equals(expected):
        raise SourceFailure("Coinbase signal day grid invalid")
    if not binance.index.equals(pd.date_range(date(2021, 8, 1),
                                               date(2026, 9, 30), tz="UTC", freq="D")):
        raise SourceFailure("Binance signal/execution day grid invalid")
    result: dict[date, float] = {}
    for ts in expected:
        d = ts.date()
        cb_btc = float(coinbase_btc.at[ts, "close"])
        cb_usdt = float(coinbase_usdt.at[ts, "close"])
        bn_btc = float(binance.at[ts, "close"])
        if not all(math.isfinite(x) and x > 0 for x in (cb_btc, cb_usdt, bn_btc)):
            raise SourceFailure(f"invalid premium components on {d}")
        result[d] = math.log(cb_btc / (bn_btc * cb_usdt))
    return result


def weekly_target(tuesday: date, premium: dict[date, float]) -> int:
    if tuesday.weekday() != 1:
        raise ValueError("signal acts on Tuesday only")
    sunday = tuesday - timedelta(days=2)
    monday = sunday - timedelta(days=6)
    try:
        values = [premium[monday + timedelta(days=i)] for i in range(7)]
    except KeyError as exc:
        raise SourceFailure(f"missing completed premium day {exc}") from exc
    if not all(math.isfinite(x) for x in values):
        raise SourceFailure("nonfinite prior seven-day premium")
    return int(sum(values) / 7 > 0)


def _cash_buy(cash: float, open_price: float, slippage: float,
              fee_rate: float) -> tuple[float, float, float, float]:
    execution = open_price * (1 + slippage)
    raw_qty = cash / (execution * (1 + fee_rate))
    qty = math.floor(raw_qty / LOT + 1e-9) * LOT
    notional = qty * execution
    fee = notional * fee_rate
    residual = cash - notional - fee
    if qty <= 0 or notional < 10 or residual < -1e-7:
        raise SourceFailure("buy failed fixed lot/notional/cash checks")
    return qty, notional, fee, max(residual, 0.0)


def simulate(premium: dict[date, float], binance: pd.DataFrame, terminal_open: float,
             *, slippage: float = SLIP, latency_days: int = 0,
             fee_rate: float = FEE) -> Simulation:
    if slippage < 0 or fee_rate < 0 or latency_days not in (0, 1):
        raise ValueError("invalid frozen scenario")
    if not math.isfinite(terminal_open) or terminal_open <= 0:
        raise SourceFailure("invalid terminal Binance open")
    expected = pd.date_range(date(2021, 8, 1), date(2026, 9, 30), tz="UTC", freq="D")
    if not binance.index.equals(expected):
        raise SourceFailure("execution price day grid invalid")
    cash, qty = INITIAL, 0.0
    total_fees, total_slip = 0.0, 0.0
    fills: list[dict] = []
    days: list[Daily] = []
    prior_equity = INITIAL
    for ts in pd.date_range(START, END - timedelta(days=1), tz="UTC", freq="D"):
        day = ts.date()
        row = binance.loc[ts]
        day_fee = 0.0
        if day.weekday() == 1 + latency_days and day - timedelta(days=latency_days) >= START:
            tuesday = day - timedelta(days=latency_days)
            target = weekly_target(tuesday, premium)
            if target != int(qty > 0):
                reference = float(row["open"])
                if target == 1:
                    new_qty, notional, fee, residual = _cash_buy(cash, reference,
                                                                 slippage, fee_rate)
                    prior_volume = binance.loc[pd.date_range(day - timedelta(days=30),
                                                             day - timedelta(days=1),
                                                             tz="UTC", freq="D"), "quote_volume"]
                    if len(prior_volume) != 30 or not np.isfinite(prior_volume.to_numpy()).all():
                        raise SourceFailure("missing prior 30-day capacity history")
                    median_volume = float(prior_volume.median())
                    if median_volume <= 0 or notional > .01 * median_volume:
                        raise SourceFailure(f"{day}: fixed 1% capacity Gate failed")
                    cash = residual
                    qty = new_qty
                    total_slip += qty * reference * slippage
                    side = "buy"
                    execution = reference * (1 + slippage)
                else:
                    execution = reference * (1 - slippage)
                    notional = qty * execution
                    if notional < 10:
                        raise SourceFailure(f"{day}: sell below 10 USDT proxy")
                    fee = notional * fee_rate
                    cash += notional - fee
                    total_slip += qty * reference * slippage
                    side = "sell"
                    new_qty = qty
                    qty = 0.0
                total_fees += fee
                day_fee += fee
                fills.append({"day": str(day), "side": side, "reference": reference,
                              "execution": execution, "quantity": new_qty,
                              "notional": notional, "fee": fee})
        end = cash + qty * float(row["close"])
        high = max(prior_equity, cash + qty * float(row["high"]), end)
        low = min(prior_equity, cash + qty * float(row["low"]), end)
        if not all(math.isfinite(x) and x > 0 for x in (prior_equity, high, low, end)):
            raise SourceFailure(f"{day}: ruined or nonfinite spot account")
        days.append(Daily(day, prior_equity, end, high, low, int(qty > 0), day_fee))
        prior_equity = end
    if qty:
        execution = terminal_open * (1 - slippage)
        notional = qty * execution
        if notional < 10:
            raise SourceFailure("terminal sell below 10 USDT proxy")
        fee = notional * fee_rate
        cash += notional - fee
        total_fees += fee
        total_slip += qty * terminal_open * slippage
        fills.append({"day": str(END), "side": "sell", "reference": terminal_open,
                      "execution": execution, "quantity": qty,
                      "notional": notional, "fee": fee, "forced_terminal": True})
        old = days[-1]
        days[-1] = Daily(old.day, old.start, cash, max(old.high, cash),
                         min(old.low, cash), old.position, old.fee + fee)
    else:
        cash = prior_equity
    if len(days) != 1826 or cash <= 0 or abs(days[-1].end - cash) > 1e-7:
        raise SourceFailure("terminal cash/daily-factor reconciliation failed")
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
        daily = factors[idx]
        equity = INITIAL * np.cumprod(daily)
        begins = np.r_[INITIAL, equity[:-1]]
        if not np.isfinite(equity).all() or (equity <= 0).any():
            ruined += 1
            continue
        peaks = np.maximum.accumulate(np.r_[INITIAL, begins * high[idx]])[1:]
        dd = float(np.max(1 - begins * low[idx] / peaks))
        rets = daily - 1
        sd = float(np.std(rets, ddof=1))
        sharpe = float(np.mean(rets) / sd * math.sqrt(365)) if sd > 0 else None
        dollars = equity - begins
        losses = float(-dollars[dollars < 0].sum())
        pf = float(dollars[dollars > 0].sum() / losses) if losses > 0 else None
        cagr = float((equity[-1] / INITIAL) ** (365.25 / n) - 1)
        metrics.append((cagr, sharpe, dd, pf))
        if sharpe is None or pf is None:
            undefined += 1
        if _numeric_pass({"cagr": cagr, "sharpe": sharpe,
                          "max_drawdown": dd, "profit_factor_daily_dollar": pf}):
            passes += 1
    def quantile(field: int) -> dict:
        values = [x[field] for x in metrics if x[field] is not None and math.isfinite(x[field])]
        return {f"p{q:02d}": float(np.percentile(values, q)) if values else None
                for q in (5, 50, 95)}
    return {"paths": MC_PATHS, "joint_passes": passes, "joint_pass_rate": passes / MC_PATHS,
            "ruined": ruined, "undefined": undefined,
            "cagr": quantile(0), "sharpe": quantile(1),
            "max_drawdown": quantile(2), "profit_factor": quantile(3)}


def evaluate(primary: Simulation, gross: Simulation, double_slip: Simulation,
             extra_day: Simulation, binance: pd.DataFrame) -> dict:
    full = _numbers(primary.days)
    annual = {str(year): _numbers([x for x in primary.days if x.day.year == year])
              for year in range(2021, 2027)}
    holdout = _numbers([x for x in primary.days if x.day >= date(2024, 1, 1)])
    monthly = _monthly(primary.days)
    states = _regimes(primary.days, binance)
    mc = _bootstrap(primary.days)
    buys = sum(x["side"] == "buy" for x in primary.fills)
    fills_per_year = {str(y): sum(x["day"][:4] == str(y) and
                                  not x.get("forced_terminal", False) for x in primary.fills)
                      for y in range(2021, 2027)}
    # The terminal boundary sale is excluded from the annual trade-count Gate.
    checks = {
        "numerical_full": _numeric_pass(full),
        "min_20_buys": buys >= 20,
        "annual_walk_forward": all(annual[y]["end"] > annual[y]["start"] and
                                   annual[y]["sharpe"] is not None and annual[y]["sharpe"] > 0 and
                                   fills_per_year[y] > 0 for y in annual),
        "historical_2024_2026_holdout": _numeric_pass(holdout),
        "monthly_stability": monthly["months"] == 60 and monthly["positive"] >= 36 and
                             monthly["longest_nonpositive_streak"] <= 3,
        "four_regimes": all(x["days"] >= 90 and x["net_dollar_pnl"] > 0
                            for x in states.values()),
        "block_monte_carlo": mc["joint_pass_rate"] >= .80,
        "double_slippage": double_slip.terminal_cash > INITIAL,
        "one_day_latency": extra_day.terminal_cash > INITIAL,
        "independent_executable_oos": False,
    }
    return {"full": full, "gross_no_cost": _numbers(gross.days),
            "holdout_2024_2026": holdout, "annual": annual,
            "monthly": monthly, "regimes": states, "monte_carlo": mc,
            "checks": checks, "all_gate_passed": all(checks.values()),
            "buys": buys, "fills_per_year": fills_per_year,
            "fills": primary.fills, "fees_usdt": primary.fees,
            "slippage_reference_drag_usdt": primary.slippage_reference_drag,
            "net_final_usdt": primary.terminal_cash,
            "gross_final_usdt": gross.terminal_cash,
            "double_slippage_final_usdt": double_slip.terminal_cash,
            "one_day_latency_final_usdt": extra_day.terminal_cash}
