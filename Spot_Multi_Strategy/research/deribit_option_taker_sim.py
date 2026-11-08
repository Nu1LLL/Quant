"""Single-rule BTC spot/cash replay for preregistered Deribit taker pressure."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import math

import numpy as np
import pandas as pd

from research.btc_exchange_supply_source import SourceFailure
from research.eth_defi_tvl_sim import Daily, _monthly, _numbers, _numeric_pass, _regimes


START = date(2021, 1, 1)
END = date(2026, 1, 1)
INITIAL = 10_000.0
FEE = .001
SLIP = .0005
LOT = .00001


@dataclass(frozen=True)
class Simulation:
    days: tuple[Daily, ...]
    fills: tuple[dict, ...]
    fees: float
    slippage_drag: float
    terminal_cash: float


def _buy(cash: float, reference: float, slip: float, fee_rate: float) -> tuple[float, float, float, float]:
    execution = reference * (1 + slip)
    qty = math.floor(cash / (execution * (1 + fee_rate)) / LOT + 1e-9) * LOT
    notional = qty * execution
    fee = notional * fee_rate
    residual = cash - notional - fee
    if qty <= 0 or notional < 10 or residual < -1e-7:
        raise SourceFailure("BTC option-flow buy not executable")
    return qty, notional, fee, max(0.0, residual)


def simulate(targets: dict[date, int], prices: pd.DataFrame, terminal_open: float,
             *, slippage: float = SLIP, fee_rate: float = FEE,
             latency_days: int = 0, buy_and_hold: bool = False) -> Simulation:
    if latency_days not in (0, 1) or slippage < 0 or fee_rate < 0:
        raise ValueError("invalid preregistered scenario")
    if not prices.index.equals(pd.date_range("2020-12-01", "2025-12-31", tz="UTC", freq="D")):
        raise SourceFailure("option-flow spot price grid invalid")
    if not math.isfinite(terminal_open) or terminal_open <= 0:
        raise SourceFailure("option-flow terminal spot opening invalid")
    if len(targets) != 261 or min(targets) != date(2021, 1, 4) or max(targets) != date(2025, 12, 29):
        raise SourceFailure("option-flow target calendar invalid")
    cash, qty, fees, drag = INITIAL, 0.0, 0.0, 0.0
    fills: list[dict] = []
    days: list[Daily] = []
    previous = INITIAL
    for stamp in pd.date_range(START, END - timedelta(days=1), tz="UTC", freq="D"):
        day = stamp.date()
        bar = prices.loc[stamp]
        day_fee = 0.0
        due = (day == START if buy_and_hold else day.weekday() == 1 + latency_days)
        if due:
            monday = day - timedelta(days=1 + latency_days)
            target = 1 if buy_and_hold else targets.get(monday)
            if target not in (0, 1):
                raise SourceFailure(f"{day}: missing frozen option-flow target")
            if target != int(qty > 0):
                reference = float(bar["open"])
                if target:
                    new_qty, notional, fee, residual = _buy(cash, reference, slippage, fee_rate)
                    previous_volume = prices.loc[pd.date_range(day - timedelta(days=30),
                                                                day - timedelta(days=1),
                                                                tz="UTC", freq="D"), "quote_volume"]
                    if (len(previous_volume) != 30 or not np.isfinite(previous_volume.to_numpy()).all()
                            or notional > .01 * float(previous_volume.median()) * (1 + 1e-12)):
                        raise SourceFailure(f"{day}: BTC 1% volume capacity failed")
                    qty, cash = new_qty, residual
                    drag += qty * reference * slippage
                    side = "buy"
                    execution = reference * (1 + slippage)
                else:
                    execution = reference * (1 - slippage)
                    notional = qty * execution
                    if notional < 10:
                        raise SourceFailure(f"{day}: BTC sale below minimum notional")
                    fee = notional * fee_rate
                    cash += notional - fee
                    drag += qty * reference * slippage
                    new_qty = qty
                    qty = 0.0
                    side = "sell"
                fees += fee
                day_fee += fee
                fills.append({"day": str(day), "side": side, "reference": reference,
                              "execution": execution, "quantity": new_qty,
                              "notional": notional, "fee": fee})
        end = cash + qty * float(bar["close"])
        high = max(previous, end, cash + qty * float(bar["high"]))
        low = min(previous, end, cash + qty * float(bar["low"]))
        if not all(math.isfinite(x) and x > 0 for x in (end, high, low)):
            raise SourceFailure(f"{day}: nonpositive/nonfinite BTC account")
        days.append(Daily(day, previous, end, high, low, int(qty > 0), day_fee))
        previous = end
    if qty:
        execution = terminal_open * (1 - slippage)
        notional = qty * execution
        if notional < 10:
            raise SourceFailure("terminal BTC sale below minimum notional")
        fee = notional * fee_rate
        cash += notional - fee
        fees += fee
        drag += qty * terminal_open * slippage
        fills.append({"day": str(END), "side": "sell", "reference": terminal_open,
                      "execution": execution, "quantity": qty, "notional": notional,
                      "fee": fee, "forced_terminal": True})
        last = days[-1]
        days[-1] = Daily(last.day, last.start, cash, max(last.high, cash),
                         min(last.low, cash), last.position, last.fee + fee)
    else:
        cash = previous
    if len(days) != 1826 or cash <= 0 or abs(days[-1].end - cash) > 1e-7:
        raise SourceFailure("option-flow final account reconciliation failed")
    return Simulation(tuple(days), tuple(fills), fees, drag, cash)


def _bootstrap(rows: tuple[Daily, ...]) -> dict:
    n = len(rows)
    factor = np.array([x.end / x.start for x in rows])
    high = np.array([x.high / x.start for x in rows])
    low = np.array([x.low / x.start for x in rows])
    rng = np.random.Generator(np.random.PCG64(20261209))
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
