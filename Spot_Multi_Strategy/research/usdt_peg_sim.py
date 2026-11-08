"""Frozen daily USDT peg risk gate, spot account, USD marks and Gate."""

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
INITIAL = 10_000.0
FEE = .001
SLIP = .0005
LOT = .00001
MC_PATHS = 5_000
MC_BLOCK = 21
MC_SEED = 20261122


@dataclass(frozen=True)
class Simulation:
    usd_days: tuple[Daily, ...]
    usdt_days: tuple[Daily, ...]
    fills: tuple[dict, ...]
    fees_usdt: float
    slippage_drag_usdt: float
    terminal_cash_usdt: float


def _target(day: date, closes: dict[date, float]) -> int:
    try:
        values = [closes[day - timedelta(days=offset)] for offset in range(8, 1, -1)]
    except KeyError as exc:
        raise SourceFailure(f"missing completed USDT peg day {exc}") from exc
    if len(values) != 7 or not all(math.isfinite(x) and x > 0 for x in values):
        raise SourceFailure("invalid seven-day USDT peg window")
    return int(all(.997 <= value <= 1.003 for value in values))


def _buy(cash: float, reference: float, slippage: float,
         fee_rate: float) -> tuple[float, float, float, float]:
    execution = reference * (1 + slippage)
    qty = math.floor(cash / (execution * (1 + fee_rate)) / LOT + 1e-9) * LOT
    notional = qty * execution
    fee = notional * fee_rate
    residual = cash - notional - fee
    if qty <= 0 or notional < 10 or residual < -1e-7:
        raise SourceFailure("buy violated lot, notional or available-cash limit")
    return qty, notional, fee, max(0.0, residual)


def simulate(coinbase: pd.DataFrame, binance: pd.DataFrame, terminal_open: float,
             *, slippage: float = SLIP, fee_rate: float = FEE,
             latency_days: int = 0, mode: str = "signal") -> Simulation:
    if slippage < 0 or fee_rate < 0 or latency_days not in (0, 1):
        raise ValueError("invalid frozen execution scenario")
    if mode not in {"signal", "buy_hold", "cash"}:
        raise ValueError("invalid frozen comparator")
    if not math.isfinite(terminal_open) or terminal_open <= 0:
        raise SourceFailure("invalid BTC terminal opening price")
    if not coinbase.index.equals(pd.date_range("2021-09-01", "2026-09-30", tz="UTC", freq="D")):
        raise SourceFailure("USDT-USD signal/mark grid invalid")
    if not binance.index.equals(pd.date_range("2021-08-01", "2026-09-30", tz="UTC", freq="D")):
        raise SourceFailure("BTCUSDT execution grid invalid")
    closes = {stamp.date(): float(value) for stamp, value in coinbase["close"].items()}
    if not all(math.isfinite(x) and x > 0 for x in closes.values()):
        raise SourceFailure("invalid Coinbase USDT/USD close")
    usd_start = INITIAL * closes[START - timedelta(days=1)]
    cash, qty = INITIAL, 0.0
    prior_usdt, prior_usd = INITIAL, usd_start
    fees = slip_drag = 0.0
    fills: list[dict] = []
    usdt_days: list[Daily] = []
    usd_days: list[Daily] = []
    for stamp in pd.date_range(START, END - timedelta(days=1), tz="UTC", freq="D"):
        day = stamp.date()
        row = binance.loc[stamp]
        peg = coinbase.loc[stamp]
        if not all(math.isfinite(float(peg[k])) and float(peg[k]) > 0
                   for k in ("open", "high", "low", "close")):
            raise SourceFailure(f"{day}: invalid Coinbase USD valuation")
        base_day = day - timedelta(days=latency_days)
        if mode == "cash":
            target = 0
        elif mode == "buy_hold":
            target = 1
        elif base_day < START:
            target = 0
        else:
            target = _target(base_day, closes)
        reference = float(row["open"])
        prefill = cash + qty * reference
        daily_fee = 0.0
        if target != int(qty > 0):
            if target:
                new_qty, notional, fee, residual = _buy(cash, reference, slippage,
                                                        fee_rate)
                previous = binance.loc[pd.date_range(day - timedelta(days=30),
                                                     day - timedelta(days=1),
                                                     tz="UTC", freq="D"), "quote_volume"]
                if len(previous) != 30 or not np.isfinite(previous.to_numpy()).all():
                    raise SourceFailure(f"{day}: 30 completed capacity bars unavailable")
                if float(previous.median()) <= 0 or notional > .01 * float(previous.median()):
                    raise SourceFailure(f"{day}: one-percent spot capacity limit failed")
                cash, qty = residual, new_qty
                execution = reference * (1 + slippage)
                side = "buy"
            else:
                execution = reference * (1 - slippage)
                notional = qty * execution
                if notional < 10:
                    raise SourceFailure(f"{day}: sell below 10 USDT")
                fee = notional * fee_rate
                cash += notional - fee
                new_qty = qty
                qty = 0.0
                side = "sell"
            slip_drag += new_qty * reference * slippage
            fees += fee
            daily_fee = fee
            fills.append({"day": str(day), "side": side, "reference": reference,
                          "execution": execution, "quantity": new_qty,
                          "notional": notional, "fee_usdt": fee})
        postfill = cash + qty * reference
        end_usdt = cash + qty * float(row["close"])
        high_usdt = max(prior_usdt, prefill, postfill,
                        cash + qty * float(row["high"]), end_usdt)
        low_usdt = min(prior_usdt, prefill, postfill,
                       cash + qty * float(row["low"]), end_usdt)
        peg_open, peg_high, peg_low, peg_close = (float(peg[k]) for k in
                                                 ("open", "high", "low", "close"))
        end_usd = end_usdt * peg_close
        high_usd = max(prior_usd, prefill * peg_open, postfill * peg_open,
                       high_usdt * peg_high, end_usd)
        low_usd = min(prior_usd, prefill * peg_open, postfill * peg_open,
                      low_usdt * peg_low, end_usd)
        if not all(math.isfinite(x) and x > 0 for x in
                   (prior_usdt, prefill, postfill, end_usdt, high_usdt, low_usdt,
                    prior_usd, end_usd, high_usd, low_usd)):
            raise SourceFailure(f"{day}: ruined or nonfinite account")
        if qty * float(row["close"]) > end_usdt + 1e-6:
            raise SourceFailure(f"{day}: spot gross exposure exceeds equity")
        usdt_days.append(Daily(day, prior_usdt, end_usdt, high_usdt, low_usdt,
                               int(qty > 0), daily_fee))
        usd_days.append(Daily(day, prior_usd, end_usd, high_usd, low_usd,
                              int(qty > 0), daily_fee * peg_open))
        prior_usdt, prior_usd = end_usdt, end_usd
    last_peg = closes[END - timedelta(days=1)]
    if qty:
        reference = terminal_open
        execution = reference * (1 - slippage)
        notional = qty * execution
        if notional < 10:
            raise SourceFailure("terminal sell below 10 USDT")
        fee = notional * fee_rate
        cash += notional - fee
        fees += fee
        slip_drag += qty * reference * slippage
        fills.append({"day": str(END), "side": "sell", "reference": reference,
                      "execution": execution, "quantity": qty, "notional": notional,
                      "fee_usdt": fee, "forced_terminal": True})
        prior = usdt_days[-1]
        usdt_days[-1] = Daily(prior.day, prior.start, cash,
                              max(prior.high, cash), min(prior.low, cash),
                              prior.position, prior.fee + fee)
        prior = usd_days[-1]
        terminal_usd = cash * last_peg
        usd_days[-1] = Daily(prior.day, prior.start, terminal_usd,
                             max(prior.high, terminal_usd),
                             min(prior.low, terminal_usd), prior.position,
                             prior.fee + fee * last_peg)
    if len(usd_days) != 1826 or len(usdt_days) != 1826:
        raise SourceFailure("account day count changed")
    if cash <= 0 or abs(usdt_days[-1].end - cash) > 1e-7:
        raise SourceFailure("terminal USDT cash reconciliation failed")
    if abs(usd_days[-1].end - cash * last_peg) > 1e-6:
        raise SourceFailure("terminal USD valuation reconciliation failed")
    if abs(sum(fill["fee_usdt"] for fill in fills) - fees) > 1e-7:
        raise SourceFailure("fill fee ledger mismatch")
    return Simulation(tuple(usd_days), tuple(usdt_days), tuple(fills), fees,
                      slip_drag, cash)


def _bootstrap(rows: tuple[Daily, ...]) -> dict:
    n = len(rows)
    factors = np.array([r.end / r.start for r in rows])
    highs = np.array([r.high / r.start for r in rows])
    lows = np.array([r.low / r.start for r in rows])
    start = rows[0].start
    rng = np.random.Generator(np.random.PCG64(MC_SEED))
    metrics: list[tuple[float, float | None, float, float | None]] = []
    passes = ruined = undefined = 0
    for _ in range(MC_PATHS):
        anchors = rng.integers(0, n, size=math.ceil(n / MC_BLOCK))
        idx = ((anchors[:, None] + np.arange(MC_BLOCK)) % n).ravel()[:n]
        factor = factors[idx]
        equity = start * np.cumprod(factor)
        begin = np.r_[start, equity[:-1]]
        if not np.isfinite(equity).all() or (equity <= 0).any():
            ruined += 1
            continue
        peak = np.maximum.accumulate(np.r_[start, begin * highs[idx]])[1:]
        dd = float(np.max(1 - begin * lows[idx] / peak))
        returns = factor - 1
        sigma = float(np.std(returns, ddof=1))
        sharpe = float(np.mean(returns) / sigma * math.sqrt(365)) if sigma > 0 else None
        dollars = equity - begin
        loss = float(-dollars[dollars < 0].sum())
        pf = float(dollars[dollars > 0].sum() / loss) if loss > 0 else None
        cagr = float((equity[-1] / start) ** (365.25 / n) - 1)
        metrics.append((cagr, sharpe, dd, pf))
        if sharpe is None or pf is None:
            undefined += 1
        if _numeric_pass({"cagr": cagr, "sharpe": sharpe,
                          "max_drawdown": dd, "profit_factor_daily_dollar": pf}):
            passes += 1

    def quantile(index: int) -> dict:
        values = [row[index] for row in metrics
                  if row[index] is not None and math.isfinite(row[index])]
        return {f"p{p:02d}": float(np.percentile(values, p)) if values else None
                for p in (5, 50, 95)}

    return {"paths": MC_PATHS, "joint_passes": passes,
            "joint_pass_rate": passes / MC_PATHS, "ruined": ruined,
            "undefined": undefined, "cagr": quantile(0), "sharpe": quantile(1),
            "max_drawdown": quantile(2), "profit_factor": quantile(3)}


def evaluate(primary: Simulation, gross: Simulation, double_slip: Simulation,
             delayed: Simulation, buy_hold: Simulation, cash_only: Simulation,
             binance: pd.DataFrame) -> dict:
    full = _numbers(primary.usd_days)
    annual = {str(year): _numbers([row for row in primary.usd_days
                                  if row.day.year == year]) for year in range(2021, 2027)}
    holdout = _numbers([row for row in primary.usd_days
                        if row.day >= date(2024, 1, 1)])
    monthly = _monthly(primary.usd_days)
    regimes = _regimes(primary.usd_days, binance)
    mc = _bootstrap(primary.usd_days)
    buys = sum(fill["side"] == "buy" for fill in primary.fills)
    yearly_fills = {str(year): sum(fill["day"][:4] == str(year)
                                   and not fill.get("forced_terminal", False)
                                   for fill in primary.fills)
                    for year in range(2021, 2027)}
    start_usd = full["start"]
    checks = {
        "numerical_full_usd": _numeric_pass(full),
        "min_20_buys": buys >= 20,
        "annual_walk_forward": all(annual[year]["end"] > annual[year]["start"]
                                   and annual[year]["sharpe"] is not None
                                   and annual[year]["sharpe"] > 0
                                   and yearly_fills[year] > 0 for year in annual),
        "historical_2024_2026_holdout": _numeric_pass(holdout),
        "monthly_stability": monthly["months"] == 60 and monthly["positive"] >= 36
                             and monthly["longest_nonpositive_streak"] <= 3,
        "four_regimes": all(value["days"] >= 90 and value["net_dollar_pnl"] > 0
                            for value in regimes.values()),
        "block_monte_carlo": mc["joint_pass_rate"] >= .80,
        "double_slippage": double_slip.usd_days[-1].end > start_usd,
        "one_day_latency": delayed.usd_days[-1].end > start_usd,
        "independent_executable_oos": False,
    }
    return {
        "usd_full": full, "usdt_full": _numbers(primary.usdt_days),
        "usd_zero_cost_same_signal": _numbers(gross.usd_days),
        "usd_same_cost_btc_buy_hold": _numbers(buy_hold.usd_days),
        "usd_all_usdt_cash": _numbers(cash_only.usd_days),
        "usd_holdout_2024_2026": holdout, "usd_annual": annual,
        "usd_monthly": monthly, "usd_regimes": regimes,
        "usd_monte_carlo": mc, "checks": checks,
        "all_gate_passed": all(checks.values()),
        "buys": buys, "fills_per_year": yearly_fills,
        "fills": primary.fills, "fees_usdt": primary.fees_usdt,
        "slippage_reference_drag_usdt": primary.slippage_drag_usdt,
        "net_final_usdt": primary.terminal_cash_usdt,
        "net_final_usd": primary.usd_days[-1].end,
        "double_slippage_final_usd": double_slip.usd_days[-1].end,
        "one_day_latency_final_usd": delayed.usd_days[-1].end,
    }
