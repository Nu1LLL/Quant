"""One-shot, frozen Snapshot governance turnout backtest; no order routing."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import math
from statistics import median

import numpy as np
import pandas as pd

from .snapshot_governance_source import END, SPACES, START, Proposal, SourceFailure


FEE = 0.001
SLIPPAGE = 0.001
INITIAL_EQUITY = 10_000.0
HOLD_DAYS = 7
BOOTSTRAP_PATHS = 5_000
BOOTSTRAP_BLOCK_DAYS = 30
BOOTSTRAP_SEED = 20261108
MIN_VOTES = 20
MIN_PRIOR = 6
TURNOUT_MULTIPLE = 2


@dataclass
class Position:
    quantity: float
    entry_cost: float
    entry_day: date
    exit_day: date


def _utc_day(seconds: int) -> date:
    return datetime.fromtimestamp(seconds, timezone.utc).date()


def signals(proposals: list[Proposal]) -> tuple[dict[date, tuple[str, ...]], dict]:
    """Fixed per-space six-prior-proposal test, with same-end proposals simultaneous."""
    starts: dict[date, set[str]] = defaultdict(set)
    diagnostics = {"source_ineligible": 0, "eligible_proposals": 0,
                   "high_turnout_events": 0, "terminal_skips": 0}
    for symbol, space in SPACES.items():
        by_end: dict[int, list[Proposal]] = defaultdict(list)
        for proposal in proposals:
            if proposal.symbol == symbol:
                if proposal.space != space:
                    raise SourceFailure("fixed space/symbol mapping changed")
                by_end[proposal.end_s].append(proposal)
        prior = deque(maxlen=MIN_PRIOR)
        for finish in sorted(by_end):
            group = sorted(by_end[finish], key=lambda item: item.proposal_id)
            for proposal in group:
                if not proposal.score_available:
                    diagnostics["source_ineligible"] += 1
                    continue
                diagnostics["eligible_proposals"] += 1
                if (len(prior) == MIN_PRIOR and proposal.votes >= MIN_VOTES
                        and proposal.votes >= TURNOUT_MULTIPLE * median(prior)):
                    entry_day = _utc_day(proposal.decision_s) + timedelta(days=1)
                    if START <= entry_day and entry_day + timedelta(days=HOLD_DAYS) <= END:
                        starts[entry_day].add(symbol)
                        diagnostics["high_turnout_events"] += 1
                    elif START <= _utc_day(proposal.end_s) < END:
                        diagnostics["terminal_skips"] += 1
            # A simultaneous vote is never in another same-end vote's history.
            for proposal in group:
                if proposal.score_available:
                    prior.append(proposal.votes)
    return {day: tuple(sorted(symbols)) for day, symbols in starts.items()}, diagnostics


def _metrics(returns: np.ndarray) -> dict:
    if len(returns) < 2 or not np.isfinite(returns).all() or (returns <= -1).any():
        return {"cagr": None, "sharpe": None, "max_drawdown": None,
                "profit_factor": None, "net_return": None}
    nav = np.r_[1.0, np.cumprod(1.0 + returns)]
    growth = float(nav[-1])
    sd = float(returns.std(ddof=1))
    positive = float(returns[returns > 0].sum())
    negative = float(-returns[returns < 0].sum())
    return {
        "cagr": growth ** (365.25 / len(returns)) - 1.0,
        "sharpe": float(returns.mean() / sd * math.sqrt(365.25)) if sd > 0 else None,
        "max_drawdown": float((nav / np.maximum.accumulate(nav) - 1.0).min()),
        "profit_factor": positive / negative if negative > 0 else None,
        "net_return": growth - 1.0,
    }


def _regime(btc: pd.DataFrame, day: date) -> str:
    timestamp = pd.Timestamp(day, tz="UTC")
    prior = timestamp - pd.Timedelta(days=1)
    old = prior - pd.Timedelta(days=90)
    change = float(btc.at[prior, "close"] / btc.at[old, "close"] - 1.0)
    return "bull" if change > 0.2 else "bear" if change < -0.2 else "sideways"


def _bootstrap(returns: np.ndarray) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(returns)
    if n < BOOTSTRAP_BLOCK_DAYS:
        raise ValueError("not enough daily returns for 30-day block bootstrap")
    values = {name: [] for name in ("cagr", "sharpe", "max_drawdown", "profit_factor")}
    joint = 0
    blocks = math.ceil(n / BOOTSTRAP_BLOCK_DAYS)
    for _ in range(BOOTSTRAP_PATHS):
        starts = rng.integers(0, n - BOOTSTRAP_BLOCK_DAYS + 1, size=blocks)
        sample = np.concatenate([returns[k:k + BOOTSTRAP_BLOCK_DAYS] for k in starts])[:n]
        metric = _metrics(sample)
        for name in values:
            values[name].append(metric[name])
        if (metric["cagr"] is not None and metric["cagr"] >= 0.40
                and metric["sharpe"] is not None and metric["sharpe"] > 1.5
                and metric["max_drawdown"] is not None and metric["max_drawdown"] >= -0.20
                and metric["profit_factor"] is not None and metric["profit_factor"] > 1.3):
            joint += 1
    return {"seed": BOOTSTRAP_SEED, "paths": BOOTSTRAP_PATHS,
            "block_days": BOOTSTRAP_BLOCK_DAYS, "joint_gate_fraction": joint / BOOTSTRAP_PATHS,
            "percentiles": {name: [float(np.percentile([x for x in sequence if x is not None], p))
                                   if any(x is not None for x in sequence) else None
                                   for p in (5, 50, 95)]
                            for name, sequence in values.items()}}


def _path_drawdown(opening_equity: float, highs: list[float], lows: list[float]) -> float:
    peak = opening_equity
    worst = 0.0
    for high, low in zip(highs, lows):
        peak = max(peak, high)
        worst = min(worst, low / peak - 1.0)
    return worst


def _passive(prices: dict[str, pd.DataFrame], terminal: dict[str, float],
             days: pd.DatetimeIndex) -> dict:
    quantities = {}
    for symbol in SPACES:
        initial_open = float(prices[symbol].at[days[0], "open"])
        spend = INITIAL_EQUITY / len(SPACES)
        quantities[symbol] = spend / ((1.0 + FEE) * initial_open * (1.0 + SLIPPAGE))
    values = [sum(quantities[symbol] * float(prices[symbol].at[day, "close"])
                  for symbol in SPACES) for day in days]
    final = sum(quantities[symbol] * terminal[symbol] * (1.0 - SLIPPAGE)
                * (1.0 - FEE) for symbol in SPACES)
    values[-1] = final
    returns = np.diff(np.asarray([INITIAL_EQUITY] + values)) / np.asarray(
        [INITIAL_EQUITY] + values[:-1])
    result = _metrics(returns)
    result["terminal_equity"] = final
    return result


def run(proposals: list[Proposal], prices: dict[str, pd.DataFrame],
        terminal: dict[str, float]) -> dict:
    if set(prices) != set(SPACES) | {"BTCUSDT"} or set(terminal) != set(prices):
        raise SourceFailure("fixed price universe missing")
    entries, diagnostics = signals(proposals)
    if diagnostics["eligible_proposals"] == 0:
        raise SourceFailure("no proposal vote count passes availability screen")
    if not entries:
        raise SourceFailure("no high-turnout event survives the fixed source and timing gates")
    days = pd.date_range(START, END - timedelta(days=1), tz="UTC", freq="D")
    cash = INITIAL_EQUITY
    positions: dict[str, Position] = {}
    prev_equity = INITIAL_EQUITY
    peak = INITIAL_EQUITY
    worst_drawdown = 0.0
    daily_returns = []
    daily_equity = []
    daily_high = []
    daily_low = []
    daily_regime = []
    closed = []
    ignored_active = 0
    traded_notional = 0.0
    for timestamp in days:
        day = timestamp.date()
        opening = {symbol: float(prices[symbol].at[timestamp, "open"]) for symbol in SPACES}
        for symbol in sorted(tuple(positions)):
            holding = positions[symbol]
            if holding.exit_day != day:
                continue
            executed = opening[symbol] * (1.0 - SLIPPAGE)
            proceeds = holding.quantity * executed * (1.0 - FEE)
            cash += proceeds
            traded_notional += holding.quantity * executed
            closed.append({"symbol": symbol, "entry_day": str(holding.entry_day),
                           "exit_day": str(day), "net_pnl": proceeds - holding.entry_cost,
                           "holding_days": HOLD_DAYS})
            del positions[symbol]
        for symbol in entries.get(day, ()):
            if symbol in positions:
                ignored_active += 1
                continue
            equity_at_open = cash + sum(holding.quantity * opening[other]
                                        for other, holding in positions.items())
            notional = min(0.2 * equity_at_open, cash / (1.0 + FEE))
            if notional <= 0 or not math.isfinite(notional):
                raise ValueError("unfillable entry")
            quantity = notional / (opening[symbol] * (1.0 + SLIPPAGE))
            cost = notional * (1.0 + FEE)
            cash -= cost
            traded_notional += notional
            if cash < -1e-7:
                raise ValueError("negative account cash")
            positions[symbol] = Position(quantity, cost, day, day + timedelta(days=HOLD_DAYS))
        closing = cash + sum(holding.quantity * float(prices[symbol].at[timestamp, "close"])
                             for symbol, holding in positions.items())
        high = cash + sum(holding.quantity * float(prices[symbol].at[timestamp, "high"])
                          for symbol, holding in positions.items())
        low = cash + sum(holding.quantity * float(prices[symbol].at[timestamp, "low"])
                         for symbol, holding in positions.items())
        if not all(math.isfinite(value) and value > 0 for value in (cash + 1e-8, closing, high, low)):
            raise ValueError("nonpositive or nonfinite account path")
        peak = max(peak, high, closing)
        worst_drawdown = min(worst_drawdown, low / peak - 1.0)
        daily_returns.append(closing / prev_equity - 1.0)
        daily_equity.append(closing)
        daily_high.append(high)
        daily_low.append(low)
        daily_regime.append(_regime(prices["BTCUSDT"], day))
        prev_equity = closing
    for symbol in sorted(tuple(positions)):
        holding = positions[symbol]
        if holding.exit_day != END:
            raise ValueError("position crosses terminal accounting boundary")
        executed = terminal[symbol] * (1.0 - SLIPPAGE)
        proceeds = holding.quantity * executed * (1.0 - FEE)
        cash += proceeds
        traded_notional += holding.quantity * executed
        closed.append({"symbol": symbol, "entry_day": str(holding.entry_day),
                       "exit_day": str(END), "net_pnl": proceeds - holding.entry_cost,
                       "holding_days": HOLD_DAYS})
        del positions[symbol]
    peak = max(peak, cash)
    worst_drawdown = min(worst_drawdown, cash / peak - 1.0)
    daily_returns[-1] = cash / (daily_equity[-2] if len(daily_equity) > 1 else INITIAL_EQUITY) - 1.0
    daily_equity[-1] = cash
    returns = np.asarray(daily_returns, dtype=float)
    if (not np.isfinite(returns).all() or (returns <= -1).any()
            or abs(np.prod(1.0 + returns) * INITIAL_EQUITY - cash) > 1e-5):
        raise ValueError("daily/account reconciliation failed")
    if abs(sum(item["net_pnl"] for item in closed) - (cash - INITIAL_EQUITY)) > 1e-5:
        raise ValueError("closed-trade/account reconciliation failed")
    metrics = _metrics(returns)
    metrics["max_drawdown"] = min(metrics["max_drawdown"], worst_drawdown)
    year_metrics = {str(year): _metrics(returns[np.asarray(days.year == year)])
                    for year in range(2021, 2026)}
    holdout = _metrics(returns[np.asarray(days.year >= 2024)])
    holdout_start = int(np.searchsorted(days.values, pd.Timestamp("2024-01-01", tz="UTC").to_datetime64()))
    holdout["max_drawdown"] = min(
        holdout["max_drawdown"],
        _path_drawdown(daily_equity[holdout_start - 1],
                       daily_high[holdout_start:], daily_low[holdout_start:]))
    months = pd.Series(returns, index=days).resample("MS").apply(lambda x: float(np.prod(1 + x) - 1))
    positive_months = int((months > 0).sum())
    negative_run = run_length = 0
    for value in months:
        run_length = run_length + 1 if value < 0 else 0
        negative_run = max(negative_run, run_length)
    regimes = {name: _metrics(returns[np.array([label == name for label in daily_regime])])
               for name in ("bull", "bear", "sideways")}
    regime_days = {name: daily_regime.count(name) for name in regimes}
    monte_carlo = _bootstrap(returns)
    numerical_gate = (
        metrics["cagr"] >= 0.40 and metrics["sharpe"] is not None
        and metrics["sharpe"] > 1.5 and metrics["max_drawdown"] >= -0.20
        and metrics["profit_factor"] is not None and metrics["profit_factor"] > 1.3
        and all(item["net_return"] is not None and item["net_return"] > 0
                for item in year_metrics.values())
        and holdout["cagr"] >= 0.40 and holdout["sharpe"] is not None
        and holdout["sharpe"] > 1.5 and holdout["max_drawdown"] >= -0.20
        and holdout["profit_factor"] is not None and holdout["profit_factor"] > 1.3
        and positive_months >= 36 and negative_run < 4
        and all(regime_days[name] >= 90 and regimes[name]["net_return"] > 0
                for name in regimes)
        and monte_carlo["joint_gate_fraction"] >= 0.95
        and len(closed) > 0)
    by_asset = {symbol: {"trades": sum(x["symbol"] == symbol for x in closed),
                         "net_pnl": sum(x["net_pnl"] for x in closed if x["symbol"] == symbol)}
                for symbol in SPACES}
    return {"sample_days": len(days), "initial_equity": INITIAL_EQUITY,
            "terminal_equity": cash, "signal_diagnostics": diagnostics,
            "closed_trades": len(closed), "ignored_while_open": ignored_active,
            "median_holding_days": HOLD_DAYS if closed else None,
            "total_traded_notional_usd": traded_notional,
            "passive_comparison": _passive(prices, terminal, days),
            "by_asset": by_asset, "metrics": metrics, "year_metrics": year_metrics,
            "holdout_metrics": holdout, "positive_months": positive_months,
            "longest_negative_month_run": negative_run,
            "regime_days": regime_days, "regime_metrics": regimes,
            "monte_carlo": monte_carlo, "numerical_gate": bool(numerical_gate),
            "independent_oos": False, "full_gate": False,
            "evidence_grade": "preregistered_retrospective_historical_holdout"}
