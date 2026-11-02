"""Frozen, isolated test of the preregistered altcoin volume-reversal idea.

This module deliberately does not import the production trading engine.  The
public entry point fixes the universe and dates from the committed preregistered
protocol; internal helpers accept shorter periods solely for synthetic tests.
Prices are 4-hour Binance trade/mark bars indexed by UTC *opening* timestamp.
Funding uses its actual exchange event time and the most recent 4-hour mark
opening price, which is a disclosed approximation when an event falls mid-bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


FIXED_SYMBOLS = (
    "AAVEUSDT", "BANDUSDT", "COMPUSDT", "DASHUSDT", "IOTAUSDT",
    "KSMUSDT", "ONTUSDT", "SNXUSDT", "SUSHIUSDT", "XMRUSDT",
    "XTZUSDT", "YFIUSDT", "ZECUSDT", "ZILUSDT",
)
START = pd.Timestamp("2021-01-01", tz="UTC")
HOLDOUT_START = pd.Timestamp("2023-01-01", tz="UTC")
END = pd.Timestamp("2026-10-01", tz="UTC")
INITIAL_CASH = 10_000.0
TAKER_FEE = 0.00045
BASE_SLIPPAGE = 0.001
STRESS_SLIPPAGE = 0.0025
DAY_YEAR = 365.25
REGIMES = (
    ("2021", "2021-01-01", "2022-01-01"),
    ("2022", "2022-01-01", "2023-01-01"),
    ("2023-2024", "2023-01-01", "2025-01-01"),
    ("2025-2026-09", "2025-01-01", "2026-10-01"),
)


class DataIntegrityError(ValueError):
    """A missing, duplicated, or invalid observation invalidates the experiment."""


@dataclass(frozen=True)
class Signal:
    date: pd.Timestamp
    eligible_count: int
    low_activity_count: int
    longs: tuple[str, ...]
    shorts: tuple[str, ...]
    reason: str


@dataclass
class BacktestResult:
    daily_nav: pd.Series
    daily_returns: pd.Series
    trade_log: pd.DataFrame
    signal_log: pd.DataFrame
    report: dict[str, Any]


def _utc_index(frame: pd.DataFrame, name: str) -> pd.DatetimeIndex:
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise DataIntegrityError(f"{name}: expected timezone-aware DatetimeIndex")
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise DataIntegrityError(f"{name}: timestamps must be sorted and unique")
    index = index.tz_convert("UTC")
    if len(index) == 0 or not (index.hour % 4 == 0).all() or not (index.minute == 0).all() or not (index.second == 0).all():
        raise DataIntegrityError(f"{name}: timestamp is not on UTC 4-hour grid")
    if len(index) > 1:
        # Compare TimedeltaIndex values directly.  DatetimeIndex.asi8 may be
        # microseconds or nanoseconds depending on pandas' datetime unit.
        steps = index[1:] - index[:-1]
        four_hours = pd.Timedelta(hours=4)
        if not ((steps == four_hours) | (steps == 2 * four_hours)).all():
            raise DataIntegrityError(f"{name}: more than one missing 4-hour bar or overlap")
        expected = int((index[-1] - index[0]) / pd.Timedelta(hours=4)) + 1
        if len(index) / expected < 0.999:
            raise DataIntegrityError(f"{name}: 4-hour grid coverage below 99.9%")
    return index


def _audit_bars(frame: pd.DataFrame, name: str, *, trade: bool) -> pd.DataFrame:
    required = ("open", "high", "low", "close") + (("quote_volume",) if trade else ())
    if any(column not in frame.columns for column in required):
        raise DataIntegrityError(f"{name}: missing required OHLC/quote-volume columns")
    index = _utc_index(frame, name)
    bars = frame.loc[:, list(required)].copy()
    bars.index = index
    try:
        numbers = bars.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise DataIntegrityError(f"{name}: nonnumeric bar") from exc
    if not np.isfinite(numbers).all():
        raise DataIntegrityError(f"{name}: nonfinite bar")
    o, h, l, c = (numbers[:, i] for i in range(4))
    if (l <= 0).any() or (h < np.maximum(o, c)).any() or (l > np.minimum(o, c)).any() or (h < l).any():
        raise DataIntegrityError(f"{name}: invalid OHLC")
    if trade and (numbers[:, 4] < 0).any():
        raise DataIntegrityError(f"{name}: negative quote volume")
    return bars.astype(float)


def _audit_funding(frame: pd.DataFrame, name: str, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if not {"funding_time", "rate"}.issubset(frame.columns):
        raise DataIntegrityError(f"{name}: missing funding_time/rate")
    events = frame.loc[:, ["funding_time", "rate"]].copy()
    try:
        times = pd.DatetimeIndex(pd.to_datetime(events["funding_time"], utc=True))
        rates = pd.to_numeric(events["rate"], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise DataIntegrityError(f"{name}: malformed funding event") from exc
    if len(events) == 0 or not np.isfinite(rates).all() or times.has_duplicates or not times.is_monotonic_increasing:
        raise DataIntegrityError(f"{name}: empty/duplicate/unsorted/nonfinite funding")
    if len(times) > 1 and ((times[1:] - times[:-1]) > pd.Timedelta(hours=24)).any():
        raise DataIntegrityError(f"{name}: funding gap exceeds 24 hours")
    if times[0] > start or times[-1] < end - pd.Timedelta(hours=24):
        raise DataIntegrityError(f"{name}: funding does not cover test period")
    events["funding_time"] = times
    events["rate"] = rates
    return events.reset_index(drop=True)


def _prepare_inputs(
    trade_bars: Mapping[str, pd.DataFrame],
    mark_bars: Mapping[str, pd.DataFrame],
    funding_events: Mapping[str, pd.DataFrame],
    symbols: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    trade: dict[str, pd.DataFrame] = {}
    mark: dict[str, pd.DataFrame] = {}
    funding: dict[str, pd.DataFrame] = {}
    last_open = end - pd.Timedelta(hours=4)
    for symbol in symbols:
        if symbol not in trade_bars or symbol not in mark_bars or symbol not in funding_events:
            raise DataIntegrityError(f"{symbol}: missing one complete data source")
        trade[symbol] = _audit_bars(trade_bars[symbol], f"{symbol} trade", trade=True)
        mark[symbol] = _audit_bars(mark_bars[symbol], f"{symbol} mark", trade=False)
        if trade[symbol].index[-1] < last_open or mark[symbol].index[-1] < last_open:
            raise DataIntegrityError(f"{symbol}: no verifiable terminal exit bar")
        # The early formation period may begin mid-month if listing starts then.
        if trade[symbol].index[0] > start - pd.Timedelta(days=62):
            raise DataIntegrityError(f"{symbol}: insufficient 2020 formation history")
        # Isolated trade and mark gaps need not coincide.  They are audited
        # separately; only a gap that affects a selected position or required
        # price invalidates the account path (see _simulate).
        funding[symbol] = _audit_funding(funding_events[symbol], f"{symbol} funding", start=start, end=end)
    return trade, mark, funding


def _daily_trade_data(trade: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    daily: dict[str, pd.DataFrame] = {}
    for symbol, bars in trade.items():
        grouped = bars.groupby(bars.index.floor("D"), sort=True)
        out = pd.DataFrame({
            "volume": grouped["quote_volume"].sum(),
            "close": grouped["close"].last(),
            "bar_count": grouped["close"].count(),
        })
        daily[symbol] = out[out["bar_count"] == 6].drop(columns="bar_count")
    return daily


def _select_for_day(day: pd.Timestamp, daily: Mapping[str, pd.DataFrame], symbols: Sequence[str]) -> Signal:
    yesterday = day - pd.Timedelta(days=1)
    previous = day - pd.Timedelta(days=2)
    baseline_first = day - pd.Timedelta(days=31)
    baseline_last = day - pd.Timedelta(days=2)
    eligible: list[tuple[str, float, float]] = []
    baseline_dates = pd.date_range(baseline_first, baseline_last, freq="D", tz="UTC")
    for symbol in symbols:
        series = daily[symbol]
        if yesterday not in series.index or previous not in series.index:
            continue
        history = series.loc[series.index <= yesterday]
        if len(history) < 90 or not baseline_dates.isin(series.index).all():
            continue
        baseline = float(series.loc[baseline_first:baseline_last, "volume"].mean())
        latest_volume = float(series.at[yesterday, "volume"])
        if baseline < 5_000_000 or latest_volume <= 0:
            continue
        yesterday_close = float(series.at[yesterday, "close"])
        previous_close = float(series.at[previous, "close"])
        if previous_close <= 0:
            raise DataIntegrityError(f"{symbol}: invalid previous close")
        activity = float(np.log(latest_volume) - np.log(baseline))
        prior_day_return = yesterday_close / previous_close - 1.0
        eligible.append((symbol, activity, prior_day_return))
    if len(eligible) < 8:
        return Signal(day, len(eligible), len(eligible) // 2, (), (), "fewer_than_8_eligible")
    low_half = sorted(eligible, key=lambda row: (row[1], row[0]))[:len(eligible) // 2]
    if len(low_half) < 4:
        return Signal(day, len(eligible), len(low_half), (), (), "fewer_than_4_low_activity")
    longs = tuple(row[0] for row in sorted(low_half, key=lambda row: (row[2], row[0]))[:2])
    # In the all-equal edge case, assigning longs first and ranking only the
    # remainder for shorts makes the alphabetic tie rule deterministic without
    # trying to hold one contract in opposite directions at once.
    remaining = [row for row in low_half if row[0] not in longs]
    shorts = tuple(row[0] for row in sorted(remaining, key=lambda row: (-row[2], row[0]))[:2])
    if set(longs) & set(shorts):
        raise DataIntegrityError("long and short selected same contract")
    return Signal(day, len(eligible), len(low_half), longs, shorts, "trade")


def _metrics(returns: pd.Series) -> dict[str, Any]:
    values = returns.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < -1).any():
        raise DataIntegrityError("invalid daily account return")
    n = len(values)
    if n == 0:
        return {"days": 0, "cagr": None, "sharpe": None, "max_drawdown": None,
                "profit_factor": None, "monthly_profit_rate": None,
                "longest_losing_months": None, "net_return": None}
    nav = np.concatenate(([1.0], np.cumprod(1.0 + values)))
    growth = float(nav[-1])
    cagr = float(growth ** (DAY_YEAR / n) - 1.0) if growth > 0 else -1.0
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    sharpe = float(values.mean() / std * np.sqrt(DAY_YEAR)) if std > 0 else None
    drawdown = nav / np.maximum.accumulate(nav) - 1.0
    gains = float(values[values > 0].sum())
    losses = float(-values[values < 0].sum())
    pf = gains / losses if losses > 0 else None
    # Return dates are UTC calendar days; removing tz only for month labels
    # avoids pandas' otherwise noisy timezone-to-Period warning.
    month_factors = (1.0 + returns).groupby(returns.index.tz_localize(None).to_period("M")).prod()
    month_returns = month_factors - 1.0
    longest = 0
    streak = 0
    for month_return in month_returns:
        streak = streak + 1 if month_return < 0 else 0
        longest = max(longest, streak)
    return {
        "days": n,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "profit_factor": pf,
        "monthly_profit_rate": float((month_returns > 0).mean()),
        "longest_losing_months": longest,
        "net_return": growth - 1.0,
    }


def _monte_carlo(returns: pd.Series, *, paths: int, seed: int) -> dict[str, Any]:
    if paths <= 0:
        return {"paths": 0, "block_days": 21, "seed": seed,
                "joint_gate_frequency": None, "p05": None}
    observed = returns.to_numpy(dtype=float)
    n = len(observed)
    if n < 21:
        raise DataIntegrityError("21-day block Monte Carlo needs >=21 daily returns")
    rng = np.random.default_rng(seed)
    store = np.empty((paths, 4), dtype=float)
    joint_count = 0
    chunks = ceil(n / 21)
    for offset in range(0, paths, 128):
        batch = min(128, paths - offset)
        starts = rng.integers(0, n, size=(batch, chunks))
        offsets = np.arange(21, dtype=np.int64)
        indices = ((starts[:, :, None] + offsets) % n).reshape(batch, -1)[:, :n]
        sample = observed[indices]
        mean = sample.mean(axis=1)
        std = sample.std(axis=1, ddof=1)
        sharpe = np.divide(mean * np.sqrt(DAY_YEAR), std,
                           out=np.full(batch, np.nan), where=std > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            growth_log = np.log1p(sample).sum(axis=1)
        cagr = np.expm1(growth_log * DAY_YEAR / n)
        nav = np.cumprod(1.0 + sample, axis=1)
        running_high = np.maximum.accumulate(np.concatenate((np.ones((batch, 1)), nav), axis=1), axis=1)[:, 1:]
        max_dd = np.min(nav / running_high - 1.0, axis=1)
        gains = np.clip(sample, 0, None).sum(axis=1)
        losses = -np.clip(sample, None, 0).sum(axis=1)
        pf = np.divide(gains, losses, out=np.full(batch, np.nan), where=losses > 0)
        store[offset:offset + batch] = np.stack((cagr, sharpe, max_dd, pf), axis=1)
        joint_count += int(((cagr >= 0.40) & (cagr <= 0.60) & (sharpe > 1.5)
                            & (max_dd >= -0.20) & (pf > 1.3)).sum())
    names = ("cagr", "sharpe", "max_drawdown", "profit_factor")
    p05 = {name: (float(np.nanquantile(store[:, i], 0.05)) if np.isfinite(store[:, i]).any() else None)
           for i, name in enumerate(names)}
    return {"paths": paths, "block_days": 21, "seed": seed,
            "joint_gate_frequency": joint_count / paths, "p05": p05}


def _passes_lower_bounds(metrics: Mapping[str, Any], *, holdout: bool) -> bool:
    cagr = metrics["cagr"]
    sharpe = metrics["sharpe"]
    dd = metrics["max_drawdown"]
    pf = metrics["profit_factor"]
    return bool(cagr is not None and cagr >= 0.40 and (not holdout or cagr <= 0.60)
                and sharpe is not None and sharpe > 1.5 and dd is not None
                and dd >= -0.20 and pf is not None and pf > 1.3)


def _period_metrics(returns: pd.Series, start: str, end: str) -> dict[str, Any]:
    period = returns.loc[(returns.index >= pd.Timestamp(start, tz="UTC"))
                         & (returns.index < pd.Timestamp(end, tz="UTC"))]
    return _metrics(period)


def _report(returns: pd.Series, accounting: Mapping[str, float], *, paths: int,
            liquidated: bool, leverage: float, slippage_rate: float) -> dict[str, Any]:
    full = _metrics(returns)
    holdout = _metrics(returns.loc[returns.index >= HOLDOUT_START])
    annual: dict[str, Any] = {}
    for year in range(2021, 2026):
        annual_return = returns.loc[(returns.index.year == year)]
        detail = _metrics(annual_return)
        detail["pass"] = bool(detail["net_return"] is not None and detail["net_return"] > 0
                              and detail["profit_factor"] is not None and detail["profit_factor"] > 1
                              and detail["max_drawdown"] is not None and detail["max_drawdown"] >= -0.20)
        annual[str(year)] = detail
    regimes = {label: _period_metrics(returns, begin, end)
               for label, begin, end in REGIMES}
    mc_full = _monte_carlo(returns, paths=paths, seed=20261102)
    mc_holdout = _monte_carlo(returns.loc[returns.index >= HOLDOUT_START], paths=paths, seed=20261102)
    wf_full = sum(bool(annual[str(year)]["pass"]) for year in range(2021, 2026)) >= 4
    wf_holdout = sum(bool(annual[str(year)]["pass"]) for year in range(2023, 2026)) >= 2
    regime_ok = (sum(bool(item["net_return"] is not None and item["net_return"] > 0)
                     for item in regimes.values()) >= 3
                 and all(item["max_drawdown"] is not None and item["max_drawdown"] >= -0.20
                         for item in regimes.values()))
    def mc_pass(item: Mapping[str, Any]) -> bool:
        return bool(item["joint_gate_frequency"] is not None and item["joint_gate_frequency"] >= 0.10
                    and item["p05"] is not None and item["p05"]["cagr"] is not None
                    and item["p05"]["cagr"] > 0)
    gate = {
        "full_lower_bounds": _passes_lower_bounds(full, holdout=False),
        "historical_holdout_targets": _passes_lower_bounds(holdout, holdout=True),
        "annual_walk_forward_full": wf_full,
        "annual_walk_forward_holdout": wf_holdout,
        "fixed_regimes": regime_ok,
        "monte_carlo_full": mc_pass(mc_full),
        "monte_carlo_holdout": mc_pass(mc_holdout),
        "no_liquidation": not liquidated,
        "prospective_independent_oos": False,
    }
    historical_numeric_passed = all(value for key, value in gate.items()
                                    if key != "prospective_independent_oos")
    return {
        "evidence_grade": "preregistered retrospective blind test; NOT prospective independent OOS",
        "test_start_utc": str(START), "test_end_exclusive_utc": str(END),
        "holdout_start_utc": str(HOLDOUT_START),
        "initial_cash_usd": INITIAL_CASH,
        "gross_leverage": leverage,
        "taker_fee_per_side": TAKER_FEE,
        "slippage_per_side": slippage_rate,
        "funding_mark_proxy": "most recent 4-hour mark open at actual funding event time",
        "intrabar_margin_model": "simultaneous long mark lows/short mark highs; 1% maintenance plus 1% liquidation fee at 1x/3x/5x",
        "full": full, "historical_holdout": holdout,
        "annual_walk_forward": annual, "fixed_regimes": regimes,
        "monte_carlo_full": mc_full, "monte_carlo_holdout": mc_holdout,
        "accounting_usd": dict(accounting), "liquidated": liquidated,
        "gate": {**gate, "historical_numeric_passed": historical_numeric_passed,
                 "all_passed": all(gate.values())},
    }


def _simulate(
    trade: Mapping[str, pd.DataFrame],
    mark: Mapping[str, pd.DataFrame],
    funding: Mapping[str, pd.DataFrame],
    signals: Mapping[pd.Timestamp, Signal],
    *, start: pd.Timestamp, end: pd.Timestamp, slippage_rate: float,
    leverage: float,
) -> tuple[pd.Series, pd.DataFrame, dict[str, float], bool]:
    symbols = tuple(trade)
    cash = INITIAL_CASH
    qty = {symbol: 0.0 for symbol in symbols}
    basis = {symbol: 0.0 for symbol in symbols}
    funding_records = sorted(
        ((pd.Timestamp(time), symbol, float(rate))
         for symbol in symbols
         for time, rate in funding[symbol].loc[:, ["funding_time", "rate"]].itertuples(index=False, name=None)
         if start <= pd.Timestamp(time) < end),
        key=lambda row: (row[0], row[1]),
    )
    event_cursor = 0
    nav: dict[pd.Timestamp, float] = {}
    trades: list[dict[str, Any]] = []
    accounting = {"funding": 0.0, "taker_fees": 0.0, "slippage": 0.0,
                  "financing": 0.0, "liquidation_fee": 0.0,
                  "traded_notional": 0.0}
    liquidated = False

    def mark_at(symbol: str, time: pd.Timestamp, field: str = "open") -> float:
        bar_time = time.floor("4h")
        if bar_time not in mark[symbol].index:
            raise DataIntegrityError(f"{symbol}: missing mark for {time}")
        return float(mark[symbol].at[bar_time, field])

    def equity_at(time: pd.Timestamp) -> float:
        return cash + sum(qty[symbol] * (mark_at(symbol, time) - basis[symbol])
                          for symbol in symbols if qty[symbol] != 0)

    def apply_funding(until: pd.Timestamp) -> None:
        nonlocal event_cursor, cash
        while event_cursor < len(funding_records) and funding_records[event_cursor][0] <= until:
            event_time, symbol, rate = funding_records[event_cursor]
            event_cursor += 1
            if qty[symbol] == 0:
                continue
            payment = -qty[symbol] * mark_at(symbol, event_time) * rate
            cash += payment
            accounting["funding"] += payment

    def trade_to(time: pd.Timestamp, target: Mapping[str, float], *, terminal: bool = False) -> None:
        nonlocal cash
        for symbol in symbols:
            old = qty[symbol]
            new = float(target.get(symbol, 0.0))
            if old == 0 and new == 0:
                continue
            bar_time = time if not terminal else end - pd.Timedelta(hours=4)
            field = "open" if not terminal else "close"
            if bar_time not in trade[symbol].index:
                raise DataIntegrityError(f"{symbol}: no executable {time} trade bar")
            price = float(trade[symbol].at[bar_time, field])
            notional = abs(new - old) * price
            fee = notional * TAKER_FEE
            slip = notional * slippage_rate
            cash += old * (price - basis[symbol]) - fee - slip
            accounting["taker_fees"] += fee
            accounting["slippage"] += slip
            accounting["traded_notional"] += notional
            qty[symbol] = new
            basis[symbol] = price if new != 0 else 0.0
            if notional > 0:
                trades.append({"time": time, "symbol": symbol, "old_qty": old,
                               "new_qty": new, "price": price, "notional": notional,
                               "fee": fee, "slippage": slip})

    grid = pd.date_range(start, end, freq="4h", inclusive="left")
    for time in grid:
        apply_funding(time)
        if time.hour == 0:
            nav[time] = equity_at(time) if not liquidated else cash
        # Archived funding calc_time may have a +2–5ms offset.  Apply events
        # through the first second of each 4-hour bar before its margin check.
        # At 04:00 this is explicitly *before* rebalance (old-position charge);
        # at 00:00 the exact-boundary NAV was already recorded, so +ms events
        # belong to the new UTC day rather than the day just completed.
        apply_funding(time + pd.Timedelta(milliseconds=999))
        if time.hour == 4 and not liquidated:
            signal = signals[time.floor("D")]
            equity = equity_at(time)
            if equity <= 0:
                cash = 0.0
                qty = {symbol: 0.0 for symbol in symbols}
                liquidated = True
            else:
                chosen = signal.longs + signal.shorts
                # A missing 04:00 *entry* price cancels the whole portfolio
                # for that day.  Any missing exit price in trade_to is fatal.
                entries_complete = all(time in trade[symbol].index and time in mark[symbol].index
                                       for symbol in chosen)
                weights = ({symbol: 0.25 for symbol in signal.longs}
                           if entries_complete else {})
                if entries_complete:
                    weights.update({symbol: -0.25 for symbol in signal.shorts})
                target = {symbol: equity * leverage * weight / float(trade[symbol].at[time, "open"])
                          for symbol, weight in weights.items()}
                trade_to(time, target)
        # Even at 1x, an isolated missing 4-hour bar in a held contract cannot
        # be silently skipped.  A missing bar in an unrelated contract is fine.
        for symbol in symbols:
            if qty[symbol] and (time not in trade[symbol].index or time not in mark[symbol].index):
                raise DataIntegrityError(f"{symbol}: missing 4-hour bar while position was open at {time}")
        if not liquidated:
            # Long lows and short highs are assumed simultaneous: deliberately
            # more adverse than any observable portfolio path within the bar.
            # The same margin rule applies at 1x: an extreme squeeze of the
            # short legs can still exhaust a nominally unlevered account.
            gross = 0.0
            adverse = cash
            for symbol in symbols:
                position = qty[symbol]
                if position == 0:
                    continue
                # Binance's margin/liquidation reference is the *mark* price,
                # not a possibly dislocated last-trade print.
                adverse_price = float(mark[symbol].at[time, "low" if position > 0 else "high"])
                gross += abs(position * adverse_price)
                adverse += position * (adverse_price - basis[symbol])
            if gross > 0 and adverse <= 0.01 * gross:
                penalty = 0.01 * gross
                exit_fee = gross * TAKER_FEE
                exit_slip = gross * slippage_rate
                accounting["liquidation_fee"] += penalty
                accounting["taker_fees"] += exit_fee
                accounting["slippage"] += exit_slip
                accounting["traded_notional"] += gross
                cash = max(0.0, adverse - penalty - exit_fee - exit_slip)
                qty = {symbol: 0.0 for symbol in symbols}
                basis = {symbol: 0.0 for symbol in symbols}
                liquidated = True
        if leverage > 1 and not liquidated:
            gross = sum(abs(qty[symbol] * mark_at(symbol, time)) for symbol in symbols if qty[symbol])
            excess = max(0.0, gross - max(0.0, equity_at(time)))
            financing = excess * 0.04 * (4.0 / 24.0 / DAY_YEAR)
            cash -= financing
            accounting["financing"] += financing

    # The September 20:00 bar closes immediately before the exclusive October
    # boundary.  It is the last verifiable price without reading October data.
    if not liquidated:
        apply_funding(end - pd.Timedelta(microseconds=1))
        trade_to(end, {}, terminal=True)
    nav[end] = cash
    if any(value < 0 or not np.isfinite(value) for value in nav.values()):
        raise DataIntegrityError("nonfinite or negative daily equity")
    return (pd.Series(nav, name="nav_usd", dtype=float),
            pd.DataFrame(trades), accounting, liquidated)


def run_reversal_backtest(
    trade_bars: Mapping[str, pd.DataFrame],
    mark_bars: Mapping[str, pd.DataFrame],
    funding_events: Mapping[str, pd.DataFrame],
    *, slippage_rate: float = BASE_SLIPPAGE,
    monte_carlo_paths: int = 5000,
    leverage: float = 1.0,
    base_result: BacktestResult | None = None,
) -> BacktestResult:
    """Run the immutable formal sample; no symbol or date override is exposed.

    3x/5x requires the actual 1x base result with historical-holdout Sharpe
    >1.5 and PF >1.3; a caller cannot bypass the frozen sequence by passing
    leverage alone.
    """
    if slippage_rate not in (BASE_SLIPPAGE, STRESS_SLIPPAGE):
        raise ValueError("only preregistered 0.10% or 0.25% slippage is allowed")
    if leverage not in (1.0, 3.0, 5.0):
        raise ValueError("only preregistered gross leverage 1x, 3x, or 5x is allowed")
    if monte_carlo_paths != 5000:
        raise ValueError("formal Monte Carlo must use 5,000 paths")
    if leverage > 1:
        if base_result is None:
            raise ValueError("3x/5x requires the 1x base result")
        base = base_result.report
        held = base.get("historical_holdout", {})
        if (base.get("gross_leverage") != 1.0
                or base.get("slippage_per_side") != BASE_SLIPPAGE
                or held.get("sharpe") is None or held["sharpe"] <= 1.5
                or held.get("profit_factor") is None or held["profit_factor"] <= 1.3):
            raise ValueError("1x historical holdout Sharpe/PF did not clear preregistered stress trigger")
    trade, mark, funding = _prepare_inputs(trade_bars, mark_bars, funding_events,
                                           FIXED_SYMBOLS, START, END)
    daily = _daily_trade_data(trade)
    signals = {day: _select_for_day(day, daily, FIXED_SYMBOLS)
               for day in pd.date_range(START, END, freq="D", inclusive="left")}
    nav, trade_log, accounting, liquidated = _simulate(
        trade, mark, funding, signals, start=START, end=END,
        slippage_rate=slippage_rate, leverage=leverage)
    returns = nav.pct_change(fill_method=None).iloc[1:]
    # NAV timestamps are *boundaries*.  Attribute each 00-to-00 result to the
    # day just completed so 31 December belongs to that year's WF window.
    returns.index = returns.index - pd.Timedelta(days=1)
    if liquidated and (nav == 0).any():
        returns = returns.fillna(0.0)
    signals_frame = pd.DataFrame([{
        "date": value.date, "eligible_count": value.eligible_count,
        "low_activity_count": value.low_activity_count,
        "longs": ",".join(value.longs), "shorts": ",".join(value.shorts),
        "reason": value.reason,
    } for value in signals.values()])
    report = _report(returns, accounting, paths=monte_carlo_paths,
                     liquidated=liquidated, leverage=leverage,
                     slippage_rate=slippage_rate)
    return BacktestResult(nav, returns, trade_log, signals_frame, report)
