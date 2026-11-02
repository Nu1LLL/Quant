"""Pure bar-level backtest for the frozen spot-flow/perpetual hypothesis.

No downloader, credential, production engine, or portfolio optimiser is used.
The public entry point accepts an already checksum-audited ArchiveBundle and
returns JSON-compatible values.  The internal period entry point exists only
to exercise the same accounting on synthetic fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt
from typing import Any, Mapping

import numpy as np
import pandas as pd


SYMBOLS = ("BTCUSDT", "ETHUSDT", "XRPUSDT", "LTCUSDT")
START = pd.Timestamp("2021-01-01", tz="UTC")
HOLDOUT_START = pd.Timestamp("2024-01-01", tz="UTC")
END = pd.Timestamp("2026-10-01", tz="UTC")
INITIAL_EQUITY = 10_000.0
ANNUAL_DAYS = 365.0
BAR = pd.Timedelta(hours=4)
DAY = pd.Timedelta(days=1)
FEE_BPS = 5.0
SLIPPAGE_BPS = 10.0


class BacktestIntegrityError(ValueError):
    """The passed bundle is incomplete or violates the frozen data contract."""


@dataclass
class Lot:
    quantity: float
    price: float
    opened: pd.Timestamp
    opening_cost: float
    funding: float = 0.0


def _json_number(value: Any) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _component(bundle: Any, name: str) -> Mapping[str, pd.DataFrame]:
    part = bundle.get(name) if isinstance(bundle, Mapping) else getattr(bundle, name, None)
    if not isinstance(part, Mapping) or set(part) != set(SYMBOLS):
        raise BacktestIntegrityError(f"{name}: exactly the four frozen symbols are required")
    return part


def _check_bars(frame: pd.DataFrame, *, symbol: str, kind: str,
                start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
        raise BacktestIntegrityError(f"{kind}/{symbol}: UTC-indexed DataFrame required")
    if frame.index.tz is None or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise BacktestIntegrityError(f"{kind}/{symbol}: timezone/duplicate/order failure")
    required = {"open", "high", "low", "close"}
    if kind == "spot":
        required |= {"quote_volume", "taker_buy_quote_volume"}
    if not required.issubset(frame.columns):
        raise BacktestIntegrityError(f"{kind}/{symbol}: missing required columns")
    expected = pd.date_range(start, end, freq="4h", inclusive="left")
    index = frame.index.tz_convert("UTC")
    if not index.equals(expected):
        raise BacktestIntegrityError(f"{kind}/{symbol}: incomplete 4h grid")
    values = frame.loc[:, list(required)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise BacktestIntegrityError(f"{kind}/{symbol}: nonfinite bar")
    prices = frame[["open", "high", "low", "close"]].to_numpy(dtype=float)
    op, hi, lo, cl = prices.T
    if (lo <= 0).any() or (hi < np.maximum(op, cl)).any() or (lo > np.minimum(op, cl)).any():
        raise BacktestIntegrityError(f"{kind}/{symbol}: invalid OHLC")
    if kind == "spot":
        q = frame["quote_volume"].to_numpy(dtype=float)
        b = frame["taker_buy_quote_volume"].to_numpy(dtype=float)
        if (q <= 0).any() or (b < 0).any() or (b > q).any():
            raise BacktestIntegrityError(f"{kind}/{symbol}: invalid taker/total quote volume")
    return frame


def _check_funding(frame: pd.DataFrame, *, symbol: str,
                   start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or not {"funding_time", "rate"}.issubset(frame.columns):
        raise BacktestIntegrityError(f"funding/{symbol}: missing event columns")
    out = frame.loc[:, ["funding_time", "rate"]].copy()
    times = pd.DatetimeIndex(pd.to_datetime(out["funding_time"], utc=True))
    rates = pd.to_numeric(out["rate"], errors="raise").to_numpy(dtype=float)
    if (len(times) == 0 or times.has_duplicates or not times.is_monotonic_increasing
            or not np.isfinite(rates).all()):
        raise BacktestIntegrityError(f"funding/{symbol}: invalid event sequence")
    if times[0] > start + pd.Timedelta(hours=24) or times[-1] < end - pd.Timedelta(hours=24):
        raise BacktestIntegrityError(f"funding/{symbol}: first/last 24h not covered")
    if len(times) > 1 and ((times[1:] - times[:-1]) > pd.Timedelta(hours=24)).any():
        raise BacktestIntegrityError(f"funding/{symbol}: event gap exceeds 24h")
    if (times < start).any() or (times >= end).any():
        raise BacktestIntegrityError(f"funding/{symbol}: event outside frozen period")
    out["funding_time"] = times
    out["rate"] = rates
    return out.reset_index(drop=True)


def _prepare(bundle: Any, start: pd.Timestamp, end: pd.Timestamp) -> tuple[dict, dict, dict, dict]:
    if start.tz is None or end.tz is None or start >= end or start.hour or end.hour:
        raise ValueError("start/end must be ordered UTC day boundaries")
    spot_input = _component(bundle, "spot_bars")
    trade_input = _component(bundle, "trade_bars")
    mark_input = _component(bundle, "mark_bars")
    funding_input = _component(bundle, "funding_events")
    spot = {s: _check_bars(spot_input[s], symbol=s, kind="spot", start=start, end=end) for s in SYMBOLS}
    trade = {s: _check_bars(trade_input[s], symbol=s, kind="trade", start=start, end=end) for s in SYMBOLS}
    mark = {s: _check_bars(mark_input[s], symbol=s, kind="mark", start=start, end=end) for s in SYMBOLS}
    funding = {s: _check_funding(funding_input[s], symbol=s, start=start, end=end) for s in SYMBOLS}
    return spot, trade, mark, funding


def _signals(spot: Mapping[str, pd.DataFrame], start: pd.Timestamp,
             end: pd.Timestamp) -> dict[pd.Timestamp, tuple[str, str]]:
    """Map next-day UTC 04:00 to (long, short), using only the previous day."""
    flows: dict[str, pd.Series] = {}
    for symbol, bars in spot.items():
        grouped = bars.groupby(bars.index.floor("D"), sort=True)
        volume = grouped["quote_volume"].sum()
        bought = grouped["taker_buy_quote_volume"].sum()
        count = grouped["quote_volume"].count()
        if (count != 6).any() or (volume <= 0).any():
            raise BacktestIntegrityError(f"spot/{symbol}: incomplete or zero-volume UTC day")
        flows[symbol] = (2.0 * bought - volume) / volume
    decisions: dict[pd.Timestamp, tuple[str, str]] = {}
    for day in pd.date_range(start + DAY, end - DAY, freq="D"):
        previous = day - DAY
        ranked = sorted(SYMBOLS, key=lambda s: (-float(flows[s].at[previous]), s))
        decisions[day + pd.Timedelta(hours=4)] = (ranked[0], ranked[-1])
    return decisions


def _regimes(spot_btc: pd.DataFrame, start: pd.Timestamp,
             end: pd.Timestamp) -> dict[pd.Timestamp, str]:
    closes = spot_btc.groupby(spot_btc.index.floor("D"))["close"].last()
    result = {}
    for day in pd.date_range(start, end, freq="D", inclusive="left"):
        last = day - DAY
        first = day - pd.Timedelta(days=31)
        if last not in closes.index or first not in closes.index:
            result[day] = "insufficient_history"
        else:
            value = float(closes.at[last] / closes.at[first] - 1.0)
            result[day] = "bull" if value > 0.10 else "bear" if value < -0.10 else "sideways"
    return result


def _period_metrics(returns: pd.Series, start: pd.Timestamp | None = None,
                    end: pd.Timestamp | None = None,
                    intrabar_drawdown: float | None = None) -> dict[str, Any]:
    selected = returns
    if start is not None:
        selected = selected.loc[selected.index >= start]
    if end is not None:
        selected = selected.loc[selected.index < end]
    x = selected.to_numpy(dtype=float)
    if len(x) == 0:
        return {"days": 0, "net_return": None, "cagr": None, "sharpe": None,
                "profit_factor_daily": None, "max_drawdown_daily": None,
                "max_drawdown_intrabar": intrabar_drawdown}
    nav = np.r_[1.0, np.cumprod(1.0 + x)]
    growth = float(nav[-1])
    cagr = -1.0 if growth <= 0 else growth ** (ANNUAL_DAYS / len(x)) - 1.0
    deviation = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    sharpe = float(x.mean() / deviation * sqrt(ANNUAL_DAYS)) if deviation > 0 else None
    gain = float(x[x > 0].sum())
    loss = float(-x[x < 0].sum())
    pf = gain / loss if loss > 0 else None
    daily_dd = float((nav / np.maximum.accumulate(nav) - 1.0).min())
    return {"days": len(x), "net_return": growth - 1.0, "cagr": cagr,
            "sharpe": sharpe, "profit_factor_daily": pf,
            "max_drawdown_daily": daily_dd,
            "max_drawdown_intrabar": intrabar_drawdown}


def _intrabar_drawdown(path: list[dict[str, Any]], start: pd.Timestamp,
                       end: pd.Timestamp) -> float | None:
    selected = [bar for bar in path if start <= bar["time"] < end]
    if not selected:
        return None
    peak = float(selected[0]["open_equity"])
    worst = 0.0
    for bar in selected:
        peak = max(peak, bar["open_equity"])
        if peak > 0:
            worst = min(worst, bar["adverse_equity"] / peak - 1.0)
        peak = max(peak, bar["close_equity"])
    return float(worst)


def block_monte_carlo(returns: pd.Series, *, block_days: int,
                      paths: int, seed: int) -> dict[str, Any]:
    """Circular contiguous-block resampling; zero-return days are retained."""
    observed = np.asarray(returns, dtype=float)
    n = len(observed)
    if n < block_days or paths < 1 or not np.isfinite(observed).all() or (observed < -1).any():
        raise ValueError("invalid Monte Carlo sample or specification")
    rng = np.random.default_rng(seed)
    samples = np.empty((paths, 4), dtype=float)
    chunks = ceil(n / block_days)
    offsets = np.arange(block_days, dtype=np.int64)
    for base in range(0, paths, 128):
        batch = min(128, paths - base)
        starts = rng.integers(0, n, size=(batch, chunks))
        indices = ((starts[:, :, None] + offsets) % n).reshape(batch, -1)[:, :n]
        x = observed[indices]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            growth = np.exp(np.log1p(x).sum(axis=1))
            cagr = np.where(growth > 0, growth ** (ANNUAL_DAYS / n) - 1.0, -1.0)
        std = x.std(axis=1, ddof=1)
        sharpe = np.divide(x.mean(axis=1) * sqrt(ANNUAL_DAYS), std,
                           out=np.full(batch, np.nan), where=std > 0)
        nav = np.cumprod(1.0 + x, axis=1)
        high = np.maximum.accumulate(np.c_[np.ones(batch), nav], axis=1)[:, 1:]
        drawdown = np.min(nav / high - 1.0, axis=1)
        gains = np.clip(x, 0, None).sum(axis=1)
        losses = -np.clip(x, None, 0).sum(axis=1)
        pf = np.divide(gains, losses, out=np.full(batch, np.nan), where=losses > 0)
        samples[base:base + batch] = np.column_stack((cagr, sharpe, drawdown, pf))
    joint3 = ((samples[:, 0] >= .40) & (samples[:, 1] > 1.5)
              & (samples[:, 2] >= -.20))
    joint4 = joint3 & (samples[:, 3] > 1.3)
    names = ("cagr", "sharpe", "max_drawdown", "profit_factor_daily")
    q05 = {name: (float(np.nanquantile(samples[:, i], .05))
                  if np.isfinite(samples[:, i]).any() else None)
           for i, name in enumerate(names)}
    return {"paths": paths, "block_days": block_days, "seed": seed,
            "joint_three_gate_frequency": float(joint3.mean()),
            "joint_four_gate_frequency": float(joint4.mean()),
            "positive_cagr_frequency": float((samples[:, 0] > 0).mean()),
            "p05": q05}


def _run_period(bundle: Any, *, start: pd.Timestamp, end: pd.Timestamp,
                fee_bps: float = FEE_BPS, slippage_bps: float = SLIPPAGE_BPS,
                monte_carlo_paths: int = 0) -> dict[str, Any]:
    """Internal identical accounting on a shorter synthetic period for tests."""
    if fee_bps < 0 or slippage_bps < 0:
        raise ValueError("negative fees/slippage are invalid")
    spot, trade, mark, funding = _prepare(bundle, start, end)
    decisions = _signals(spot, start, end)
    regimes = _regimes(spot["BTCUSDT"], start, end)
    lots: dict[str, list[Lot]] = {s: [] for s in SYMBOLS}
    cash = INITIAL_EQUITY
    accounting = {"funding_usd": 0.0, "taker_fees_usd": 0.0,
                  "slippage_usd": 0.0, "traded_notional_usd": 0.0}
    orders: list[dict[str, Any]] = []
    closed_lots: list[dict[str, Any]] = []
    bar_path: list[dict[str, Any]] = []
    nav: dict[pd.Timestamp, float] = {}
    exposure_sum: dict[pd.Timestamp, float] = {}
    cycles: list[dict[str, Any]] = []
    cycle_start: pd.Timestamp | None = None
    cycle_equity = 0.0
    events = sorted(((pd.Timestamp(t), s, float(r))
                     for s in SYMBOLS
                     for t, r in funding[s][["funding_time", "rate"]].itertuples(index=False, name=None)),
                    key=lambda row: (row[0], row[1]))
    cursor = 0
    cost_rate = (fee_bps + slippage_bps) / 10_000.0
    terminal = end - pd.Timedelta(hours=20)  # Last 04:00 before the Sep/Oct boundary.

    def quantity(symbol: str) -> float:
        return sum(lot.quantity for lot in lots[symbol])

    def marked_equity(time: pd.Timestamp, field: str = "open") -> float:
        return cash + sum(lot.quantity * (float(mark[s].at[time, field]) - lot.price)
                          for s in SYMBOLS for lot in lots[s])

    def gross_at(time: pd.Timestamp) -> float:
        return sum(abs(quantity(s) * float(mark[s].at[time, "open"])) for s in SYMBOLS)

    def pay_funding(stop: pd.Timestamp, *, inclusive: bool) -> None:
        nonlocal cursor, cash
        while cursor < len(events) and (events[cursor][0] <= stop if inclusive else events[cursor][0] < stop):
            event_time, symbol, rate = events[cursor]
            cursor += 1
            bar_time = event_time.floor("4h")
            mark_price = float(mark[symbol].at[bar_time, "open"])
            for lot in lots[symbol]:
                payment = -lot.quantity * mark_price * rate
                cash += payment
                lot.funding += payment
                accounting["funding_usd"] += payment

    def transact(time: pd.Timestamp, symbol: str, target: float) -> None:
        nonlocal cash
        old = quantity(symbol)
        delta = target - old
        if abs(delta) <= 1e-12:
            return
        price = float(trade[symbol].at[time, "open"])
        turnover = abs(delta) * price
        fee = turnover * fee_bps / 10_000.0
        slip = turnover * slippage_bps / 10_000.0
        cash -= fee + slip
        accounting["taker_fees_usd"] += fee
        accounting["slippage_usd"] += slip
        accounting["traded_notional_usd"] += turnover
        orders.append({"time": time.isoformat(), "symbol": symbol, "old_quantity": old,
                       "new_quantity": target, "trade_open": price,
                       "traded_notional_usd": turnover, "fee_usd": fee, "slippage_usd": slip})
        remaining = abs(delta)
        if old * delta < 0:
            while remaining > 1e-12 and lots[symbol]:
                lot = lots[symbol][0]
                closed_abs = min(abs(lot.quantity), remaining)
                signed_closed = np.sign(lot.quantity) * closed_abs
                ratio = closed_abs / abs(lot.quantity)
                pnl = signed_closed * (price - lot.price)
                entry_cost = lot.opening_cost * ratio
                alloc_funding = lot.funding * ratio
                exit_cost = closed_abs * price * cost_rate
                cash += pnl
                closed_lots.append({"entry_time": lot.opened.isoformat(),
                                    "exit_time": time.isoformat(), "symbol": symbol,
                                    "signed_quantity": float(signed_closed),
                                    "price_pnl_usd": float(pnl),
                                    "funding_usd": float(alloc_funding),
                                    "net_pnl_usd": float(pnl + alloc_funding - entry_cost - exit_cost)})
                lot.quantity -= signed_closed
                lot.opening_cost -= entry_cost
                lot.funding -= alloc_funding
                remaining -= closed_abs
                if abs(lot.quantity) <= 1e-12:
                    lots[symbol].pop(0)
        if remaining > 1e-12:
            new_quantity = float(np.sign(delta) * remaining)
            lots[symbol].append(Lot(new_quantity, price, time,
                                    opening_cost=remaining * price * cost_rate))

    for time in pd.date_range(start, end, freq="4h", inclusive="left"):
        pay_funding(time, inclusive=True)  # Exact-boundary settlement belongs to the old holding.
        day = time.floor("D")
        opening_equity = marked_equity(time)
        if time.hour == 0:
            nav[time] = opening_equity
        if time.hour == 4:
            if cycle_start is not None:
                # The final close cost belongs to the ending 04:00-to-04:00 cycle.
                if time == terminal:
                    for symbol in SYMBOLS:
                        transact(time, symbol, 0.0)
                    ending_equity = marked_equity(time)
                else:
                    ending_equity = opening_equity
                cycles.append({"start": cycle_start.isoformat(), "end": time.isoformat(),
                               "net_pnl_usd": ending_equity - cycle_equity})
                cycle_start = None
            if time in decisions and time != terminal:
                cycle_start = time
                cycle_equity = opening_equity
                long, short = decisions[time]
                targets = {s: 0.0 for s in SYMBOLS}
                targets[long] = .5 * opening_equity / float(trade[long].at[time, "open"])
                targets[short] = -.5 * opening_equity / float(trade[short].at[time, "open"])
                for symbol in SYMBOLS:
                    transact(time, symbol, targets[symbol])
        post_trade_cash = cash
        exposure_sum[day] = exposure_sum.get(day, 0.0) + (
            gross_at(time) / max(marked_equity(time), 1e-12)) / 6.0
        pay_funding(time + BAR, inclusive=False)
        # All long lows and short highs are treated as simultaneous; this is
        # an adverse bar-level drawdown bound, not a reconstructable tick path.
        adverse = min(post_trade_cash, cash)
        for symbol in SYMBOLS:
            for lot in lots[symbol]:
                field = "low" if lot.quantity > 0 else "high"
                adverse += lot.quantity * (float(mark[symbol].at[time, field]) - lot.price)
        close_equity = marked_equity(time, "close")
        bar_path.append({"time": time, "open_equity": opening_equity,
                         "adverse_equity": adverse, "close_equity": close_equity})
        if adverse <= 0 or close_equity <= 0:
            raise BacktestIntegrityError("possible account insolvency under intrabar adverse mark bound")
    if any(lots.values()):
        raise BacktestIntegrityError("nonflat position at frozen terminal boundary")
    nav[end] = cash
    nav_series = pd.Series(nav, name="equity_usd", dtype=float)
    daily_returns = nav_series.pct_change(fill_method=None).iloc[1:]
    daily_returns.index = daily_returns.index - DAY
    if len(daily_returns) != int((end - start) / DAY):
        raise BacktestIntegrityError("missing daily account return")
    daily_rows = [{"date": day.date().isoformat(), "start_equity_usd": float(nav_series.at[day]),
                   "end_equity_usd": float(nav_series.at[day + DAY]),
                   "net_return": float(value), "gross_exposure": float(exposure_sum.get(day, 0.0)),
                   "regime": regimes[day]}
                  for day, value in daily_returns.items()]
    cycle_values = np.asarray([row["net_pnl_usd"] for row in cycles], dtype=float)
    gain = float(cycle_values[cycle_values > 0].sum())
    loss = float(-cycle_values[cycle_values < 0].sum())
    cycle_pf = gain / loss if loss > 0 else None
    full_mdd = _intrabar_drawdown(bar_path, start, end)
    full = _period_metrics(daily_returns, intrabar_drawdown=full_mdd)
    if start < HOLDOUT_START < end:
        holdout_mdd = _intrabar_drawdown(bar_path, HOLDOUT_START, end)
        holdout = _period_metrics(daily_returns, HOLDOUT_START, end, holdout_mdd)
        holdout_returns = daily_returns.loc[daily_returns.index >= HOLDOUT_START]
    else:
        holdout = None
        holdout_returns = pd.Series(dtype=float)
    monthly_factors = (1.0 + daily_returns).groupby(daily_returns.index.tz_localize(None).to_period("M")).prod()
    months = {str(month): float(factor - 1.0) for month, factor in monthly_factors.items()}
    holdout_months = [value for month, value in months.items() if month >= "2024-01"]
    annual = {}
    for year in range(max(2022, start.year), min(2027, end.year + 1)):
        first = max(start, pd.Timestamp(f"{year}-01-01", tz="UTC"))
        last = min(end, pd.Timestamp(f"{year + 1}-01-01", tz="UTC"))
        if first >= last:
            continue
        metric = _period_metrics(daily_returns, first, last,
                                 _intrabar_drawdown(bar_path, first, last))
        metric["complete_calendar_year"] = last == pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
        annual[str(year)] = metric
    regime_report = {}
    for label in ("bull", "bear", "sideways", "insufficient_history"):
        idx = [day for day, state in regimes.items() if state == label]
        subset = daily_returns.loc[daily_returns.index.isin(idx)]
        regime_report[label] = {**_period_metrics(subset),
                                "mean_gross_exposure": (float(np.mean([exposure_sum.get(day, 0.0) for day in idx]))
                                                        if idx else None)}
    diagnostics: dict[str, Any] = {
        "full": full, "holdout": holdout, "annual": annual,
        "regimes": regime_report, "monthly_returns": months,
        "holdout_positive_month_fraction": (float(np.mean(np.array(holdout_months) > 0))
                                            if holdout_months else None),
        "holdout_worst_month": min(holdout_months) if holdout_months else None,
        "basket_cycle_profit_factor": cycle_pf,
        "basket_cycles": len(cycles), "accounting_usd": accounting,
        "evidence_grade": "preregistered retrospective holdout; not prospective independent OOS",
        "strict_v2_oos_years_satisfied": False,
        "strict_v2_five_complete_oos_folds_satisfied": False,
    }
    if monte_carlo_paths:
        if len(holdout_returns) < 30:
            raise BacktestIntegrityError("holdout too short for frozen Monte Carlo")
        diagnostics["monte_carlo_30d_10000"] = block_monte_carlo(
            holdout_returns, block_days=30, paths=monte_carlo_paths, seed=20261102)
        diagnostics["monte_carlo_v2_21d_10000"] = block_monte_carlo(
            holdout_returns, block_days=21, paths=monte_carlo_paths, seed=20261102)
    equity_rows = [{"time": time.isoformat(), "equity_usd": float(value)}
                   for time, value in nav_series.items()]
    return {"start": start.isoformat(), "end_exclusive": end.isoformat(),
            "initial_equity_usd": INITIAL_EQUITY,
            "fee_bps_per_side": fee_bps, "slippage_bps_per_side": slippage_bps,
            "daily": daily_rows, "equity": equity_rows, "orders": orders,
            "basket_cycles": cycles, "closed_lots": closed_lots,
            "diagnostics": diagnostics}


def run_backtest(bundle: Any, *, fee_bps: float = FEE_BPS,
                 slippage_bps: float = SLIPPAGE_BPS) -> dict[str, Any]:
    """Run the single frozen 1x experiment; official data are never loaded here."""
    if fee_bps != FEE_BPS or slippage_bps != SLIPPAGE_BPS:
        raise ValueError("formal rule fixes 5bp taker fee and 10bp slippage per side")
    return _run_period(bundle, start=START, end=END,
                       fee_bps=fee_bps, slippage_bps=slippage_bps,
                       monte_carlo_paths=10_000)
