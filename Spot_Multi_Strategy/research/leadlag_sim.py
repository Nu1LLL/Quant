"""Frozen ten-coin delayed lead-lag model and five-minute account replay.

The only model is the preregistered monthly ridge fit. There is no runtime
parameter optimizer, future-label access, order router or live account hook.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
import math
from statistics import median

import numpy as np

from .btc_eth_pairs_sim import DayRecord
from .leadlag_data import PriceTape, SYMBOLS


UTC = timezone.utc
BASE = datetime(2020, 12, 1, tzinfo=UTC)
FORMAL_START = datetime(2021, 1, 1, tzinfo=UTC)
FORMAL_END = datetime(2026, 1, 1, tzinfo=UTC)
BAR = timedelta(minutes=5)
FORMAL_FIRST = 31 * 288
FORMAL_DAYS = 1826
EXPECTED_BARS = FORMAL_FIRST + FORMAL_DAYS * 288
INITIAL_EQUITY = 10_000.0
TAKER_FEE = 0.0005
PREDICTED_SPREAD_MIN = 0.004
RIDGE_PENALTY = 100.0
SCENARIOS = (
    ("base_1x", 1.0, 0.0005, 1),
    ("base_2x", 2.0, 0.0005, 1),
    ("double_slippage_2x", 2.0, 0.0010, 1),
    ("latency_2x", 2.0, 0.0005, 2),
)


class LeadLagSimulationError(ValueError):
    pass


class LeadLagStrategyFailure(LeadLagSimulationError):
    pass


@dataclass(frozen=True, slots=True)
class _Model:
    feature_mean: np.ndarray
    feature_sd: np.ndarray
    intercept: np.ndarray
    coefficients: np.ndarray
    training_rows: int

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.feature_mean) / self.feature_sd
        result = self.intercept + z @ self.coefficients
        if not np.all(np.isfinite(result)):
            raise LeadLagSimulationError("nonfinite cross-asset model prediction")
        return result


@dataclass(frozen=True, slots=True)
class LeadLagTrade:
    entry_signal_time: datetime
    entry_fill_time: datetime
    exit_fill_time: datetime
    long_symbol: str
    short_symbol: str
    regime: str
    predicted_spread: float
    net_pnl: float
    gross_price_pnl: float
    fee_pnl: float
    funding_pnl: float = 0.0


@dataclass(frozen=True, slots=True)
class SimulationResult:
    daily_records: tuple[DayRecord, ...]
    trades: tuple[LeadLagTrade, ...]
    max_drawdown: float
    final_equity: float
    decision_opportunities: int
    monthly_fits: int
    min_training_rows: int


@dataclass
class _Account:
    name: str
    gross: float
    adverse_rate: float
    entry_delay_bars: int
    cash: float = INITIAL_EQUITY
    peak: float = INITIAL_EQUITY
    max_drawdown: float = 0.0
    last_regular: float = INITIAL_EQUITY
    day: date | None = None
    day_start: float = INITIAL_EQUITY
    day_regular_peak: float = INITIAL_EQUITY
    day_max_regular_factor: float = 1.0
    day_min_adverse_factor: float = 1.0
    day_max_drawdown: float = 0.0
    daily_records: list[DayRecord] = field(default_factory=list)
    trades: list[LeadLagTrade] = field(default_factory=list)

    def _account(self, regular: float, adverse: float) -> None:
        if not math.isfinite(regular + adverse) or min(regular, adverse) <= 0:
            raise LeadLagStrategyFailure(f"{self.name}: nonpositive five-minute equity")
        self.last_regular = regular
        self.peak = max(self.peak, regular)
        self.max_drawdown = max(self.max_drawdown, 1 - adverse / self.peak)
        if self.day is not None:
            self.day_regular_peak = max(self.day_regular_peak, regular)
            self.day_max_regular_factor = max(self.day_max_regular_factor,
                                              regular / self.day_start)
            self.day_min_adverse_factor = min(self.day_min_adverse_factor,
                                              adverse / self.day_start)
            self.day_max_drawdown = max(self.day_max_drawdown,
                                        1 - adverse / self.day_regular_peak)

    def start_day(self, day: date) -> None:
        if self.day is not None:
            self._record_day()
        if not math.isclose(self.last_regular, self.cash, rel_tol=1e-12, abs_tol=1e-8):
            raise LeadLagSimulationError("position spans a UTC day boundary")
        self.day = day
        self.day_start = self.cash
        self.day_regular_peak = self.cash
        self.day_max_regular_factor = 1.0
        self.day_min_adverse_factor = 1.0
        self.day_max_drawdown = 0.0

    def _record_day(self) -> None:
        if self.day is None:
            raise LeadLagSimulationError("day not started")
        self.daily_records.append(DayRecord(
            self.day, self.day_start, self.cash,
            self.day_max_regular_factor, self.day_min_adverse_factor,
            self.day_max_drawdown,
        ))

    def finish_day(self) -> None:
        self._record_day()

    def execute(self, tape: PriceTape, decision_index: int, long_index: int,
                short_index: int, regime: str, predicted_spread: float) -> None:
        if (self.day is None or long_index == short_index or self.cash <= 0
                or self.entry_delay_bars not in (1, 2)):
            raise LeadLagSimulationError("invalid short-hold entry")
        pre = self.cash
        entry = decision_index + self.entry_delay_bars
        exit_index = decision_index + 3
        if entry >= exit_index or exit_index >= len(tape.opening):
            raise LeadLagSimulationError("trade exceeds fixed data interval")
        entry_time = BASE + BAR * entry
        exit_time = BASE + BAR * exit_index
        if (entry_time.replace(minute=0) != exit_time.replace(minute=0)
                or entry_time.minute < 5 or exit_time.minute > 45):
            raise LeadLagSimulationError("trade overlaps possible funding-settlement hour")
        opens = tape.opening
        long_entry = float(opens[entry, long_index]) * (1 + self.adverse_rate)
        short_entry = float(opens[entry, short_index]) * (1 - self.adverse_rate)
        long_quantity = pre * self.gross / 2 / long_entry
        short_quantity = pre * self.gross / 2 / short_entry
        entry_fee = TAKER_FEE * (long_quantity * long_entry + short_quantity * short_entry)
        self.cash -= entry_fee

        def marks(long_price: float, short_price: float) -> float:
            return (self.cash + long_quantity * (long_price - long_entry)
                    + short_quantity * (short_entry - short_price))

        # Capture immediate fee/slippage before the first held bar closes.
        self._account(marks(float(opens[entry, long_index]),
                            float(opens[entry, short_index])),
                      marks(float(opens[entry, long_index]),
                            float(opens[entry, short_index])))
        for bar in range(entry, exit_index):
            regular = marks(float(tape.close[bar, long_index]),
                            float(tape.close[bar, short_index]))
            adverse = marks(float(tape.low[bar, long_index]),
                            float(tape.high[bar, short_index]))
            self._account(regular, adverse)
        long_exit = float(opens[exit_index, long_index]) * (1 - self.adverse_rate)
        short_exit = float(opens[exit_index, short_index]) * (1 + self.adverse_rate)
        price_pnl = (long_quantity * (long_exit - long_entry)
                     + short_quantity * (short_entry - short_exit))
        exit_fee = TAKER_FEE * (long_quantity * long_exit + short_quantity * short_exit)
        self.cash += price_pnl - exit_fee
        self._account(self.cash, self.cash)
        decision_time = BASE + BAR * decision_index
        self.trades.append(LeadLagTrade(
            decision_time, entry_time, exit_time,
            SYMBOLS[long_index], SYMBOLS[short_index], regime,
            predicted_spread, self.cash - pre, price_pnl,
            -entry_fee - exit_fee,
        ))

    def result(self, opportunities: int, fits: int, min_rows: int) -> SimulationResult:
        return SimulationResult(tuple(self.daily_records), tuple(self.trades),
                                self.max_drawdown, self.cash, opportunities,
                                fits, min_rows)


def _monthly_model(opening: np.ndarray, returns: np.ndarray, cutoff: int) -> _Model:
    indices = np.arange(cutoff - 30 * 288, cutoff - 3, dtype=np.int64)
    if len(indices) < 8000 or indices[0] < 2 or indices[-1] + 3 >= cutoff:
        raise LeadLagSimulationError("monthly training/embargo window incomplete")
    x = returns[indices - 1]
    y = np.log(opening[indices + 3] / opening[indices + 1])
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise LeadLagSimulationError("nonfinite monthly training sample")
    mean = np.mean(x, axis=0)
    sd = np.std(x, axis=0, ddof=0)
    if not np.all(np.isfinite(sd)) or np.any(sd <= 0):
        raise LeadLagSimulationError("zero or nonfinite feature SD")
    z = (x - mean) / sd
    intercept = np.mean(y, axis=0)
    coefficients = np.zeros((len(SYMBOLS), len(SYMBOLS)), dtype=np.float64)
    for target in range(len(SYMBOLS)):
        other = [j for j in range(len(SYMBOLS)) if j != target]
        predictors = z[:, other]
        gram = predictors.T @ predictors + RIDGE_PENALTY * np.eye(len(other))
        rhs = predictors.T @ (y[:, target] - intercept[target])
        coefficients[other, target] = np.linalg.solve(gram, rhs)
    if not np.all(np.isfinite(coefficients)):
        raise LeadLagSimulationError("nonfinite monthly ridge coefficient")
    return _Model(mean, sd, intercept, coefficients, len(indices))


def _rolling_volatility(returns: np.ndarray) -> np.ndarray:
    common = np.mean(returns, axis=1)
    if not np.all(np.isfinite(common[1:])):
        raise LeadLagSimulationError("nonfinite equal-weight five-minute return")
    sums = np.concatenate(([0.0], np.cumsum(common[1:])))
    squares = np.concatenate(([0.0], np.cumsum(common[1:] ** 2)))
    index = np.arange(288, len(common), dtype=np.int64)
    total = sums[index] - sums[index - 288]
    sq = squares[index] - squares[index - 288]
    variance = np.maximum(0.0, (sq - total * total / 288) / 287)
    result = np.full(len(common), np.nan, dtype=np.float64)
    result[index] = np.sqrt(variance)
    return result


def run_leadlag_simulation(tape: PriceTape) -> dict[str, SimulationResult]:
    if (tape.first_open_ms != int(BASE.timestamp() * 1000)
            or len(tape.opening) != EXPECTED_BARS):
        raise LeadLagSimulationError("fixed five-year price tape is incomplete")
    returns = np.empty_like(tape.close)
    returns[0] = np.nan
    returns[1:] = np.log(tape.close[1:] / tape.close[:-1])
    volatility = _rolling_volatility(returns)
    accounts = {name: _Account(name, gross, adverse, delay)
                for name, gross, adverse, delay in SCENARIOS}
    model: _Model | None = None
    fits = 0
    min_training_rows = 10**9
    opportunities = 0
    for day_number in range(FORMAL_DAYS):
        day_index = FORMAL_FIRST + day_number * 288
        day = (FORMAL_START + timedelta(days=day_number)).date()
        if day.day == 1:
            model = _monthly_model(tape.opening, returns, day_index)
            fits += 1
            min_training_rows = min(min_training_rows, model.training_rows)
        if model is None:
            raise LeadLagSimulationError("no fitted model at formal decision")
        snapshot = day_index - np.arange(30, 0, -1) * 288 + 1
        previous_vol = volatility[snapshot - 1]
        if not np.all(np.isfinite(previous_vol)):
            raise LeadLagSimulationError("thirty prior daily volatility snapshots unavailable")
        threshold = median(float(x) for x in previous_vol)
        for account in accounts.values():
            account.start_day(day)
        for hour in range(24):
            for quarter in (0, 3, 6):
                i = day_index + 12 * hour + quarter
                opportunities += 1
                predicted = model.predict(returns[i - 1])
                ranking = sorted(range(len(SYMBOLS)),
                                 key=lambda k: (float(predicted[k]), SYMBOLS[k]))
                short_index, long_index = ranking[0], ranking[-1]
                spread = float(predicted[long_index] - predicted[short_index])
                if spread <= PREDICTED_SPREAD_MIN:
                    continue
                prior_close = tape.close[i - 1 - 30 * 288]
                current_close = tape.close[i - 1]
                trend = float(np.mean(np.log(current_close / prior_close)))
                sigma = float(volatility[i - 1])
                if not math.isfinite(trend + sigma):
                    raise LeadLagSimulationError("nonfinite trade-entry market state")
                regime = ("bull" if trend >= 0 else "bear") + ("_high" if sigma >= threshold else "_low")
                for account in accounts.values():
                    account.execute(tape, i, long_index, short_index, regime, spread)
    for account in accounts.values():
        account.finish_day()
    if opportunities != 72 * FORMAL_DAYS or fits != 60:
        raise LeadLagSimulationError("wrong number of decisions or monthly fits")
    if any(len(account.daily_records) != FORMAL_DAYS for account in accounts.values()):
        raise LeadLagSimulationError("missing five-year daily equity path")
    return {name: account.result(opportunities, fits, min_training_rows)
            for name, account in accounts.items()}
