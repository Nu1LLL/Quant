"""Deterministic, PAPER-ONLY accounting for linear USDT crypto perpetuals.

This module has no network, credentials, order-routing, or live-trading code.  It
accepts already timestamped observations and fills.  A fill is not an instruction
to an exchange.  Prices, fill quantities, commission, and funding are supplied by
the caller; missing information is never estimated as zero.

If a mark implies nonpositive bucket/account equity, the event fails closed and
is not recorded.  This is *not* a liquidation simulator: it cannot assert a
liquidation price, fill, insurance-fund outcome, or bounded loss without real
venue-specific order and margin events.

The instrument type and venue are mandatory, but a symbol string is not proof
that a contract is a cryptocurrency.  A separate point-in-time instrument
catalog must verify venue, contract type, quote currency, and listing status
before supplying any OrderIntent to this accounting module.

The three capital buckets are isolated for risk sizing.  The default reserve
bucket cannot open positions.  Equity is cash plus marked unrealized PnL, whereas
external deposits/withdrawals are recorded separately and excluded from returns.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import MappingProxyType
from typing import Mapping


ZERO = Decimal("0")
ONE = Decimal("1")
FIVE = Decimal("5")


def _decimal(value: Decimal | int | str, name: str) -> Decimal:
    # Floats quietly lose cents and would make the paper ledger irreproducible.
    if isinstance(value, bool) or not isinstance(value, (Decimal, int, str)):
        raise TypeError(f"{name} must be a finite Decimal, int, or decimal string")
    try:
        number = Decimal(value)
    except Exception as exc:
        raise ValueError(f"{name} is not decimal") from exc
    if not number.is_finite():
        raise ValueError(f"{name} must be finite")
    return number


def _utc(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware UTC")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must use UTC")
    return value.astimezone(timezone.utc)


def _positive(value: Decimal | int | str, name: str) -> Decimal:
    number = _decimal(value, name)
    if number <= ZERO:
        raise ValueError(f"{name} must be positive")
    return number


@dataclass(frozen=True)
class AccountConfig:
    initial_equity: Decimal = Decimal("10000")
    bucket_weights: Mapping[str, Decimal] = field(
        default_factory=lambda: {
            "core": Decimal("0.33"),
            "tactical": Decimal("0.33"),
            "reserve": Decimal("0.34"),
        }
    )
    reserve_bucket: str = "reserve"
    reserve_can_open: bool = False
    max_gross_leverage: Decimal = FIVE
    max_symbol_leverage: Decimal = ONE
    max_cluster_leverage: Decimal = Decimal("2")
    max_drawdown: Decimal = Decimal("0.20")
    fractional_kelly: Decimal = Decimal("0.25")
    max_other_mark_age: timedelta = timedelta(minutes=5)
    max_drawdown_mark_gap: timedelta = timedelta(hours=1)
    # Unlisted coins conservatively share one ALT cluster.  A higher-level
    # eligible-universe gate must separately exclude tokenized stocks/metals.
    symbol_clusters: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "initial_equity", _positive(self.initial_equity, "initial_equity"))
        weights = {str(k): _decimal(v, f"bucket_weights[{k}]") for k, v in self.bucket_weights.items()}
        if set(weights) != {"core", "tactical", "reserve"}:
            raise ValueError("exactly core, tactical, and reserve buckets are required")
        if any(v <= ZERO for v in weights.values()) or sum(weights.values(), ZERO) != ONE:
            raise ValueError("three bucket weights must be positive and sum exactly to 1")
        if self.reserve_bucket not in weights:
            raise ValueError("reserve_bucket must name an existing bucket")
        object.__setattr__(self, "bucket_weights", MappingProxyType(weights))
        for name in ("max_gross_leverage", "max_symbol_leverage", "max_cluster_leverage"):
            value = _positive(getattr(self, name), name)
            if value > FIVE:
                raise ValueError(f"{name} cannot exceed the hard 5x limit")
            object.__setattr__(self, name, value)
        drawdown = _decimal(self.max_drawdown, "max_drawdown")
        fraction = _decimal(self.fractional_kelly, "fractional_kelly")
        if not ZERO < drawdown <= Decimal("0.20"):
            raise ValueError("max_drawdown must be in (0, 0.20]")
        if not ZERO < fraction <= ONE:
            raise ValueError("fractional_kelly must be in (0, 1]")
        object.__setattr__(self, "max_drawdown", drawdown)
        object.__setattr__(self, "fractional_kelly", fraction)
        if not isinstance(self.max_other_mark_age, timedelta) or self.max_other_mark_age <= timedelta(0):
            raise ValueError("max_other_mark_age must be a positive timedelta")
        if (not isinstance(self.max_drawdown_mark_gap, timedelta)
                or not timedelta(0) < self.max_drawdown_mark_gap <= timedelta(hours=1)):
            raise ValueError("max_drawdown_mark_gap must be in (0, 1 hour]")
        clusters = {str(k): str(v) for k, v in self.symbol_clusters.items()}
        if any(not k or not v for k, v in clusters.items()):
            raise ValueError("symbol_clusters cannot contain empty symbols or clusters")
        object.__setattr__(self, "symbol_clusters", MappingProxyType(clusters))


@dataclass(frozen=True)
class OrderIntent:
    client_order_id: str
    timestamp: datetime
    symbol: str
    bucket: str
    cluster: str
    side: str
    reference_price: Decimal
    baseline_notional: Decimal
    p_win_lower: Decimal
    payoff_ratio_lower: Decimal
    model_error_haircut: Decimal
    requested_leverage: Decimal
    instrument_type: str
    venue: str

    def __post_init__(self) -> None:
        if not self.client_order_id or not isinstance(self.client_order_id, str):
            raise ValueError("client_order_id is required")
        object.__setattr__(self, "timestamp", _utc(self.timestamp, "intent timestamp"))
        if not self.symbol or not isinstance(self.symbol, str):
            raise ValueError("symbol is required")
        if self.instrument_type != "CRYPTO_LINEAR_USDT_PERP":
            raise ValueError("only CRYPTO_LINEAR_USDT_PERP instruments are accepted")
        if not isinstance(self.venue, str) or not self.venue.strip():
            raise ValueError("nonempty venue is required")
        if not self.cluster or not isinstance(self.cluster, str):
            raise ValueError("cluster is required")
        if self.side not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")
        for name in ("reference_price", "baseline_notional", "payoff_ratio_lower", "requested_leverage"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if self.requested_leverage > FIVE:
            raise ValueError("10x/20x and every leverage above 5x are prohibited")
        for name in ("p_win_lower", "model_error_haircut"):
            value = _decimal(getattr(self, name), name)
            if not ZERO <= value <= ONE:
                raise ValueError(f"{name} must be in [0, 1]")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class RiskDecision:
    allowed_notional: Decimal
    kelly_fraction: Decimal
    reason: str
    closing_notional: Decimal
    opening_notional: Decimal


@dataclass(frozen=True)
class Fill:
    timestamp: datetime
    intent: OrderIntent
    quantity: Decimal
    execution_price: Decimal
    commission: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", _utc(self.timestamp, "fill timestamp"))
        if self.timestamp < self.intent.timestamp:
            raise ValueError("fill cannot precede its signal")
        object.__setattr__(self, "quantity", _positive(self.quantity, "quantity"))
        object.__setattr__(self, "execution_price", _positive(self.execution_price, "execution_price"))
        fee = _decimal(self.commission, "commission")
        if fee < ZERO:
            raise ValueError("commission cannot be negative")
        object.__setattr__(self, "commission", fee)


@dataclass(frozen=True)
class Mark:
    timestamp: datetime
    symbol: str
    price: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", _utc(self.timestamp, "mark timestamp"))
        if not self.symbol:
            raise ValueError("symbol is required")
        object.__setattr__(self, "price", _positive(self.price, "mark price"))


@dataclass(frozen=True)
class Funding:
    timestamp: datetime
    symbol: str
    bucket: str
    amount: Decimal  # positive received, negative paid

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", _utc(self.timestamp, "funding timestamp"))
        if not self.symbol or not self.bucket:
            raise ValueError("funding symbol and bucket are required")
        object.__setattr__(self, "amount", _decimal(self.amount, "funding amount"))


@dataclass(frozen=True)
class ExternalFlow:
    timestamp: datetime
    bucket: str
    amount: Decimal  # deposit positive, withdrawal negative

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", _utc(self.timestamp, "cash-flow timestamp"))
        if not self.bucket:
            raise ValueError("cash-flow bucket is required")
        amount = _decimal(self.amount, "cash-flow amount")
        if amount == ZERO:
            raise ValueError("zero cash flow is not an event")
        object.__setattr__(self, "amount", amount)


@dataclass(frozen=True)
class InternalTransfer:
    timestamp: datetime
    source_bucket: str
    target_bucket: str
    amount: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", _utc(self.timestamp, "transfer timestamp"))
        if self.source_bucket == self.target_bucket:
            raise ValueError("transfer buckets must differ")
        object.__setattr__(self, "amount", _positive(self.amount, "transfer amount"))


@dataclass(frozen=True)
class LedgerEntry:
    timestamp: datetime
    kind: str
    amount: Decimal
    bucket: str | None = None
    symbol: str | None = None


@dataclass(frozen=True)
class ClosedTrade:
    timestamp: datetime
    symbol: str
    bucket: str
    quantity: Decimal
    closed_notional: Decimal
    holding_seconds: Decimal
    net_pnl: Decimal  # FIFO fragment, including allocated entry/exit fees and funding


@dataclass(frozen=True)
class FillRecord:
    timestamp: datetime
    symbol: str
    notional: Decimal
    latency_ms: Decimal
    adverse_slippage: Decimal


@dataclass(frozen=True)
class EquityPoint:
    timestamp: datetime
    equity: Decimal
    nav: Decimal
    external_flow: Decimal
    gross_notional: Decimal
    unrealized_pnl: Decimal
    valuation_fresh: bool


@dataclass(frozen=True)
class AccountSnapshot:
    timestamp: datetime
    paper_only: bool
    equity: Decimal
    valuation_fresh: bool
    cash: Decimal
    bucket_cash: Mapping[str, Decimal]
    unrealized_pnl: Decimal
    realized_price_pnl: Decimal
    commission_paid: Decimal
    funding_net: Decimal
    external_flows_net: Decimal
    gross_notional: Decimal
    gross_leverage: Decimal
    current_drawdown: Decimal
    max_drawdown: Decimal  # observed-event drawdown; not a tick/intrabar guarantee
    new_entries_halted: bool


@dataclass(frozen=True)
class Performance30d:
    as_of: datetime
    paper_only: bool
    covered_30d: bool
    reason: str
    time_weighted_return: Decimal | None
    profit_ex_external_flows: Decimal | None
    realized_net_pnl: Decimal | None
    unrealized_pnl_now: Decimal | None
    unrealized_pnl_change: Decimal | None
    win_rate: Decimal | None  # fraction of profitable closed FIFO fragments
    average_holding_seconds: Decimal | None  # exit-notional-weighted FIFO fragments
    traded_notional: Decimal | None
    max_drawdown: Decimal | None  # hourly-sampled only; None if position marks are sparse
    external_flows_net: Decimal | None
    closed_trade_count: int | None
    mean_latency_ms: Decimal | None
    adverse_slippage_total: Decimal | None


@dataclass
class _Lot:
    quantity: Decimal  # signed
    entry_price: Decimal
    opened_at: datetime
    cluster: str
    venue: str
    entry_fee_remaining: Decimal
    funding_remaining: Decimal = ZERO


class PaperAccount:
    """Event-sourced in-memory risk and PnL, with no execution capability."""

    paper_only = True

    def __init__(self, config: AccountConfig, start_time: datetime):
        if not isinstance(config, AccountConfig):
            raise TypeError("config must be AccountConfig")
        self.config = config
        start = _utc(start_time, "start_time")
        if start > datetime.now(timezone.utc):
            raise ValueError("future start_time is prohibited")
        self._cash = {b: config.initial_equity * w for b, w in config.bucket_weights.items()}
        self._lots: dict[tuple[str, str], list[_Lot]] = {}
        self._marks: dict[str, Decimal] = {}
        self._mark_times: dict[str, datetime] = {}
        self._journal: list[LedgerEntry] = []
        self._fills: list[FillRecord] = []
        self._closed: list[ClosedTrade] = []
        self._points: list[EquityPoint] = [
            EquityPoint(start, config.initial_equity, ONE, ZERO, ZERO, ZERO, True)
        ]
        self._seen_order_ids: set[str] = set()
        self._last_time = start
        self._nav = ONE
        self._peak_nav = ONE
        self._max_drawdown_seen = ZERO
        self._halt_new_entries = False

    @property
    def ledger(self) -> tuple[LedgerEntry, ...]:
        return tuple(self._journal)

    @property
    def equity_points(self) -> tuple[EquityPoint, ...]:
        return tuple(self._points)

    @property
    def closed_trades(self) -> tuple[ClosedTrade, ...]:
        return tuple(self._closed)

    def _check_time(self, timestamp: datetime) -> None:
        timestamp = _utc(timestamp, "event timestamp")
        if timestamp < self._last_time:
            raise ValueError("out-of-order event, including within the same day")
        if timestamp > datetime.now(timezone.utc):
            raise ValueError("future event is prohibited")

    def _position_qty(self, bucket: str, symbol: str) -> Decimal:
        return sum((lot.quantity for lot in self._lots.get((bucket, symbol), [])), ZERO)

    def _unrealized(self, price_override: Mapping[str, Decimal] | None = None) -> Decimal:
        total = ZERO
        override = price_override or {}
        for (_, symbol), lots in self._lots.items():
            if symbol not in self._marks and symbol not in override:
                raise ValueError(f"missing mark for open position {symbol}")
            mark = override.get(symbol, self._marks.get(symbol))
            total += sum((lot.quantity * (mark - lot.entry_price) for lot in lots), ZERO)
        return total

    def _equity(self, price_override: Mapping[str, Decimal] | None = None) -> Decimal:
        return sum(self._cash.values(), ZERO) + self._unrealized(price_override)

    def _bucket_equity(self, bucket: str, price_override: Mapping[str, Decimal] | None = None) -> Decimal:
        unrealized = ZERO
        override = price_override or {}
        for (lot_bucket, symbol), lots in self._lots.items():
            if lot_bucket == bucket:
                mark = override[symbol] if symbol in override else self._marks[symbol]
                unrealized += sum((lot.quantity * (mark - lot.entry_price) for lot in lots), ZERO)
        return self._cash[bucket] + unrealized

    def _gross(self, *, bucket: str | None = None, symbol: str | None = None,
               cluster: str | None = None,
               price_override: Mapping[str, Decimal] | None = None) -> Decimal:
        gross = ZERO
        override = price_override or {}
        for (lot_bucket, lot_symbol), lots in self._lots.items():
            if bucket is not None and lot_bucket != bucket:
                continue
            if symbol is not None and lot_symbol != symbol:
                continue
            if cluster is not None and lots[0].cluster != cluster:
                continue
            mark = override[lot_symbol] if lot_symbol in override else self._marks[lot_symbol]
            gross += abs(sum((lot.quantity for lot in lots), ZERO)) * mark
        return gross

    def _canonical_cluster(self, symbol: str) -> str:
        return self.config.symbol_clusters.get(symbol, "ALT")

    def _valuation_fresh(self, timestamp: datetime) -> bool:
        return all(self._mark_times.get(symbol) == timestamp for _, symbol in self._lots)

    def _assert_post_action_caps(self) -> None:
        """For voluntary risk additions/withdrawals, enforce caps after fees."""
        equity = self._equity()
        if equity <= ZERO or self._gross() > equity * self.config.max_gross_leverage:
            raise ValueError("post-action total leverage exceeds cap")
        for symbol in {symbol for _, symbol in self._lots}:
            if self._gross(symbol=symbol) > equity * self.config.max_symbol_leverage:
                raise ValueError("post-action symbol leverage exceeds cap")
        for cluster in {lots[0].cluster for lots in self._lots.values()}:
            if self._gross(cluster=cluster) > equity * self.config.max_cluster_leverage:
                raise ValueError("post-action correlated-cluster leverage exceeds cap")
        for bucket in self._cash:
            if self._gross(bucket=bucket) > self._bucket_equity(bucket) * FIVE:
                raise ValueError("post-action bucket leverage exceeds hard 5x cap")

    def assess_order(self, intent: OrderIntent) -> RiskDecision:
        """Return the maximum *paper* notional; never sends or reserves an order."""
        if not isinstance(intent, OrderIntent):
            raise TypeError("intent must be OrderIntent")
        self._check_time(intent.timestamp)
        if intent.bucket not in self._cash:
            raise ValueError("unknown bucket")
        if intent.cluster != self._canonical_cluster(intent.symbol):
            raise ValueError("cluster differs from configured canonical mapping")
        if intent.client_order_id in self._seen_order_ids:
            raise ValueError("client_order_id already filled")
        return self._risk_for_intent(intent, intent.reference_price, intent.timestamp)

    def _risk_for_intent(self, intent: OrderIntent, price: Decimal, observed_at: datetime) -> RiskDecision:
        if intent.requested_leverage > self.config.max_gross_leverage:
            raise ValueError("requested leverage exceeds configured or hard cap")
        for (bucket, symbol), lots in self._lots.items():
            if symbol == intent.symbol and lots[0].cluster != intent.cluster:
                raise ValueError("existing symbol uses another cluster")
            if symbol == intent.symbol and lots[0].venue != intent.venue:
                raise ValueError("existing symbol uses another venue")
        current_qty = self._position_qty(intent.bucket, intent.symbol)
        direction = ONE if intent.side == "BUY" else -ONE
        reducible = abs(current_qty) * price if current_qty * direction < ZERO else ZERO
        close_amt = min(intent.baseline_notional, reducible)
        open_request = intent.baseline_notional - close_amt
        raw_kelly = max(ZERO, intent.p_win_lower - (ONE - intent.p_win_lower) / intent.payoff_ratio_lower)
        kelly_fraction = raw_kelly * intent.model_error_haircut * self.config.fractional_kelly
        if open_request == ZERO:
            return RiskDecision(close_amt, kelly_fraction, "reduce_only", close_amt, ZERO)
        if intent.bucket == self.config.reserve_bucket and not self.config.reserve_can_open:
            return RiskDecision(close_amt, kelly_fraction, "reserve_bucket_no_new_entries", close_amt, ZERO)
        if self._halt_new_entries:
            return RiskDecision(close_amt, kelly_fraction, "drawdown_stop_new_entries", close_amt, ZERO)
        if any(
            observed_at - self._mark_times[symbol] > self.config.max_other_mark_age
            for _, symbol in self._lots if symbol != intent.symbol
        ):
            return RiskDecision(close_amt, kelly_fraction, "stale_valuation_no_new_entries", close_amt, ZERO)
        if kelly_fraction <= ZERO:
            return RiskDecision(close_amt, kelly_fraction, "no_conservative_kelly_edge", close_amt, ZERO)
        price_override = {intent.symbol: price}
        equity = self._equity(price_override)
        bucket_equity = self._bucket_equity(intent.bucket, price_override)
        if equity <= ZERO or bucket_equity <= ZERO:
            return RiskDecision(close_amt, kelly_fraction, "nonpositive_equity", close_amt, ZERO)
        total_after_close = self._gross(price_override=price_override) - close_amt
        symbol_after_close = self._gross(symbol=intent.symbol, price_override=price_override) - close_amt
        cluster_after_close = self._gross(cluster=intent.cluster, price_override=price_override) - close_amt
        bucket_after_close = self._gross(bucket=intent.bucket, price_override=price_override) - close_amt
        allowed_open = min(
            open_request,
            equity * kelly_fraction * intent.requested_leverage,
            max(ZERO, equity * self.config.max_gross_leverage - total_after_close),
            max(ZERO, equity * self.config.max_symbol_leverage - symbol_after_close),
            max(ZERO, equity * self.config.max_cluster_leverage - cluster_after_close),
            max(ZERO, bucket_equity * intent.requested_leverage - bucket_after_close),
        )
        allowed_open = max(ZERO, allowed_open)
        reason = "approved" if allowed_open == open_request else "risk_capped"
        return RiskDecision(close_amt + allowed_open, kelly_fraction, reason, close_amt, allowed_open)

    def _commit_point(self, timestamp: datetime, external_flow: Decimal = ZERO) -> None:
        new_equity = self._equity()
        if new_equity <= ZERO or any(self._bucket_equity(b) <= ZERO for b in self._cash):
            raise ValueError("nonpositive account or bucket equity; fail closed")
        previous = self._points[-1]
        pre_flow_equity = new_equity - external_flow
        if pre_flow_equity <= ZERO:
            raise ValueError("cash flow cannot conceal a nonpositive pre-flow account")
        new_nav = self._nav * pre_flow_equity / previous.equity
        if new_nav <= ZERO:
            raise ValueError("invalid nonpositive unitized NAV")
        self._nav = new_nav
        self._peak_nav = max(self._peak_nav, new_nav)
        current_dd = ONE - new_nav / self._peak_nav
        self._max_drawdown_seen = max(self._max_drawdown_seen, current_dd)
        if current_dd >= self.config.max_drawdown:
            self._halt_new_entries = True
        self._points.append(
            EquityPoint(timestamp, new_equity, new_nav, external_flow, self._gross(),
                        self._unrealized(), self._valuation_fresh(timestamp))
        )
        self._last_time = timestamp

    def _atomic(self, timestamp: datetime, apply, external_flow: Decimal = ZERO) -> None:
        self._check_time(timestamp)
        old = (
            deepcopy(self._cash), deepcopy(self._lots), dict(self._marks), dict(self._mark_times),
            len(self._journal), len(self._fills), len(self._closed), len(self._points),
            set(self._seen_order_ids), self._last_time, self._nav,
            self._peak_nav, self._max_drawdown_seen, self._halt_new_entries,
        )
        try:
            apply()
            self._commit_point(timestamp, external_flow)
        except Exception:
            (self._cash, self._lots, self._marks, self._mark_times, journal_len, fills_len, closed_len,
             points_len, self._seen_order_ids, self._last_time, self._nav,
             self._peak_nav, self._max_drawdown_seen, self._halt_new_entries) = old
            del self._journal[journal_len:]
            del self._fills[fills_len:]
            del self._closed[closed_len:]
            del self._points[points_len:]
            raise

    def record_fill(self, fill: Fill) -> None:
        """Account for a supplied paper fill, including FIFO closes and fees."""
        if not isinstance(fill, Fill):
            raise TypeError("fill must be Fill")
        self._check_time(fill.timestamp)
        intent = fill.intent
        if intent.bucket not in self._cash or intent.cluster != self._canonical_cluster(intent.symbol):
            raise ValueError("invalid bucket or noncanonical cluster")
        if intent.client_order_id in self._seen_order_ids:
            raise ValueError("duplicate client_order_id")
        executed_notional = fill.quantity * fill.execution_price
        if executed_notional > intent.baseline_notional:
            raise ValueError("execution exceeds 1x baseline notional")
        decision = self._risk_for_intent(intent, fill.execution_price, fill.timestamp)
        if executed_notional > decision.allowed_notional:
            raise ValueError(f"fill exceeds risk approval: {decision.reason}")
        direction = ONE if intent.side == "BUY" else -ONE
        adverse = direction * (fill.execution_price - intent.reference_price) * fill.quantity
        latency_ms = Decimal(str((fill.timestamp - intent.timestamp).total_seconds())) * Decimal("1000")

        def apply() -> None:
            key = (intent.bucket, intent.symbol)
            lots = self._lots.setdefault(key, [])
            remaining = fill.quantity
            price_pnl = ZERO
            fee_per_unit = fill.commission / fill.quantity
            while remaining > ZERO and lots and lots[0].quantity * direction < ZERO:
                lot = lots[0]
                prior_abs = abs(lot.quantity)
                matched = min(remaining, prior_abs)
                matched_fraction = matched / prior_abs
                entry_fee = lot.entry_fee_remaining * matched_fraction
                lot_funding = lot.funding_remaining * matched_fraction
                close_fee = fee_per_unit * matched
                pnl = matched * (fill.execution_price - lot.entry_price) * (ONE if lot.quantity > ZERO else -ONE)
                price_pnl += pnl
                self._closed.append(
                    ClosedTrade(
                        fill.timestamp, intent.symbol, intent.bucket, matched,
                        matched * fill.execution_price,
                        Decimal(str((fill.timestamp - lot.opened_at).total_seconds())),
                        pnl - entry_fee - close_fee + lot_funding,
                    )
                )
                lot.entry_fee_remaining -= entry_fee
                lot.funding_remaining -= lot_funding
                lot.quantity += direction * matched
                remaining -= matched
                if lot.quantity == ZERO:
                    lots.pop(0)
            if remaining > ZERO:
                lots.append(_Lot(direction * remaining, fill.execution_price, fill.timestamp,
                                 intent.cluster, intent.venue, fee_per_unit * remaining))
            if not lots:
                del self._lots[key]
            self._cash[intent.bucket] += price_pnl - fill.commission
            self._marks[intent.symbol] = fill.execution_price
            self._mark_times[intent.symbol] = fill.timestamp
            if remaining > ZERO:
                self._assert_post_action_caps()
                if self._gross(bucket=intent.bucket) > self._bucket_equity(intent.bucket) * intent.requested_leverage:
                    raise ValueError("post-fee bucket leverage exceeds requested limit")
            self._journal.append(LedgerEntry(fill.timestamp, "REALIZED_PRICE_PNL", price_pnl, intent.bucket, intent.symbol))
            self._journal.append(LedgerEntry(fill.timestamp, "COMMISSION", -fill.commission, intent.bucket, intent.symbol))
            self._fills.append(FillRecord(fill.timestamp, intent.symbol, executed_notional, latency_ms, adverse))
            self._seen_order_ids.add(intent.client_order_id)

        self._atomic(fill.timestamp, apply)

    def record_mark(self, mark: Mark) -> None:
        if not isinstance(mark, Mark):
            raise TypeError("mark must be Mark")

        def apply() -> None:
            self._marks[mark.symbol] = mark.price
            self._mark_times[mark.symbol] = mark.timestamp
            self._journal.append(LedgerEntry(mark.timestamp, "MARK", ZERO, symbol=mark.symbol))

        self._atomic(mark.timestamp, apply)

    def record_funding(self, funding: Funding) -> None:
        if not isinstance(funding, Funding):
            raise TypeError("funding must be Funding")
        if funding.bucket not in self._cash:
            raise ValueError("unknown bucket")
        key = (funding.bucket, funding.symbol)
        if key not in self._lots:
            raise ValueError("funding requires an open position; no fabricated cash PnL")

        def apply() -> None:
            lots = self._lots[key]
            total_qty = sum((abs(lot.quantity) for lot in lots), ZERO)
            for lot in lots:
                lot.funding_remaining += funding.amount * abs(lot.quantity) / total_qty
            self._cash[funding.bucket] += funding.amount
            self._journal.append(LedgerEntry(funding.timestamp, "FUNDING", funding.amount,
                                             funding.bucket, funding.symbol))

        self._atomic(funding.timestamp, apply)

    def record_external_flow(self, flow: ExternalFlow) -> None:
        if not isinstance(flow, ExternalFlow):
            raise TypeError("flow must be ExternalFlow")
        if flow.bucket not in self._cash:
            raise ValueError("unknown bucket")

        def apply() -> None:
            if self._cash[flow.bucket] + flow.amount <= ZERO:
                raise ValueError("cash flow would exhaust bucket cash")
            self._cash[flow.bucket] += flow.amount
            if flow.amount < ZERO:
                self._assert_post_action_caps()
            self._journal.append(LedgerEntry(flow.timestamp, "EXTERNAL_FLOW", flow.amount, flow.bucket))

        self._atomic(flow.timestamp, apply, flow.amount)

    def record_internal_transfer(self, transfer: InternalTransfer) -> None:
        if not isinstance(transfer, InternalTransfer):
            raise TypeError("transfer must be InternalTransfer")
        if transfer.source_bucket not in self._cash or transfer.target_bucket not in self._cash:
            raise ValueError("unknown bucket")

        def apply() -> None:
            if self._cash[transfer.source_bucket] - transfer.amount <= ZERO:
                raise ValueError("transfer would exhaust source cash")
            self._cash[transfer.source_bucket] -= transfer.amount
            self._cash[transfer.target_bucket] += transfer.amount
            self._assert_post_action_caps()
            self._journal.append(LedgerEntry(transfer.timestamp, "INTERNAL_TRANSFER_OUT", -transfer.amount,
                                             transfer.source_bucket))
            self._journal.append(LedgerEntry(transfer.timestamp, "INTERNAL_TRANSFER_IN", transfer.amount,
                                             transfer.target_bucket))

        self._atomic(transfer.timestamp, apply)

    def snapshot(self) -> AccountSnapshot:
        equity = self._equity()
        current_dd = ONE - self._nav / self._peak_nav
        return AccountSnapshot(
            timestamp=self._last_time,
            paper_only=True,
            equity=equity,
            valuation_fresh=self._valuation_fresh(self._last_time),
            cash=sum(self._cash.values(), ZERO),
            bucket_cash=MappingProxyType(dict(self._cash)),
            unrealized_pnl=self._unrealized(),
            realized_price_pnl=sum((e.amount for e in self._journal if e.kind == "REALIZED_PRICE_PNL"), ZERO),
            commission_paid=-sum((e.amount for e in self._journal if e.kind == "COMMISSION"), ZERO),
            funding_net=sum((e.amount for e in self._journal if e.kind == "FUNDING"), ZERO),
            external_flows_net=sum((e.amount for e in self._journal if e.kind == "EXTERNAL_FLOW"), ZERO),
            gross_notional=self._gross(),
            gross_leverage=self._gross() / equity,
            current_drawdown=current_dd,
            max_drawdown=self._max_drawdown_seen,
            new_entries_halted=self._halt_new_entries,
        )

    def report_30d(self, as_of: datetime) -> Performance30d:
        """Exact 30-day endpoint PnL; drawdown only with hourly full-book marks.

        Even with that coverage, drawdown is sampled hourly, not the true tick
        or intrabar maximum.  Sparse valuation refuses the drawdown field.
        """
        as_of = _utc(as_of, "as_of")
        if as_of != self._last_time:
            raise ValueError("as_of must equal the latest recorded event time")
        boundary = as_of - timedelta(days=30)
        start_point = next((p for p in reversed(self._points) if p.timestamp == boundary), None)
        if start_point is None or not start_point.valuation_fresh or not self._points[-1].valuation_fresh:
            reason = "exact_30d_boundary_missing" if start_point is None else "stale_open_position_mark"
            return Performance30d(
                as_of=as_of, paper_only=True, covered_30d=False,
                reason=reason,
                time_weighted_return=None, profit_ex_external_flows=None,
                realized_net_pnl=None, unrealized_pnl_now=None,
                unrealized_pnl_change=None, win_rate=None,
                average_holding_seconds=None, traded_notional=None,
                max_drawdown=None, external_flows_net=None,
                closed_trade_count=None, mean_latency_ms=None,
                adverse_slippage_total=None,
            )
        current = self._points[-1]
        entries = [e for e in self._journal if boundary < e.timestamp <= as_of]
        closes = [e for e in self._closed if boundary < e.timestamp <= as_of]
        fills = [e for e in self._fills if boundary < e.timestamp <= as_of]
        points = [p for p in self._points if boundary <= p.timestamp <= as_of]
        fresh_points = [p for p in points if p.valuation_fresh]
        drawdown_covered = all(
            # A flat-to-open interval had no exposure before the opening fill.
            # An open-to-flat interval did, so it still needs intraperiod marks.
            previous.gross_notional == ZERO
            or current_point.timestamp - previous.timestamp <= self.config.max_drawdown_mark_gap
            for previous, current_point in zip(fresh_points, fresh_points[1:])
        )
        ext_flow = sum((e.amount for e in entries if e.kind == "EXTERNAL_FLOW"), ZERO)
        realized_net = sum((e.amount for e in entries if e.kind in {
            "REALIZED_PRICE_PNL", "COMMISSION", "FUNDING"
        }), ZERO)
        win_rate = (Decimal(sum(e.net_pnl > ZERO for e in closes)) / Decimal(len(closes))) if closes else None
        total_closed_notional = sum((e.closed_notional for e in closes), ZERO)
        avg_holding = (
            sum((e.holding_seconds * e.closed_notional for e in closes), ZERO) / total_closed_notional
        ) if closes else None
        max_dd: Decimal | None = None
        if drawdown_covered:
            maximum = fresh_points[0].nav
            max_dd = ZERO
            for point in fresh_points:
                maximum = max(maximum, point.nav)
                max_dd = max(max_dd, ONE - point.nav / maximum)
        return Performance30d(
            as_of=as_of, paper_only=True, covered_30d=True,
            reason="ok_hourly_sampled_drawdown" if drawdown_covered else "sparse_intraperiod_marks_drawdown_unknown",
            time_weighted_return=current.nav / start_point.nav - ONE,
            profit_ex_external_flows=current.equity - start_point.equity - ext_flow,
            realized_net_pnl=realized_net,
            unrealized_pnl_now=current.unrealized_pnl,
            unrealized_pnl_change=current.unrealized_pnl - start_point.unrealized_pnl,
            win_rate=win_rate, average_holding_seconds=avg_holding,
            traded_notional=sum((e.notional for e in fills), ZERO),
            max_drawdown=max_dd, external_flows_net=ext_flow,
            closed_trade_count=len(closes),
            mean_latency_ms=(sum((e.latency_ms for e in fills), ZERO) / len(fills)) if fills else None,
            adverse_slippage_total=sum((e.adverse_slippage for e in fills), ZERO),
        )
