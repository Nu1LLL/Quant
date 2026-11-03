"""Pure, research-only cost and latency arithmetic for a $10,000 account.

All inputs are explicit, nonnegative and point-in-time. The gross edge and its
uncertainty haircut must come from a separate, independently validated model;
this module cannot establish a confidence level, predict returns, route orders,
or assert that a historical bar was actually visible at the time shown.
Every price cost is expressed in basis points of the same position notional.
Funding is an adverse total over the proposed holding period. A favorable
funding estimate is not credited against the conservative cost floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation


ACCOUNT_EQUITY_USD = Decimal("10000")
MAX_GROSS_LEVERAGE = Decimal("5")
BASIS_POINTS = Decimal("10000")
YEAR_SECONDS = Decimal("31557600")  # 365.25 * 24 * 60 * 60
ZERO = Decimal("0")


def _nonnegative(value: Decimal | int | str, name: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (Decimal, int, str)):
        raise TypeError(f"{name} must be an explicit Decimal, int or decimal string")
    try:
        number = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} is not decimal") from exc
    if not number.is_finite() or number < ZERO:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def _utc(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be a timezone-aware UTC datetime")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must use UTC")
    return value.astimezone(timezone.utc)


def _seconds(delta: timedelta) -> Decimal:
    return (Decimal(delta.days * 86400 + delta.seconds)
            + Decimal(delta.microseconds) / Decimal("1000000"))


@dataclass(frozen=True)
class CausalTimeline:
    """Observed signal, entry and exit times; all eight are required."""

    signal_event_at: datetime
    signal_available_at: datetime
    decision_at: datetime
    entry_submitted_at: datetime
    entry_filled_at: datetime
    exit_decision_at: datetime
    exit_submitted_at: datetime
    exit_filled_at: datetime

    def __post_init__(self) -> None:
        names = (
            "signal_event_at", "signal_available_at", "decision_at",
            "entry_submitted_at", "entry_filled_at", "exit_decision_at",
            "exit_submitted_at", "exit_filled_at",
        )
        times = tuple(_utc(getattr(self, name), name) for name in names)
        for name, when in zip(names, times):
            object.__setattr__(self, name, when)
        if any(later < earlier for earlier, later in zip(times, times[1:])):
            raise ValueError("event, availability, decisions, submissions and fills must be causal")
        if self.exit_filled_at == self.entry_filled_at:
            raise ValueError("holding period must be positive")

    @property
    def entry_delay_seconds(self) -> Decimal:
        """Elapsed time from signal event to actual entry fill."""
        return _seconds(self.entry_filled_at - self.signal_event_at)

    @property
    def exit_delay_seconds(self) -> Decimal:
        return _seconds(self.exit_filled_at - self.exit_decision_at)

    @property
    def holding_seconds(self) -> Decimal:
        return _seconds(self.exit_filled_at - self.entry_filled_at)


@dataclass(frozen=True)
class CostInputs:
    """All entry/exit cost components must be supplied, including explicit zeroes.

    `entry_spread_bps` and `exit_spread_bps` are full quoted spreads; each
    crossing consumes half. Slippage and impact are additional adverse amounts
    beyond that half spread, so a source must avoid double counting them.
    Funding is the adverse expected cash flow over the entire holding period.
    Borrow APR is annual bps on the whole position notional.
    Delay drift rates are adverse bps per elapsed second.
    """

    entry_fee_bps: Decimal
    exit_fee_bps: Decimal
    entry_spread_bps: Decimal
    exit_spread_bps: Decimal
    entry_slippage_bps: Decimal
    exit_slippage_bps: Decimal
    entry_impact_bps: Decimal
    exit_impact_bps: Decimal
    funding_adverse_bps: Decimal
    borrow_apr_bps: Decimal
    entry_delay_adverse_bps_per_second: Decimal
    exit_delay_adverse_bps_per_second: Decimal

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))


@dataclass(frozen=True)
class CostBreakdown:
    entry_fee_bps: Decimal
    exit_fee_bps: Decimal
    entry_half_spread_bps: Decimal
    exit_half_spread_bps: Decimal
    entry_slippage_bps: Decimal
    exit_slippage_bps: Decimal
    entry_impact_bps: Decimal
    exit_impact_bps: Decimal
    funding_adverse_bps: Decimal
    borrow_bps: Decimal
    entry_delay_adverse_bps: Decimal
    exit_delay_adverse_bps: Decimal
    total_bps: Decimal


@dataclass(frozen=True)
class CostAssessment:
    """Arithmetic screen. Its values never authorize trading."""

    position_notional_usd: Decimal
    gross_leverage: Decimal
    gross_edge_point_bps: Decimal
    gross_edge_uncertainty_haircut_bps: Decimal
    gross_edge_lower_bps: Decimal
    cost: CostBreakdown
    net_edge_lower_bps: Decimal
    net_edge_lower_usd: Decimal
    net_edge_lower_pct_account: Decimal
    minimum_viable_gross_edge_bps: Decimal
    minimum_viable_gross_profit_usd: Decimal
    required_net_edge_bps: Decimal
    research_only: bool = field(default=True, init=False)
    deployable: bool = field(default=False, init=False)


def assess_cost_latency(
    *,
    timeline: CausalTimeline,
    costs: CostInputs,
    position_notional_usd: Decimal | int | str,
    gross_edge_point_bps: Decimal | int | str,
    gross_edge_uncertainty_haircut_bps: Decimal | int | str,
    required_net_edge_bps: Decimal | int | str,
) -> CostAssessment:
    """Decompose conservative round-trip costs and compute a net lower bound.

    The lower bound is conditional on the caller's independently justified
    uncertainty haircut, not a statistical confidence interval certified here.
    `minimum_viable_gross_edge_bps` is the point gross margin needed to cover
    all costs, uncertainty and the caller's required lower-bound net margin.
    """
    if not isinstance(timeline, CausalTimeline) or not isinstance(costs, CostInputs):
        raise TypeError("complete CausalTimeline and CostInputs are required")
    notional = _nonnegative(position_notional_usd, "position_notional_usd")
    if notional <= ZERO:
        raise ValueError("position_notional_usd must be positive")
    leverage = notional / ACCOUNT_EQUITY_USD
    if leverage > MAX_GROSS_LEVERAGE:
        raise ValueError("single-position notional exceeds the hard 5x account cap")
    point = _nonnegative(gross_edge_point_bps, "gross_edge_point_bps")
    haircut = _nonnegative(
        gross_edge_uncertainty_haircut_bps,
        "gross_edge_uncertainty_haircut_bps",
    )
    required = _nonnegative(required_net_edge_bps, "required_net_edge_bps")

    entry_half_spread = costs.entry_spread_bps / 2
    exit_half_spread = costs.exit_spread_bps / 2
    borrow = costs.borrow_apr_bps * timeline.holding_seconds / YEAR_SECONDS
    entry_delay = (costs.entry_delay_adverse_bps_per_second
                   * timeline.entry_delay_seconds)
    exit_delay = (costs.exit_delay_adverse_bps_per_second
                  * timeline.exit_delay_seconds)
    pieces = (
        costs.entry_fee_bps, costs.exit_fee_bps,
        entry_half_spread, exit_half_spread,
        costs.entry_slippage_bps, costs.exit_slippage_bps,
        costs.entry_impact_bps, costs.exit_impact_bps,
        costs.funding_adverse_bps, borrow, entry_delay, exit_delay,
    )
    total = sum(pieces, ZERO)
    breakdown = CostBreakdown(
        costs.entry_fee_bps, costs.exit_fee_bps,
        entry_half_spread, exit_half_spread,
        costs.entry_slippage_bps, costs.exit_slippage_bps,
        costs.entry_impact_bps, costs.exit_impact_bps,
        costs.funding_adverse_bps, borrow, entry_delay, exit_delay, total,
    )
    gross_lower = point - haircut
    net_lower = gross_lower - total
    minimum_gross = haircut + total + required
    return CostAssessment(
        position_notional_usd=notional,
        gross_leverage=leverage,
        gross_edge_point_bps=point,
        gross_edge_uncertainty_haircut_bps=haircut,
        gross_edge_lower_bps=gross_lower,
        cost=breakdown,
        net_edge_lower_bps=net_lower,
        net_edge_lower_usd=net_lower * notional / BASIS_POINTS,
        net_edge_lower_pct_account=(net_lower * notional / BASIS_POINTS
                                    / ACCOUNT_EQUITY_USD * 100),
        minimum_viable_gross_edge_bps=minimum_gross,
        minimum_viable_gross_profit_usd=minimum_gross * notional / BASIS_POINTS,
        required_net_edge_bps=required,
    )
