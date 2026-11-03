"""Point-in-time, research-only crypto archetype target proposals.

These are deterministic hypotheses inspired by *descriptions* of two trading
styles, not implementations of a third-party product or evidence of alpha.
They consume already-available features and return proposed portfolio weights;
they never fetch data, place orders, or enable live trading.  A separate,
point-in-time instrument catalog must establish that each input really is a
crypto instrument.  The ``market`` guard here is a second, not sole, check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from math import isfinite
from numbers import Real
from typing import Iterable, Literal

from .schemas import CRYPTO_MARKETS


Side = Literal["LONG", "SHORT", "FLAT"]


def _utc(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be a timezone-aware UTC datetime")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be a timezone-aware UTC datetime")
    return value.astimezone(timezone.utc)


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _range(value: float, name: str, low: float, high: float) -> None:
    number = _finite(value, name)
    if not low <= number <= high:
        raise ValueError(f"{name} must be within [{low}, {high}]")


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _crypto_identity(market: str, venue: str, symbol: str) -> None:
    if market not in CRYPTO_MARKETS:
        raise ValueError("only crypto instruments are permitted")
    if not isinstance(venue, str) or not venue.strip():
        raise ValueError("venue must be nonempty")
    if (not isinstance(symbol, str) or not symbol.strip()
            or any(character.isspace() for character in symbol)):
        raise ValueError("symbol must be nonempty and contain no whitespace")


@dataclass(frozen=True)
class ObservedValue:
    """Feature value with the instant it became usable, not its bar start."""

    value: float
    available_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _finite(self.value, "feature value"))
        object.__setattr__(self, "available_at", _utc(self.available_at, "available_at"))


def _seen(feature: ObservedValue, decision_at: datetime,
          max_age_minutes: int, name: str) -> float:
    if not isinstance(feature, ObservedValue):
        raise TypeError(f"{name} must be an ObservedValue")
    if feature.available_at > decision_at:
        raise ValueError(f"{name} is not available at decision time")
    if decision_at - feature.available_at > timedelta(minutes=max_age_minutes):
        raise ValueError(f"{name} is stale at decision time")
    return feature.value


@dataclass(frozen=True)
class Candidate:
    """A target proposal, never an order or a deployable trading signal."""

    archetype: Literal["flow", "rising_tide"]
    market: str  # Canonical schemas.CRYPTO_MARKETS value.
    venue: str
    symbol: str
    side: Side
    target_weight: float  # Signed fraction of current account equity; no leverage.
    horizon_minutes: int
    reason: str
    theme: str | None = None
    entry_layers: int = 0
    decision_at: datetime | None = None
    research_only: bool = field(default=True, init=False)
    deployable: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        _crypto_identity(self.market, self.venue, self.symbol)
        if self.archetype not in {"flow", "rising_tide"}:
            raise ValueError("unknown research archetype")
        if self.side not in {"LONG", "SHORT", "FLAT"}:
            raise ValueError("unknown side")
        _range(self.target_weight, "target_weight", -1.0, 1.0)
        if ((self.side == "LONG" and self.target_weight <= 0)
                or (self.side == "SHORT" and self.target_weight >= 0)
                or (self.side == "FLAT" and self.target_weight != 0)):
            raise ValueError("side and target_weight are inconsistent")
        if self.market == "crypto_spot" and self.side == "SHORT":
            raise ValueError("spot short requires a separate borrow model")
        if self.market == "crypto_options" and self.side != "FLAT":
            raise ValueError("options require a separate payoff model")
        _positive_int(self.horizon_minutes, "horizon_minutes")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("candidate requires an audit reason")
        if self.decision_at is not None:
            object.__setattr__(self, "decision_at", _utc(self.decision_at, "decision_at"))
        if isinstance(self.entry_layers, bool) or not isinstance(self.entry_layers, int):
            raise ValueError("entry_layers must be an integer")
        if not 0 <= self.entry_layers <= 3:
            raise ValueError("entry_layers must be between zero and three")
        if self.archetype == "flow" and self.entry_layers:
            raise ValueError("flow has no layered entry")


@dataclass(frozen=True)
class FlowSnapshot:
    symbol: str
    venue: str
    market: str
    ofi: ObservedValue  # Normalized order-flow imbalance in [-1, 1].
    spread_bps: ObservedValue
    anchor_deviation_bps: ObservedValue  # Mid minus trailing anchor, in bps.
    volatility_bps: ObservedValue  # Realized risk over the intended horizon.
    liquidity_usd: ObservedValue  # Point-in-time executable depth, not 24h volume.


@dataclass(frozen=True)
class FlowConfig:
    horizon_minutes: int = 60
    max_feature_age_minutes: int = 5
    min_abs_ofi: float = 0.25
    min_abs_anchor_deviation_bps: float = 15.0
    max_spread_bps: float = 5.0
    max_volatility_bps: float = 150.0
    min_liquidity_usd: float = 500_000.0
    reversion_capture_fraction: float = 0.25
    roundtrip_fee_bps: float = 8.0
    roundtrip_slippage_bps: float = 4.0
    funding_buffer_bps: float = 1.0
    min_net_edge_bps: float = 2.0
    risk_budget_bps: float = 20.0
    volatility_floor_bps: float = 1.0
    max_abs_weight: float = 0.25

    def __post_init__(self) -> None:
        _positive_int(self.horizon_minutes, "horizon_minutes")
        _positive_int(self.max_feature_age_minutes, "max_feature_age_minutes")
        _range(self.min_abs_ofi, "min_abs_ofi", 0.0, 1.0)
        _range(self.reversion_capture_fraction, "reversion_capture_fraction", 0.0, 1.0)
        _range(self.max_abs_weight, "max_abs_weight", 0.0, 1.0)
        for name in ("min_abs_anchor_deviation_bps", "max_spread_bps",
                     "max_volatility_bps", "min_liquidity_usd",
                     "roundtrip_fee_bps", "roundtrip_slippage_bps",
                     "funding_buffer_bps", "min_net_edge_bps", "risk_budget_bps"):
            _range(getattr(self, name), name, 0.0, float("inf"))
        _range(self.volatility_floor_bps, "volatility_floor_bps", 1e-12, float("inf"))


def flow_candidate(snapshot: FlowSnapshot, decision_at: datetime,
                   config: FlowConfig = FlowConfig()) -> Candidate:
    """Propose a short-horizon reversal only when flow opposes dislocation.

    Gross edge is an explicitly assumed capture fraction of the observed
    dislocation, **not** an estimated expected return.  It exists solely to
    screen obvious transaction-cost failures before formal OOS testing.
    """

    if not isinstance(snapshot, FlowSnapshot) or not isinstance(config, FlowConfig):
        raise TypeError("FlowSnapshot and FlowConfig are required")
    _crypto_identity(snapshot.market, snapshot.venue, snapshot.symbol)
    at = _utc(decision_at, "decision_at")
    features = {
        name: _seen(getattr(snapshot, name), at, config.max_feature_age_minutes, name)
        for name in ("ofi", "spread_bps", "anchor_deviation_bps",
                     "volatility_bps", "liquidity_usd")
    }
    ofi = features["ofi"]
    spread = features["spread_bps"]
    deviation = features["anchor_deviation_bps"]
    volatility = features["volatility_bps"]
    liquidity = features["liquidity_usd"]
    _range(ofi, "ofi", -1.0, 1.0)
    for name in ("spread_bps", "volatility_bps", "liquidity_usd"):
        _range(features[name], name, 0.0, float("inf"))

    def flat(reason: str) -> Candidate:
        return Candidate("flow", snapshot.market, snapshot.venue, snapshot.symbol,
                         "FLAT", 0.0, config.horizon_minutes, reason,
                         decision_at=at)

    if liquidity < config.min_liquidity_usd:
        return flat("insufficient_point_in_time_liquidity")
    if spread > config.max_spread_bps:
        return flat("spread_limit")
    if volatility > config.max_volatility_bps:
        return flat("volatility_limit")
    if abs(deviation) < config.min_abs_anchor_deviation_bps:
        return flat("anchor_dislocation_too_small")
    if abs(ofi) < config.min_abs_ofi or deviation * ofi >= 0:
        return flat("flow_does_not_confirm_reversal")
    assumed_gross_bps = abs(deviation) * config.reversion_capture_fraction
    estimated_cost_bps = (spread + config.roundtrip_fee_bps
                          + config.roundtrip_slippage_bps
                          + config.funding_buffer_bps)
    if assumed_gross_bps - estimated_cost_bps <= config.min_net_edge_bps:
        return flat("assumed_edge_does_not_clear_costs")
    side: Side = "LONG" if deviation < 0 else "SHORT"
    if side == "SHORT" and snapshot.market == "crypto_spot":
        return flat("spot_short_requires_separate_borrow_and_cost_model")
    if snapshot.market == "crypto_options":
        return flat("options_require_separate_payoff_and_cost_model")
    weight = min(config.max_abs_weight,
                 config.risk_budget_bps / max(volatility, config.volatility_floor_bps))
    if weight <= 0:
        return flat("position_budget_zero")
    weight = weight if side == "LONG" else -weight
    return Candidate("flow", snapshot.market, snapshot.venue, snapshot.symbol,
                     side, weight, config.horizon_minutes,
                     "opposing_flow_and_dislocation_cost_screen_passed",
                     decision_at=at)


@dataclass(frozen=True)
class RisingSnapshot:
    symbol: str
    venue: str
    market: str
    theme: str
    theme_available_at: datetime
    strength: ObservedValue  # Frozen upstream, point-in-time score in [0, 1].
    beta: ObservedValue  # Trailing beta to a crypto benchmark.
    spread_bps: ObservedValue
    liquidity_usd: ObservedValue
    volatility_bps: ObservedValue
    stress_score: ObservedValue  # Frozen upstream state score in [0, 1].


@dataclass(frozen=True)
class RisingConfig:
    horizon_minutes: int = 720
    max_feature_age_minutes: int = 60
    min_strength: float = 0.65
    second_layer_strength: float = 0.80
    third_layer_strength: float = 0.90
    min_beta: float = 1.10
    max_beta: float = 4.0
    min_liquidity_usd: float = 1_000_000.0
    max_spread_bps: float = 15.0
    max_volatility_bps: float = 600.0
    max_stress_score: float = 0.40
    max_assets: int = 4
    max_per_theme: int = 2
    max_weight_per_asset: float = 0.20
    max_theme_weight: float = 0.30
    max_total_weight: float = 0.60
    entry_layer_fractions: tuple[float, float, float] = (0.50, 0.30, 0.20)

    def __post_init__(self) -> None:
        for name in ("horizon_minutes", "max_feature_age_minutes",
                     "max_assets", "max_per_theme"):
            _positive_int(getattr(self, name), name)
        for name in ("min_strength", "second_layer_strength",
                     "third_layer_strength", "max_stress_score",
                     "max_weight_per_asset", "max_theme_weight", "max_total_weight"):
            _range(getattr(self, name), name, 0.0, 1.0)
        if not (self.min_strength <= self.second_layer_strength
                <= self.third_layer_strength):
            raise ValueError("layer strength thresholds must be ordered")
        _range(self.min_beta, "min_beta", 0.0, float("inf"))
        _range(self.max_beta, "max_beta", self.min_beta, float("inf"))
        for name in ("min_liquidity_usd", "max_spread_bps", "max_volatility_bps"):
            _range(getattr(self, name), name, 0.0, float("inf"))
        fractions = self.entry_layer_fractions
        if not isinstance(fractions, tuple) or len(fractions) != 3:
            raise ValueError("entry_layer_fractions must be a three-item tuple")
        for fraction in fractions:
            _range(fraction, "entry layer fraction", 0.0, 1.0)
        if sum(fractions) > 1.0 + 1e-12 or fractions[0] <= 0:
            raise ValueError("entry layers must have a positive first slice and total at most one")


def rising_tide_candidates(snapshots: Iterable[RisingSnapshot],
                           decision_at: datetime,
                           config: RisingConfig = RisingConfig()) -> tuple[Candidate, ...]:
    """Rank crypto strength, then allocate capped, layered long-only targets.

    One proposal is returned per input symbol, including explicit FLATs.  All
    eligibility features and theme labels must be known at ``decision_at``.
    The input is not an alpha claim; strength/beta/stress construction and
    universe selection must be preregistered and independently validated.
    """

    if not isinstance(config, RisingConfig):
        raise TypeError("RisingConfig is required")
    at = _utc(decision_at, "decision_at")
    rows = tuple(snapshots)
    seen_symbols: set[str] = set()
    eligible: list[tuple[RisingSnapshot, float, int]] = []
    results: dict[str, Candidate] = {}

    def flat(row: RisingSnapshot, reason: str) -> Candidate:
        return Candidate("rising_tide", row.market, row.venue, row.symbol,
                         "FLAT", 0.0, config.horizon_minutes, reason,
                         theme=row.theme, decision_at=at)

    for row in rows:
        if not isinstance(row, RisingSnapshot):
            raise TypeError("all snapshots must be RisingSnapshot")
        _crypto_identity(row.market, row.venue, row.symbol)
        key = row.symbol.casefold()
        if key in seen_symbols:
            raise ValueError("duplicate crypto symbol")
        seen_symbols.add(key)
        if not isinstance(row.theme, str) or not row.theme.strip():
            raise ValueError("theme must be nonempty")
        if _utc(row.theme_available_at, "theme_available_at") > at:
            raise ValueError("theme is not available at decision time")
        features = {
            name: _seen(getattr(row, name), at, config.max_feature_age_minutes, name)
            for name in ("strength", "beta", "spread_bps", "liquidity_usd",
                         "volatility_bps", "stress_score")
        }
        for name in ("strength", "stress_score"):
            _range(features[name], name, 0.0, 1.0)
        for name in ("beta", "spread_bps", "liquidity_usd", "volatility_bps"):
            _range(features[name], name, 0.0, float("inf"))
        reason = None
        if row.market == "crypto_options":
            reason = "options_require_separate_payoff_and_cost_model"
        elif features["liquidity_usd"] < config.min_liquidity_usd:
            reason = "insufficient_point_in_time_liquidity"
        elif features["spread_bps"] > config.max_spread_bps:
            reason = "spread_limit"
        elif features["volatility_bps"] > config.max_volatility_bps:
            reason = "volatility_limit"
        elif features["stress_score"] > config.max_stress_score:
            reason = "stress_state"
        elif not config.min_beta <= features["beta"] <= config.max_beta:
            reason = "outside_beta_band"
        elif features["strength"] < config.min_strength:
            reason = "insufficient_strength"
        if reason is not None:
            results[key] = flat(row, reason)
            continue
        layers = 1 + int(features["strength"] >= config.second_layer_strength)
        layers += int(features["strength"] >= config.third_layer_strength)
        eligible.append((row, features["strength"], layers))

    eligible.sort(key=lambda item: (-item[1], item[0].symbol.casefold()))
    total_weight = 0.0
    theme_weight: dict[str, float] = {}
    theme_count: dict[str, int] = {}
    selected_count = 0
    for row, _, layers in eligible:
        key = row.symbol.casefold()
        theme_key = row.theme.casefold()
        if selected_count >= config.max_assets:
            results[key] = flat(row, "portfolio_asset_limit")
            continue
        if theme_count.get(theme_key, 0) >= config.max_per_theme:
            results[key] = flat(row, "theme_asset_limit")
            continue
        layer_fraction = sum(config.entry_layer_fractions[:layers])
        desired = config.max_weight_per_asset * layer_fraction
        weight = min(desired,
                     config.max_theme_weight - theme_weight.get(theme_key, 0.0),
                     config.max_total_weight - total_weight)
        if weight <= 1e-12:
            results[key] = flat(row, "theme_or_portfolio_weight_limit")
            continue
        results[key] = Candidate(
            "rising_tide", row.market, row.venue, row.symbol, "LONG", weight,
            config.horizon_minutes, "strength_beta_liquidity_and_risk_gates_passed",
            theme=row.theme, entry_layers=layers, decision_at=at,
        )
        total_weight += weight
        theme_weight[theme_key] = theme_weight.get(theme_key, 0.0) + weight
        theme_count[theme_key] = theme_count.get(theme_key, 0) + 1
        selected_count += 1

    return tuple(results[row.symbol.casefold()] for row in rows)
