"""Point-in-time matching of crypto hypotheses to sealed evidence.

This module ranks *research-only* candidate/evidence pairs.  The evidence
fields are attestations supplied by a separately audited experiment, not a
way to manufacture proof; the return value can never authorize an order.
Ambiguous results for the same candidate fail closed to avoid cherry-picking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from math import isfinite
from typing import Iterable

from .archetypes import Candidate
from .schemas import CRYPTO_MARKETS, ObservationClassification, require_utc


MIN_HISTORY = timedelta(days=365 * 5)
MIN_OOS = timedelta(days=365 * 5)
MIN_TRADES = 30
MIN_WALK_FORWARD_YEARS = 5
MIN_MONTE_CARLO_PATHS = 5_000
MONTE_CARLO_BLOCK_DAYS = 21
CONFIDENCE_MULTIPLIER = 1.96


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty")


@dataclass(frozen=True)
class EvidenceRecord:
    """One immutable, preregistered result for one exact research candidate.

    ``independent_oos`` and other pass flags are *claims* from the sealed
    report. The matcher cannot independently verify the report or its data.
    The two commits must point to different stages, with preregistration
    preceding result inspection in the external audit trail.
    """

    evidence_id: str
    prereg_commit: str
    result_commit: str
    license_id: str
    market: str
    venue: str
    symbol: str
    archetype: str
    side: str
    horizon_minutes: int
    preregistered_at: datetime
    history_start: datetime
    history_end: datetime
    oos_start: datetime
    oos_end: datetime
    available_at: datetime
    expires_at: datetime
    mean_gross_bps: float
    roundtrip_cost_bps: float
    stderr_bps: float
    trades: int
    cagr: float
    sharpe: float
    max_drawdown: float  # Nonnegative loss magnitude, e.g. 0.18 for -18%.
    daily_profit_factor: float
    leverage: float
    walk_forward_years: int
    walk_forward_success_fraction: float
    monte_carlo_paths: int
    monte_carlo_block_days: int
    monte_carlo_joint_pass_fraction: float
    independent_oos: bool
    annual_walk_forward: bool
    walk_forward_passed: bool
    regimes_passed: bool
    block_monte_carlo_passed: bool
    data_integrity_passed: bool
    fees_slippage_included: bool
    funding_included_if_applicable: bool
    executable_for_10000_usd: bool

    def __post_init__(self) -> None:
        for name in ("evidence_id", "prereg_commit", "result_commit", "license_id",
                     "venue", "symbol"):
            _nonempty(getattr(self, name), name)
        if self.prereg_commit == self.result_commit:
            raise ValueError("preregistration and result commits must differ")
        if self.market not in CRYPTO_MARKETS:
            raise ValueError("evidence market must be crypto")
        if self.archetype not in {"flow", "rising_tide"}:
            raise ValueError("unsupported archetype")
        if self.side not in {"LONG", "SHORT"}:
            raise ValueError("evidence side must be LONG or SHORT")
        if type(self.horizon_minutes) is not int or self.horizon_minutes <= 0:
            raise ValueError("horizon_minutes must be positive")
        for name in ("trades", "walk_forward_years", "monte_carlo_paths",
                     "monte_carlo_block_days"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("preregistered_at", "history_start", "history_end", "oos_start", "oos_end",
                     "available_at", "expires_at"):
            require_utc(getattr(self, name), name)
        if not (self.history_start <= self.oos_start < self.oos_end
                <= self.history_end <= self.available_at < self.expires_at
                and self.preregistered_at <= self.available_at):
            raise ValueError("evidence timestamps are not causal")
        for name in ("mean_gross_bps", "roundtrip_cost_bps", "stderr_bps",
                     "cagr", "sharpe", "max_drawdown", "daily_profit_factor",
                     "leverage", "walk_forward_success_fraction",
                     "monte_carlo_joint_pass_fraction"):
            _finite(getattr(self, name), name)
        if self.roundtrip_cost_bps < 0 or self.stderr_bps < 0:
            raise ValueError("cost and stderr must be nonnegative")
        if not 0 <= self.max_drawdown <= 1 or self.daily_profit_factor < 0:
            raise ValueError("drawdown or profit factor outside range")
        if self.leverage <= 0:
            raise ValueError("leverage must be positive")
        if not (0 <= self.walk_forward_success_fraction <= 1
                and 0 <= self.monte_carlo_joint_pass_fraction <= 1):
            raise ValueError("pass fractions must be in [0, 1]")
        for name in ("independent_oos", "annual_walk_forward", "walk_forward_passed",
                     "regimes_passed",
                     "block_monte_carlo_passed", "data_integrity_passed",
                     "fees_slippage_included", "funding_included_if_applicable",
                     "executable_for_10000_usd"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be boolean")

    @property
    def match_key(self) -> tuple[str, str, str, str, str, int]:
        return (self.market, self.venue, self.symbol, self.archetype,
                self.side, self.horizon_minutes)


def reported_gate_failures(record: EvidenceRecord) -> tuple[str, ...]:
    """Apply the research gate to reported values; not an independent audit."""
    failed: list[str] = []
    if record.history_end - record.history_start < MIN_HISTORY:
        failed.append("history_under_five_years")
    if record.oos_end - record.oos_start < MIN_OOS:
        failed.append("oos_under_five_years")
    if record.preregistered_at > record.oos_start:
        failed.append("historical_holdout_not_prospective_oos")
    if record.trades < MIN_TRADES:
        failed.append("too_few_trades")
    if record.cagr < 0.40:
        failed.append("cagr_below_40_percent")
    if record.sharpe <= 1.5:
        failed.append("sharpe_not_above_1_5")
    if record.max_drawdown > 0.20:
        failed.append("drawdown_above_20_percent")
    if record.daily_profit_factor <= 1.3:
        failed.append("daily_profit_factor_not_above_1_3")
    if record.leverage != 1.0:
        failed.append("base_signal_not_tested_at_one_x")
    if record.walk_forward_years < MIN_WALK_FORWARD_YEARS:
        failed.append("annual_walk_forward_folds_not_sufficient")
    if (record.monte_carlo_paths < MIN_MONTE_CARLO_PATHS
            or record.monte_carlo_block_days != MONTE_CARLO_BLOCK_DAYS):
        failed.append("block_monte_carlo_protocol_not_passed")
    for field_name in ("independent_oos", "annual_walk_forward", "walk_forward_passed",
                       "regimes_passed",
                       "block_monte_carlo_passed", "data_integrity_passed",
                       "fees_slippage_included", "funding_included_if_applicable",
                       "executable_for_10000_usd"):
        if not getattr(record, field_name):
            failed.append(field_name + "_not_passed")
    return tuple(failed)


@dataclass(frozen=True)
class MatchDecision:
    """A non-executable ranking/audit record, even if every gate is reported passed."""

    candidate: Candidate
    status: str
    reasons: tuple[str, ...]
    evidence_id: str | None = None
    net_lower_bound_bps: float | None = None
    alert_label: str | None = None
    research_only: bool = field(default=True, init=False)
    deployable: bool = field(default=False, init=False)


def rank_shadow_candidates(
    candidates: Iterable[Candidate],
    evidence: Iterable[EvidenceRecord],
    decision_at: datetime,
    classifications: Iterable[ObservationClassification] = (),
) -> tuple[MatchDecision, ...]:
    """Rank exact candidate/evidence pairs without selecting a live order.

    Only one eligible sealed result per exact candidate is accepted.  This
    prevents ranking multiple backtests and quietly choosing their winner.
    Classification labels are audit context, not predictive features here.
    """
    require_utc(decision_at, "decision_at")
    rows = tuple(candidates)
    records = tuple(evidence)
    labels = tuple(classifications)
    for row in rows:
        if not isinstance(row, Candidate):
            raise TypeError("candidates must contain Candidate objects")
        if row.market not in CRYPTO_MARKETS or row.deployable or not row.research_only:
            raise ValueError("only research-only crypto candidates are accepted")
        if row.decision_at != decision_at:
            raise ValueError("candidate must be generated for the exact decision time")
    for record in records:
        if not isinstance(record, EvidenceRecord):
            raise TypeError("evidence must contain EvidenceRecord objects")
    for label in labels:
        if not isinstance(label, ObservationClassification):
            raise TypeError("classifications must contain ObservationClassification objects")
        if label.asof_time > decision_at:
            raise ValueError("future classification is unavailable")

    def latest_alert(row: Candidate) -> str | None:
        matches = (label for label in labels
                   if (label.market, label.venue, label.symbol)
                   == (row.market, row.venue, row.symbol))
        latest = max(matches, key=lambda label: label.asof_time, default=None)
        return latest.label if latest is not None else None

    output: list[MatchDecision] = []
    for row in rows:
        alert = latest_alert(row)
        if row.side == "FLAT":
            output.append(MatchDecision(row, "FLAT", (row.reason,), alert_label=alert))
            continue
        key = (row.market, row.venue, row.symbol, row.archetype,
               row.side, row.horizon_minutes)
        matching = [record for record in records if record.match_key == key
                    and record.available_at <= decision_at < record.expires_at]
        if not matching:
            output.append(MatchDecision(row, "NO_EVIDENCE",
                                        ("no_asof_sealed_result_for_exact_candidate",),
                                        alert_label=alert))
            continue
        if len(matching) > 1:
            output.append(MatchDecision(row, "AMBIGUOUS_EVIDENCE",
                                        ("multiple_results_require_prior_selection",),
                                        alert_label=alert))
            continue
        record = matching[0]
        failures = reported_gate_failures(record)
        if failures:
            output.append(MatchDecision(row, "GATE_FAILED", failures,
                                        evidence_id=record.evidence_id,
                                        alert_label=alert))
            continue
        lower_bound = (record.mean_gross_bps - record.roundtrip_cost_bps
                       - CONFIDENCE_MULTIPLIER * record.stderr_bps)
        if lower_bound <= 0:
            output.append(MatchDecision(row, "NO_POSITIVE_NET_EDGE",
                                        ("cost_adjusted_lower_bound_not_positive",),
                                        evidence_id=record.evidence_id,
                                        net_lower_bound_bps=lower_bound,
                                        alert_label=alert))
            continue
        output.append(MatchDecision(
            row, "SHADOW_RANKED", ("reported_gate_pass_requires_independent_audit",),
            evidence_id=record.evidence_id, net_lower_bound_bps=lower_bound,
            alert_label=alert,
        ))
    return tuple(sorted(
        output,
        key=lambda item: (
            item.status != "SHADOW_RANKED",
            -(item.net_lower_bound_bps if item.net_lower_bound_bps is not None else float("-inf")),
            item.candidate.venue, item.candidate.symbol, item.candidate.archetype,
        ),
    ))
