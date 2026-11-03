"""Deterministic, non-executable multi-strategy/multi-coin shadow allocation.

The allocation is an audit artifact for separately verified research. It
never places orders or supersedes the paper-account risk checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from math import isfinite
from typing import Iterable, Mapping

from .matching import MatchDecision
from .schemas import require_utc


@dataclass(frozen=True)
class AllocationConfig:
    max_symbol_gross: float = 0.20
    max_cluster_gross: float = 0.30
    max_portfolio_gross: float = 0.60

    def __post_init__(self) -> None:
        for name in ("max_symbol_gross", "max_cluster_gross", "max_portfolio_gross"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be finite in (0, 1]")


@dataclass(frozen=True)
class ShadowAllocation:
    market: str
    venue: str
    symbol: str
    cluster: str
    archetype: str
    side: str
    target_weight: float
    evidence_id: str
    net_lower_bound_bps: float


@dataclass(frozen=True)
class ShadowPortfolio:
    decision_at: datetime
    allocations: tuple[ShadowAllocation, ...]
    gross_weight: float
    cash_weight: float
    blocked: tuple[str, ...]
    research_only: bool = field(default=True, init=False)
    deployable: bool = field(default=False, init=False)


def allocate_shadow_portfolio(
    decisions: Iterable[MatchDecision],
    decision_at: datetime,
    *,
    cluster_by_symbol: Mapping[str, str] | None = None,
    config: AllocationConfig = AllocationConfig(),
) -> ShadowPortfolio:
    """Apply fixed diversification caps to reported-gate shadow rankings.

    If one instrument has opposing candidate directions, all candidates for
    that instrument are blocked. Same-direction duplicate archetypes are
    reduced to the best *predefined ranking* before allocation to avoid
    double counting one market exposure. Unknown symbols share ALT.
    """
    require_utc(decision_at, "decision_at")
    if not isinstance(config, AllocationConfig):
        raise TypeError("AllocationConfig is required")
    clusters = dict(cluster_by_symbol or {})
    if any(not isinstance(key, str) or not key or not isinstance(value, str) or not value
           for key, value in clusters.items()):
        raise ValueError("cluster mapping needs nonempty symbol and cluster")
    rows = tuple(decisions)
    for row in rows:
        if not isinstance(row, MatchDecision):
            raise TypeError("decisions must contain MatchDecision objects")
        if row.candidate.decision_at != decision_at or row.deployable or not row.research_only:
            raise ValueError("only same-time non-deployable decisions are accepted")

    eligible: list[MatchDecision] = []
    blocked: list[str] = []
    for row in rows:
        if (row.status != "SHADOW_RANKED" or row.net_lower_bound_bps is None
                or row.net_lower_bound_bps <= 0 or row.evidence_id is None):
            blocked.append(f"{row.candidate.symbol}:{row.status}")
        else:
            eligible.append(row)
    by_instrument: dict[tuple[str, str, str], list[MatchDecision]] = {}
    for row in eligible:
        candidate = row.candidate
        key = candidate.market, candidate.venue, candidate.symbol
        by_instrument.setdefault(key, []).append(row)

    chosen: list[MatchDecision] = []
    for key, same_instrument in by_instrument.items():
        if len({row.candidate.side for row in same_instrument}) > 1:
            blocked.append(f"{key[2]}:opposing_directions")
            continue
        same_instrument.sort(key=lambda row: (
            -row.net_lower_bound_bps, row.candidate.archetype, row.evidence_id,
        ))
        chosen.append(same_instrument[0])
        for duplicate in same_instrument[1:]:
            blocked.append(f"{duplicate.candidate.symbol}:duplicate_exposure")
    chosen.sort(key=lambda row: (
        -row.net_lower_bound_bps,
        row.candidate.venue, row.candidate.symbol, row.candidate.archetype,
    ))

    allocations: list[ShadowAllocation] = []
    total = 0.0
    cluster_totals: dict[str, float] = {}
    for row in chosen:
        candidate = row.candidate
        cluster = clusters.get(candidate.symbol, "ALT")
        desired = abs(candidate.target_weight)
        allowed = min(
            desired,
            config.max_symbol_gross,
            max(0.0, config.max_cluster_gross - cluster_totals.get(cluster, 0.0)),
            max(0.0, config.max_portfolio_gross - total),
        )
        if allowed <= 1e-12:
            blocked.append(f"{candidate.symbol}:diversification_cap")
            continue
        signed = allowed if candidate.side == "LONG" else -allowed
        allocations.append(ShadowAllocation(
            candidate.market, candidate.venue, candidate.symbol, cluster,
            candidate.archetype, candidate.side, signed,
            row.evidence_id, row.net_lower_bound_bps,
        ))
        total += allowed
        cluster_totals[cluster] = cluster_totals.get(cluster, 0.0) + allowed
    return ShadowPortfolio(
        decision_at, tuple(allocations), total, max(0.0, 1.0 - total),
        tuple(blocked),
    )
