"""Divergence, three buy/sell classes and interval nesting for Task 1."""
from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from chan_full_structure import (
    Center, Leg, Segment, leg_momentum, momentum_diverges,
)


Element = Leg | Segment


@dataclass(frozen=True)
class Divergence:
    level: int
    mode: str  # trend or consolidation
    direction: int
    entry_leg: int
    exit_leg: int
    start_idx: int
    end_idx: int
    confirmed_at: int
    entry_energy: tuple[float, float]
    exit_energy: tuple[float, float]


@dataclass(frozen=True)
class BuySellPoint:
    level: int
    point_class: int
    side: int  # 1 buy, -1 sell
    element_idx: int
    price: float
    confirmed_at: int
    reference_price: float
    held_reference: bool


@dataclass(frozen=True)
class NestedSignal:
    side: int
    outer_level: int
    inner_level: int
    start_idx: int
    end_idx: int
    confirmed_at: int
    depth: int


def _element_price(element: Element, direction: int) -> float:
    return element.high if direction > 0 else element.low


def detect_divergences(elements: Sequence[Element], centers: Sequence[Center],
                       macd_frame: pd.DataFrame, level: int) -> list[Divergence]:
    """Compare only structure-aligned legs on the two sides of a center."""
    elements, centers = list(elements), list(centers)
    if not centers:
        return []
    candidates: list[tuple[str, Center, int]] = []
    trend_center_positions = set()
    run_direction = 0
    for pos, (left_center, right_center) in enumerate(zip(centers, centers[1:]), 1):
        if right_center.zd > left_center.zg:
            pair_direction = 1
        elif right_center.zg < left_center.zd:
            pair_direction = -1
        else:
            pair_direction = 0
        if pair_direction == 0:
            run_direction = 0
            continue
        if run_direction not in (0, pair_direction):
            run_direction = pair_direction
        else:
            run_direction = pair_direction
        candidates.append(("trend", right_center, pair_direction))
        trend_center_positions.add(pos)

    # A lone/non-trending center may still have consolidation divergence, but
    # it must stay labelled separately and can never create a class-1 point.
    for pos, center in enumerate(centers):
        if pos in trend_center_positions:
            continue
        left, right = center.start_leg - 1, center.end_leg + 1
        if left >= 0 and right < len(elements):
            direction = elements[right].direction
            if direction == elements[left].direction:
                candidates.append(("consolidation", center, direction))

    result = []
    for mode, center, direction in candidates:
        entry_idx, exit_idx = center.start_leg - 1, center.end_leg + 1
        if entry_idx < 0 or exit_idx >= len(elements):
            continue
        entry, exit_ = elements[entry_idx], elements[exit_idx]
        if entry.direction != direction or exit_.direction != direction:
            continue
        before = leg_momentum(macd_frame, entry.start.source_idx,
                              entry.end.source_idx, direction)
        after = leg_momentum(macd_frame, exit_.start.source_idx,
                             exit_.end.source_idx, direction)
        if momentum_diverges(_element_price(entry, direction),
                             _element_price(exit_, direction), before, after,
                             direction):
            result.append(Divergence(
                level, mode, direction, entry_idx, exit_idx,
                entry.start.source_idx, exit_.end.source_idx,
                int(exit_.confirmed_at), before, after,
            ))
    return result


def first_and_second_points(elements: Sequence[Element],
                            divergences: Sequence[Divergence],
                            level: int) -> list[BuySellPoint]:
    """Trend divergence gives class 1; first completed pullback gives class 2."""
    elements = list(elements)
    result = []
    for divergence in divergences:
        if divergence.mode != "trend":
            continue
        first_element = elements[divergence.exit_leg]
        side = -divergence.direction
        first_price = (first_element.end.price)
        result.append(BuySellPoint(level, 1, side, divergence.exit_leg,
                                   first_price, divergence.confirmed_at,
                                   first_price, True))
        pullback_idx = divergence.exit_leg + 2
        if pullback_idx >= len(elements):
            continue
        pullback = elements[pullback_idx]
        expected = -side  # buy's pullback is down; sell's is up
        if pullback.direction != expected:
            continue
        price = pullback.end.price
        held = price >= first_price if side > 0 else price <= first_price
        result.append(BuySellPoint(level, 2, side, pullback_idx, price,
                                   int(pullback.confirmed_at), first_price,
                                   held))
    return result


def third_points(elements: Sequence[Element], centers: Sequence[Center],
                 level: int) -> list[BuySellPoint]:
    """Departure plus completed retest that stays outside the center."""
    elements = list(elements)
    result = []
    for center in centers:
        for departure_idx in range(center.end_leg + 1, len(elements) - 1):
            departure = elements[departure_idx]
            retest = elements[departure_idx + 1]
            if departure.direction > 0 and departure.high > center.zg:
                if retest.direction < 0 and retest.low > center.zg:
                    result.append(BuySellPoint(
                        level, 3, 1, departure_idx + 1, retest.end.price,
                        int(retest.confirmed_at), center.zg, True,
                    ))
                    break
            if departure.direction < 0 and departure.low < center.zd:
                if retest.direction > 0 and retest.high < center.zd:
                    result.append(BuySellPoint(
                        level, 3, -1, departure_idx + 1, retest.end.price,
                        int(retest.confirmed_at), center.zd, True,
                    ))
                    break
    return result


def interval_nesting(divergences: Sequence[Divergence]) -> list[NestedSignal]:
    """Return maximal same-side chains whose intervals are geometrically nested."""
    events = sorted(divergences, key=lambda x: (-x.level, x.start_idx, x.end_idx))
    result = []
    for outer in events:
        chain = [outer]
        current = outer
        for inner in sorted(events, key=lambda x: -x.level):
            if inner.level >= current.level or inner.direction != current.direction:
                continue
            if (inner.start_idx >= current.start_idx
                    and inner.end_idx <= current.end_idx):
                chain.append(inner)
                current = inner
        if len(chain) > 1:
            result.append(NestedSignal(
                -outer.direction, outer.level, chain[-1].level,
                chain[-1].start_idx, chain[-1].end_idx,
                max(item.confirmed_at for item in chain), len(chain),
            ))
    return result
