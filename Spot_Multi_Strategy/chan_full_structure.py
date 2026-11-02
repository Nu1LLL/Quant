"""Causal structural core for Task 1's full Chan-theory experiment.

This module deliberately keeps interpretation choices explicit.  Objects are
immutable after emission and carry the first bar at which they were knowable.
It contains no portfolio logic and makes no claim that the structures are alpha.
"""
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from alphas import chan_theory


@dataclass(frozen=True)
class Point:
    source_idx: int
    kind: int  # 1 top, -1 bottom
    price: float
    confirmed_at: int


@dataclass(frozen=True)
class Leg:
    start: Point
    end: Point
    level: int = 0

    @property
    def direction(self):
        return 1 if self.end.price > self.start.price else -1

    @property
    def low(self):
        return min(self.start.price, self.end.price)

    @property
    def high(self):
        return max(self.start.price, self.end.price)

    @property
    def confirmed_at(self):
        return self.end.confirmed_at


@dataclass(frozen=True)
class Feature:
    start_pos: int
    end_pos: int
    low: float
    high: float
    confirmed_at: int


@dataclass(frozen=True)
class Segment:
    start: Point
    end: Point
    start_pos: int
    end_pos: int
    confirmed_at: int
    gap_case: bool
    level: int = 1

    @property
    def direction(self):
        return 1 if self.end.price > self.start.price else -1

    @property
    def low(self):
        return min(self.start.price, self.end.price)

    @property
    def high(self):
        return max(self.start.price, self.end.price)


@dataclass(frozen=True)
class Center:
    level: int
    start_leg: int
    end_leg: int
    zg: float
    zd: float
    gg: float
    dd: float
    confirmed_at: int


def confirmed_stroke_points(frame: pd.DataFrame) -> list[Point]:
    """Return causally confirmed stroke endpoints from adjusted OHLC."""
    high, low, start = chan_theory.merge_inclusive_bars(frame)
    fractals = chan_theory.detect_fractals(high, low, start)
    strokes = chan_theory.build_strokes(fractals)
    # Stroke tuples identify a merged-bar position.  Momentum alignment needs
    # the corresponding original-bar position, not that compressed ordinal.
    return [Point(int(start[int(i)]), int(k), float(p), int(c))
            for i, k, p, c in strokes]


def _validate_points(points: Sequence[Point]):
    for left, right in zip(points, points[1:]):
        if right.source_idx <= left.source_idx:
            raise ValueError("Point source indices must increase")
        if right.kind == left.kind:
            raise ValueError("Points must alternate top and bottom")
        if right.confirmed_at < left.confirmed_at:
            raise ValueError("Confirmation times must not go backwards")


def legs_from_points(points: Sequence[Point], level=0) -> list[Leg]:
    _validate_points(points)
    return [Leg(a, b, level) for a, b in zip(points, points[1:])]


def _included(a: Feature, b: Feature) -> bool:
    return ((a.low <= b.low and a.high >= b.high)
            or (b.low <= a.low and b.high >= a.high))


def _merge_features(features: Sequence[Feature]) -> list[Feature]:
    """Apply the same inclusion principle to a feature sequence."""
    out: list[Feature] = []
    direction = 0
    for item in features:
        if not out:
            out.append(item)
            continue
        prev = out[-1]
        if _included(prev, item):
            if direction >= 0:
                low, high = max(prev.low, item.low), max(prev.high, item.high)
            else:
                low, high = min(prev.low, item.low), min(prev.high, item.high)
            out[-1] = Feature(prev.start_pos, item.end_pos, low, high,
                              item.confirmed_at)
        else:
            direction = 1 if item.high > prev.high else -1
            out.append(item)
    return out


def _opposing_features(points: Sequence[Point], start_pos: int,
                        direction: int) -> list[Feature]:
    result = []
    for pos in range(start_pos, len(points) - 1):
        a, b = points[pos], points[pos + 1]
        leg_direction = 1 if b.price > a.price else -1
        if leg_direction == -direction:
            result.append(Feature(pos, pos + 1, min(a.price, b.price),
                                  max(a.price, b.price), b.confirmed_at))
    return _merge_features(result)


def _feature_fractal(a: Feature, b: Feature, c: Feature, direction: int) -> bool:
    if direction > 0:  # upward segment ends at top feature fractal
        return b.high > a.high and b.high > c.high and b.low > a.low and b.low > c.low
    return b.low < a.low and b.low < c.low and b.high < a.high and b.high < c.high


def _has_gap(a: Feature, b: Feature) -> bool:
    return min(a.high, b.high) < max(a.low, b.low)


def build_segments(points: Sequence[Point]) -> list[Segment]:
    """Build locked segments from feature-sequence fractals.

    The gap branch deliberately requires a fourth opposing feature.  This is a
    conservative live rule for the original text's second case: it delays a
    disputed endpoint instead of repairing it with future chart knowledge.
    """
    points = list(points)
    _validate_points(points)
    if len(points) < 4:
        return []
    segments = []
    start_pos = 0
    while start_pos + 3 < len(points):
        direction = 1 if points[start_pos + 1].price > points[start_pos].price else -1
        found = None
        # Evaluate prefixes in chronological order.  Once a candidate has a
        # separate post-fractal feature, lock it before exposing later points.
        # This prevents a future containing feature from rewriting history.
        for stop in range(start_pos + 4, len(points) + 1):
            features = _opposing_features(points[:stop], start_pos, direction)
            for j in range(1, len(features) - 1):
                a, b, c = features[j - 1:j + 2]
                if not _feature_fractal(a, b, c, direction):
                    continue
                endpoint_pos = b.start_pos
                if endpoint_pos - start_pos < 3:
                    continue
                gap = _has_gap(a, b)
                # c itself can still be altered by inclusion.  One separate
                # feature locks the ordinary case; the disputed gap case uses
                # one additional feature, as frozen in the preregistration.
                confirmation_pos = j + (3 if gap else 2)
                if confirmation_pos >= len(features):
                    continue
                confirmed_at = features[confirmation_pos].confirmed_at
                if segments:
                    # A later segment cannot be knowable before the endpoint
                    # that starts it was itself promoted by the prior segment.
                    confirmed_at = max(confirmed_at, segments[-1].confirmed_at)
                found = Segment(points[start_pos], points[endpoint_pos], start_pos,
                                endpoint_pos, confirmed_at, gap)
                break
            if found is not None:
                break
        if found is None:
            break
        segments.append(found)
        start_pos = found.end_pos
    return segments


def segment_points(segments: Sequence[Segment]) -> list[Point]:
    if not segments:
        return []
    points = [segments[0].start]
    for segment in segments:
        points.append(Point(segment.end.source_idx, segment.end.kind,
                            segment.end.price, segment.confirmed_at))
    return points


def build_centers(elements: Sequence[Leg | Segment], level: int) -> list[Center]:
    """Build non-duplicated centers and extend while overlap persists."""
    elements = list(elements)
    centers: list[Center] = []
    k = 0
    while k + 2 < len(elements):
        triple = elements[k:k + 3]
        zg = min(x.high for x in triple)
        zd = max(x.low for x in triple)
        if zg <= zd:
            k += 1
            continue
        end = k + 2
        gg = max(x.high for x in triple)
        dd = min(x.low for x in triple)
        while end + 1 < len(elements):
            nxt = elements[end + 1]
            if min(zg, nxt.high) <= max(zd, nxt.low):
                break
            end += 1
            gg, dd = max(gg, nxt.high), min(dd, nxt.low)
        centers.append(Center(level, k, end, float(zg), float(zd),
                              float(gg), float(dd),
                              int(elements[end].confirmed_at)))
        k = end + 1
    return centers


def classify_center_sequence(centers: Sequence[Center]) -> tuple[str, int]:
    """Return (consolidation/trend/none, direction)."""
    centers = list(centers)
    if not centers:
        return "none", 0
    if len(centers) == 1:
        return "consolidation", 0
    directions = []
    for a, b in zip(centers, centers[1:]):
        if b.zd > a.zg:
            directions.append(1)
        elif b.zg < a.zd:
            directions.append(-1)
        else:
            return "consolidation", 0
    if directions and len(set(directions)) == 1:
        return "trend", directions[0]
    return "consolidation", 0


def recursive_levels(points: Sequence[Point], max_levels=4) -> dict[int, list[Point]]:
    """Recursively promote confirmed segment endpoints to higher levels."""
    levels = {0: list(points)}
    current = list(points)
    for level in range(1, max_levels + 1):
        segments = build_segments(current)
        promoted = segment_points(segments)
        if len(promoted) < 2:
            break
        levels[level] = promoted
        current = promoted
    return levels


def macd(close: Iterable[float], fast=12, slow=26, signal=9) -> pd.DataFrame:
    close = pd.Series(close, dtype=float)
    dif = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    dea = dif.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"dif": dif, "dea": dea, "hist": 2.0 * (dif - dea)})


def leg_momentum(macd_frame: pd.DataFrame, start: int, end: int,
                 direction: int) -> tuple[float, float]:
    hist = macd_frame["hist"].iloc[start:end + 1].to_numpy(dtype=float)
    signed = np.clip(hist, 0.0, None) if direction > 0 else np.clip(-hist, 0.0, None)
    return float(signed.sum()), float(signed.max(initial=0.0))


def momentum_diverges(price_before: float, price_after: float,
                      energy_before: tuple[float, float],
                      energy_after: tuple[float, float], direction: int) -> bool:
    price_extends = price_after > price_before if direction > 0 else price_after < price_before
    weaker = (energy_after[0] < energy_before[0]
              or energy_after[1] < energy_before[1])
    return bool(price_extends and weaker)


def analyze_frame(frame: pd.DataFrame, max_levels=4) -> dict:
    """Run the complete structural pipeline without creating a trade target."""
    required = {"open", "high", "low", "close"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Frame must contain {sorted(required)}")
    from chan_full_signals import (detect_divergences, first_and_second_points,
                                   third_points)

    base_points = confirmed_stroke_points(frame)
    levels = recursive_levels(base_points, max_levels=max_levels)
    macd_frame = macd(frame["close"].reset_index(drop=True))
    analyses = {}
    for level, level_points in levels.items():
        elements = legs_from_points(level_points, level=level)
        centers = build_centers(elements, level=level)
        divergences = detect_divergences(elements, centers, macd_frame, level)
        buy_sell = (first_and_second_points(elements, divergences, level)
                    + third_points(elements, centers, level))
        analyses[level] = {
            "points": level_points,
            "elements": elements,
            "centers": centers,
            "divergences": divergences,
            "buy_sell_points": sorted(
                buy_sell, key=lambda x: (x.confirmed_at, x.point_class)
            ),
        }
    return {"macd": macd_frame, "levels": analyses}
