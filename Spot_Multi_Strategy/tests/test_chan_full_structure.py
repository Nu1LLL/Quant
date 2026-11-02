import numpy as np
import pandas as pd

import chan_full_signals as signals
import chan_full_structure as structure


def points(values):
    return [structure.Point(i, -1 if i % 2 == 0 else 1, float(value), i + 1)
            for i, value in enumerate(values)]


def test_feature_sequence_builds_locked_up_and_down_segments():
    sample = points([5, 10, 7, 12, 9, 11, 8, 10, 9, 10.5,
                     7.5, 11, 8.5, 10, 9, 10.2])
    segments = structure.build_segments(sample)
    assert [(x.start_pos, x.end_pos) for x in segments] == [(0, 3), (3, 6)]
    assert segments[0].direction == 1 and segments[1].direction == -1
    assert segments[0].confirmed_at == sample[12].confirmed_at
    assert segments[1].confirmed_at == sample[13].confirmed_at


def test_gap_case_needs_an_extra_feature_before_locking():
    incomplete = points([5, 10, 7, 13, 11, 12, 8, 11, 9, 12.5, 9.5, 12])
    assert structure.build_segments(incomplete) == []
    complete = incomplete + [structure.Point(12, -1, 9.0, 13)]
    segment = structure.build_segments(complete)[0]
    assert segment.gap_case
    assert segment.confirmed_at == complete[12].confirmed_at


def test_center_extension_and_center_sequence_classification():
    sample = points([0, 10, 4, 8, 5])
    legs = structure.legs_from_points(sample)
    centers = structure.build_centers(legs, level=0)
    assert len(centers) == 1
    assert centers[0].zd == 4 and centers[0].zg == 8
    assert centers[0].end_leg == 3
    low = structure.Center(1, 0, 2, 6, 4, 8, 2, 5)
    high = structure.Center(1, 4, 6, 12, 10, 14, 9, 10)
    assert structure.classify_center_sequence([low]) == ("consolidation", 0)
    assert structure.classify_center_sequence([low, high]) == ("trend", 1)


def test_momentum_divergence_requires_price_extension_and_weaker_energy():
    assert structure.momentum_diverges(10, 12, (20, 5), (15, 6), 1)
    assert not structure.momentum_diverges(10, 9, (20, 5), (10, 3), 1)
    assert structure.momentum_diverges(8, 7, (20, 5), (21, 4), -1)


def test_trend_divergence_is_aligned_to_last_center_not_any_macd_cross():
    sample = points([0, 5, 1, 6, 2, 7, 3, 8, 4, 10])
    legs = structure.legs_from_points(sample)
    centers = [
        structure.Center(0, 0, 2, 4, 2, 6, 0, 4),
        structure.Center(0, 5, 7, 8, 6, 9, 5, 8),
    ]
    histogram = np.zeros(10)
    histogram[4:6] = 5.0
    histogram[8:10] = 1.0
    macd_frame = pd.DataFrame({"hist": histogram})
    events = signals.detect_divergences(legs, centers, macd_frame, level=0)
    assert len(events) == 1
    assert events[0].mode == "trend" and events[0].direction == 1
    assert events[0].entry_leg == 4 and events[0].exit_leg == 8


def test_local_trend_is_detected_even_when_later_centers_reverse_direction():
    sample = points([0, 5, 1, 6, 2, 7, 3, 8, 4, 10])
    legs = structure.legs_from_points(sample)
    centers = [
        structure.Center(0, 0, 2, 4, 2, 6, 0, 4),
        structure.Center(0, 5, 7, 8, 6, 9, 5, 8),
        structure.Center(0, 8, 8, 3, 1, 4, 0, 9),
    ]
    histogram = np.zeros(10)
    histogram[4:6] = 5.0
    histogram[8:10] = 1.0
    events = signals.detect_divergences(
        legs, centers, pd.DataFrame({"hist": histogram}), level=0
    )
    assert any(event.mode == "trend" and event.direction == 1
               for event in events)


def test_third_point_uses_whole_retest_range_outside_center():
    sample = points([0, 10, 4, 8, 5, 12, 9])
    legs = structure.legs_from_points(sample)
    center = structure.Center(0, 0, 3, 8, 4, 10, 0, 5)
    events = signals.third_points(legs, [center], level=0)
    assert len(events) == 1
    assert events[0].point_class == 3 and events[0].side == 1
    assert events[0].price == 9


def test_first_second_points_and_failed_reference_are_not_deleted():
    sample = points([12, 5, 10, 4, 9])
    legs = structure.legs_from_points(sample)
    divergence = signals.Divergence(0, "trend", -1, 0, 0, 0, 1, 2,
                                    (10, 5), (5, 2))
    events = signals.first_and_second_points(legs, [divergence], 0)
    assert [event.point_class for event in events] == [1, 2]
    assert events[1].side == 1
    assert not events[1].held_reference


def test_interval_nesting_requires_same_direction_and_containment():
    def event(level, direction, start, end):
        return signals.Divergence(level, "trend", direction, 0, 0, start, end,
                                  end + 1, (2, 2), (1, 1))
    nested = signals.interval_nesting([
        event(2, -1, 0, 20), event(1, -1, 5, 15), event(0, -1, 8, 10),
        event(0, 1, 8, 10),
    ])
    assert nested[0].side == 1 and nested[0].depth == 3
    assert nested[0].outer_level == 2 and nested[0].inner_level == 0


def test_future_mutation_cannot_change_prior_confirmed_strokes():
    index = pd.RangeIndex(200)
    x = np.arange(200, dtype=float)
    close = 100 + np.sin(x / 3) * 5 + x * 0.02
    frame = pd.DataFrame({"open": close, "high": close + 1,
                          "low": close - 1, "close": close}, index=index)
    cutoff = 150
    before = [p for p in structure.confirmed_stroke_points(frame)
              if p.confirmed_at < cutoff]
    changed = frame.copy()
    changed.loc[cutoff:, ["open", "high", "low", "close"]] *= 2.0
    after = [p for p in structure.confirmed_stroke_points(changed)
             if p.confirmed_at < cutoff]
    assert before == after


def test_end_to_end_analysis_exposes_recursive_structures_without_positions():
    x = np.arange(300, dtype=float)
    close = 100 + np.sin(x / 4) * 8 + np.sin(x / 17) * 3
    frame = pd.DataFrame({"open": close, "high": close + 1,
                          "low": close - 1, "close": close})
    result = structure.analyze_frame(frame)
    assert 0 in result["levels"]
    assert len(result["macd"]) == len(frame)
    assert set(result["levels"][0]) == {
        "points", "elements", "centers", "divergences", "buy_sell_points"
    }


def test_appending_future_points_does_not_move_locked_segments():
    prefix = points([5, 10, 7, 12, 9, 11, 8, 10, 9, 10.5,
                     7.5, 11, 8.5, 10, 9, 10.2])
    locked = structure.build_segments(prefix)
    extended = prefix + [structure.Point(16, -1, -20.0, 17),
                         structure.Point(17, 1, 30.0, 18),
                         structure.Point(18, -1, -25.0, 19)]
    cutoff = prefix[-1].confirmed_at
    comparable = [segment for segment in structure.build_segments(extended)
                  if segment.confirmed_at <= cutoff]
    assert locked == comparable


def test_promoted_segment_confirmation_times_never_go_backwards():
    sample = points([5, 10, 7, 13, 11, 12, 8, 11, 9, 12.5, 9.5, 12,
                     9, 11.5, 9.2, 11, 9.4, 10.8, 9.6, 10.6])
    promoted = structure.segment_points(structure.build_segments(sample))
    assert all(right.confirmed_at >= left.confirmed_at
               for left, right in zip(promoted, promoted[1:]))
