"""Historical CFTC reporting disruptions must not leak into decisions."""

from datetime import date, datetime, timezone

from research.crypto_blackbox.cftc_release_calendar import (
    conservative_eligible_at, latest_eligible_report,
)


def test_ion_cyber_delay_2023():
    jan24, jan31, feb7 = date(2023, 1, 24), date(2023, 1, 31), date(2023, 2, 7)
    assert latest_eligible_report((jan24, jan31, feb7),
                                  datetime(2023, 2, 13, tzinfo=timezone.utc)) == jan24
    assert latest_eligible_report((jan24, jan31, feb7),
                                  datetime(2023, 2, 27, tzinfo=timezone.utc)) == jan31
    assert conservative_eligible_at(feb7) == datetime(2023, 3, 4, tzinfo=timezone.utc)


def test_appropriations_lapse_2025():
    sep23, sep30, oct7 = date(2025, 9, 23), date(2025, 9, 30), date(2025, 10, 7)
    assert latest_eligible_report((sep23, sep30, oct7),
                                  datetime(2025, 10, 13, tzinfo=timezone.utc)) == sep23
    assert latest_eligible_report((sep23, sep30, oct7),
                                  datetime(2025, 11, 20, tzinfo=timezone.utc)) == sep30
    assert conservative_eligible_at(oct7) == datetime(2025, 11, 22, tzinfo=timezone.utc)


def test_normal_rule_is_conservative_and_inputs_fail_closed():
    assert conservative_eligible_at(date(2024, 7, 2)) == datetime(2024, 7, 15, tzinfo=timezone.utc)
    assert conservative_eligible_at(date(2024, 7, 2), minimum_lag_days=8) == datetime(2024, 7, 10, tzinfo=timezone.utc)
    assert conservative_eligible_at(date(2025, 9, 30), minimum_lag_days=8) == datetime(2025, 11, 20, tzinfo=timezone.utc)
    try:
        latest_eligible_report((date(2024, 7, 2),), datetime(2024, 7, 15))
    except ValueError:
        pass
    else:
        raise AssertionError("naive decision timestamp accepted")
    try:
        latest_eligible_report((date(2024, 7, 2), date(2024, 7, 2)),
                               datetime(2024, 7, 15, tzinfo=timezone.utc))
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate report date accepted")
