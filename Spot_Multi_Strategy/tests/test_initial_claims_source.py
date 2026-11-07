"""Official ICSA CSV contract with synthetic counts only."""

from datetime import timedelta
import unittest

from research.initial_claims_source import (
    BASE, FIRST, LAST, ROWS, ClaimsSourceError, _parse,
)


URL = BASE + "?id=ICSA&cosd=2020-11-01&coed=2026-09-30"


def _fixture():
    lines = ["DATE,ICSA"]
    for i in range(ROWS):
        lines.append(f"{FIRST + timedelta(days=7*i)},{300000-i}")
    return ("\n".join(lines) + "\n").encode()


class ClaimsSourceTests(unittest.TestCase):
    def test_weekly_coverage_and_release_proxy(self):
        result = _parse(_fixture(), URL)
        self.assertEqual(len(result.observations), 308)
        self.assertEqual(result.observations[0].observed_date, FIRST)
        self.assertEqual(result.observations[-1].observed_date, LAST)
        self.assertEqual(result.observations[0].proxy_available_at.date(),
                         FIRST + timedelta(days=9))

    def test_missing_week_or_nonnumeric_claims_fails(self):
        lines = _fixture().decode().splitlines()
        with self.assertRaisesRegex(ClaimsSourceError, "row count"):
            _parse(("\n".join(lines[:-1]) + "\n").encode(), URL)
        lines[12] = lines[12].split(",")[0] + ",."
        with self.assertRaisesRegex(ClaimsSourceError, "invalid"):
            _parse(("\n".join(lines) + "\n").encode(), URL)

    def test_wrong_host_or_schema_fails(self):
        with self.assertRaisesRegex(ClaimsSourceError, "untrusted"):
            _parse(_fixture(), "https://example.com/data.csv")
        with self.assertRaisesRegex(ClaimsSourceError, "header"):
            _parse(_fixture().replace(b"DATE,ICSA", b"date,ICSA"), URL)


if __name__ == "__main__":
    unittest.main()
