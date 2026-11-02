"""Synthetic main-chain fixtures for the registered difficulty-only source."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from hashlib import sha256
import json
import unittest

from research.btc_difficulty_blocks import (
    BASE_URL,
    DIFF_ONE_TARGET,
    DifficultySourceError,
    _fetch_range,
    difficulty_from_bits,
    target_from_bits,
)


UTC = timezone.utc
EASY = 0x1D00FFFF
HARD = 0x1C00FFFF
FIRST_HEIGHT = 2016 * 300
BASE_DAY = datetime(2020, 1, 1, tzinfo=UTC)


def block_hash(height: int) -> str:
    return f"{height:064x}"


def page(
    index: int,
    bits: int,
    previous_bits: int,
    *,
    confirm_hours: int = 1,
) -> bytes:
    h = FIRST_HEIGHT + index * 2016
    anchor = BASE_DAY + timedelta(days=index * 10)
    rows = []
    for height in range(h + 6, h - 4, -1):
        offset = height - h
        seconds = int((anchor + timedelta(hours=confirm_hours * offset / 6)).timestamp())
        rows.append({
            "id": block_hash(height),
            "height": height,
            "previousblockhash": block_hash(height - 1),
            "timestamp": seconds,
            "bits": bits if offset >= 0 else previous_bits,
            # A deliberately wrong API float proves it is not used.
            "difficulty": 123.456,
        })
    return json.dumps(rows, separators=(",", ":")).encode("utf-8")


class FakeResponse:
    def __init__(self, content: bytes = b"", status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def iter_content(self, *, chunk_size: int):
        for offset in range(0, len(self.content), chunk_size):
            yield self.content[offset:offset + chunk_size]


class FakeSession:
    def __init__(self, pages: dict[int, bytes], *, first_status: int | None = None):
        self.pages = pages
        self.first_status = first_status
        self.urls: list[str] = []

    def get(self, url: str, *, timeout, stream, allow_redirects):
        assert timeout == (10, 30)
        assert stream is True
        assert allow_redirects is False
        assert url.startswith(f"{BASE_URL}/blocks/")
        self.urls.append(url)
        if self.first_status is not None:
            status, self.first_status = self.first_status, None
            return FakeResponse(status_code=status)
        return FakeResponse(self.pages[int(url.rsplit("/", 1)[1])])


def fixture_pages(*, confirm_hours: int = 1) -> dict[int, bytes]:
    # Hard -> easy is a negative adjustment; easy -> hard is recovery.
    bits = (HARD, EASY, HARD, HARD)
    return {
        FIRST_HEIGHT + index * 2016 + 6: page(
            index, current, bits[index - 1] if index else current,
            confirm_hours=confirm_hours if index == 2 else 1,
        )
        for index, current in enumerate(bits)
    }


def fetch_fixture(session: FakeSession):
    return _fetch_range(
        opener=session,
        pause=lambda _delay: None,
        start_height=FIRST_HEIGHT,
        start=BASE_DAY + timedelta(days=15),
        end_exclusive=BASE_DAY + timedelta(days=30),
        max_pages=4,
        min_interval_seconds=0,
    )


class DifficultyBitsTests(unittest.TestCase):
    def test_compact_target_exact_and_decimal_difficulty(self):
        self.assertEqual(target_from_bits(EASY), DIFF_ONE_TARGET)
        self.assertEqual(target_from_bits(HARD), DIFF_ONE_TARGET >> 8)
        self.assertEqual(difficulty_from_bits(EASY), Decimal(1))
        self.assertEqual(difficulty_from_bits(HARD), Decimal(256))

    def test_invalid_compact_targets_fail_closed(self):
        for bits in (0, -1, 0xFFFFFFFF, 0x1D80FFFF, 0x1D000000, 0x1E00FFFF, True):
            with self.subTest(bits=bits), self.assertRaises(DifficultySourceError):
                target_from_bits(bits)


class DifficultySourceTests(unittest.TestCase):
    def test_recovery_knowledge_time_and_manifest(self):
        pages = fixture_pages()
        session = FakeSession(pages)
        result = fetch_fixture(session)
        self.assertEqual(len(result.events), 1)
        event = result.events[0]
        self.assertEqual(event.height, FIRST_HEIGHT + 2 * 2016)
        self.assertEqual(event.timestamp, BASE_DAY + timedelta(days=20))
        self.assertEqual(event.confirm_timestamp, event.timestamp + timedelta(hours=1))
        self.assertEqual(event.knowledge_time, event.timestamp + timedelta(hours=48))
        self.assertEqual(event.bits, HARD)
        self.assertEqual(event.difficulty, Decimal(256))
        self.assertEqual(event.adjustment, Decimal(255))
        self.assertTrue(event.recovery)
        self.assertEqual(event.successor_hashes, tuple(
            block_hash(event.height + offset) for offset in range(1, 7)
        ))
        self.assertEqual(len(result.manifest), 4)
        self.assertTrue(all(request.status_code == 200 for request in result.manifest))
        self.assertEqual(event.source_url, session.urls[2])
        self.assertEqual(event.source_sha256, sha256(pages[event.height + 6]).hexdigest())
        self.assertEqual(result.manifest[2].chain_height, event.height)
        self.assertEqual(result.manifest[2].block_hash, event.block_hash)
        self.assertEqual(result.manifest[2].confirm_hash, event.confirm_hash)
        with self.assertRaises(FrozenInstanceError):
            event.recovery = False
        with self.assertRaises(FrozenInstanceError):
            result.events = ()

    def test_late_sixth_successor_controls_knowledge_time(self):
        event = fetch_fixture(FakeSession(fixture_pages(confirm_hours=49))).events[0]
        self.assertEqual(event.knowledge_time, event.confirm_timestamp)
        self.assertEqual(event.knowledge_time, event.timestamp + timedelta(hours=49))

    def test_retry_attempt_is_recorded(self):
        session = FakeSession(fixture_pages(), first_status=429)
        result = fetch_fixture(session)
        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.manifest[0].status_code, 429)
        self.assertIn("HTTP 429", result.manifest[0].error)
        self.assertEqual(result.manifest[1].status_code, 200)
        self.assertEqual(session.urls[0], session.urls[1])

    def test_height_hash_missing_bits_and_successor_conflicts_fail_closed(self):
        for mutate, expected in (
            (lambda rows: rows[0].update(height=rows[0]["height"] - 1), "height mismatch"),
            (lambda rows: rows[0].update(previousblockhash="0" * 64), "hash link"),
            (lambda rows: rows[6].pop("bits"), "missing or invalid block bits"),
            (lambda rows: rows[3].update(bits=EASY), "successor blocks"),
            (lambda rows: rows[7].update(bits=HARD), "preceding epoch bits disagree"),
        ):
            pages = fixture_pages()
            key = FIRST_HEIGHT + 2 * 2016 + 6
            rows = json.loads(pages[key])
            mutate(rows)
            pages[key] = json.dumps(rows).encode("utf-8")
            with self.subTest(expected=expected), self.assertRaisesRegex(
                DifficultySourceError, expected
            ) as caught:
                fetch_fixture(FakeSession(pages))
            self.assertEqual(len(caught.exception.manifest), 3)
            self.assertIn(expected, caught.exception.manifest[-1].error)
            self.assertEqual(
                caught.exception.manifest[-1].response_sha256,
                sha256(pages[key]).hexdigest(),
            )

    def test_missing_page_or_insufficient_window_is_ineligible(self):
        session = FakeSession(fixture_pages())
        with self.assertRaisesRegex(DifficultySourceError, "page limit") as caught:
            _fetch_range(
                opener=session, pause=lambda _delay: None,
                start_height=FIRST_HEIGHT,
                start=BASE_DAY + timedelta(days=15),
                end_exclusive=BASE_DAY + timedelta(days=100),
                max_pages=4, min_interval_seconds=0,
            )
        self.assertEqual(len(caught.exception.manifest), 4)


if __name__ == "__main__":
    unittest.main()
