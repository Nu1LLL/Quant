"""Fixed Bitcoin-mainnet difficulty events for the preregistered BTC study.

Only Blockstream Esplora's canonical ``/blocks/{height}`` summaries are used.
Responses stay in memory; the returned manifest contains URL, digest, chain
heights/hashes and retry errors, but no raw response or strategy outcomes.
Miner-supplied block timestamps are not historical first-seen timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, localcontext
from hashlib import sha256
import json
import re
import time
from typing import Callable

import requests


UTC = timezone.utc
BASE_URL = "https://blockstream.info/api"
RETARGET_INTERVAL = 2016
START = datetime(2017, 8, 1, tzinfo=UTC)
END_EXCLUSIVE = datetime(2026, 10, 1, tzinfo=UTC)
# Three preregistration warm-up epochs precede the first August 2017 event.
FIRST_HEIGHT = 473_760
MAX_PAGES = 300
MAX_RESPONSE_BYTES = 128 * 1024
MAX_ATTEMPTS = 4
MIN_REQUEST_INTERVAL_SECONDS = 0.25
DIFF_ONE_TARGET = 0xFFFF << 208
_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class SourceRequest:
    """One HTTP attempt; successful attempts include the raw-response digest."""

    url: str
    requested_height: int
    status_code: int | None
    response_sha256: str | None = None
    chain_height: int | None = None
    block_hash: str | None = None
    confirm_hash: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class DifficultyEvent:
    height: int
    timestamp: datetime
    confirm_timestamp: datetime
    bits: int
    target: int
    difficulty: Decimal
    adjustment: Decimal
    knowledge_time: datetime
    recovery: bool
    block_hash: str
    confirm_hash: str
    successor_hashes: tuple[str, ...]
    source_url: str
    source_sha256: str


@dataclass(frozen=True)
class DifficultyFetch:
    events: tuple[DifficultyEvent, ...]
    manifest: tuple[SourceRequest, ...]


class DifficultySourceError(RuntimeError):
    """A missing or inconsistent block page makes the full study ineligible."""

    def __init__(self, message: str, manifest: tuple[SourceRequest, ...] = ()) -> None:
        super().__init__(message)
        self.manifest = manifest


@dataclass(frozen=True)
class _Block:
    height: int
    block_hash: str
    previous_hash: str
    timestamp: datetime
    bits: int


def target_from_bits(bits: int) -> int:
    """Decode Bitcoin's compact nBits into an exact positive mainnet target."""

    if isinstance(bits, bool) or not isinstance(bits, int) or not 0 <= bits <= 0xFFFFFFFF:
        raise DifficultySourceError(f"invalid compact bits: {bits!r}")
    size = bits >> 24
    word = bits & 0x007FFFFF
    if word == 0 or bits & 0x00800000:
        raise DifficultySourceError(f"zero or negative compact target: {bits:#x}")
    target = word >> (8 * (3 - size)) if size <= 3 else word << (8 * (size - 3))
    if target == 0 or target > DIFF_ONE_TARGET:
        raise DifficultySourceError(f"target outside Bitcoin-mainnet proof-of-work limit: {bits:#x}")
    return target


def difficulty_from_bits(bits: int) -> Decimal:
    """Calculate difficulty from the decoded target, never API float difficulty."""

    target = target_from_bits(bits)
    with localcontext() as ctx:
        ctx.prec = 80
        return Decimal(DIFF_ONE_TARGET) / Decimal(target)


def _hash(value: object, field: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise DifficultySourceError(f"invalid {field} in block summary")
    return value


def _block(row: object) -> _Block:
    if not isinstance(row, dict):
        raise DifficultySourceError("block summary is not an object")
    height, seconds, bits = row.get("height"), row.get("timestamp"), row.get("bits")
    if isinstance(height, bool) or not isinstance(height, int) or height < 0:
        raise DifficultySourceError("missing or invalid block height")
    if isinstance(seconds, bool) or not isinstance(seconds, int):
        raise DifficultySourceError("missing or invalid block timestamp")
    if isinstance(bits, bool) or not isinstance(bits, int):
        raise DifficultySourceError("missing or invalid block bits; source-format correction required")
    target_from_bits(bits)
    try:
        stamp = datetime.fromtimestamp(seconds, UTC)
    except (OverflowError, OSError, ValueError) as exc:
        raise DifficultySourceError("block timestamp outside datetime range") from exc
    return _Block(
        height=height,
        block_hash=_hash(row.get("id"), "block hash"),
        previous_hash=_hash(row.get("previousblockhash"), "previous block hash"),
        timestamp=stamp,
        bits=bits,
    )


def _parse_page(payload: bytes, requested_height: int) -> tuple[_Block, ...]:
    try:
        rows = json.loads(payload)
    except (ValueError, UnicodeDecodeError) as exc:
        raise DifficultySourceError("invalid Blockstream JSON") from exc
    if not isinstance(rows, list) or len(rows) != 10:
        raise DifficultySourceError("Blockstream page must have exactly ten blocks")
    blocks = tuple(_block(row) for row in rows)
    hashes = set()
    for index, block in enumerate(blocks):
        expected = requested_height - index
        if block.height != expected:
            raise DifficultySourceError(f"height mismatch: expected {expected}, got {block.height}")
        if block.block_hash in hashes:
            raise DifficultySourceError("duplicate block hash in page")
        hashes.add(block.block_hash)
        if index < 9 and block.previous_hash != blocks[index + 1].block_hash:
            raise DifficultySourceError(f"broken main-chain hash link at {block.height}")
        if int(block.block_hash, 16) > target_from_bits(block.bits):
            raise DifficultySourceError(f"block hash fails compact proof-of-work target at {block.height}")
    retarget = blocks[6]
    if retarget.height % RETARGET_INTERVAL != 0:
        raise DifficultySourceError("requested block is not a retarget height")
    if any(block.bits != retarget.bits for block in blocks[:7]):
        raise DifficultySourceError("retarget bits disagree across six successor blocks")
    return blocks


def _read_bounded(response: requests.Response) -> bytes:
    payload = bytearray()
    for chunk in response.iter_content(chunk_size=16 * 1024):
        payload.extend(chunk)
        if len(payload) > MAX_RESPONSE_BYTES:
            raise DifficultySourceError("Blockstream response exceeds byte limit")
    return bytes(payload)


def _fetch_page(
    opener: requests.Session,
    requested_height: int,
    manifest: list[SourceRequest],
    pause: Callable[[float], None],
    min_interval_seconds: float,
) -> tuple[tuple[_Block, ...], str, str]:
    url = f"{BASE_URL}/blocks/{requested_height}"
    for attempt in range(MAX_ATTEMPTS):
        if min_interval_seconds:
            pause(min_interval_seconds)
        try:
            with opener.get(url, timeout=(10, 30), stream=True, allow_redirects=False) as response:
                status = response.status_code
                if status != 200:
                    message = f"Blockstream HTTP {status} at {url}"
                    manifest.append(SourceRequest(url, requested_height, status, error=message))
                    if status == 429 or 500 <= status < 600:
                        if attempt + 1 < MAX_ATTEMPTS:
                            pause(0.5 * 2**attempt)
                            continue
                    raise DifficultySourceError(message)
                try:
                    payload = _read_bounded(response)
                except DifficultySourceError as exc:
                    manifest.append(SourceRequest(
                        url, requested_height, status, error=str(exc)
                    ))
                    raise
                digest = sha256(payload).hexdigest()
                try:
                    blocks = _parse_page(payload, requested_height)
                except DifficultySourceError as exc:
                    manifest.append(SourceRequest(
                        url, requested_height, status, digest, error=str(exc)
                    ))
                    raise
                h, h6 = blocks[6], blocks[0]
                manifest.append(SourceRequest(
                    url, requested_height, status, digest,
                    chain_height=h.height, block_hash=h.block_hash,
                    confirm_hash=h6.block_hash,
                ))
                return blocks, url, digest
        except DifficultySourceError:
            raise
        except (requests.Timeout, requests.ConnectionError) as exc:
            manifest.append(SourceRequest(
                url, requested_height, None, error=f"{type(exc).__name__}: {exc}"
            ))
            if attempt + 1 < MAX_ATTEMPTS:
                pause(0.5 * 2**attempt)
                continue
            raise DifficultySourceError(f"Blockstream transport failed at {url}") from exc
        except requests.RequestException as exc:
            manifest.append(SourceRequest(
                url, requested_height, None, error=f"{type(exc).__name__}: {exc}"
            ))
            raise DifficultySourceError(f"Blockstream request failed at {url}") from exc
    raise AssertionError("unreachable retry state")


def _fetch_range(
    *,
    opener: requests.Session,
    pause: Callable[[float], None],
    start_height: int,
    start: datetime,
    end_exclusive: datetime,
    max_pages: int,
    min_interval_seconds: float,
) -> DifficultyFetch:
    if (isinstance(start_height, bool) or not isinstance(start_height, int)
            or start_height < RETARGET_INTERVAL
            or start_height % RETARGET_INTERVAL != 0):
        raise ValueError("start_height must be a positive retarget height")
    if (start.tzinfo is None or end_exclusive.tzinfo is None
            or start >= end_exclusive or max_pages < 3 or min_interval_seconds < 0):
        raise ValueError("invalid UTC window or request bounds")
    manifest: list[SourceRequest] = []
    events: list[DifficultyEvent] = []
    previous_target: int | None = None
    previous_bits: int | None = None
    previous_adjustment: Decimal | None = None
    reached_end = False
    try:
        for page_index in range(max_pages):
            h = start_height + RETARGET_INTERVAL * page_index
            blocks, url, digest = _fetch_page(
                opener, h + 6, manifest, pause, min_interval_seconds
            )
            block = blocks[6]
            if previous_bits is not None and blocks[7].bits != previous_bits:
                manifest[-1] = replace(manifest[-1], error="preceding epoch bits disagree with prior retarget")
                raise DifficultySourceError("preceding epoch bits disagree with prior retarget")
            target = target_from_bits(block.bits)
            with localcontext() as ctx:
                ctx.prec = 80
                difficulty = Decimal(DIFF_ONE_TARGET) / Decimal(target)
                adjustment = (
                    Decimal(previous_target) / Decimal(target) - 1
                    if previous_target is not None else None
                )
            if block.timestamp >= end_exclusive:
                reached_end = True
                break
            if block.timestamp >= start:
                if adjustment is None or previous_adjustment is None:
                    raise DifficultySourceError("insufficient earlier retargets for recovery warm-up")
                knowledge_time = max(block.timestamp + timedelta(hours=48), blocks[0].timestamp)
                events.append(DifficultyEvent(
                    height=block.height,
                    timestamp=block.timestamp,
                    confirm_timestamp=blocks[0].timestamp,
                    bits=block.bits,
                    target=target,
                    difficulty=difficulty,
                    adjustment=adjustment,
                    knowledge_time=knowledge_time,
                    recovery=previous_adjustment < 0 <= adjustment,
                    block_hash=block.block_hash,
                    confirm_hash=blocks[0].block_hash,
                    successor_hashes=tuple(b.block_hash for b in reversed(blocks[:6])),
                    source_url=url,
                    source_sha256=digest,
                ))
            previous_target = target
            previous_bits = block.bits
            previous_adjustment = adjustment
        if not reached_end:
            raise DifficultySourceError("end of registered window was not reached within page limit")
        if not events:
            raise DifficultySourceError("no retarget events in registered window")
    except DifficultySourceError as exc:
        raise DifficultySourceError(str(exc), tuple(manifest)) from exc
    return DifficultyFetch(tuple(events), tuple(manifest))


def fetch_events(
    *,
    opener: requests.Session | None = None,
    pause: Callable[[float], None] = time.sleep,
) -> DifficultyFetch:
    """Fetch the fixed 2017-08 through 2026-09 mainnet event sequence.

    This function performs no price fetch, PnL calculation or raw-file write.
    Any missing/inconsistent page raises with the attempts in ``exc.manifest``.
    """

    owned = opener is None
    session = opener if opener is not None else requests.Session()
    try:
        return _fetch_range(
            opener=session, pause=pause, start_height=FIRST_HEIGHT,
            start=START, end_exclusive=END_EXCLUSIVE, max_pages=MAX_PAGES,
            min_interval_seconds=MIN_REQUEST_INTERVAL_SECONDS,
        )
    finally:
        if owned:
            session.close()
