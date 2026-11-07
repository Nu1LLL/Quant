"""Explicit one-shot entry point for preregistered, non-executable research.

Importing this module performs no network request. Call `run_one_shot` with
an explicit chunk fetcher only after the implementation has been reviewed.
No strategy result is written automatically.
"""

from __future__ import annotations

from contextlib import closing
from hashlib import sha256
import time

from .btc_jump_reversal_data import (
    MANIFEST_PATH, iter_price_bars, load_funding,
)
from .btc_jump_reversal_sim import report, simulate
from .crypto_blackbox.binance_vision_5m import (
    ArchiveIntegrityError, ChunkFetcher, MAX_FETCH_CHUNK_BYTES, MAX_ZIP_BYTES,
)


def retry_complete_transport(
    fetch_chunks: ChunkFetcher, *, attempts: int = 4,
) -> ChunkFetcher:
    """Retry transient transport failures before any archive byte is exposed.

    This changes only downloading reliability. It neither changes the
    checksum-pinned archive content nor any strategy decision or cost.
    Each request is capped at the existing 64 MiB ZIP bound in memory,
    never retained on disk, and emitted only after a complete fetch.
    """
    if type(attempts) is not int or not 1 <= attempts <= 4:
        raise ValueError("attempts must be between 1 and 4")

    def complete(url: str):
        for index in range(attempts):
            payload = bytearray()
            try:
                for chunk in fetch_chunks(url):
                    if not isinstance(chunk, bytes) or len(chunk) > MAX_FETCH_CHUNK_BYTES:
                        raise ValueError("transport yielded an invalid chunk")
                    if len(payload) + len(chunk) > MAX_ZIP_BYTES:
                        raise ValueError("transport exceeded the archive byte limit")
                    payload.extend(chunk)
            except ArchiveIntegrityError:
                if index == attempts - 1:
                    raise
                time.sleep(index + 1)
            else:
                for offset in range(0, len(payload), MAX_FETCH_CHUNK_BYTES):
                    yield bytes(payload[offset:offset + MAX_FETCH_CHUNK_BYTES])
                return

    return complete


def run_one_shot(*, fetch_chunks: ChunkFetcher) -> dict:
    """Return the sealed candidate result without persisting raw market data."""
    funding_hashes: dict[str, str] = {}
    funding = load_funding(fetch_chunks=fetch_chunks, audit_hashes=funding_hashes)
    with closing(iter_price_bars(fetch_chunks=fetch_chunks)) as price_bars:
        simulation = simulate(price_bars, funding)
    result = report(simulation)
    result["archive_manifest_sha256"] = sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    result["funding_archive_sha256"] = funding_hashes
    return result
