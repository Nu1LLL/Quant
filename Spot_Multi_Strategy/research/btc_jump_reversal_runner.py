"""Explicit one-shot entry point for preregistered, non-executable research.

Importing this module performs no network request. Call `run_one_shot` with
an explicit chunk fetcher only after the implementation has been reviewed.
No strategy result is written automatically.
"""

from __future__ import annotations

from contextlib import closing
from hashlib import sha256

from .btc_jump_reversal_data import (
    MANIFEST_PATH, iter_price_bars, load_funding,
)
from .btc_jump_reversal_sim import report, simulate
from .crypto_blackbox.binance_vision_5m import ChunkFetcher


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
