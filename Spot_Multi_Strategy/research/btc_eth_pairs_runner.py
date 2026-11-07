"""Explicit, one-shot entry point for preregistered BTC/ETH pairs research.

Importing this module performs no network request and writes no result. A
caller must supply a bounded official-archive fetcher. This module never
routes orders or tunes the fixed rule after seeing historical performance.
"""

from __future__ import annotations

from contextlib import closing
from hashlib import sha256

from .btc_eth_pairs_data import (
    FUNDING_MANIFEST_PATH,
    KLINE_MANIFEST_PATH,
    iter_paired_bars,
    load_funding_events,
)
from .btc_eth_pairs_metrics import evaluate_gate
from .btc_eth_pairs_sim import run_pairs_simulation
from .crypto_blackbox.binance_vision_5m import ChunkFetcher


PREREGISTRATION_COMMIT = "ce92889"
PRE_RESULT_AMENDMENT_COMMITS = ("73dfb0f", "ea4847c")
EXPERIMENT_ID = "btc_eth_distance_pairs_2026_11_07"


def run_one_shot(*, fetch_chunks: ChunkFetcher) -> dict:
    """Return the complete research report after all fixed data checks pass.

    Price archives are streamed through validated month pairs; only the
    parsed funding events and cumulative diagnostics are retained. A caller
    must seal any exception as a failure rather than accepting partial PnL.
    """
    funding_hashes: dict[tuple[str, str], str] = {}
    kline_hashes: dict[tuple[str, str], str] = {}
    funding_events = load_funding_events(
        fetch_chunks=fetch_chunks, audit_hashes=funding_hashes,
    )
    with closing(iter_paired_bars(
        fetch_chunks=fetch_chunks, audit_hashes=kline_hashes,
    )) as bars:
        scenarios = run_pairs_simulation(bars, funding_events)
    result = evaluate_gate(scenarios)
    result["experiment_id"] = EXPERIMENT_ID
    result["preregistration_commit"] = PREREGISTRATION_COMMIT
    result["pre_result_amendment_commits"] = PRE_RESULT_AMENDMENT_COMMITS
    result["evidence_grade"] = "pair_mechanism_internal_blind_historical_holdout"
    result["archive_manifest_sha256"] = {
        "kline": sha256(KLINE_MANIFEST_PATH.read_bytes()).hexdigest(),
        "funding": sha256(FUNDING_MANIFEST_PATH.read_bytes()).hexdigest(),
    }
    result["measured_archive_sha256"] = {
        "kline": {
            f"{month}:{symbol}": digest
            for (month, symbol), digest in sorted(kline_hashes.items())
        },
        "funding": {
            f"{month}:{symbol}": digest
            for (month, symbol), digest in sorted(funding_hashes.items())
        },
    }
    if len(kline_hashes) != 144 or len(funding_hashes) != 120:
        raise ValueError("complete archive digest lists are required")
    return result
