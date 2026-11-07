"""One-shot frozen three-altcoin archival experiment; never routes orders."""

from __future__ import annotations

from contextlib import closing
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Iterator

from .alt_common_flow_data import (
    AltFlowDataError, FUNDING_MANIFEST, KLINE_MANIFEST, iter_minutes, load_funding,
)
from .alt_common_flow_metrics import evaluate_alt_flow_gate
from .alt_common_flow_sim import FlowSimulationError, run_alt_flow_simulation
from .crypto_blackbox.binance_vision_5m import ArchiveIntegrityError, ChunkFetcher


EXPERIMENT_ID = "alt_common_flow_reversal_2026_11_07"
PREREGISTRATION_COMMIT = "690a7f5"
PRE_RESULT_CODE_COMMITS = ("716595a", "732ff0f")
REPORT_PATH = Path(__file__).parent / "results" / f"{EXPERIMENT_ID}.json"


def _verified_curl_chunks(url: str) -> Iterator[bytes]:
    """Use the host TLS trust store when Python's CA bundle is unavailable."""
    command = ("curl", "--fail", "--location", "--silent", "--show-error",
               "--max-time", "180", "--proto", "=https", "--proto-redir",
               "=https", url)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        if process.stdout is None or process.stderr is None:
            raise ArchiveIntegrityError("curl pipes unavailable")
        while True:
            block = process.stdout.read(64 * 1024)
            if not block:
                break
            yield block
        error = process.stderr.read(1000).decode("utf-8", errors="replace")
        if process.wait() != 0:
            raise ArchiveIntegrityError(f"archive transport failed: {url}: {error.strip()}")
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()


def run_one_shot(*, fetch_chunks: ChunkFetcher) -> dict:
    funding_hashes: dict[tuple[str, str], str] = {}
    price_hashes: dict[tuple[str, str], str] = {}
    funding = load_funding(fetch_chunks=fetch_chunks, audit_hashes=funding_hashes)
    with closing(iter_minutes(fetch_chunks=fetch_chunks, audit_hashes=price_hashes)) as bars:
        scenarios = run_alt_flow_simulation(bars, funding)
    if len(price_hashes) != 192 or len(funding_hashes) != 180:
        raise ValueError("all 372 pinned archive hashes are mandatory")
    result = evaluate_alt_flow_gate(scenarios)
    result.update({
        "experiment_id": EXPERIMENT_ID,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "pre_result_code_commits": PRE_RESULT_CODE_COMMITS,
        "archive_manifest_sha256": {
            "kline": sha256(KLINE_MANIFEST.read_bytes()).hexdigest(),
            "funding": sha256(FUNDING_MANIFEST.read_bytes()).hexdigest(),
        },
        "measured_archive_sha256": {
            "kline": {f"{month}:{symbol}": digest
                      for (month, symbol), digest in sorted(price_hashes.items())},
            "funding": {f"{month}:{symbol}": digest
                        for (month, symbol), digest in sorted(funding_hashes.items())},
        },
    })
    return result


def main() -> None:
    try:
        report = run_one_shot(fetch_chunks=_verified_curl_chunks)
    except (AltFlowDataError, FlowSimulationError, ValueError) as exc:
        # A failed predeclared scenario invalidates the whole path. No partial
        # equity curve, candidate promotion, or tuned retry is written.
        report = {
            "experiment_id": EXPERIMENT_ID,
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "status": ("transport_interrupted_no_result"
                       if isinstance(exc, ArchiveIntegrityError)
                       and str(exc).startswith("archive transport failed:")
                       else "sealed_failed_run"),
            "failure_type": type(exc).__name__,
            "failure_reason": str(exc),
            "gate": {"full": False},
        }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp = REPORT_PATH.with_suffix(".json.tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    temp.replace(REPORT_PATH)
    print(json.dumps({"report": str(REPORT_PATH),
                      "gate": report["gate"],
                      "metrics": report.get("metrics"),
                      "completed_baskets": report.get("completed_baskets"),
                      "failure_reason": report.get("failure_reason")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
