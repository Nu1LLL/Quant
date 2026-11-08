# Future source transport audit (no strategy result)

At 2026-11-08 14:52 UTC, the two latest formal studies were already sealed
as `DATA_FAILURE`: the U.S. weekly-claims study timed out reading FRED via
Python `requests`, and the BTC/ETH transfer-growth study saw one Binance
`.CHECKSUM` TLS handshake error via `curl`. No source, signal, numerical Gate
or result in those sealed studies was altered or rerun. A subsequent
**read-only** GET of that failed Binance checksum sidecar succeeded, so a
single failed connection is insufficient to establish a persistent source
outage.

`research/frozen_source_transport.py` is an additive module for **future,
separately preregistered** experiments. It limits HTTPS hosts, response
bytes, request time and attempts; the default policy makes at most four
attempts on the **same URL** with bounded backoff. An unexpected pinned
SHA-256 is a terminal integrity failure, never a retry or alternate-source
trigger. It writes no raw responses, ZIPs or market values to disk. Older
formal runners remain byte-for-byte unchanged.

The checksum-only audit validated all **62 BTC** and **63 ETH** entries in
the two previously committed Binance Vision manifests. It fetched only the
official `.CHECKSUM` sidecars, checked their filenames and hashes against
the frozen manifests, and returned **125/125 verified, zero retries**. It did
not fetch or decode historical prices, calculate an economic signal, or
generate PnL. Three synthetic transport tests passed, covering a transient
TLS error followed by success on the same URL, immediate rejection of a
changed content hash, disallowed hosts and manifest/sidecar validation.

This audit proves the listed sidecars were reachable during the audit window;
it does not guarantee later availability, ZIP content integrity, depth,
historical fills or independent OOS. Each future hypothesis must pin its
sources and retry policy **before** opening results, verify the full ZIP
hashes and all sample gates, and seal any unresolved failure. No orders.
