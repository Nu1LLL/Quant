"""Bounded, uncached HTTPS transport for future preregistered research sources.

Source bytes are kept in memory only. Transport failures may be retried under
a policy fixed before an experiment; a changed pinned digest never is.
Existing sealed experiments do not import this module retroactively.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import csv
import io
import re
import subprocess
import time
from urllib.parse import urlsplit


ALLOWED_HOSTS = frozenset({
    "data.binance.vision", "community-api.coinmetrics.io",
    "fred.stlouisfed.org", "www.dol.gov", "oui.doleta.gov",
})


class TransportFailure(RuntimeError):
    pass


class IntegrityFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class FetchReceipt:
    body: bytes
    sha256: str
    attempts: int


@dataclass(frozen=True)
class ChecksumAudit:
    checked: int
    retries: int
    all_verified: bool


def _valid_url(url: str) -> bool:
    if not isinstance(url, str) or len(url) > 2048 or any(ord(c) < 32 for c in url):
        return False
    try:
        parsed = urlsplit(url)
        return (parsed.scheme == "https" and parsed.hostname in ALLOWED_HOSTS
                and parsed.username is None and parsed.password is None
                and parsed.port in (None, 443) and not parsed.fragment)
    except ValueError:
        return False


def fetch_pinned(url: str, *, expected_sha256: str | None = None,
                 max_bytes: int = 4_000_000, attempts: int = 4,
                 timeout_seconds: int = 25) -> FetchReceipt:
    """Retry only transport errors; fail immediately on content disagreement."""
    if not _valid_url(url):
        raise ValueError("source URL must be HTTPS on a frozen allowed host")
    if (expected_sha256 is not None and
            (not isinstance(expected_sha256, str) or
             re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None)):
        raise ValueError("expected SHA-256 must be 64 lowercase hex characters")
    if (type(max_bytes) is not int or not 1 <= max_bytes <= 64 * 1024 * 1024
            or type(attempts) is not int or not 1 <= attempts <= 5
            or type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 60):
        raise ValueError("invalid bounded source fetch policy")
    last_error = ""
    for number in range(1, attempts + 1):
        try:
            result = subprocess.run(
                ["curl", "--fail", "--location", "--silent", "--show-error",
                 "--proto", "=https", "--proto-redir", "=https",
                 "--max-redirs", "3", "--connect-timeout", "8",
                 "--max-time", str(timeout_seconds), "--max-filesize",
                 str(max_bytes), url],
                capture_output=True, check=False, timeout=timeout_seconds + 3,
            )
            if result.returncode == 63 or len(result.stdout) > max_bytes:
                raise IntegrityFailure("source exceeded the preregistered byte limit")
            if result.returncode == 0 and 0 < len(result.stdout) <= max_bytes:
                digest = sha256(result.stdout).hexdigest()
                if expected_sha256 is not None and digest != expected_sha256:
                    raise IntegrityFailure("frozen source SHA-256 changed")
                return FetchReceipt(result.stdout, digest, number)
            last_error = (f"curl exit {result.returncode}; "
                          f"{result.stderr.decode('utf-8', 'replace')[:180]}")
        except (OSError, subprocess.TimeoutExpired) as exc:
            last_error = f"transport exception {type(exc).__name__}: {exc}"
        if number < attempts:
            time.sleep(min(2.0, .25 * 2 ** (number - 1)))
    raise TransportFailure(f"source failed after {attempts} bounded attempts: {last_error}")


def audit_checksum_manifest(path: str | Path, expected_manifest_sha256: str,
                            *, workers: int = 4) -> ChecksumAudit:
    """Check official Binance checksum sidecars; never download price ZIPs."""
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("workers must be in [1,8]")
    if (not isinstance(expected_manifest_sha256, str) or
            re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256) is None):
        raise ValueError("manifest SHA-256 invalid")
    try:
        body = Path(path).read_bytes()
    except OSError as exc:
        raise IntegrityFailure(f"frozen price manifest missing: {exc}") from exc
    if sha256(body).hexdigest() != expected_manifest_sha256:
        raise IntegrityFailure("frozen price manifest SHA-256 changed")
    try:
        reader = csv.DictReader(io.StringIO(body.decode("utf-8-sig")))
        if tuple(reader.fieldnames or ()) != ("month", "url", "official_zip_sha256"):
            raise IntegrityFailure("price manifest schema changed")
        rows = list(reader)
        if not rows:
            raise IntegrityFailure("empty price manifest")
        for row in rows:
            if (set(row) != {"month", "url", "official_zip_sha256"}
                    or not re.fullmatch(r"20\d{2}-(0[1-9]|1[0-2])", row["month"])
                    or not re.fullmatch(r"[0-9a-f]{64}", row["official_zip_sha256"])
                    or not _valid_url(row["url"])
                    or urlsplit(row["url"]).hostname != "data.binance.vision"
                    or not row["url"].endswith(f"-1d-{row['month']}.zip")):
                raise IntegrityFailure("price manifest row invalid")

        def check(row: dict[str, str]) -> int:
            receipt = fetch_pinned(row["url"] + ".CHECKSUM", max_bytes=500)
            text = receipt.body.decode("ascii").strip()
            expected_name = row["url"].rsplit("/", 1)[-1]
            if re.fullmatch(r"[0-9a-f]{64}\s+" + re.escape(expected_name), text) is None:
                raise IntegrityFailure("official Binance checksum sidecar format changed")
            if text.split()[0] != row["official_zip_sha256"]:
                raise IntegrityFailure("official Binance checksum no longer matches manifest")
            return receipt.attempts - 1

        with ThreadPoolExecutor(max_workers=workers) as pool:
            retries = sum(pool.map(check, rows))
        return ChecksumAudit(len(rows), retries, True)
    except IntegrityFailure:
        raise
    except (csv.Error, UnicodeError, TypeError, KeyError, ValueError) as exc:
        raise IntegrityFailure(f"price manifest audit invalid: {exc}") from exc
