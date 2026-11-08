"""Search *all tracked* prior research documents before a new preregistration.

This is a collision screen, not a proof of economic independence. Researchers
must inspect the matches and read the full prior registrations. In particular,
source-name changes must not masquerade as a new economic mechanism.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import unicodedata


FAMILY_ALIASES = {
    "real_yield": (
        "实际利率", "real yield", "real-yield", "TIPS", "DFII10",
        "inflation-indexed treasury", "real discount rate",
    ),
    "credit_spread": (
        "信用利差", "credit spread", "credit risk premium", "HYG/LQD",
        "BAA10Y", "high yield option-adjusted spread",
    ),
    "funding_carry": (
        "资金费率套利", "funding carry", "funding spread carry",
        "cash-and-carry", "cash and carry", "basis arbitrage",
    ),
    "exchange_supply": (
        "交易所储备", "exchange supply", "exchange reserves",
        "exchange-wallet inventory", "SplyExNtv",
    ),
}


def _fold(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def tracked_documents(repo: Path) -> tuple[Path, ...]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "Spot_Multi_Strategy/reports",
         "Spot_Multi_Strategy/research/preregistrations"],
        cwd=repo, capture_output=True, check=True,
    )
    files = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        path = Path(raw.decode("utf-8"))
        pieces = path.parts
        if (path.suffix == ".md" and
                ((len(pieces) >= 4 and pieces[1:3] == ("research", "preregistrations")) or
                 (len(pieces) >= 4 and pieces[1] == "reports" and
                  path.name in {"PREREGISTRATION.md", "RESULT.md", "REPORT.md"}))):
            files.append(path)
    return tuple(sorted(files))


def find_mentions(repo: Path, documents: tuple[Path, ...],
                  phrases: tuple[str, ...], *, exclude: Path | None = None
                  ) -> list[dict[str, object]]:
    folded = tuple(_fold(phrase) for phrase in phrases)
    if not folded or any(not phrase.strip() for phrase in folded):
        raise ValueError("at least one nonempty mechanism phrase is required")
    hits = []
    for path in documents:
        if exclude is not None and path == exclude:
            continue
        for number, line in enumerate((repo / path).read_text(encoding="utf-8").splitlines(), 1):
            matched = [phrase for phrase, normalized in zip(phrases, folded)
                       if normalized in _fold(line)]
            if matched:
                hits.append({"path": str(path), "line": number,
                             "phrases": matched, "excerpt": line[:220]})
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description="Screen tracked prior research for a mechanism")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--family", choices=sorted(FAMILY_ALIASES))
    parser.add_argument("--phrase", action="append", default=[])
    parser.add_argument("--exclude", type=Path,
                        help="new preregistration path relative to repository root")
    args = parser.parse_args()
    phrases = tuple((*FAMILY_ALIASES.get(args.family, ()), *args.phrase))
    docs = tracked_documents(args.repo)
    hits = find_mentions(args.repo, docs, phrases, exclude=args.exclude)
    print(json.dumps({
        "tracked_research_documents": len(docs),
        "phrases": phrases,
        "matching_documents": len({hit["path"] for hit in hits}),
        "matches": hits,
        "conclusion": "manual_prior_registration_review_required" if hits else
                      "no_text_match_economic_independence_still_unproven",
    }, ensure_ascii=False, indent=2))
    return 1 if hits else 0


if __name__ == "__main__":
    raise SystemExit(main())
