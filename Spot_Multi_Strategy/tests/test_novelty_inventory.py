"""Synthetic search behavior and a read-only tracked-inventory smoke check."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from research.novelty_inventory import FAMILY_ALIASES, find_mentions, tracked_documents


class NoveltyInventoryTests(unittest.TestCase):
    def test_synonyms_find_source_names_and_chinese_mechanism(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prior = Path("prior.md")
            current = Path("current.md")
            (root / prior).write_text("美国十年期 TIPS 实际利率下降\n", encoding="utf-8")
            (root / current).write_text("FRED DFII10 real yield easing\n", encoding="utf-8")
            found = find_mentions(root, (prior, current),
                                  FAMILY_ALIASES["real_yield"], exclude=current)
            self.assertEqual({hit["path"] for hit in found}, {str(prior)})
            self.assertTrue(any("TIPS" in hit["phrases"] for hit in found))
            self.assertTrue(any("实际利率" in hit["phrases"] for hit in found))

    def test_empty_query_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                find_mentions(Path(tmp), (), ())

    def test_tracked_inventory_covers_legacy_and_new_preregistrations(self):
        repo = Path(__file__).resolve().parents[2]
        paths = tracked_documents(repo)
        self.assertTrue(any("mini_medallion_tips_real_yield_crypto_oos"
                            in str(path) for path in paths))
        self.assertTrue(any(path.name == "2026-11-08_us_real_yield_crypto_risk.md"
                            for path in paths))
        self.assertGreaterEqual(len(paths), 176)
        current = Path("Spot_Multi_Strategy/research/preregistrations/"
                       "2026-11-08_us_real_yield_crypto_risk.md")
        hits = find_mentions(repo, paths, FAMILY_ALIASES["real_yield"],
                             exclude=current)
        self.assertTrue(any("mini_medallion_tips_real_yield_crypto_oos/"
                            "PREREGISTRATION.md" in hit["path"] for hit in hits))

    def test_dollar_index_source_change_finds_prior_dxy_experiment(self):
        repo = Path(__file__).resolve().parents[2]
        paths = tracked_documents(repo)
        hits = find_mentions(repo, paths, FAMILY_ALIASES["dollar_strength"])
        self.assertTrue(any(
            "mini_medallion_dxy_trend_oos/PREREGISTRATION.md" in hit["path"]
            for hit in hits
        ))


if __name__ == "__main__":
    unittest.main()
