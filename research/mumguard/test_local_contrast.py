"""Geometric/photometric software tests. Artificial pixels are NOT biological evidence."""
import unittest
import numpy as np
from local_contrast import analyze


class ContrastTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.full((224, 224, 3), [30, 170, 80], np.uint8)
        self.mask = np.ones((224, 224), bool)

    def test_uniform_and_abstention(self):
        result, _, _ = analyze(self.rgb, self.mask)
        self.assertAlmostEqual(result["local_contrast_score"], 0)
        self.assertIsNone(result["research_binary_class"])
        self.assertEqual(result["decision_status"], "ABSTAIN_UNCALIBRATED")

    def test_focal_contrast(self):
        self.rgb[80:144, 80:144] = [180, 40, 50]
        result, heat, support = analyze(self.rgb, self.mask)
        self.assertGreater(result["local_contrast_score"], 0)
        self.assertGreater(heat[100:124, 100:124].mean(), heat[:16].mean())
        self.assertGreater(support.sum(), 0)

    def test_no_field_is_missing_not_healthy(self):
        result, _, _ = analyze(self.rgb, np.zeros_like(self.mask))
        self.assertIsNone(result["local_contrast_score"])
        self.assertIn("NO_USABLE_LOCAL_COMPARATORS", result["quality_reasons"])

    def test_dark_contact_is_not_normal(self):
        result, _, _ = analyze(np.zeros_like(self.rgb), self.mask)
        self.assertIsNone(result["local_contrast_score"])
        self.assertIn("LOW_CHROMATIC_SUPPORT", result["quality_reasons"])

    def test_excluded_artifact_does_not_become_evidence(self):
        self.rgb[80:144, 80:144] = [180, 40, 50]
        exclude = np.zeros_like(self.mask)
        exclude[80:144, 80:144] = True
        result, _, _ = analyze(self.rgb, self.mask, exclude)
        self.assertAlmostEqual(result["local_contrast_score"], 0)

    def test_tokens(self):
        tokens = np.ones((16, 16, 384))
        result, _, _ = analyze(self.rgb, self.mask, tokens=tokens)
        scores = [r["dino_local_cosine_distance"] for r in result["candidates"]
                  if r["dino_local_cosine_distance"] is not None]
        self.assertTrue(scores)
        self.assertLess(max(scores), 1e-12)

    def test_invalid_tokens_rejected(self):
        with self.assertRaises(ValueError):
            analyze(self.rgb, self.mask, tokens=np.zeros((16, 16, 384)))
        with self.assertRaises(ValueError):
            analyze(self.rgb, self.mask, tokens=np.full((16, 16, 384), np.nan))

    def test_hue_wrap_is_finite(self):
        self.rgb[:112] = [190, 5, 10]
        self.rgb[112:] = [190, 10, 5]
        result, heat, _ = analyze(self.rgb, self.mask)
        self.assertTrue(np.isfinite(heat).all())
        self.assertLess(result["local_contrast_score"], .02)


if __name__ == "__main__":
    unittest.main()
