"""Algorithm/contract tests using artificial numeric features, not disease data."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from fit_subject_heads import fit, score, load_manifest


class SubjectHeadTests(unittest.TestCase):
    def test_ridge_finite_and_ordered(self):
        X = np.array([[0.], [1.], [2.]])
        y = np.array([0, 1, 1])
        model = fit(X, y)
        scores = score(model, X)
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreater(scores[2], scores[0])

    def test_one_class_rejected(self):
        with self.assertRaises(ValueError):
            fit(np.ones((3, 2)), np.ones(3))

    def make_manifest(self, folder, *, label="TUMOR_BEARING", species="mouse", duplicate_hash=False):
        rows = []
        for i in range(2):
            source = "same" if duplicate_hash else str(i)
            evidence = {"provenance": {"subject_id": "one-subject", "species": species,
                "tlc_profile_id": "profile", "device_profile_id": "device", "source_sha256": source},
                "candidates": [{"physical_contrast": .1, "hue_js": .1,
                    "saturation_signed_delta": .1, "value_signed_delta": -.1,
                    "dino_local_cosine_distance": None}]}
            (folder/f"{i}.json").write_text(json.dumps(evidence))
            rows.append({"subject_id":"one-subject", "species":species, "label":label,
                         "tlc_profile_id":"profile", "device_profile_id":"device", "evidence_path":f"{i}.json"})
        path = folder/"manifest.csv"
        with path.open("w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_repeats_are_one_subject(self):
        with tempfile.TemporaryDirectory() as folder:
            ids, X, y, _ = load_manifest(self.make_manifest(Path(folder)))
            self.assertEqual(len(ids), 1)
            self.assertEqual(X.shape, (1,6))
            self.assertTrue(np.isnan(X[0,5]))

    def test_duplicate_original_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                load_manifest(self.make_manifest(Path(folder), duplicate_hash=True))

    def test_benign_is_not_no_tumor(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                load_manifest(self.make_manifest(Path(folder), label="BENIGN"))

    def test_human_cannot_use_mouse_lane(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                load_manifest(self.make_manifest(Path(folder), species="human"))


if __name__ == "__main__":
    unittest.main()
