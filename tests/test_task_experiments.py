import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from dura.task_experiments import (RELAXED_LIMITS, STEP_SIZES, absolute_rows,
                                   evaluate_one, variant)


class TaskExperimentTests(unittest.TestCase):
    def test_task_filter_preserves_only_requested_instruction_and_absolute_images(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "a.png").write_bytes(b"x")
            manifest = root / "all.jsonl"
            manifest.write_text(
                json.dumps({"image": "a.png", "instruction": "one"}) + "\n" +
                json.dumps({"image": "a.png", "instruction": "two"}) + "\n"
            )
            rows = absolute_rows(manifest, "two")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["instruction"], "two")
            self.assertEqual(rows[0]["image"], str((root / "a.png").resolve()))

    def test_evaluate_one_reuses_report_from_same_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            final = root / "final"
            final.mkdir()
            (final / "patch.png").write_bytes(b"patch")
            comparison = {
                "conditions": {
                    "adversarial": {"asr": 0.6},
                    "seed": {"asr": 0.2},
                    "clean": {"asr": 0.1},
                },
                "conditional_on_both_baselines_succeeding": {"rate": 0.5},
            }
            report = {
                "raw_evaluation": str((root / "evaluations" / "example").resolve()),
                "patch_sha256": hashlib.sha256(b"patch").hexdigest(),
                "comparison": comparison,
            }
            (final / "asr_report.json").write_text(json.dumps(report))
            result = evaluate_one({"name": "example", "final": str(final)},
                                  root / "baseline", root, "/unused")
            self.assertEqual(result["status"], "evaluated")
            self.assertEqual(result["asr"], 0.6)
            self.assertEqual(result["conditional_asr"], 0.5)

    def test_variant_sets_relaxed_profile_and_does_not_mutate_base(self):
        base = {
            "policy": {"checkpoint": "old", "revision": "old", "unnorm_key": "old"},
            "diffusion": {"local_files_only": False},
            "naturalness": {"fixed_limits": True, "limits": {}, "step_sizes": []},
        }
        original = json.loads(json.dumps(base))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            validation = root / "val.jsonl"
            validation.write_text(json.dumps({"image": "/tmp/x", "instruction": "x"}) + "\n")
            cfg = variant(base, manifest=root / "train.jsonl", validation=validation,
                          seed=root / "seed.png", scene_seed=root / "scene.png",
                          output=root / "out", final=root / "final",
                          policy={"checkpoint": "new", "revision": "pin", "unnorm_key": "suite"})
        self.assertEqual(base, original)
        self.assertEqual(cfg["naturalness"]["limits"], RELAXED_LIMITS)
        self.assertEqual(cfg["naturalness"]["step_sizes"], STEP_SIZES)
        self.assertFalse(cfg["naturalness"]["fixed_limits"])
        self.assertFalse(cfg["policy"]["local_files_only"])
        self.assertFalse(cfg["diffusion"]["local_files_only"])


if __name__ == "__main__":
    unittest.main()
