import json
import unittest
from pathlib import Path

import numpy as np
import torch

from fruitbowl import retarget, worldmodel
from fruitbowl.env import HALF, fit_layout
from fruitbowl.perception import target_from_text


class GeometryTests(unittest.TestCase):
    def test_missing_hand_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "fewer than 2"):
            retarget.fill_and_smooth(np.full((12, 3), np.nan))

    def test_short_hand_gap_is_interpolated(self):
        trajectory = np.column_stack([np.arange(10, dtype=float)] * 3)
        trajectory[3:5] = np.nan
        restored, metadata = retarget.fill_and_smooth(trajectory, window=1)
        np.testing.assert_allclose(restored[:, 0], np.arange(10))
        self.assertEqual(metadata["longest_gap"], 2)
        self.assertTrue(metadata["gap_ok"])

    def test_long_hand_gap_is_flagged(self):
        trajectory = np.zeros((22, 3))
        trajectory[1:21] = np.nan
        restored, metadata = retarget.fill_and_smooth(trajectory)
        self.assertTrue(np.isfinite(restored).all())
        self.assertFalse(metadata["gap_ok"])

    def test_grasp_release_follow_aperture(self):
        pinch = np.zeros((30, 3))
        aperture = np.full(30, 0.06)
        aperture[10:21] = 0.01
        self.assertEqual(retarget.find_events(pinch, aperture), (10, 21))

    def test_yaw_is_symmetric_under_finger_reversal(self):
        thumb = np.zeros((2, 3))
        index = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        np.testing.assert_allclose(retarget.closing_yaw(thumb, index), retarget.closing_yaw(index, thumb))

    def test_layout_fitting_preserves_a_feasible_layout(self):
        bowl = np.array([0.0, 0.0])
        fruits = np.array([[0.1, 0.1], [-0.1, 0.1], [0.0, -0.15]])
        fitted_bowl, fitted_fruits = fit_layout(bowl, fruits)
        np.testing.assert_allclose(fitted_bowl, bowl)
        np.testing.assert_allclose(fitted_fruits, fruits)
        self.assertTrue((np.abs(fitted_fruits) <= HALF - 0.04).all())


class InterfaceTests(unittest.TestCase):
    def test_auc_gives_tied_predictions_average_ranks(self):
        scores = np.array([0.0, 0.0, 1.0, 1.0])
        labels = np.array([False, True, False, True])
        self.assertAlmostEqual(worldmodel.binary_auc(scores, labels), 0.5)

    def test_target_synonyms(self):
        self.assertEqual(target_from_text("Put the nashi in the bowl"), "pear")
        self.assertEqual(target_from_text("Move the tangerine"), "mandarin")
        self.assertEqual(target_from_text("Pick up the kaki"), "persimmon")
        self.assertIsNone(target_from_text("Move the banana"))

    def test_world_model_feature_dimensions(self):
        states = torch.zeros(3, 22)
        actions = torch.zeros(3, 6)
        self.assertEqual(tuple(worldmodel.features(states, actions).shape), (3, 43))


class EvidenceTests(unittest.TestCase):
    def test_world_model_auc_audit_recomputes(self):
        path = Path(__file__).resolve().parents[1] / "results/worldmodel_mine_auc_audit.json"
        record = json.loads(path.read_text())
        self.assertEqual(len(record["scores"]), record["episodes"])
        self.assertEqual(len(record["success_labels"]), record["episodes"])
        self.assertAlmostEqual(worldmodel.binary_auc(record["scores"], record["success_labels"]), record["tie_aware_auc"])

    def test_source_ablation_matches_episode_records(self):
        root = Path(__file__).resolve().parents[1] / "results"
        for source in ("expert", "egodex", "mine"):
            for split in ("random", "heldout", "crowded"):
                with self.subTest(source=source, split=split):
                    record = json.loads((root / f"abl_{source}_{split}.json").read_text())
                    episodes = record["episodes"]
                    self.assertEqual(len(episodes), 150)
                    self.assertEqual(record["summary"]["episodes"], len(episodes))
                    self.assertAlmostEqual(record["summary"]["success"], np.mean([episode["success"] for episode in episodes]))
                    self.assertAlmostEqual(record["summary"]["wrong_fruit_in_bowl"], np.mean([episode["others_in_bowl"] for episode in episodes]))

    def test_own_heldout_set_has_six_distinct_clips(self):
        path = Path(__file__).resolve().parents[1] / "results/session2_heldout_layouts.json"
        layouts = json.loads(path.read_text())
        self.assertEqual(len(layouts), 6)
        self.assertEqual(len({layout["clip"] for layout in layouts}), 6)


if __name__ == "__main__":
    unittest.main()
