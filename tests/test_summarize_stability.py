"""Pooling arithmetic and provenance checks; requires only the standard library."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location("summarize_stability", Path(__file__).resolve().parents[1] / "scripts/summarize_stability.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class StabilitySummaryTests(unittest.TestCase):
    def metrics(self, samples, mean, rms, traversals, seconds, speed):
        return {
            "sample_mean": mean, "body_roll_rms_rad": rms,
            "applied_vs_clipped_raw_action_rms": rms,
            "stairs_traversal_time_mean_s": seconds,
            "stairs_traversal_speed_mean_m_s": speed,
            "metric_element_counts": {"sample_mean": samples, "body_roll_rms_rad": samples,
                "applied_vs_clipped_raw_action_rms": samples,
                "stairs_traversal_time_mean_s": traversals, "stairs_traversal_speed_mean_m_s": traversals},
            "valid_env_samples": samples, "stairs_valid_env_samples": samples,
            "stairs_completed_traversals": traversals, "nonfinite_metric_elements": 0,
            "raw_action_abs_max": mean, "lateral_metric_thresholds": {"hip_abduction_max_rad": .22},
            "selection": "all_environments",
        }

    def report(self, seed, episodes=1, successes=1, controller="stairs", height=.18, speed=.4):
        return {
            "settings": {"seed": seed, "terrain": {"height_min": height, "height_max": height},
                         "commands": {"forward_min": speed, "forward_max": speed}},
            "controller": controller, "checkpoint": "/models/model_600.pt",
            "episodes_per_env_target": 1, "episodes": episodes, "successes": successes,
            "failures": episodes-successes, "timeouts": 0, "envs_incomplete": 0,
            "success_rate": successes/episodes, "stop_reason": "episode_target_reached",
        }

    def write(self, root, name, report):
        p = Path(root) / name
        p.write_text(json.dumps(report))
        return p

    def test_pool_uses_element_counts_squared_rms_and_traversal_counts(self):
        first = self.metrics(1, 2., 1., 1, 10., .2)
        second = self.metrics(9, 4., 3., 2, 4., .5)
        r = summary.pool_metrics([first, second])
        self.assertAlmostEqual(r["sample_mean"], 3.8)
        self.assertAlmostEqual(r["body_roll_rms_rad"], math.sqrt(8.2))
        self.assertAlmostEqual(r["applied_vs_clipped_raw_action_rms"], math.sqrt(8.2))
        self.assertAlmostEqual(r["stairs_traversal_time_mean_s"], 6.)
        self.assertAlmostEqual(r["stairs_traversal_speed_mean_m_s"], .4)
        self.assertEqual(r["metric_element_counts"]["stairs_traversal_time_mean_s"], 3)
        self.assertEqual(r["stairs_completed_traversals"], 3)
        self.assertEqual(r["valid_env_samples"], 10)
        self.assertEqual(r["raw_action_abs_max"], 4.)

    def test_zero_samples_do_not_dilute_means_and_no_traversals_stay_null(self):
        empty = self.metrics(0, None, None, 0, None, None)
        empty["raw_action_abs_max"] = None
        populated = self.metrics(3, 5., 2., 0, None, None)
        r = summary.pool_metrics([empty, populated])
        self.assertEqual(r["sample_mean"], 5.)
        self.assertEqual(r["body_roll_rms_rad"], 2.)
        self.assertIsNone(r["stairs_traversal_time_mean_s"])
        self.assertIsNone(r["stairs_traversal_speed_mean_m_s"])
        self.assertEqual(r["metric_element_counts"]["stairs_traversal_time_mean_s"], 0)
        self.assertIsNone(summary.pool_metrics([None, None]))
        json.dumps(r, allow_nan=False)

    def test_group_success_denominators_and_all_vs_successful_metrics_remain_separate(self):
        a = self.report(42)
        b = self.report(43, episodes=3, successes=2)
        a["gait_metrics"] = self.metrics(1, 2., 1., 1, 10., .2)
        b["gait_metrics"] = self.metrics(9, 4., 3., 2, 4., .5)
        a["gait_metrics_successful_episodes"] = self.metrics(2, 10., 1., 1, 10., .2)
        b["gait_metrics_successful_episodes"] = self.metrics(2, 20., 3., 2, 4., .5)
        other = self.report(42, controller="hybrid", height=0., speed=.6)
        with tempfile.TemporaryDirectory() as folder:
            paths = [self.write(folder, "after_stairs_0.18m_0.4mps_seed42.json", a),
                     self.write(folder, "after_stairs_0.18m_0.4mps_seed43.json", b),
                     self.write(folder, "after_hybrid_0m_0.6mps_seed42.json", other),
                     self.write(folder, "comparison.json", {"schema_version": 1, "groups": []})]
            result = summary.summarize(paths)
        self.assertEqual(len(result["groups"]), 2)
        r = next(g for g in result["groups"] if g["controller"] == "stairs")
        self.assertEqual(r["successes"], 3)
        self.assertEqual(r["episodes"], 4)
        self.assertEqual(r["success_rate"], .75)
        self.assertAlmostEqual(r["gait_metrics"]["sample_mean"], 3.8)
        self.assertEqual(r["gait_metrics_successful_episodes"]["sample_mean"], 15.)
        self.assertEqual(r["seeds"], [42, 43])
        self.assertEqual(r["per_seed"][1]["success_rate"], 2/3)
        self.assertEqual(r["per_seed"][1]["gait_metrics"], b["gait_metrics"])
        self.assertEqual(len(result["skipped_non_evaluation_files"]), 1)

    def test_label_rules_override_filename_but_settings_determine_conditions(self):
        report = self.report(43, height=.18, speed=.6)
        rules = summary.parse_label_rules(["unusual*=candidate600"])
        key, seed = summary.report_identity(Path("unusual.json"), report, rules)
        self.assertEqual(key, ("candidate600", "stairs", .18, .6))
        self.assertEqual(seed, 43)
        with self.assertRaises(ValueError):
            summary.report_identity(Path("unusual.json"), report, rules + [("*", "other")])
        report["settings"]["commands"]["forward_max"] = .8
        with self.assertRaises(ValueError):
            summary.report_identity(Path("unusual.json"), report, rules)

    def test_duplicate_seed_mismatched_counts_and_nonfinite_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            report = self.report(42)
            a = self.write(folder, "before_stairs_0.18m_0.4mps_seed42.json", report)
            b = self.write(folder, "copy_stairs_0.18m_0.4mps_seed42.json", report)
            with self.assertRaisesRegex(ValueError, "Duplicate seed"):
                summary.summarize([a, b], [("*", "same")])
        invalid = self.metrics(1, 2., 1., 1, 10., .2)
        invalid["stairs_completed_traversals"] = 2
        with self.assertRaisesRegex(ValueError, "traversals"):
            summary.pool_metrics([invalid])
        invalid = self.metrics(1, float("nan"), 1., 1, 10., .2)
        with self.assertRaisesRegex(ValueError, "finite"):
            summary.pool_metrics([invalid])

    def test_different_diagnostic_thresholds_cannot_be_silently_pooled(self):
        first = self.metrics(1, 2., 1., 1, 10., .2)
        other = copy.deepcopy(first)
        other["lateral_metric_thresholds"]["hip_abduction_max_rad"] = .3
        with self.assertRaisesRegex(ValueError, "lateral_metric_thresholds"):
            summary.pool_metrics([first, other])

    def test_policy_switch_counts_keep_seed_environment_identity_and_missing_data(self):
        a = {"walk_transition_stairs_samples": [10, 5, 20], "mode_transitions_per_env": [2, 1],
             "flat_stairs_flat_completed_per_env": [True, False],
             "flat_stairs_flat_completion_counts_per_env": [1, 0], "max_joint_target_delta_rad": .1}
        b = {"walk_transition_stairs_samples": [30, 4, 40], "mode_transitions_per_env": [0],
             "max_joint_target_delta_rad": .2}  # older schema: sequence is unknown, not false
        r = summary.pool_switch([(Path("a"), 42, {"policy_switch": a}),
                                 (Path("b"), 43, {"policy_switch": b})])
        self.assertEqual(r["walk_transition_stairs_samples"], [40, 9, 60])
        self.assertEqual(r["mode_transitions_total"], 3)
        self.assertEqual(r["flat_stairs_flat_completed_envs"], 1)
        self.assertEqual(r["environments_with_sequence_report"], 2)
        self.assertEqual(r["flat_stairs_flat_completed_fraction"], .5)
        self.assertEqual(r["flat_stairs_flat_completions_total"], 1)
        self.assertEqual(r["per_environment"][-1], {"seed": 43, "env_id": 0, "mode_transitions": 0})
        self.assertEqual(r["max_joint_target_delta_rad"], .2)


if __name__ == "__main__":
    unittest.main()
