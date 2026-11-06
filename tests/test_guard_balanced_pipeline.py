"""Protect acceptance decisions against missing evidence and task regressions."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("guard_pipeline", ROOT / "scripts/run_guard_balanced_training.py")
PIPELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PIPELINE)


def complete_rows(successes=60):
    rows = []
    for label in PIPELINE.LABELS:
        for height, tread, controller in PIPELINE.CONDITIONS:
            for seed in PIPELINE.SEEDS:
                joint = {
                    "audit_invalid": False, "joint_limits_passed": True,
                    "physics_samples_per_joint": 2000, "startup_physics_samples_per_joint": 400,
                    "velocity_limit_rad_s": 13.0, "tracking_error_limit_rad": 0.9,
                    "joints": {},
                }
                for kind in PIPELINE.JOINT_KINDS:
                    joint.update({f"{kind}_excess_joint_samples": 0,
                                  f"startup_{kind}_excess_joint_samples": 0,
                                  f"{kind}_excess_fraction": 0.0,
                                  f"startup_{kind}_excess_fraction": 0.0,
                                  f"{kind}_peak_ratio": 0.8})
                for name in PIPELINE.JOINT_NAMES:
                    lower, upper = PIPELINE.position_bounds(name)
                    joint["joints"][name] = {
                        "position_lower_rad": lower, "position_upper_rad": upper,
                        "velocity_peak_rad_s": 12.5, "tracking_error_peak_rad": 0.7,
                        "position_peak_overshoot_rad": 0.0, "target_position_peak_overshoot_rad": 0.0,
                        **{f"{kind}_excess_samples": 0 for kind in PIPELINE.JOINT_KINDS},
                    }
                torque = {
                    "physics_samples_per_joint": 2000, "startup_physics_samples_per_joint": 400,
                    "torque_limits_passed": True, "requested_excess_joint_samples": 0,
                    "applied_excess_joint_samples": 0, "startup_excess_joint_samples": 0,
                    "requested_excess_fraction": 0.0,
                    "joints": {name: {"limit_nm": 45.43 if "_calf_" in name else 23.7,
                                      "requested_peak_nm": 15.0, "applied_peak_nm": 15.0,
                                      "requested_excess_samples": 0}
                               for name in PIPELINE.JOINT_NAMES},
                }
                rows.append({"label": label,
                             "condition": PIPELINE.condition_name(label, height, controller, seed),
                             "height_m": height, "tread_m": tread, "controller": controller,
                             "seed": seed, "command_m_s": 0.6,
                             "episodes": 64, "successes": successes, "envs_incomplete": 0,
                             "joint_audit": joint, "torque_audit": torque,
                             "joint_acceptance_passed": True, "torque_acceptance_passed": True})
    return rows


class GuardBalancedAcceptanceTest(unittest.TestCase):
    def test_no_regression_can_be_eligible_without_perfect_climbing(self):
        rows = complete_rows(successes=60)
        original = copy.deepcopy(rows)
        result = PIPELINE.build_summary(rows)
        self.assertEqual(result["constraint_eligible_candidates"], ["guard500", "guard1000"])
        self.assertEqual(len(result["scene_comparisons"]), 6)
        self.assertEqual(result["scene_comparisons"][0]["successes"], 120)
        self.assertEqual(rows, original)

    def test_task_and_guard_results_stay_separate_and_each_scene_must_pass(self):
        rows = complete_rows()
        for row in rows:
            if row["label"] == "guard500" and row["height_m"] == 0.15:
                row["successes"] = 59
            if row["label"] == "guard500" and row["height_m"] == 0.18:
                row["successes"] = 64
        result = PIPELINE.build_summary(rows)
        self.assertEqual(result["constraint_eligible_candidates"], ["guard1000"])
        self.assertTrue(result["candidate_results"]["guard500"]["joint_and_torque_acceptance_passed"])
        self.assertFalse(result["candidate_results"]["guard500"]["task_nonregression_passed"])

    def test_zero_success_is_not_eligible_or_improved_gait(self):
        result = PIPELINE.build_summary(complete_rows(successes=0))
        self.assertEqual(result["constraint_eligible_candidates"], [])
        self.assertTrue(all(value is None for value in result["scene_comparisons"][0]["successful_episode_gait"].values()))

    def test_zero_rounded_fractions_do_not_hide_any_guard_peak(self):
        base = complete_rows()[0]
        for field, value in (("velocity_peak_rad_s", 13.001),
                             ("tracking_error_peak_rad", 0.901),
                             ("position_peak_overshoot_rad", 0.00001),
                             ("target_position_peak_overshoot_rad", 0.00001)):
            with self.subTest(field=field):
                row = copy.deepcopy(base)
                row["joint_audit"]["joints"]["RL_calf_joint"][field] = value
                self.assertFalse(PIPELINE.joint_acceptance(row))

    def test_missing_nonfinite_wrong_bound_and_incomplete_audits_fail_closed(self):
        base = complete_rows()[0]
        mutations = (
            lambda row: row.pop("joint_audit"),
            lambda row: row["joint_audit"].pop("audit_invalid"),
            lambda row: row["joint_audit"].update(audit_invalid=True),
            lambda row: row["joint_audit"].update(velocity_limit_rad_s=30.1),
            lambda row: row["joint_audit"].update(tracking_error_limit_rad=1.0),
            lambda row: row["joint_audit"].update(physics_samples_per_joint=0),
            lambda row: row["joint_audit"].update(startup_physics_samples_per_joint=0),
            lambda row: row["joint_audit"].update(position_peak_ratio=float("nan")),
            lambda row: row["joint_audit"].update(target_position_excess_fraction=float("inf")),
            lambda row: row["joint_audit"].pop("tracking_error_excess_joint_samples"),
            lambda row: row["joint_audit"].update(startup_velocity_excess_joint_samples=1),
            lambda row: row["joint_audit"]["joints"]["RL_thigh_joint"].update(position_lower_rad=-1.5708),
            lambda row: row["joint_audit"]["joints"]["FL_hip_joint"].update(velocity_peak_rad_s=float("inf")),
            lambda row: row["joint_audit"]["joints"].pop("FR_calf_joint"),
            lambda row: row.update(envs_incomplete=1),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                row = copy.deepcopy(base)
                mutate(row)
                self.assertFalse(PIPELINE.joint_acceptance(row))

    def test_torque_requests_and_applied_effort_both_checked(self):
        base = complete_rows()[0]
        mutations = (
            lambda row: row["torque_audit"]["joints"]["FL_hip_joint"].update(requested_peak_nm=24.0),
            lambda row: row["torque_audit"]["joints"]["FL_hip_joint"].update(applied_peak_nm=float("inf")),
            lambda row: row["torque_audit"]["joints"]["FL_hip_joint"].update(limit_nm=45.43),
            lambda row: row["torque_audit"].update(applied_excess_joint_samples=1),
            lambda row: row["torque_audit"].update(startup_excess_joint_samples=1),
            lambda row: row["torque_audit"].update(physics_samples_per_joint=1999),
            lambda row: row["torque_audit"].pop("requested_excess_fraction"),
            lambda row: row.pop("torque_audit"),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                row = copy.deepcopy(base)
                mutate(row)
                self.assertFalse(PIPELINE.torque_acceptance(row))

    def test_sample_weighting_and_peak_max_are_not_seed_means(self):
        rows = complete_rows()[:2]
        rows[1]["joint_audit"]["physics_samples_per_joint"] = 6000
        rows[0]["joint_audit"]["velocity_excess_joint_samples"] = 12
        rows[1]["joint_audit"]["velocity_excess_joint_samples"] = 12
        rows[1]["joint_audit"]["joints"]["RL_calf_joint"]["velocity_peak_rad_s"] = 14.0
        rows[0]["successful_episode_gait"] = {"stairs_body_roll_rms_rad": 1.0,
            "metric_element_counts": {"stairs_body_roll_rms_rad": 2}}
        rows[1]["successful_episode_gait"] = {"stairs_body_roll_rms_rad": 3.0,
            "metric_element_counts": {"stairs_body_roll_rms_rad": 6}}
        scene = PIPELINE.aggregate_scene(rows)
        self.assertEqual(scene["joint_audit"]["velocity_excess_fraction"], 24 / (8000 * 12))
        self.assertEqual(scene["joint_audit"]["velocity_peak_rad_s"], 14.0)
        self.assertAlmostEqual(scene["successful_episode_gait"]["stairs_body_roll_rms_rad"], 7 ** 0.5)

    def test_incomplete_duplicate_or_mislabeled_reports_cannot_finish(self):
        rows = complete_rows()
        mismatched = copy.deepcopy(rows)
        mismatched[0]["label"] = "guard1000"
        for bad in (rows[:-1], rows[:-1] + [copy.deepcopy(rows[0])], mismatched):
            with self.assertRaises(RuntimeError):
                PIPELINE.build_summary(bad)

    def test_reservation_never_overwrites_an_existing_job(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            names = {"OUT": out, "JOB": out / "job.json", "CONSOLE": out / "training.console.log",
                     "EVAL_CONSOLE": out / "evaluation.console.log", "SUMMARY": out / "comparison.json",
                     "SNAPSHOT": out / "snapshot.tar.gz", "MANIFEST": out / "source_manifest.json"}
            with patch.multiple(PIPELINE, **names):
                PIPELINE.reserve_job({"pid": 1})
                first = PIPELINE.JOB.read_bytes()
                with self.assertRaises(FileExistsError):
                    PIPELINE.reserve_job({"pid": 2})
                self.assertEqual(PIPELINE.JOB.read_bytes(), first)


if __name__ == "__main__":
    unittest.main()
