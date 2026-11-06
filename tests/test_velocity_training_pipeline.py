"""Acceptance regressions for the velocity-focused training experiment.

These tests exercise report evidence, not simulator motion. In particular, a
safe speed report can coexist with position failures, whereas missing or
nonfinite speed evidence must never become a pass.
"""
import copy
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("velocity_pipeline", ROOT / "scripts/run_velocity_training.py")
PIPELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PIPELINE)


def complete_rows():
    rows = []
    for label in PIPELINE.LABELS:
        for height, tread, controller in PIPELINE.CONDITIONS:
            for seed in PIPELINE.SEEDS:
                joint = {
                    "audit_invalid": False, "joint_limits_passed": False,
                    "physics_samples_per_joint": 2000, "startup_physics_samples_per_joint": 400,
                    "velocity_limit_rad_s": 13.0, "velocity_excess_joint_samples": 0,
                    "startup_velocity_excess_joint_samples": 0,
                    "velocity_excess_fraction": 0.0, "startup_velocity_excess_fraction": 0.0,
                    "velocity_peak_ratio": 12.5 / 13.0,
                    # These remain visible despite not being this experiment's objective.
                    "position_excess_joint_samples": 20, "position_excess_fraction": 20 / 24000,
                    "target_position_excess_joint_samples": 30,
                    "tracking_error_excess_joint_samples": 40,
                    "joints": {name: {"velocity_peak_rad_s": 12.5, "velocity_excess_samples": 0,
                                      "position_peak_overshoot_rad": 0.01,
                                      "target_position_peak_overshoot_rad": 0.02,
                                      "tracking_error_peak_rad": 0.95}
                               for name in PIPELINE.JOINT_NAMES},
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
                             "episodes": 64, "successes": 64, "envs_incomplete": 0,
                             "joint_audit": joint, "torque_audit": torque,
                             "joint_acceptance_passed": False, "torque_acceptance_passed": True})
    return rows


class VelocityEligibilityTest(unittest.TestCase):
    def test_secondary_joint_violations_stay_visible_without_vetoing_speed(self):
        rows = complete_rows()
        before = copy.deepcopy(rows)
        result = PIPELINE.build_summary(rows)
        self.assertEqual(result["constraint_eligible_candidates"], ["velocity500", "velocity1000"])
        self.assertTrue(all(row["speed_acceptance_passed"] for row in result["rows"]))
        self.assertTrue(all(not row["joint_acceptance_passed"] for row in result["rows"]))
        self.assertTrue(all(row["joint_audit"]["position_excess_joint_samples"] == 20
                            for row in result["rows"]))
        self.assertEqual(rows, before)

    def test_a_single_failed_seed_or_climb_prevents_candidate_eligibility(self):
        rows = complete_rows()
        row = next(row for row in rows if row["label"] == "velocity500" and row["seed"] == 43)
        row["successes"] = 63
        self.assertEqual(PIPELINE.build_summary(rows)["constraint_eligible_candidates"], ["velocity1000"])

    def test_peak_over_limit_cannot_hide_behind_zero_rounded_fraction(self):
        rows = complete_rows()
        row = next(row for row in rows if row["label"] == "velocity1000")
        row["joint_audit"]["joints"]["RL_calf_joint"]["velocity_peak_rad_s"] = 13.001
        self.assertFalse(PIPELINE.speed_acceptance(row))
        self.assertEqual(PIPELINE.build_summary(rows)["constraint_eligible_candidates"], ["velocity500"])

    def test_missing_or_invalid_speed_evidence_fails_closed(self):
        base = complete_rows()[0]
        mutations = (
            lambda row: row.pop("joint_audit"),
            lambda row: row["joint_audit"].pop("audit_invalid"),
            lambda row: row["joint_audit"].update(audit_invalid=True),
            lambda row: row["joint_audit"].pop("velocity_excess_joint_samples"),
            lambda row: row["joint_audit"].update(velocity_excess_joint_samples=1),
            lambda row: row["joint_audit"].update(physics_samples_per_joint=0),
            lambda row: row["joint_audit"].update(startup_physics_samples_per_joint=0),
            lambda row: row["joint_audit"].update(velocity_limit_rad_s=30.0),
            lambda row: row["joint_audit"].update(velocity_peak_ratio=float("nan")),
            lambda row: row["joint_audit"].update(velocity_excess_fraction=float("inf")),
            lambda row: row["joint_audit"].update(joints={}),
            lambda row: row["joint_audit"]["joints"]["FL_hip_joint"].update(velocity_peak_rad_s=float("inf")),
            lambda row: row.update(envs_incomplete=1),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                row = copy.deepcopy(base)
                mutate(row)
                self.assertFalse(PIPELINE.speed_acceptance(row))

    def test_torque_clipping_does_not_excuse_requested_effort_violation(self):
        row = complete_rows()[0]
        row["torque_audit"]["joints"]["FL_hip_joint"]["requested_peak_nm"] = 24.0
        self.assertFalse(PIPELINE.torque_acceptance(row))

    def test_incomplete_nonfinite_or_mismatched_torque_evidence_fails_closed(self):
        base = complete_rows()[0]
        mutations = (
            lambda row: row.pop("torque_audit"),
            lambda row: row["torque_audit"].pop("applied_excess_joint_samples"),
            lambda row: row["torque_audit"].update(applied_excess_joint_samples=1),
            lambda row: row["torque_audit"].update(physics_samples_per_joint=1999),
            lambda row: row["torque_audit"].update(requested_excess_fraction=float("nan")),
            lambda row: row["torque_audit"]["joints"]["FL_hip_joint"].update(applied_peak_nm=float("inf")),
            lambda row: row["torque_audit"]["joints"].pop("FR_calf_joint"),
            lambda row: row["torque_audit"]["joints"]["FL_hip_joint"].update(limit_nm=45.43),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                row = copy.deepcopy(base)
                mutate(row)
                self.assertFalse(PIPELINE.torque_acceptance(row))

    def test_incomplete_or_duplicate_comparison_cannot_be_final(self):
        rows = complete_rows()
        for bad in (rows[:-1], rows[:-1] + [copy.deepcopy(rows[0])]):
            with self.assertRaises(RuntimeError):
                PIPELINE.build_summary(bad)


if __name__ == "__main__":
    unittest.main()
