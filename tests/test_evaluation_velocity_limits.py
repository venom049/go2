"""Check name mapping and failure behavior without Isaac Lab or PyTorch."""
import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from go2_him.evaluation_velocity_limits import (
    GO2_JOINT_NAMES, load_evaluation_velocity_profile,
    verify_evaluation_velocity_readback,
)


class ArrayStub:
    def __init__(self, rows):
        self.rows = rows

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return copy.deepcopy(self.rows)


class EvaluationVelocityLimitsTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "profile.json"
        self.data = {"joints": [{"joint_name": name,
            "description_velocity_limit_rad_s": 15.7 if "calf" in name else 30.1}
            for name in reversed(GO2_JOINT_NAMES)]}

    def load(self, data=None):
        self.path.write_text(json.dumps(self.data if data is None else data))
        return load_evaluation_velocity_profile(self.path)

    def robot(self, profile):
        # PhysX and actuator orders deliberately differ from each other and
        # from the profile's input order. Every cap must follow the joint name.
        names = list(GO2_JOINT_NAMES)
        actuator_names = list(reversed(names))
        physics = ArrayStub([[profile.limits[name] for name in names] for _ in range(3)])
        actuator_rows = [[profile.limits[name] for name in actuator_names] for _ in range(3)]
        actuator = SimpleNamespace(joint_names=actuator_names,
            velocity_limit=ArrayStub(copy.deepcopy(actuator_rows)),
            velocity_limit_sim=ArrayStub(copy.deepcopy(actuator_rows)))
        return SimpleNamespace(joint_names=names, actuators={"legs": actuator},
            root_physx_view=SimpleNamespace(get_dof_max_velocities=lambda: physics)), physics

    def test_verified_profile_caps_and_source_hash(self):
        verified = Path(__file__).resolve().parents[1] / "validation/onboard_joint_limits_20261104/joint_limits.json"
        profile = load_evaluation_velocity_profile(verified)
        self.assertEqual(profile.sha256, hashlib.sha256(verified.read_bytes()).hexdigest())
        for name in GO2_JOINT_NAMES:
            self.assertEqual(profile.limits[name], 15.7 if "calf" in name else 30.1)
        self.assertFalse(profile.metadata()["measured_velocity_clipping"])
        self.assertEqual(profile.metadata()["audit_velocity_limit_rad_s"], 13.)

    def test_profile_mapping_applies_only_velocity_caps(self):
        profile = self.load()
        cfg = SimpleNamespace(velocity_limit=30., velocity_limit_sim=30.,
                              joint_velocity_limit=13., effort_limit=23.7, stiffness=40.)
        profile.apply(cfg)
        self.assertEqual(cfg.velocity_limit_sim["FR_calf_joint"], 15.7)
        self.assertEqual(cfg.velocity_limit["RL_thigh_joint"], 30.1)
        self.assertEqual((cfg.joint_velocity_limit, cfg.effort_limit, cfg.stiffness), (13., 23.7, 40.))
        cfg.velocity_limit["FR_calf_joint"] = 99.
        self.assertEqual(profile.limits["FR_calf_joint"], 15.7)
        self.assertEqual(cfg.velocity_limit_sim["FR_calf_joint"], 15.7)

    def test_invalid_names_counts_and_duplicate_entries_rejected(self):
        for mutation in ("missing", "extra", "duplicate", "unknown", "nonstring"):
            with self.subTest(mutation=mutation):
                data = copy.deepcopy(self.data)
                rows = data["joints"]
                if mutation == "missing":
                    rows.pop()
                elif mutation == "extra":
                    rows.append(copy.deepcopy(rows[0]))
                elif mutation == "duplicate":
                    rows[0]["joint_name"] = rows[1]["joint_name"]
                elif mutation == "unknown":
                    rows[0]["joint_name"] = "FR_ankle_joint"
                else:
                    rows[0]["joint_name"] = ["FR_calf_joint"]
                with self.assertRaises(ValueError):
                    self.load(data)

    def test_nonpositive_nonfinite_and_nonnumeric_limits_rejected(self):
        for value in (0, -1, float("nan"), float("inf"), True, "15.7", None):
            with self.subTest(value=value):
                data = copy.deepcopy(self.data)
                data["joints"][0]["description_velocity_limit_rad_s"] = value
                with self.assertRaises(ValueError):
                    self.load(data)

    def test_duplicate_json_keys_rejected(self):
        self.path.write_text('{"joints": [], "joints": []}')
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            load_evaluation_velocity_profile(self.path)

    def test_runtime_name_mapping_and_float32_roundoff(self):
        profile = self.load()
        robot, physics = self.robot(profile)
        physics.rows[1][0] = 30.100000381469727
        report = verify_evaluation_velocity_readback(profile, robot, 13.)
        self.assertEqual(report["physics"]["verified_environments"], 3)
        self.assertEqual(report["actuator_sim"]["per_joint"]["FR_calf_joint"]["max_rad_s"], 15.7)
        self.assertEqual(report["physics"]["per_joint"][GO2_JOINT_NAMES[0]]["max_rad_s"], physics.rows[1][0])

    def test_mismatch_in_later_environment_fails(self):
        profile = self.load()
        robot, physics = self.robot(profile)
        physics.rows[2][2] = 30.
        with self.assertRaisesRegex(ValueError, "physics velocity cap mismatch"):
            verify_evaluation_velocity_readback(profile, robot, 13.)

    def test_bad_actuator_readback_and_changed_audit_fail(self):
        profile = self.load()
        for kind in ("nan", "width", "count", "audit", "joint_names"):
            with self.subTest(kind=kind):
                robot, _ = self.robot(profile)
                actuator = robot.actuators["legs"]
                if kind == "nan":
                    actuator.velocity_limit_sim.rows[-1][0] = float("nan")
                elif kind == "width":
                    actuator.velocity_limit.rows[-1].pop()
                elif kind == "count":
                    actuator.velocity_limit.rows.pop()
                elif kind == "joint_names":
                    actuator.joint_names[-1] = actuator.joint_names[0]
                with self.assertRaises(ValueError):
                    verify_evaluation_velocity_readback(profile, robot, 30. if kind == "audit" else 13.)


if __name__ == "__main__":
    unittest.main()
