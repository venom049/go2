"""Offline tests only: SDK imports and any robot access are excluded."""
import ast
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "deploy/go2_stairs/read_state.py"
SPEC = importlib.util.spec_from_file_location("read_state", SCRIPT)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def message(tick=42):
    return NS(
        tick=tick, power_v=28.5, power_a=1., level_flag=255, bit_flag=0,
        sn=[123456, 654321], wireless_remote=[0]*40, secret="excluded",
        bms_state=NS(status=0, soc=90),
        imu_state=NS(quaternion=[1., 0., 0., 0.], gyroscope=[0., 0., 0.],
                     accelerometer=[0., 0., 9.81], rpy=[0., 0., 0.], temperature=29),
        motor_state=[NS(q=.1*i, dq=0., tau_est=0., temperature=30, mode=1, lost=0)
                     for i in range(20)],
    )


class SanitizationTests(unittest.TestCase):
    def test_joint_order_and_allowlist(self):
        report = probe.sanitize_sample(message())
        self.assertEqual(len(report["motors"]), 12)
        self.assertEqual(report["motors"][0]["joint_name"], "FR_hip_joint")
        self.assertEqual(report["motors"][3]["joint_name"], "FL_hip_joint")
        self.assertEqual(report["motors"][9]["joint_name"], "RL_hip_joint")
        self.assertEqual(report["motors"][9]["q_rad"], .9)
        encoded = json.dumps(report, allow_nan=False)
        for excluded in ("123456", "654321", "wireless_remote", "secret"):
            self.assertNotIn(excluded, encoded)
        self.assertTrue(report["checks"]["policy_input_fields_finite"])

    def test_nonfinite_becomes_null_and_is_reported(self):
        msg = message()
        msg.motor_state[1].q = float("nan")
        msg.imu_state.gyroscope[2] = float("inf")
        msg.power_v = float("-inf")
        report = probe.sanitize_sample(msg)
        self.assertIsNone(report["motors"][1]["q_rad"])
        self.assertIn("motor_state[1].q", report["checks"]["nonfinite_fields"])
        self.assertFalse(report["checks"]["policy_input_fields_finite"])
        json.dumps(report, allow_nan=False)

    def test_missing_values_not_fabricated_as_zero(self):
        report = probe.sanitize_sample(NS())
        self.assertIsNone(report["tick_raw"])
        self.assertIsNone(report["imu"]["quaternion_wxyz"])
        self.assertIsNone(report["motors"][0]["q_rad"])
        self.assertIsNone(report["status_raw"]["bms_status"])
        self.assertFalse(report["checks"]["policy_input_fields_finite"])

    def test_truncated_motor_sequence_marks_missing_joint(self):
        msg = message()
        msg.motor_state = msg.motor_state[:5]
        report = probe.sanitize_sample(msg)
        self.assertIsNone(report["motors"][5]["q_rad"])
        self.assertIn("motor_state[5].q", report["checks"]["missing_fields"])

    def test_invalid_sequence_and_zero_quaternion_do_not_pass(self):
        msg = message()
        msg.imu_state.quaternion = [0.]*4
        msg.imu_state.gyroscope = "0,0,0"
        report = probe.sanitize_sample(msg)
        self.assertFalse(report["imu"]["quaternion_unit_within_0p01"])
        self.assertIn("imu.gyroscope", report["checks"]["invalid_fields"])
        self.assertFalse(report["checks"]["policy_input_fields_finite"])

    def test_missing_error_field_stays_unknown_and_status_raw(self):
        msg = message()
        msg.motor_state[0].lost = 4
        report = probe.sanitize_sample(msg)
        self.assertEqual(report["motors"][0]["lost_raw"], 4)
        self.assertNotIn("error_code", report["motors"][0])
        self.assertIn("no fault-clear", report["status_raw"]["interpretation"])

    def test_extreme_finite_quaternion_cannot_make_nonfinite_json(self):
        msg = message()
        msg.imu_state.quaternion = [1e308]*4
        report = probe.sanitize_sample(msg)
        self.assertIsNone(report["imu"]["quaternion_norm"])
        self.assertIn("imu.quaternion_norm", report["checks"]["nonfinite_fields"])
        json.dumps(report, allow_nan=False)


class SummaryTests(unittest.TestCase):
    def test_no_data_is_unknown_and_not_fresh(self):
        summary = probe.StateSummary(clock=lambda: 10.)
        report = summary.report(interface="test0", max_age=.2)
        self.assertEqual(report["sample_count"], 0)
        self.assertIsNone(report["fresh_receipt"])
        self.assertIsNone(report["observed_receive_rate_hz"])
        self.assertIsNone(report["latest"])

    def test_freshness_tick_progress_and_bounded_latest(self):
        clock = [10.]
        summary = probe.StateSummary(clock=lambda: clock[0])
        summary.receive(message(tick=2**32-1))
        clock[0] = 10.002
        summary.receive(message(tick=0))
        clock[0] = 10.5
        report = summary.report(interface="test0", max_age=.2)
        self.assertEqual(report["sample_count"], 2)
        self.assertFalse(report["fresh_receipt"])
        self.assertTrue(report["tick_progress_observed"])
        self.assertAlmostEqual(report["observed_receive_rate_hz"], 500.)
        self.assertIsNone(report["source_sample_age_s"])
        self.assertFalse(report["control_or_motion_performed"])
        self.assertIn("not verify mechanical safety", report["assessment_scope"])
        self.assertEqual(report["latest"]["tick_raw"], 0)

    def test_repeated_tick_is_not_progress(self):
        clock = [1.]
        summary = probe.StateSummary(clock=lambda: clock[0])
        summary.receive(message())
        clock[0] = 1.01
        summary.receive(message())
        report = summary.report(interface="test0", max_age=.2)
        self.assertTrue(report["fresh_receipt"])
        self.assertFalse(report["tick_progress_observed"])

    def test_callback_copies_input(self):
        summary = probe.StateSummary(clock=lambda: 1.)
        msg = message()
        summary.receive(msg)
        msg.motor_state[0].q = 99
        report = summary.report(interface="test0", max_age=.2)
        self.assertEqual(report["latest"]["motors"][0]["q_rad"], 0.)


class BoundaryTests(unittest.TestCase):
    def test_no_actuator_or_rpc_calls_and_sdk_import_is_lazy(self):
        tree = ast.parse(SCRIPT.read_text())
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        forbidden = {"ChannelPublisher", "LowCmd_", "SportClient", "MotionSwitcherClient",
                     "Write", "ReleaseMode", "SelectMode", "StandDown", "Move"}
        self.assertFalse(forbidden.intersection(names | attrs))
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                self.assertFalse((node.module or "").startswith("unitree"))

    def test_bad_duration_cannot_start_dds(self):
        with patch.object(probe, "open_subscription") as open_sub:
            for duration in (0., -1., 61., float("inf"), float("nan")):
                with self.subTest(duration=duration), self.assertRaises(ValueError):
                    probe.collect("test0", duration, .2)
            open_sub.assert_not_called()

    def test_bad_interface_cannot_import_or_start_dds(self):
        with patch.object(probe.socket, "if_nametoindex") as lookup:
            for interface in ("", "abc\"/>", "name with space", "x"*16):
                with self.subTest(interface=interface), self.assertRaises(ValueError):
                    probe.open_subscription(interface, lambda x: None)
            lookup.assert_not_called()

    def test_main_returns_sanitized_json_on_missing_dependency(self):
        with patch.object(probe, "collect", side_effect=ModuleNotFoundError("sensitive text")), \
                patch("sys.stdout", new_callable=io.StringIO) as stdout:
            code = probe.main(["--interface", "test0"])
        report = json.loads(stdout.getvalue())
        self.assertEqual(code, 1)
        self.assertEqual(report["error_type"], "ModuleNotFoundError")
        self.assertNotIn("sensitive text", stdout.getvalue())
        self.assertFalse(report["control_or_motion_performed"])

    def test_collect_closes_fake_subscription_and_ends(self):
        class FakeSubscriber:
            closed = False
            def Close(self):
                self.closed = True
        fake = FakeSubscriber()
        with patch.object(probe, "open_subscription", return_value=fake):
            report = probe.collect("test0", .001, .2)
        self.assertTrue(fake.closed)
        self.assertEqual(report["sample_count"], 0)


if __name__ == "__main__":
    unittest.main()
