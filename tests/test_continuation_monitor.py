"""Monitor protocol tests; no simulator, torch, SSH, or training is required."""
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_continuation.py"
SPEC = importlib.util.spec_from_file_location("continuation_monitor", SCRIPT)
monitor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(monitor)


def iteration(completed, loss="0.1", include_footer=True):
    return (f"\x1b[1m Learning iteration {completed - 1}/1600 \x1b[0m\n"
            f"Mean value_function loss: {loss}\n"
            "Mean surrogate loss: -0.002\nMean estimation loss: 0.01\nMean swap loss: 3.2\n"
            "Curriculum/stairs/success_rate: 0.98\nCurriculum/stairs/terrain_level: 8.9\n"
            "Mean reward: 22.1\n"
            + ("Total timesteps: 12288000\nTime elapsed: 00:10:00\nETA: 00:10:00\n" if include_footer else ""))


class ParseTests(unittest.TestCase):
    def test_zero_based_log_translates_to_additional_updates(self):
        record = monitor.parse_iterations(iteration(1100))[0]
        self.assertEqual(record["completed_updates"], 1100)
        self.assertEqual(record["new_updates"], 500)
        self.assertEqual(record["metrics"]["ETA"], "00:10:00")
        self.assertEqual(record["numeric_metrics"]["Curriculum/stairs/success_rate"], .98)
        self.assertTrue(record["losses_finite"])

    def test_partial_last_iteration_is_not_reported(self):
        records = monitor.parse_iterations(iteration(1099) + iteration(1100, include_footer=False))
        self.assertEqual([record["completed_updates"] for record in records], [1099])

    def test_nonfinite_losses_remain_valid_json_and_are_flagged(self):
        for nonfinite in ("nan", "-inf", "+Infinity"):
            with self.subTest(nonfinite=nonfinite):
                record = monitor.parse_iterations(iteration(1100, nonfinite))[0]
                self.assertFalse(record["losses_finite"])
                self.assertFalse(record["metrics_finite"])
                self.assertEqual(record["nonfinite_metrics"], ["Mean value_function loss"])
                self.assertIsNone(record["numeric_metrics"]["Mean value_function loss"])
                json.dumps(record, allow_nan=False)

    def test_missing_loss_is_unknown_not_healthy(self):
        text = iteration(1100).replace("Mean swap loss: 3.2\n", "")
        self.assertIsNone(monitor.parse_iterations(text)[0]["losses_finite"])


class JointTensorBoardTests(unittest.TestCase):
    @staticmethod
    def accumulator(events):
        return SimpleNamespace(Tags=lambda: {"scalars": list(events)}, Scalars=events.__getitem__)

    @staticmethod
    def event(step, value, wall_time=1):
        return SimpleNamespace(step=step, value=value, wall_time=wall_time)

    def test_exact_step_keeps_peaks_and_small_violations_without_averaging(self):
        fraction = "JointLimits/velocity_excess_fraction"
        peak = "JointLimits/velocity_peak_ratio"
        accumulator = self.accumulator({
            fraction: [self.event(499, .0000001), self.event(500, .8)],
            peak: [self.event(499, 1.4), self.event(499, 1.1, 2), self.event(500, 2.0)],
            "Torque/requested_peak_ratio": [self.event(499, 3)],
            "Torque/requested_excess_fraction": [self.event(499, 1e-8), self.event(500, .1)],
            "GuardBalance/scale": [self.event(499, 1.0)],
            "Loss/surrogate": [self.event(499, .01)],
        })
        data = monitor.joint_metrics_at_updates(accumulator, [500, 1000])
        self.assertEqual(data[500]["step"], 499)
        self.assertEqual(data[500]["metrics"][fraction], .0000001)
        self.assertEqual(data[500]["metrics"][peak], 1.4)
        self.assertEqual(data[500]["metrics"]["Torque/requested_peak_ratio"], 3)
        self.assertEqual(data[500]["metrics"]["Torque/requested_excess_fraction"], 1e-8)
        self.assertEqual(data[500]["metrics"]["GuardBalance/scale"], 1.0)
        self.assertNotIn("Loss/surrogate", data[500]["metrics"])
        self.assertEqual(data[1000]["metrics"], {})
        self.assertIsNone(data[1000]["all_metrics_finite"])

    def test_nonfinite_duplicate_is_not_erased_and_json_stays_valid(self):
        tag = "JointLimits/position_peak_ratio"
        accumulator = self.accumulator({tag: [self.event(499, float("nan")), self.event(499, .9, 2)]})
        data = monitor.joint_metrics_at_updates(accumulator, [500])[500]
        self.assertEqual(data["nonfinite_metrics"], [tag])
        self.assertIsNone(data["metrics"][tag])
        self.assertFalse(data["all_metrics_finite"])
        json.dumps(data, allow_nan=False)

    def test_completeness_and_latest_duplicate_fraction(self):
        events = {tag: [self.event(499, 0)] for tag in monitor.JOINT_METRICS}
        tag = "JointLimits/violation_environment_fraction"
        events[tag] = [self.event(499, .3, 3), self.event(499, .1, 1)]
        data = monitor.joint_metrics_at_updates(self.accumulator(events), [500])[500]
        self.assertEqual(data["metrics"][tag], .3)
        self.assertTrue(data["all_metrics_finite"])
        self.assertEqual(data["missing_metrics"], [])


class AcknowledgementTests(unittest.TestCase):
    def test_ack_cli_writes_only_local_state_and_never_fetches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "connection.json"
            config.write_text(json.dumps({"remote_job": "/remote/job.json"}))
            state, snapshot = root / "state.json", root / "snapshot.json"
            with patch.object(monitor, "state_paths", return_value=(state, snapshot)), \
                    patch.object(monitor, "fetch_snapshot") as fetch, \
                    patch("sys.stdout", new_callable=io.StringIO):
                monitor.main(["--config", str(config), "--ack", "1100", "--ack-event", "evaluating"])
            fetch.assert_not_called()
            self.assertEqual(json.loads(state.read_text())["last_reported_milestone"], 1100)
            self.assertFalse(snapshot.exists())

    def test_ack_is_monotonic_and_rejects_old_baseline_milestones(self):
        state = monitor.acknowledge({}, 1600)
        self.assertEqual(monitor.acknowledge(state, 1100)["last_reported_milestone"], 1600)
        for invalid in (500, 600, 1000, 1500, 1601):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                monitor.acknowledge({}, invalid)

    def test_remote_job_paths_have_independent_state(self):
        first = monitor.state_paths("/runs/job-a.json", "/tmp")
        self.assertEqual(first, monitor.state_paths("/runs/job-a.json", "/tmp"))
        self.assertNotEqual(first, monitor.state_paths("/runs/job-b.json", "/tmp"))
        self.assertNotEqual(first[0].name, "report_state.json")

    def test_milestone_filter_preserves_each_unreported_node(self):
        data, _ = monitor.report_filter({"status": "running", "milestones": [
            {"completed_updates": 1100}, {"completed_updates": 1600}]}, {})
        self.assertEqual(len(data["pending_milestones"]), 2)
        data, _ = monitor.report_filter({"status": "running", "milestones": data["pending_milestones"]},
                                        {"last_reported_milestone": 1100})
        self.assertEqual(data["pending_milestones"], [{"completed_updates": 1600}])
        self.assertIsNone(data["event_to_report"])

    def test_unreachable_repeat_suppressed_and_recurring_outage_reportable(self):
        state = monitor.acknowledge({}, event="unreachable")
        data, state = monitor.report_filter({"status": "unreachable"}, state)
        self.assertIsNone(data["event_to_report"])
        data, state = monitor.report_filter({"status": "running"}, state)
        self.assertTrue(data["connection_recovered"])
        data, _ = monitor.report_filter({"status": "unreachable"}, state)
        self.assertEqual(data["event_to_report"], "unreachable")

    def test_evaluating_event_only_once_even_across_connection_outage(self):
        state = monitor.acknowledge({}, event="evaluating")
        state = monitor.acknowledge(state, event="unreachable")
        data, _ = monitor.report_filter({"status": "evaluating"}, state)
        self.assertIsNone(data["event_to_report"])


class SnapshotTests(unittest.TestCase):
    def test_joint_tensorboard_is_opt_in_and_uses_new_run_update_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            console = root / "console.log"
            console.write_text(iteration(500) + iteration(501))
            job = root / "job.json"
            metadata = {"console_log": str(console), "run_directory": str(root),
                        "start_iteration": 0, "additional_updates": 1000, "target_iteration": 1000}
            job.write_text(json.dumps(metadata))
            exact = {500: {"step": 499}, 501: {"step": 500}}
            with patch.object(monitor, "read_joint_tensorboard", return_value=exact) as read:
                legacy = monitor.remote_snapshot(str(job))
                read.assert_not_called()
                self.assertNotIn("joint_limits", legacy["latest"])
                job.write_text(json.dumps({**metadata, "tensorboard_joint_metrics": True}))
                data = monitor.remote_snapshot(str(job))
            self.assertEqual(read.call_args.args[:2], (root, {500, 501}))
            self.assertEqual(data["latest"]["joint_limits"], {"step": 500})
            self.assertEqual(data["milestones"][0]["joint_limits"], {"step": 499})
            self.assertEqual(data["milestones"][0]["new_updates"], 500)

    def test_ssh_timeout_is_unreachable_and_does_not_claim_training_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "connection.json"
            config.write_text(json.dumps({"remote_job": "/remote/job.json"}))
            state, snapshot = root / "state.json", root / "snapshot.json"
            with patch.object(monitor, "state_paths", return_value=(state, snapshot)), \
                    patch.object(monitor, "fetch_snapshot", side_effect=monitor.subprocess.TimeoutExpired("ssh", 40)), \
                    patch("sys.stdout", new_callable=io.StringIO):
                monitor.main(["--config", str(config)])
            result = json.loads(snapshot.read_text())
            self.assertEqual(result["status"], "unreachable")
            self.assertEqual(result["event_to_report"], "unreachable")
            self.assertTrue(result["connection_error"])
            self.assertIsNone(result["completion"])

    def test_completion_requires_both_metadata_and_file(self):
        valid = {"iterations": 1600, "all_model_tensors_finite": True}
        self.assertTrue(monitor.verified_completion(valid, True, 1600))
        self.assertFalse(monitor.verified_completion(valid, False, 1600))
        self.assertFalse(monitor.verified_completion({**valid, "iterations": 1599}, True, 1600))
        self.assertFalse(monitor.verified_completion({**valid, "all_model_tensors_finite": "true"}, True, 1600))
        self.assertFalse(monitor.verified_completion(None, True, 1600))

    def test_evaluation_not_misreported_as_stopped_training(self):
        self.assertEqual(monitor.classify_status("evaluating", True, False, True, 900, 10, 10), "evaluating")
        self.assertEqual(monitor.classify_status("evaluating", False, False, True, 900, 10, 10), "stopped")
        self.assertEqual(monitor.classify_status("training", True, True, False, 301, 0, 301), "stalled")
        self.assertEqual(monitor.classify_status("training", True, False, False, 10, 10, 10), "running")
        self.assertEqual(monitor.classify_status("completed", False, False, True, 900, 900, 900), "completed")
        self.assertEqual(monitor.classify_status("completed", False, False, False, 900, 900, 900), "stopped")

    def test_snapshot_discovers_run_and_two_milestones_without_running_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            console = root / "console.log"
            console.write_text(f"GO2_RUN_DIR={run}\n" + iteration(1100) + iteration(1600))
            for node in (1100, 1600):
                (run / f"model_{node}.pt").write_bytes(b"checkpoint")
            (run / "completed.json").write_text(json.dumps({"iterations": 1600, "all_model_tensors_finite": True}))
            job = root / "job.json"
            job.write_text(json.dumps({"pid": 123, "train_pid": 456, "console_log": str(console),
                                       "pipeline_status": "completed", "output_dir": str(root)}))
            with patch.object(monitor, "process_info", return_value={"alive": False}):
                data = monitor.remote_snapshot(str(job))
            self.assertEqual(data["status"], "completed")
            self.assertEqual(data["run_directory"], str(run))
            self.assertEqual(data["new_updates"], 1000)
            self.assertTrue(data["completion_verified"])
            self.assertEqual([r["completed_updates"] for r in data["milestones"]], [1100, 1600])
            self.assertTrue(all(r["checkpoint_exists"] for r in data["milestones"]))

    def test_rotated_console_does_not_invent_milestone_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "console.log").write_text("")
            (root / "model_1600.pt").write_bytes(b"checkpoint")
            (root / "completed.json").write_text(json.dumps({"iterations": 1600, "all_model_tensors_finite": True}))
            job = root / "job.json"
            job.write_text(json.dumps({"console_log": str(root / "console.log"), "run_directory": str(root),
                                       "pipeline_status": "completed"}))
            data = monitor.remote_snapshot(str(job))
            self.assertEqual(data["latest"]["completed_updates"], 1600)
            self.assertTrue(data["latest"]["metrics_unavailable"])
            self.assertEqual(data["milestones"][0]["metrics"], {})
            self.assertFalse(data["milestones"][0]["checkpoint_exists"])


if __name__ == "__main__":
    unittest.main()
