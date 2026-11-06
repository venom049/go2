"""Protect continuation provenance, completion evidence and calf-only counting."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PIPELINE = load_script("run_moe_calf_training")
MONITOR = load_script("check_moe_calf_training")
NAMES = tuple(f"{leg}_{joint}" for leg in ("FL", "FR", "RL", "RR") for joint in ("hip", "thigh", "calf"))
LIMIT = 15.699999809265137


def metric(samples, count=0, peak=10., ratio=.8, excess=0.):
    return {"sample_count": samples, "excess_count": count, "excess_fraction": count / samples,
            "peak_absolute_value": peak, "max_ratio": ratio, "max_excess": excess}


def episode(update, samples, calf_counts=(0, 0, 0, 0), peak=10.):
    per_joint = {name: metric(samples) for name in NAMES}
    for name, count in zip(MONITOR.CALVES, calf_counts):
        per_joint[name] = metric(samples, count, peak, peak / LIMIT, max(peak - LIMIT, 0.))
    velocity = metric(samples * 12, sum(calf_counts), peak, peak / LIMIT, max(peak - LIMIT, 0.))
    velocity["per_joint"] = per_joint
    return {"update": update, "guard": {
        "profile": {"joint_names": list(NAMES),
                    "velocity_limits_rad_s": [LIMIT if name.endswith("_calf") else 30.100000381469727 for name in NAMES]},
        "metrics": {kind: copy.deepcopy(velocity) for kind in MONITOR.KINDS}}}


def comparison():
    return {"rows": [{"label": label, "episodes": 129, "successes": 129,
                       "calf_speed": {"excess_count": 1, "sample_count": 100, "calf_speed_acceptance_passed": False},
                       "calf_speed_acceptance_passed": False, "guard": {"metrics": {}},
                       "joint_and_torque_acceptance_passed": False} for label in PIPELINE.LABELS]}


class CalfMonitorTest(unittest.TestCase):
    def test_four_axis_denominator_and_sample_weighted_counts(self):
        window = MONITOR.window_guard([episode(1, 100, (10, 0, 0, 0), 16.),
                                       episode(2, 900, (0, 2, 0, 0), 16.2)], 500)
        self.assertEqual(window["calf_speed"]["sample_count"], 4000)
        self.assertEqual(window["calf_speed"]["excess_count"], 12)
        self.assertEqual(window["calf_speed"]["excess_fraction"], 12 / 4000)
        self.assertEqual(window["metrics"]["velocity"]["sample_count"], 12000)
        self.assertEqual(window["calf_speed"]["peak_absolute_value"], 16.2)
        self.assertEqual(window["calf_speed"]["per_joint"]["FL_calf"]["excess_fraction"], .01)
        self.assertFalse(window["calf_speed_acceptance_passed"])

    def test_window_uses_completed_episodes_at_node_and_only_latest_256(self):
        entries = [episode(1, 10, (1, 0, 0, 0), 17.)]
        entries.extend(episode(500, 10) for _ in range(256))
        entries.append(episode(501, 10, (1, 0, 0, 0), 18.))
        window = MONITOR.window_guard(entries, 500)
        self.assertEqual(window["episodes"], 256)
        self.assertEqual(window["calf_speed"]["sample_count"], 256 * 10 * 4)
        self.assertTrue(window["calf_speed_acceptance_passed"])

    def test_missing_calf_or_velocity_evidence_is_unknown_not_zero(self):
        for key in ("velocity", "FL_calf"):
            row = episode(500, 10)
            if key == "velocity":
                del row["guard"]["metrics"][key]
            else:
                del row["guard"]["metrics"]["velocity"]["per_joint"][key]
            window = MONITOR.window_guard([row], 500)
            self.assertIsNone(window["calf_speed"]["excess_fraction"])
            self.assertIsNone(window["calf_speed_acceptance_passed"])

    def test_wrong_profile_incomplete_audit_and_empty_window_cannot_pass(self):
        altered = episode(500, 10)
        altered["guard"]["profile"]["velocity_limits_rad_s"][2] = 13.
        incomplete = episode(500, 10)
        incomplete["guard"]["incomplete_due_to_numeric_failure"] = True
        for entries in ([altered], [incomplete], []):
            self.assertIsNone(MONITOR.window_guard(entries, 500)["calf_speed_acceptance_passed"])

    def test_observed_violation_remains_failure_with_other_missing_evidence(self):
        known = episode(400, 10, (1, 0, 0, 0), 17.)
        unknown = episode(500, 10)
        del unknown["guard"]["metrics"]["velocity"]
        result = MONITOR.window_guard([known, unknown], 500)["calf_speed"]
        self.assertIsNone(result["excess_fraction"])
        self.assertFalse(result["calf_speed_acceptance_passed"])

    def test_tiny_peak_violation_is_not_hidden_by_zero_count_or_fraction(self):
        row = episode(500, 10, peak=LIMIT + 1e-8)
        self.assertFalse(MONITOR.window_guard([row], 500)["calf_speed_acceptance_passed"])

    def test_incomplete_trailing_jsonl_record_is_ignored_but_corruption_is_not(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "progress.jsonl"
            file.write_text('{"update": 499}\n{"update": 500')
            self.assertEqual(MONITOR.read_jsonl(file), [{"update": 499}])
            file.write_text('{"update": 499}\nbroken\n')
            with self.assertRaises(json.JSONDecodeError):
                MONITOR.read_jsonl(file)

    def test_remote_program_is_stdlib_only_and_uses_new_process_identity(self):
        code = MONITOR.remote_code("/tmp/not-a-real-run")
        compile(code, "remote-monitor", "exec")
        self.assertNotIn("import numpy", code)
        self.assertIn("run_moe_calf_training.py", code)
        self.assertNotIn("b\"run_moe_finetune.py\"", code)


class CalfPipelineTest(unittest.TestCase):
    def test_continuation_command_matches_bounded_authorized_run(self):
        command = PIPELINE.training_command(Path("source.pt"), Path("training"))
        self.assertIn("--resume-training", command)
        for option, value in (("--updates", "500"), ("--guard-objective", "calf_velocity"),
                              ("--critic-warmup", "0"), ("--num-envs", "128"), ("--workers", "32"),
                              ("--guard-ramp-start", ".5"), ("--guard-ramp-updates", "100")):
            self.assertEqual(command[command.index(option) + 1], value)

    def test_hash_mismatch_blocks_training_and_keeps_failed_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "unexpected.pt"
            source.write_bytes(b"not the approved source")
            out = root / "new-run"
            with patch.object(PIPELINE, "SOURCE_CHECKPOINT", source), patch.object(PIPELINE.subprocess, "run") as launch:
                with self.assertRaises(ValueError):
                    PIPELINE.run(out)
                launch.assert_not_called()
            job = json.loads((out / "job.json").read_text())
            self.assertEqual(job["status"], "failed")
            self.assertIn("unexpected MoE1000", job["error"])

    def test_existing_run_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            marker = out / "job.json"
            marker.write_text("original")
            with self.assertRaises(FileExistsError):
                PIPELINE.run(out)
            self.assertEqual(marker.read_text(), "original")

    def test_completion_checks_structure_not_successful_acceptance(self):
        summary = comparison()
        self.assertTrue(PIPELINE.validate_comparison(summary))
        for bad in (summary["rows"][:1], [summary["rows"][0]] * 2):
            with self.assertRaises(ValueError):
                PIPELINE.validate_comparison({"rows": bad})
        for field in ("calf_speed", "calf_speed_acceptance_passed", "guard", "joint_and_torque_acceptance_passed"):
            bad = copy.deepcopy(summary)
            del bad["rows"][1][field]
            with self.assertRaises(ValueError):
                PIPELINE.validate_comparison(bad)
        bad = copy.deepcopy(summary)
        bad["rows"][1]["episodes"] = 128
        with self.assertRaises(ValueError):
            PIPELINE.validate_comparison(bad)

    def test_training_failure_remains_visible_and_does_not_launch_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "model_1000.pt"
            source.write_bytes(b"approved fixture")
            out = root / "new-run"
            with patch.multiple(PIPELINE, SOURCE_CHECKPOINT=source, SOURCE_SHA256=PIPELINE.digest(source), SOURCE_FILES=()), \
                    patch.object(PIPELINE.subprocess, "run", side_effect=subprocess.CalledProcessError(2, ["trainer"])) as launch:
                with self.assertRaises(subprocess.CalledProcessError):
                    PIPELINE.run(out)
                self.assertEqual(launch.call_count, 1)
            self.assertEqual(source.read_bytes(), b"approved fixture")
            self.assertEqual(json.loads((out / "job.json").read_text())["status"], "failed")
            self.assertFalse((out / "evaluation.log").exists())

    def test_single_training_then_declared_comparison_completes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "model_1000.pt"
            source.write_bytes(b"approved fixture")
            out = root / "new-run"
            def simulated_process(command, **kwargs):
                destination = Path(command[command.index("--out") + 1])
                destination.mkdir()
                if "--resume-training" in command:
                    (destination / "model_500.pt").write_bytes(b"new model")
                else:
                    self.assertIn("--episodes-per-seed", command)
                    (destination / "comparison.json").write_text(json.dumps(comparison()))
            with patch.multiple(PIPELINE, SOURCE_CHECKPOINT=source, SOURCE_SHA256=PIPELINE.digest(source), SOURCE_FILES=()), \
                    patch.object(PIPELINE.subprocess, "run", side_effect=simulated_process) as launch:
                PIPELINE.run(out)
                self.assertEqual(launch.call_count, 2)
            job = json.loads((out / "job.json").read_text())
            self.assertEqual(job["status"], "completed")
            self.assertTrue(job["completion_verified"])
            self.assertEqual(source.read_bytes(), b"approved fixture")


if __name__ == "__main__":
    unittest.main()
