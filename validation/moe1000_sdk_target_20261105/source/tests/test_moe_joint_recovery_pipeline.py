"""Check recovery provenance, additional-update numbering and both evaluations."""
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


PIPELINE = load_script("run_moe_joint_recovery")
MONITOR = load_script("check_moe_joint_recovery")


def metric(samples, count=0, peak=.7, ratio=.8, excess=0.):
    return {"sample_count": samples, "excess_count": count,
            "excess_fraction": count / samples if samples else None,
            "peak_absolute_value": peak if samples else None,
            "max_ratio": ratio if samples else None, "max_excess": excess if samples else None,
            "min_value": -.7 if samples else None, "max_value": .7 if samples else None}


def episode(update, samples=10, violation=None):
    metrics = {}
    for kind in MONITOR.KINDS:
        per_joint = {name: metric(samples) for name in MONITOR.JOINT_NAMES}
        if violation and kind == violation[0]:
            per_joint["RL_calf"] = metric(samples, violation[1], peak=1.3, ratio=1.3, excess=.1)
        count = sum(row["excess_count"] for row in per_joint.values())
        row = metric(samples * 12, count, ratio=1.3 if count else .8, excess=.1 if count else 0.)
        row["per_joint"] = per_joint
        row["observed_state_samples"], row["unobserved_state_samples"] = samples, 0
        metrics[kind] = row
    return {"update": update, "guard": {"all_metrics_observed": True,
            "profile": {"source_sha256": MONITOR.PROFILE_SHA256, "joint_names": list(MONITOR.JOINT_NAMES)},
            "metrics": metrics}}


def comparison(mode="audit", unknown_applied=False):
    metrics = copy.deepcopy(episode(1)["guard"]["metrics"])
    if unknown_applied:
        metrics["applied_torque"] = metric(0)
    return {"guard_mode": mode, "rows": [
        {"label": label, "episodes": 129, "successes": 0, "gait_success_only": {"roll_rms_rad": None},
         "guard": {"all_metrics_observed": not unknown_applied, "metrics": copy.deepcopy(metrics)},
         "joint_and_torque_acceptance_passed": False} for label in PIPELINE.LABELS]}


class JointMonitorTest(unittest.TestCase):
    def test_all_metrics_and_per_joint_counts_are_sample_weighted(self):
        window = MONITOR.window_guard([episode(1, 10, ("target_position", 2)),
                                       episode(500, 90, ("target_position", 1))], 500)
        target = window["metrics"]["target_position"]
        self.assertEqual(target["sample_count"], 1200)
        self.assertEqual(target["excess_count"], 3)
        self.assertEqual(target["excess_fraction"], 3 / 1200)
        self.assertEqual(target["per_joint"]["RL_calf"]["excess_fraction"], 3 / 100)
        self.assertEqual(target["per_joint"]["RL_calf"]["peak_absolute_value"], 1.3)
        self.assertEqual(target["per_joint"]["RL_calf"]["min_value"], -.7)
        self.assertFalse(window["joint_and_torque_acceptance_passed"])
        self.assertEqual(set(window["metrics"]), set(MONITOR.KINDS))

    def test_window_uses_latest_256_completed_at_node(self):
        entries = [episode(1, violation=("position", 1))] + [episode(500) for _ in range(256)]
        entries.append(episode(501, violation=("position", 1)))
        window = MONITOR.window_guard(entries, 500)
        self.assertEqual(window["episodes"], 256)
        self.assertTrue(window["joint_and_torque_acceptance_passed"])

    def test_missing_metric_axis_profile_or_numeric_evidence_cannot_pass(self):
        missing_metric, missing_joint, wrong_profile, incomplete, missing_axes = [episode(500) for _ in range(5)]
        del missing_metric["guard"]["metrics"]["requested_torque"]
        del missing_joint["guard"]["metrics"]["target_position"]["per_joint"]["RL_calf"]
        wrong_profile["guard"]["profile"]["source_sha256"] = "different"
        incomplete["guard"]["incomplete_due_to_numeric_failure"] = True
        del missing_axes["guard"]["metrics"]["velocity"]["per_joint"]
        for rows in ([], [missing_metric], [missing_joint], [wrong_profile], [incomplete], [missing_axes]):
            self.assertIsNone(MONITOR.window_guard(rows, 500)["joint_and_torque_acceptance_passed"])
        self.assertIsNone(MONITOR.window_guard([missing_metric], 500)["metrics"]["requested_torque"]["excess_fraction"])

    def test_observed_violation_stays_failure_despite_other_unknown_evidence(self):
        unknown = episode(500)
        del unknown["guard"]["metrics"]["position"]
        result = MONITOR.window_guard([episode(400, violation=("position", 1)), unknown], 500)
        self.assertIsNone(result["metrics"]["position"]["excess_fraction"])
        self.assertFalse(result["joint_and_torque_acceptance_passed"])

    def test_missing_applied_boundaries_are_unknown_not_zero(self):
        partial = episode(500)
        partial["guard"]["metrics"]["applied_torque"]["unobserved_state_samples"] = 1
        result = MONITOR.window_guard([partial], 500)
        self.assertIsNone(result["metrics"]["applied_torque"]["excess_fraction"])
        self.assertIsNone(result["joint_and_torque_acceptance_passed"])

    def test_trailing_partial_jsonl_ignored_but_complete_corruption_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "progress.jsonl"
            file.write_text('{"update": 499}\n{"update": 500')
            self.assertEqual(MONITOR.read_jsonl(file), [{"update": 499}])
            file.write_text('broken\n')
            with self.assertRaises(json.JSONDecodeError):
                MONITOR.read_jsonl(file)

    def test_remote_program_is_standalone_and_process_identity_is_new(self):
        code = MONITOR.remote_code("/tmp/not-a-real-run")
        compile(code, "remote-monitor", "exec")
        self.assertNotIn("numpy", code)
        self.assertIn("run_moe_joint_recovery.py", code)
        self.assertNotIn("run_moe_calf_training.py", code)

    def test_completion_requires_both_modes_and_final_checkpoint_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, audit, strict = root / "training", root / "evaluation", root / "strict_evaluation"
            for path in (train, audit, strict):
                path.mkdir()
            final = train / "model_1000.pt"
            final.write_bytes(b"final checkpoint")
            job = {"status": "completed", "source_update": 1000, "training_dir": str(train),
                   "eval_summary_path": str(audit / "comparison.json"),
                   "strict_eval_summary_path": str(strict / "comparison.json"),
                   "final_checkpoint": str(final), "final_checkpoint_sha256": PIPELINE.digest(final),
                   "completion_verified": True}
            (root / "job.json").write_text(json.dumps(job))
            (train / "progress.jsonl").write_text('{"update":500}\n{"update":1000}\n')
            (audit / "comparison.json").write_text(json.dumps(comparison("audit")))
            self.assertFalse(MONITOR.inspect_run(root)["completion_verified"])
            (strict / "comparison.json").write_text(json.dumps(comparison("strict", unknown_applied=True)))
            state = MONITOR.inspect_run(root)
            self.assertTrue(state["completion_verified"])
            self.assertEqual([row["cumulative_update"] for row in state["milestones"]], [1500, 2000])
            final.write_bytes(b"changed")
            self.assertFalse(MONITOR.inspect_run(root)["completion_verified"])


class JointPipelineTest(unittest.TestCase):
    def test_continuation_parameters_and_reference_names(self):
        command = PIPELINE.training_command(Path("source.pt"), Path("training"))
        self.assertIn("--resume-training", command)
        for option, value in (("--updates", "1000"), ("--guard-objective", "joint_recovery"),
                              ("--critic-warmup", "20"), ("--num-envs", "128"), ("--workers", "32"),
                              ("--guard-ramp-start", ".25"), ("--guard-ramp-updates", "200")):
            self.assertEqual(command[command.index(option) + 1], value)
        strict = PIPELINE.strict_evaluation_command(Path("source.pt"), Path("training"), Path("strict"))
        self.assertEqual(strict[strict.index("--guard-mode") + 1], "strict")
        self.assertIn("joint1500=training/model_500.pt", strict)
        self.assertIn("joint2000=training/model_1000.pt", strict)
        self.assertIn("reference_moe1000=source.pt", strict)
        self.assertIn("go2_him/joint_recovery.py", PIPELINE.SOURCE_FILES)

    def test_invalid_source_prevents_any_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pt"
            source.write_bytes(b"not authorized checkpoint")
            with patch.object(PIPELINE, "SOURCE_CHECKPOINT", source), patch.object(PIPELINE.subprocess, "run") as launch:
                with self.assertRaises(ValueError):
                    PIPELINE.run(root / "out")
                launch.assert_not_called()
            self.assertEqual(json.loads((root / "out/job.json").read_text())["status"], "failed")

    def test_existing_artifacts_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "job.json").write_text("original")
            with self.assertRaises(FileExistsError):
                PIPELINE.run(out)
            self.assertEqual((out / "job.json").read_text(), "original")

    def test_strict_unknown_applied_evidence_is_a_result_but_missing_records_fail(self):
        summary = comparison("strict", unknown_applied=True)
        self.assertTrue(PIPELINE.validate_comparison(summary, "strict"))
        bad_cases = []
        bad = copy.deepcopy(summary); bad["guard_mode"] = "audit"; bad_cases.append(bad)
        bad = copy.deepcopy(summary); bad["rows"] = bad["rows"][:2]; bad_cases.append(bad)
        bad = copy.deepcopy(summary); bad["rows"][1]["episodes"] = 128; bad_cases.append(bad)
        bad = copy.deepcopy(summary); del bad["rows"][1]["guard"]["metrics"]["position"]; bad_cases.append(bad)
        bad = copy.deepcopy(summary); bad["rows"][1]["guard"]["metrics"]["tracking"]["max_ratio"] = float("nan"); bad_cases.append(bad)
        for bad in bad_cases:
            with self.assertRaises(ValueError):
                PIPELINE.validate_comparison(bad, "strict")

    def test_exactly_one_training_and_each_declared_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pt"
            source.write_bytes(b"approved source fixture")
            out = root / "out"
            def simulate(command, **kwargs):
                destination = Path(command[command.index("--out") + 1])
                destination.mkdir()
                if "--resume-training" in command:
                    for node in (500, 1000):
                        (destination / f"model_{node}.pt").write_bytes(f"checkpoint{node}".encode())
                else:
                    mode = command[command.index("--guard-mode") + 1]
                    (destination / "comparison.json").write_text(json.dumps(comparison(mode, mode == "strict")))
            with patch.multiple(PIPELINE, SOURCE_CHECKPOINT=source, SOURCE_SHA256=PIPELINE.digest(source), SOURCE_FILES=()), \
                    patch.object(PIPELINE.subprocess, "run", side_effect=simulate) as launch:
                PIPELINE.run(out)
                self.assertEqual(launch.call_count, 3)
            job = json.loads((out / "job.json").read_text())
            self.assertEqual(job["status"], "completed")
            self.assertTrue(job["completion_verified"])
            self.assertTrue(job["strict_eval_summary_exists"])
            self.assertEqual(job["final_cumulative_update"], 2000)
            self.assertEqual(source.read_bytes(), b"approved source fixture")
            self.assertTrue(MONITOR.inspect_run(out)["completion_verified"])

    def test_strict_evaluation_failure_cannot_be_marked_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pt"
            source.write_bytes(b"approved source fixture")
            out = root / "out"
            def simulate(command, **kwargs):
                destination = Path(command[command.index("--out") + 1])
                destination.mkdir()
                if "--resume-training" in command:
                    for node in (500, 1000):
                        (destination / f"model_{node}.pt").write_bytes(b"model")
                elif command[command.index("--guard-mode") + 1] == "audit":
                    (destination / "comparison.json").write_text(json.dumps(comparison()))
                else:
                    raise subprocess.CalledProcessError(2, command)
            with patch.multiple(PIPELINE, SOURCE_CHECKPOINT=source, SOURCE_SHA256=PIPELINE.digest(source), SOURCE_FILES=()), \
                    patch.object(PIPELINE.subprocess, "run", side_effect=simulate):
                with self.assertRaises(subprocess.CalledProcessError):
                    PIPELINE.run(out)
            self.assertEqual(json.loads((out / "job.json").read_text())["status"], "failed")
            self.assertFalse(MONITOR.inspect_run(out)["completion_verified"])


if __name__ == "__main__":
    unittest.main()
