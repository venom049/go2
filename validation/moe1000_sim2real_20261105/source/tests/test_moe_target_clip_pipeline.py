"""Provenance, checkpoint and command-bound evidence for target-clipped training."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("target_pipeline", ROOT / "scripts/run_moe_target_clip_training.py")
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


def comparison(mode="audit", stair_steps=2):
    guard = {kind: {"sample_count": 12, "excess_count": 0, "peak_absolute_value": .7,
                    "max_ratio": .8, "max_excess": 0.} for kind in P.KINDS}
    if mode == "strict":
        guard["applied_torque"] = {"sample_count": 0, "excess_count": 0,
                                  "peak_absolute_value": None, "max_ratio": None, "max_excess": None}
    targets = {"raw": {"sample_count": 12, "excess_count": 1, "excess_fraction": 1 / 12,
                       "max_excess_rad": .2},
               "sent": {"sample_count": 12, "excess_count": 0, "excess_fraction": 0.,
                        "max_excess_rad": 0.}}
    return {"guard_mode": mode, "target_mode_requested": "bounded", "stair_steps": stair_steps,
            "environment_contract": P.environment_contract(stair_steps), "rows": [
        {"label": label, "target_mode": "bounded", "episodes": 129, "successes": 0,
         "environment_contract": P.environment_contract(stair_steps),
         "gait_success_only": {}, "joint_and_torque_acceptance_passed": False,
         "position_constraints":{"mode":"hard","accepted_excess_count":0},
         "guard": {"metrics": copy.deepcopy(guard)}, "target_commands": copy.deepcopy(targets)}
        for label in P.LABELS]}


def checkpoint(update=1000, continuation=False, stair_steps=2):
    return {"format": "moe_cts_ppo_finetune_v1", "update": update,
            "cumulative_update": update + 1000 if continuation else update,
            "guard_objective": "target_clipped", "frozen_moe_encoder": True, "std": .03,
            "stair_steps": stair_steps, "environment_contract": P.environment_contract(stair_steps),
            "actor_state_dict": {"student_moe_encoder.weight": torch.tensor([.5]), "actor.weight": torch.tensor([.1])},
            "critic_state_dict": {"weight": torch.tensor([.3])},
            "optimizer_state_dict": {"state": {0: {"exp_avg": torch.tensor([.2])}}},
            "initialization": {"source_cumulative_update": 1000, "full_training_state_resumed": True,
                               "actor_restored": True, "critic_restored": True, "optimizer_restored": True}}


def preflight(stair_steps=2):
    return {"format": "moe1000_target_clip_abc_v1", "checkpoint_unchanged": True,
            "stair_steps": stair_steps, "environment_contract": P.environment_contract(stair_steps),
            "source_files_unchanged": True, "real_robot_driven": False, "training_performed": False,
            "rows": [{"arm": arm, "guard_mode": mode, "episodes": 17, "successes": 0,
                      "guard":{"metrics":{"position":{"sample_count":12,"excess_count":0,"max_excess":0.}}}}
                     for mode in ("audit", "strict")
                     for arm in ("A_original", "B_clip_raw_history", "C_clip_sent_history")]}


class TargetClipPipelineTest(unittest.TestCase):
    def test_commands_use_exact_continuation_and_cpu_bounded_evaluation(self):
        command = P.training_command(Path("source"), Path("train"), workers=8, stair_steps=6)
        self.assertIn("--resume-training", command)
        for key, value in (("--updates", "1000"), ("--device", "cpu"), ("--workers", "8"),
                           ("--num-envs", "128"), ("--guard-objective", "target_clipped"),
                           ("--critic-warmup", "20"), ("--anchor-coef", ".2"), ("--stair-steps", "6")):
            self.assertEqual(command[command.index(key) + 1], value)
        command = P.evaluation_command(Path("source"), Path("train"), Path("out"), "strict", stair_steps=6)
        for key, value in (("--target-mode", "bounded"), ("--guard-mode", "strict"),
                           ("--device", "cpu"), ("--episodes-per-seed", "64"), ("--stair-steps", "6")):
            self.assertEqual(command[command.index(key) + 1], value)
        self.assertIn("reference_moe1000_clipped=source", command)
        self.assertIn("target_clip2000=train/model_1000.pt", command)
        self.assertFalse(P.ACTION_CONTRACT["target_slew_limiter_present"])

    def test_six_step_request_changes_only_the_stair_flag_in_commands(self):
        for command_fn, args, kwargs in (
                (P.training_command, (Path("source"), Path("train")), {"workers": 8}),
                (P.evaluation_command, (Path("source"), Path("train"), Path("out")), {})):
            previous = command_fn(*args, **kwargs)
            changed = command_fn(*args, stair_steps=6, **kwargs)
            self.assertEqual([(a, b) for a, b in zip(previous, changed) if a != b], [("2", "6")])

    def test_preflight_stair_mismatch_and_missing_hard_contract_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.json"
            data = preflight()
            del data["stair_steps"]; del data["environment_contract"]
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                P.validate_preflight(path)
            with self.assertRaises(ValueError):
                P.validate_preflight(path, 6)
            path.write_text(json.dumps(preflight(6)))
            P.validate_preflight(path, 6)
            bad=preflight(6);bad['rows'][0]['guard']['metrics']['position']['excess_count']=1
            path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):P.validate_preflight(path,6)
            data = preflight(6); data["environment_contract"] = P.environment_contract(2)
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                P.validate_preflight(path, 6)

    def test_comparison_geometry_and_each_actor_must_match(self):
        self.assertTrue(P.validate_comparison(comparison(stair_steps=6), "audit", 6))
        for field in ("stair_steps", "environment_contract"):
            bad = comparison(stair_steps=6)
            bad[field] = comparison()[field]
            with self.assertRaises(ValueError):
                P.validate_comparison(bad, "audit", 6)
        bad = comparison(stair_steps=6)
        bad["rows"][0]["environment_contract"] = P.environment_contract(2)
        with self.assertRaises(ValueError):
            P.validate_comparison(bad, "audit", 6)

    def test_complete_failures_are_results_but_missing_or_violating_sent_targets_reject(self):
        self.assertTrue(P.validate_comparison(comparison("strict"), "strict"))
        mutations = []
        bad = comparison(); bad["target_mode_requested"] = "legacy"; mutations.append(bad)
        bad = comparison(); bad["rows"][0]["target_mode"] = "legacy"; mutations.append(bad)
        bad = comparison(); del bad["rows"][0]["target_commands"]["raw"]; mutations.append(bad)
        bad = comparison(); bad["rows"][0]["target_commands"]["sent"]["max_excess_rad"] = 1e-12; mutations.append(bad)
        bad = comparison(); bad["rows"][0]["target_commands"]["sent"].update(excess_count=1, excess_fraction=1/12); mutations.append(bad)
        bad = comparison(); bad["rows"][0]["episodes"] = 128; mutations.append(bad)
        bad = comparison(); bad["rows"][0]["guard"]["metrics"]["position"]["excess_count"] = 1; mutations.append(bad)
        bad = comparison(); bad["rows"][0]["position_constraints"] = None; mutations.append(bad)
        bad = comparison(); bad["rows"][0]["guard"]["metrics"]["tracking"]["max_ratio"] = float("nan"); mutations.append(bad)
        for bad in mutations:
            with self.assertRaises(ValueError):
                P.validate_comparison(bad, "audit")

    def test_checkpoint_checks_update_finite_optimizer_and_frozen_encoder(self):
        with tempfile.TemporaryDirectory() as directory:
            source, final = Path(directory) / "source.pt", Path(directory) / "final.pt"
            torch.save(checkpoint(), source)
            good = checkpoint(1000, True); torch.save(good, final)
            self.assertTrue(P.validate_checkpoint(final, source, 1000)["all_tensors_finite"])
            mutations = []
            bad = copy.deepcopy(good); bad["cumulative_update"] = 1000; mutations.append(bad)
            bad = copy.deepcopy(good); bad["actor_state_dict"]["student_moe_encoder.weight"] += .1; mutations.append(bad)
            bad = copy.deepcopy(good); bad["optimizer_state_dict"]["state"][0]["exp_avg"][0] = float("nan"); mutations.append(bad)
            bad = copy.deepcopy(good); bad["initialization"]["optimizer_restored"] = False; mutations.append(bad)
            bad = copy.deepcopy(good); bad["environment_contract"] = P.environment_contract(6); mutations.append(bad)
            for bad in mutations:
                torch.save(bad, final)
                with self.assertRaises(ValueError):
                    P.validate_checkpoint(final, source, 1000)

    def test_source_mismatch_and_existing_directory_never_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "source.pt"; source.write_bytes(b"unexpected")
            with patch.object(P, "SOURCE_CHECKPOINT", source), patch.object(P.subprocess, "run") as launch:
                with self.assertRaises(ValueError):
                    P.run(root / "out")
                launch.assert_not_called()
            self.assertEqual(json.loads((root / "out/job.json").read_text())["status"], "failed")
            with self.assertRaises(FileExistsError):
                P.run(root / "out")

    def test_exactly_one_training_and_both_evaluations_record_failed_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "source.pt"; torch.save(checkpoint(), source)
            profile = root / "profile.json"; profile.write_text("{}")
            evidence = root / "preflight.json"; evidence.write_text(json.dumps(preflight(6)))
            def simulate(command, **kwargs):
                out = Path(command[command.index("--out") + 1]); out.mkdir()
                self.assertEqual(command[command.index("--stair-steps") + 1], "6")
                if "--resume-training" in command:
                    for node in P.MILESTONES:
                        torch.save(checkpoint(node, True, 6), out / f"model_{node}.pt")
                else:
                    mode = command[command.index("--guard-mode") + 1]
                    (out / "comparison.json").write_text(json.dumps(comparison(mode, 6)))
            with patch.multiple(P, SOURCE_CHECKPOINT=source, SOURCE_SHA256=P.digest(source), SOURCE_FILES=(),
                                PROFILE_PATH=profile, PROFILE_SHA256=P.digest(profile)), \
                    patch.object(P.subprocess, "run", side_effect=simulate) as launch:
                job = P.run(root / "out", preflight_comparison=evidence, workers=8, stair_steps=6)
                self.assertEqual(launch.call_count, 3)
            self.assertEqual(job["status"], "completed")
            self.assertEqual(job["stair_steps"], 6)
            self.assertEqual(job["environment_contract"], P.environment_contract(6))
            self.assertTrue(job["completion_verified"])
            self.assertFalse(job["final_policy_task_and_joint_acceptance_passed"])
            report = json.loads((root / "out/validation_report.json").read_text())
            self.assertTrue(report["source_unchanged"])
            self.assertTrue(report["sent_target_bounds_passed"])
            self.assertEqual(report["environment_contract"], P.environment_contract(6))
            self.assertEqual(set(report["evaluations"]), {"audit", "strict"})


if __name__ == "__main__":
    unittest.main()
