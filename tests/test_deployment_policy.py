"""CPU contract/export regressions; never import the SDK or contact a robot."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest

try:
    import torch
    import yaml
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(torch is None, "CPU PyTorch, NumPy and PyYAML are required")
class DeploymentPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        from deploy.go2_stairs import policy
        cls.p = policy
        spec = importlib.util.spec_from_file_location("export_stairs", ROOT / "scripts/export_stairs.py")
        cls.exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.exporter)
        _, cls.agent = cls.exporter.validate_source(cls.exporter.DEFAULT_CHECKPOINT)
        cls.network, _ = policy.load_checkpoint(cls.exporter.DEFAULT_CHECKPOINT)
        cls.reference = cls.exporter.load_reference(cls.exporter.DEFAULT_CHECKPOINT, cls.agent)

    def states(self, n=1):
        return [torch.tensor([[.57, -.03, .11]]).repeat(n, 1),
                torch.tensor([[.2, -.4, .8]]).repeat(n, 1),
                torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1),
                torch.tensor([self.p.DEFAULT_JOINT_POSITION]).repeat(n, 1), torch.zeros(n, 12)]

    def constant_network(self, values):
        class Constant(torch.nn.Module):
            def forward(self, history):
                return values.expand(len(history), -1)
        return Constant()

    def test_observation_has_physical_commands_and_expected_scales(self):
        cmd, gyro, quat, q, dq = self.states()
        delta = torch.arange(12).float()[None, :]*.01
        q += delta
        dq[:] = torch.arange(12)*2.
        applied = torch.arange(12).float()[None, :]-.3
        frame = self.p.build_observation(cmd, gyro, quat, q, dq, applied)
        torch.testing.assert_close(frame[:, :3], cmd)
        torch.testing.assert_close(frame[:, 3:6], torch.tensor([[.05, -.1, .2]]))
        torch.testing.assert_close(frame[:, 6:9], torch.tensor([[0., 0., -1.]]))
        torch.testing.assert_close(frame[:, 9:21], delta)
        torch.testing.assert_close(frame[:, 21:33], torch.arange(12).float()[None, :]*.1)
        torch.testing.assert_close(frame[:, 33:45], applied)

    def test_gravity_matches_independent_rotation_matrix_for_wxyz(self):
        # Rz(yaw) Ry(pitch) Rx(roll), body-to-world; gravity is minus row 3.
        roll, pitch, yaw = .31, -.22, .74
        cr, sr = math.cos(roll/2), math.sin(roll/2)
        cp, sp = math.cos(pitch/2), math.sin(pitch/2)
        cy, sy = math.cos(yaw/2), math.sin(yaw/2)
        quat = torch.tensor([[cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                              cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy]])
        expected = torch.tensor([[math.sin(pitch), -math.cos(pitch)*math.sin(roll),
                                  -math.cos(pitch)*math.cos(roll)]])
        torch.testing.assert_close(self.p.projected_gravity(quat), expected, atol=1e-6, rtol=0)
        torch.testing.assert_close(self.p.projected_gravity(-quat), expected, atol=1e-6, rtol=0)
        # Unit norm alone cannot detect a caller silently passing xyzw. It
        # changes the physical result, which this explicit convention catches.
        self.assertFalse(torch.allclose(self.p.projected_gravity(quat[:, [1, 2, 3, 0]]), expected))

    def test_sdk_mapping_changes_observations_and_output_physical_legs(self):
        names = self.exporter.SDK_JOINT_NAMES
        indices = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        action = torch.arange(12).float()[None, :]/10
        adapter = self.p.StairsPolicy(self.constant_network(action), names)
        cmd, gyro, quat, q, dq = self.states()
        q += torch.arange(12).float()[None, :]*.01
        dq += torch.arange(12).float()[None, :]
        frame = adapter.observe(cmd, gyro, quat, q[:, indices], dq[:, indices])
        torch.testing.assert_close(frame[:, 9:21], torch.arange(12).float()[None, :]*.01)
        torch.testing.assert_close(frame[:, 21:33], dq*.05)
        result = adapter.infer()
        expected = torch.tensor([self.p.DEFAULT_JOINT_POSITION])+.25*action
        torch.testing.assert_close(result.joint_target, expected[:, indices])
        torch.testing.assert_close(result.raw_action, action[:, indices])
        torch.testing.assert_close(result.stiffness, torch.full((1, 12), 40.))
        torch.testing.assert_close(result.damping, torch.ones(1, 12))
        torch.testing.assert_close(result.effort_limit, torch.tensor([[23.7, 23.7, 45.43]*4]))

    def test_history_repeats_initial_state_then_advances_newest_first(self):
        adapter = self.p.StairsPolicy(self.network, num_envs=2)
        states = self.states(2)
        first = adapter.observe(*states)
        torch.testing.assert_close(adapter.history, first.repeat(1, 6))
        old_history = adapter.history.clone()
        states[0][:, 0] += .02
        target = torch.tensor([self.p.DEFAULT_JOINT_POSITION]).repeat(2, 1)+.025
        second = adapter.observe(*states, previous_applied_joint_target=target)
        torch.testing.assert_close(adapter.history[:, :45], second)
        torch.testing.assert_close(adapter.history[:, 45:], old_history[:, :-45])
        adapter.reset(torch.tensor([True, False]))
        prior = adapter.history[1].clone()
        third = adapter.observe(*states, previous_applied_joint_target=target)
        torch.testing.assert_close(adapter.history[0], third[0].repeat(6))
        torch.testing.assert_close(adapter.history[1, 45:], prior[:-45])

    def test_previous_action_uses_final_applied_target_without_reclipping(self):
        adapter = self.p.StairsPolicy(self.constant_network(torch.full((1, 12), 9.)))
        states = self.states()
        adapter.observe(*states)
        proposal = adapter.infer()
        torch.testing.assert_close(proposal.raw_action, torch.full((1, 12), 9.))
        torch.testing.assert_close(proposal.clipped_action, torch.full((1, 12), 4.))
        default = torch.tensor([self.p.DEFAULT_JOINT_POSITION])
        torch.testing.assert_close(proposal.joint_target, default+1.)
        # An external controller chose another target. The next frame must
        # contain its encoding, not 9/raw, 4/clipped, or measured joint state.
        applied = default + torch.linspace(-.2, .2, 12)[None, :]
        applied[0, 0] = default[0, 0]+1.25
        frame = adapter.observe(*states, previous_applied_joint_target=applied)
        torch.testing.assert_close(frame[:, 33:], (applied-default)/.25)
        self.assertEqual(float(frame[0, 33]), 5.)

    def test_real_checkpoint_and_jit_match_actual_training_actor_multistep(self):
        scripted = torch.jit.script(self.network)
        report = self.exporter.verify_against_reference(self.network, scripted, self.reference)
        self.assertEqual(report["input_histories_checked"], 176)
        self.assertEqual(report["multistep_frames"], 40)

    def test_rejects_invalid_contract_inputs_and_uninitialized_inference(self):
        for kwargs in ({"joint_names": ["FL_hip_joint"]*12}, {"control_dt": .01},
                       {"stiffness": float("nan")}, {"action_scale": 0.}, {"num_envs": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.p.StairsPolicy(self.network, **kwargs)
        adapter = self.p.StairsPolicy(self.network)
        with self.assertRaises(RuntimeError):
            adapter.infer()
        states = self.states()
        for index, bad in ((0, torch.zeros(1, 4)), (1, torch.full((1, 3), float("nan"))),
                           (2, torch.zeros(1, 4)), (3, torch.zeros(1, 12, dtype=torch.int64))):
            modified = list(states)
            modified[index] = bad
            with self.subTest(index=index), self.assertRaises(ValueError):
                adapter.observe(*modified)
        adapter.observe(*states)
        with self.assertRaises(ValueError):
            adapter.observe(*states)  # Missing actually applied previous target.
        for history in (torch.zeros(1, 269), torch.zeros(1, 270, dtype=torch.float64),
                        torch.full((1, 270), float("inf"))):
            with self.subTest(shape=history.shape), self.assertRaises(ValueError):
                self.network(history)
        bad = self.p.StairsPolicy(self.constant_network(torch.full((1, 12), float("nan"))))
        bad.observe(*states)
        with self.assertRaises(ValueError):
            bad.infer()

    def test_export_roundtrip_and_checksum_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/"export"
            output.mkdir()  # Explicitly support an existing empty directory.
            report = self.exporter.export_policy(self.exporter.DEFAULT_CHECKPOINT, output)
            self.assertEqual(report["checkpoint_iteration"], 1100)
            self.assertEqual(report["output_shape"], ["batch", 12])
            self.assertEqual(set(p.name for p in output.iterdir()), {"stairs_policy.pt", "manifest.json"})
            loaded = self.p.load_exported_policy(output, self.exporter.SDK_JOINT_NAMES)
            self.assertEqual(loaded.to_policy, [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])
            history = torch.linspace(-1, 1, 270)[None, :]
            torch.testing.assert_close(loaded.network(history), self.reference.act_inference(history))
            manifest = json.loads((output/"manifest.json").read_text())
            for key, value in (("action_scale", .5), ("action_clip", 8.),
                               ("stiffness", 20.), ("damping", 2.), ("control_dt", .01),
                               ("damping", True)):
                changed = json.loads(json.dumps(manifest))
                changed["control"][key] = value
                (output/"manifest.json").write_text(json.dumps(changed))
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "trained stair contract"):
                    self.p.load_exported_policy(output)
            changed = json.loads(json.dumps(manifest))
            changed["default_joint_position"][0] = 0.
            (output/"manifest.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "interface"):
                self.p.load_exported_policy(output)
            manifest["export_sha256"] = "0"*64
            (output/"manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "checksum"):
                self.p.load_exported_policy(output)

    def test_export_refuses_existing_files_and_source_contract_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            output = path/"keep"
            output.mkdir()
            sentinel = output/"existing.txt"
            sentinel.write_text("preserve this")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                self.exporter.export_policy(self.exporter.DEFAULT_CHECKPOINT, output)
            self.assertEqual(sentinel.read_text(), "preserve this")
            source = path/"source"
            source.mkdir()
            checkpoint = source/"model_1100.pt"
            checkpoint.write_bytes(b"different checkpoint")
            with self.assertRaisesRegex(ValueError, "Checkpoint SHA256"):
                self.exporter.export_policy(checkpoint, path/"bad-export")
            self.assertFalse((path/"bad-export").exists())
            shutil.copy2(self.exporter.DEFAULT_CHECKPOINT, checkpoint)
            for filename in self.exporter.SOURCE_HASHES:
                shutil.copy2(self.exporter.DEFAULT_CHECKPOINT.parent/filename, source/filename)
            settings = yaml.safe_load((source/"settings.yaml").read_text())
            settings["control"]["action_scale"] = .5
            (source/"settings.yaml").write_text(yaml.safe_dump(settings))
            with self.assertRaisesRegex(ValueError, "settings.yaml SHA256"):
                self.exporter.export_policy(checkpoint, path/"bad-export")
            self.assertFalse((path/"bad-export").exists())


if __name__ == "__main__":
    unittest.main()
