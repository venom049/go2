"""Hash-pinned candidate exports; offline CPU only, no SDK or robot access."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import torch
    import yaml
    import onnx
    import onnxruntime
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "CPU torch, YAML, ONNX and ONNX Runtime are required")
class SourceRecordExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import export_stairs
        import export_stairs_legacy_adapter
        cls.e = export_stairs
        cls.legacy = export_stairs_legacy_adapter
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temporary.name)
        cls.source = cls.directory / "source"
        cls.source.mkdir()
        for filename in cls.e.SOURCE_HASHES:
            shutil.copy2(cls.e.DEFAULT_CHECKPOINT.parent / filename, cls.source / filename)
        # This is a synthetic exporter test fixture, not a newly trained model.
        # Distinct weights ensure exports cannot silently reuse the default.
        saved = torch.load(cls.e.DEFAULT_CHECKPOINT, map_location="cpu", weights_only=True)
        saved["iter"] = 1000
        saved["model_state_dict"]["actor.6.bias"] = saved["model_state_dict"]["actor.6.bias"] + .01
        cls.checkpoint = cls.source / "model_1000.pt"
        torch.save(saved, cls.checkpoint)
        cls.record = {"checkpoint_sha256": cls.e.sha256(cls.checkpoint),
                      "checkpoint_iteration": 1000,
                      "source_hashes": {name: cls.e.sha256(cls.source / name) for name in cls.e.SOURCE_HASHES}}
        cls.record_path = cls.directory / "source_record.json"
        cls.record_path.write_text(json.dumps(cls.record))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_default_selection_still_only_accepts_original(self):
        self.e.validate_source(self.e.DEFAULT_CHECKPOINT)
        self.assertEqual(self.e.load_source_record()["checkpoint_sha256"], self.e.CHECKPOINT_SHA256)
        with self.assertRaisesRegex(ValueError, "Checkpoint SHA256"):
            self.e.validate_source(self.checkpoint)
        with self.assertRaisesRegex(ValueError, "Checkpoint SHA256"):
            self.legacy.export(self.directory / "rejected-default", self.checkpoint)
        self.assertFalse((self.directory / "rejected-default").exists())

    def test_record_schema_rejects_missing_hash_bad_digest_and_noninteger_iteration(self):
        variants = []
        missing = copy.deepcopy(self.record)
        del missing["source_hashes"]["env.yaml"]
        variants.append(missing)
        for value in (True, "1000", 0, -1, 1000.0):
            bad = copy.deepcopy(self.record)
            bad["checkpoint_iteration"] = value
            variants.append(bad)
        for value in ("0" * 63, "z" * 64, "A" * 64, None):
            bad = copy.deepcopy(self.record)
            bad["checkpoint_sha256"] = value
            variants.append(bad)
        for bad in variants:
            with self.subTest(record=bad), self.assertRaises(ValueError):
                self.e.load_source_record(bad)

    def test_every_source_hash_is_enforced(self):
        self.e.validate_source(self.checkpoint, self.record_path)
        for name in self.e.SOURCE_HASHES:
            bad = copy.deepcopy(self.record)
            bad["source_hashes"][name] = "0" * 64
            with self.subTest(file=name), self.assertRaisesRegex(ValueError, "SHA256"):
                self.e.validate_source(self.checkpoint, bad)
        bad = copy.deepcopy(self.record)
        bad["checkpoint_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Checkpoint SHA256"):
            self.e.validate_source(self.checkpoint, bad)

    def test_repinning_does_not_relax_control_observation_or_architecture(self):
        changes = (
            ("settings.yaml", lambda d: d["control"].update(stiffness=30.)),
            ("manifest.json", lambda d: d.update(history_frames=5)),
            ("agent.json", lambda d: d["policy"].update(actor_hidden_dims=[128, 128])),
        )
        for name, mutate in changes:
            with self.subTest(file=name), tempfile.TemporaryDirectory() as temporary:
                source = Path(temporary)
                for filename in self.e.SOURCE_HASHES:
                    shutil.copy2(self.source / filename, source / filename)
                checkpoint = source / "model_1000.pt"
                checkpoint.symlink_to(self.checkpoint)
                path = source / name
                data = yaml.safe_load(path.read_text()) if name.endswith("yaml") else json.loads(path.read_text())
                mutate(data)
                path.write_text(yaml.safe_dump(data) if name.endswith("yaml") else json.dumps(data))
                identity = copy.deepcopy(self.record)
                identity["source_hashes"][name] = self.e.sha256(path)
                with self.assertRaisesRegex(ValueError, "does not match|do not match"):
                    self.e.validate_source(checkpoint, identity)

    def test_completed_iteration_is_checked_independently_of_hash(self):
        _, agent = self.e.validate_source(self.checkpoint, self.record_path)
        with self.assertRaisesRegex(ValueError, "1100 completed updates"):
            self.e.load_reference(self.checkpoint, agent)
        reference = self.e.load_reference(self.checkpoint, agent, 1000)
        self.assertEqual(tuple(reference.act_inference(torch.zeros(1, 270)).shape), (1, 12))
        wrong = copy.deepcopy(self.record)
        wrong["checkpoint_iteration"] = 999
        with self.assertRaisesRegex(ValueError, "999 completed updates"):
            self.e.export_policy(self.checkpoint, self.directory / "wrong-iteration", wrong)
        self.assertFalse((self.directory / "wrong-iteration").exists())

    def test_exports_pin_new_identity_and_match_new_weights(self):
        native_output = self.directory / "native"
        manifest = self.e.export_policy(self.checkpoint, native_output, self.record_path)
        self.assertEqual(manifest["checkpoint_iteration"], 1000)
        self.assertEqual(manifest["checkpoint_sha256"], self.record["checkpoint_sha256"])
        self.assertEqual(manifest["source_training_files_sha256"], self.record["source_hashes"])
        exported = torch.jit.load(str(native_output / "stairs_policy.pt"))
        network, _ = self.e.load_checkpoint(self.checkpoint)
        previous, _ = self.e.load_checkpoint(self.e.DEFAULT_CHECKPOINT)
        sample = torch.linspace(-.2, .3, 270).reshape(1, 270)
        torch.testing.assert_close(exported(sample), network(sample))
        self.assertFalse(torch.allclose(exported(sample), previous(sample)))
        legacy_output = self.directory / "legacy"
        result = self.legacy.export(legacy_output, self.checkpoint, self.record_path)
        provenance = json.loads((legacy_output / "provenance.json").read_text())
        self.assertFalse(result["active"])
        self.assertFalse(provenance["active"])
        self.assertEqual(provenance["checkpoint_iteration"], 1000)
        self.assertEqual(provenance["checkpoint_sha256"], self.record["checkpoint_sha256"])
        self.assertEqual(provenance["source_training_files_sha256"], self.record["source_hashes"])
        self.assertEqual(provenance["offline_verification"]["samples"], 160)
        self.assertIn("1000", (legacy_output / "README.md").read_text().splitlines()[0])
        with self.assertRaisesRegex(ValueError, "refusing overwrite"):
            self.legacy.export(legacy_output, self.checkpoint, self.record_path)


if __name__ == "__main__":
    unittest.main()
