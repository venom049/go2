"""Check conversion failure modes, history mapping and trainability."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from go2_him.moe_cts_actor import MoECTSActor, load_actor_checkpoint
from convert_moe_cts_actor import DEFAULT_SOURCE, reconstruct, check_parity


class TestMoECTSActor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.model, cls.graph = reconstruct(DEFAULT_SOURCE)

    def test_history_term_to_frame_order(self):
        obs = torch.arange(225, dtype=torch.float32)[None, :]
        history, latest = self.model._history(obs, None)
        terms = np.split(np.arange(225), [15, 30, 45, 105, 165])
        expected = np.concatenate([t.reshape(5, -1) for t in terms], axis=1).reshape(1, -1)
        np.testing.assert_array_equal(history.numpy(), expected)
        np.testing.assert_array_equal(latest.numpy(), expected[:, -45:])

    def test_all_computed_onnx_nodes(self):
        report = check_parity(self.model, self.graph, samples=8)
        self.assertTrue(report["all_computed_operators_covered"])
        self.assertGreater(report["computed_node_outputs_checked"], 75)

    def test_both_encoder_and_actor_have_finite_gradients(self):
        model, _ = reconstruct(DEFAULT_SOURCE)
        inputs = torch.randn(3, 225)
        actions, weights, latent = model.forward_with_aux(inputs)
        actions.square().mean().backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(sum(p.grad.abs().sum().item() for p in model.student_moe_encoder.parameters()), 0)
        self.assertGreater(sum(p.grad.abs().sum().item() for p in model.actor.parameters()), 0)
        torch.testing.assert_close(weights.sum(1), torch.ones(3))
        torch.testing.assert_close(latent.norm(dim=1), torch.ones(3))

    def test_source_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / "wrong.onnx"
            source.write_bytes(b"not the authorized source")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                reconstruct(source)

    def test_wrong_input_layout_and_dtype_rejected(self):
        for inputs in (torch.zeros(1, 270), torch.zeros(225), torch.zeros(1, 225, dtype=torch.float64)):
            with self.assertRaises(ValueError):
                self.model(inputs)

    def test_checkpoint_requires_conversion_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bad.pt"
            torch.save({"actor_state_dict": self.model.state_dict()}, path)
            with self.assertRaisesRegex(ValueError, "converted"):
                load_actor_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
