"""PPO bookkeeping and warm-start preservation without a physics rollout."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_moe_stairs as training
from go2_him.moe_cts_actor import load_actor_checkpoint

ACTOR_PATH = ROOT / "validation/moe_cts_actor_conversion_20261104/actor.pt"


class SyntheticVec:
    """Deterministic observations/rewards to exercise the actual PPO loop."""
    def __init__(self, n, workers, seed, stair_steps=2, position_constraint='hard'):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.obs = self.rng.normal(0, .15, (n, 225)).astype(np.float32)
        self.steps = 0
        self.closed = False

    def settings(self, frontier, ramp):
        pass

    def step(self, actions):
        self.steps += 1
        self.obs = self.rng.normal(0, .15, (self.n, 225)).astype(np.float32)
        rewards = (.1 * actions[:, 0] + np.arange(self.n) * .05).astype(np.float32)
        done = np.zeros(self.n, dtype=bool)
        return self.obs.copy(), rewards, done, [{"guard_cost": .1} for _ in range(self.n)]

    def close(self):
        self.closed = True


class MoEPPOTests(unittest.TestCase):
    def test_gae_terminations_do_not_bootstrap_from_reset_episode(self):
        reward = torch.tensor([[1.], [2.], [3.]])
        values = torch.tensor([[.5], [.6], [.7]])
        done = torch.tensor([[0.], [1.], [0.]])
        advantage, returns = training.gae(reward, values, done, torch.tensor([4.]), gamma=1., lam=1.)
        torch.testing.assert_close(returns, torch.tensor([[3.], [2.], [7.]]))
        torch.testing.assert_close(advantage, torch.tensor([[2.5], [1.4], [6.3]]))

    def test_gae_lambda_zero_uses_one_step_td_and_handles_multiple_envs(self):
        rewards = torch.tensor([[1., 2.], [3., 4.]])
        values = torch.tensor([[.1, .2], [.3, .4]])
        dones = torch.tensor([[0., 1.], [1., 0.]])
        _, returns = training.gae(rewards, values, dones, torch.tensor([5., 6.]), gamma=.5, lam=0.)
        torch.testing.assert_close(returns, torch.tensor([[1.15, 2.], [3., 7.]]))

    @unittest.skipUnless(ACTOR_PATH.is_file(), "Requires the independently verified conversion artifact")
    def test_actual_training_loop_preserves_warmup_actor_and_frozen_encoder(self):
        torch.set_num_threads(1)
        source = load_actor_checkpoint(ACTOR_PATH)
        initial = {name: value.clone() for name, value in source.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "training"
            args = SimpleNamespace(actor=ACTOR_PATH, out=output, updates=2,
                                   num_envs=4, workers=1, steps=4, epochs=1,
                                   minibatches=1, critic_warmup=1, actor_lr=3e-6,
                                   std=.03, target_kl=1., anchor_coef=.2,
                                   seed=42, device="cpu")
            with patch.object(training, "Vec", SyntheticVec), contextlib.redirect_stdout(io.StringIO()):
                training.train(args)
            warm = torch.load(output / "model_1.pt", map_location="cpu", weights_only=True)
            final = torch.load(output / "model_2.pt", map_location="cpu", weights_only=True)
            self.assertTrue(warm["training_record"]["critic_only_warmup"])
            self.assertFalse(final["training_record"]["critic_only_warmup"])
            for name, value in initial.items():
                torch.testing.assert_close(warm["actor_state_dict"][name], value, rtol=0, atol=0)
                if name.startswith("student_moe_encoder."):
                    torch.testing.assert_close(final["actor_state_dict"][name], value, rtol=0, atol=0)
            self.assertTrue(any(not torch.equal(final["actor_state_dict"][name], value)
                                for name, value in initial.items() if name.startswith("actor.")))
            # Adam must have updated only the critic during warmup. Frozen
            # actor parameters have no optimizer state and cannot drift.
            warm_opt = warm["optimizer_state_dict"]
            actor_ids = warm_opt["param_groups"][0]["params"]
            critic_ids = warm_opt["param_groups"][1]["params"]
            self.assertTrue(all(index not in warm_opt["state"] for index in actor_ids))
            self.assertTrue(all(index in warm_opt["state"] for index in critic_ids))
            self.assertTrue(any(not torch.equal(warm["critic_state_dict"][name], value)
                                for name, value in final["critic_state_dict"].items()))
            loaded = training.load_policy(output / "model_2.pt")
            # requires_grad is deliberately not part of state_dict. A frozen
            # training encoder remains fully present in the reloaded policy.
            self.assertTrue(all(parameter.requires_grad for parameter in loaded.parameters()))
            for name, value in final["actor_state_dict"].items():
                torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)
            with torch.no_grad():
                self.assertTrue(torch.isfinite(loaded(torch.zeros(2, 225))).all())

    @unittest.skipUnless(ACTOR_PATH.is_file(), "Requires the independently verified conversion artifact")
    def test_rejected_actor_step_restores_parameters_and_existing_adam_but_keeps_critic(self):
        """Force a post-step trust-region failure after Adam has real moments."""
        torch.set_num_threads(1)
        real_step = torch.optim.Adam.step
        calls = []

        def step_with_one_overshoot(optimizer, *args, **kwargs):
            result = real_step(optimizer, *args, **kwargs)
            calls.append(True)
            if len(calls) == 2:
                # A controlled post-update disturbance, confined to the action
                # output bias. It is finite and guarantees a KL rejection.
                with torch.no_grad():
                    optimizer.param_groups[0]["params"][-1].add_(100.)
            return result

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rollback"
            args = SimpleNamespace(actor=ACTOR_PATH, out=output, updates=2,
                                   num_envs=4, workers=1, steps=4, epochs=1,
                                   minibatches=1, critic_warmup=0, actor_lr=3e-6,
                                   std=.03, target_kl=1e6, anchor_coef=.2,
                                   seed=42, device="cpu")
            with patch.object(training, "Vec", SyntheticVec), \
                    patch.object(torch.optim.Adam, "step", step_with_one_overshoot), \
                    contextlib.redirect_stdout(io.StringIO()):
                training.train(args)
            self.assertEqual(len(calls), 2)
            before = torch.load(output / "model_1.pt", map_location="cpu", weights_only=True)
            after = torch.load(output / "model_2.pt", map_location="cpu", weights_only=True)
            self.assertEqual(before["training_record"]["rejected_actor_updates"], 0)
            self.assertEqual(after["training_record"]["rejected_actor_updates"], 1)
            self.assertGreater(after["training_record"]["kl_max"], 2 * args.target_kl)
            self.assertEqual(after["training_record"]["accepted_minibatch_post_kl_max"], 0.)
            self.assertEqual(after["training_record"]["actor_learning_rate"], args.actor_lr / 2)
            for name, value in before["actor_state_dict"].items():
                torch.testing.assert_close(after["actor_state_dict"][name], value, rtol=0, atol=0)
            old_opt, new_opt = before["optimizer_state_dict"], after["optimizer_state_dict"]
            for index in old_opt["param_groups"][0]["params"]:
                self.assertTrue(old_opt["state"][index])  # existing moments, not just empty state
                self.assertEqual(set(old_opt["state"][index]), set(new_opt["state"][index]))
                for key, value in old_opt["state"][index].items():
                    torch.testing.assert_close(new_opt["state"][index][key], value, rtol=0, atol=0)
            self.assertTrue(any(not torch.equal(after["critic_state_dict"][name], value)
                                for name, value in before["critic_state_dict"].items()))
            for index in old_opt["param_groups"][1]["params"]:
                self.assertEqual(float(new_opt["state"][index]["step"]),
                                 float(old_opt["state"][index]["step"]) + 1.)


if __name__ == "__main__":
    unittest.main()
