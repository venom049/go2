"""CPU tests of the real HIM learner and Lab boundary adapter, without Isaac Sim.

Run with any Python environment containing PyTorch and NumPy:
    python -m unittest discover -s tests -p test_him.py -v

The environment doubles below exercise the actual wrapper/learner.  They do not
claim to validate physics or robot performance.
"""
from __future__ import annotations

import importlib
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "upstream/himloco_lab/source/himloco_lab"))

from himloco_lab.rsl_rl.algorithms import HIMPPO
from himloco_lab.rsl_rl.modules import HIMActorCritic, HIMEstimator
from himloco_lab.rsl_rl.runners import HIMOnPolicyRunner
from himloco_lab.rsl_rl.storage import HIMRolloutStorage


class FakeLabEnv:
    """Two-environment auto-reset boundary with distinguishable terminal states."""

    def __init__(self):
        self.unwrapped = self
        self.num_envs = 2
        self.device = "cpu"
        self.max_episode_length = 100
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        self.action_manager = SimpleNamespace(total_action_dim=2)
        self.observation_manager = SimpleNamespace(group_obs_dim={"policy": (3,), "critic": (6,)})
        self.cfg = SimpleNamespace(is_finite_horizon=False)

    def reset(self):
        self.policy = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        self.critic = torch.cat((self.policy, self.policy + 20), dim=-1)
        return {"policy": self.policy.clone(), "critic": self.critic.clone()}, {}

    def step(self, actions):
        self.received_actions = actions.clone()
        policy_before = self.policy + 10
        critic_before = self.critic + 10
        self.policy = policy_before.clone()
        self.critic = critic_before.clone()
        self.policy[0] = -1
        self.critic[0] = -2
        return (
            {"policy": self.policy.clone(), "critic": self.critic.clone()},
            {"policy": policy_before, "critic": critic_before},
            torch.ones(2), torch.tensor([True, False]), torch.zeros(2, dtype=torch.bool), {},
        )


class FakeRunnerEnv:
    num_envs = 4
    num_obs = 270
    num_one_step_obs = 45
    num_privileged_obs = 48
    num_actions = 12
    max_episode_length = 3

    def __init__(self):
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.obs = torch.zeros(self.num_envs, self.num_obs)
        self.critic = torch.zeros(self.num_envs, self.num_privileged_obs)

    def get_observations(self):
        return self.obs

    def get_privileged_observations(self):
        return self.critic

    def step(self, actions):
        self.episode_length_buf += 1
        newest = torch.randn(self.num_envs, self.num_one_step_obs) * .05
        self.obs = torch.cat((newest, self.obs[:, :-self.num_one_step_obs]), dim=1)
        self.critic = torch.cat((newest, torch.randn(self.num_envs, 3) * .1), dim=1)
        dones = self.episode_length_buf >= self.max_episode_length
        ids = dones.nonzero().flatten()
        terminal_critic = self.critic[ids].clone()
        self.episode_length_buf[ids] = 0
        self.obs[ids] = 0
        self.critic[ids] = 0
        rewards = 1 - .01 * actions.square().sum(-1)
        return self.obs, self.critic, rewards, dones, {"time_outs": dones}, ids, terminal_critic


def small_runner_cfg():
    return {
        "policy_class_name": "HIMActorCritic", "algorithm_class_name": "HIMPPO",
        "policy": {"actor_hidden_dims": [32, 16], "critic_hidden_dims": [32, 16]},
        "algorithm": {"num_learning_epochs": 2, "num_mini_batches": 2, "learning_rate": 1.e-3},
        "num_steps_per_env": 4, "save_interval": 1, "logger": "tensorboard",
    }


class HIMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        # Only the nominal environment base type is replaced.  All history and
        # terminal-state logic under test comes from the real wrapper module.
        fake_module = ModuleType("himloco_lab.envs")
        fake_module.HimlocoManagerBasedRLEnv = FakeLabEnv
        with patch.dict(sys.modules, {"himloco_lab.envs": fake_module}):
            wrapper = importlib.import_module("himloco_lab.rsl_rl.wrappers.himloco_vec_env_wrapper")
        cls.wrapper_type = wrapper.HimlocoVecEnvWrapper

    def setUp(self):
        torch.manual_seed(23)

    def test_algorithm_import_does_not_load_tasks_or_assets(self):
        self.assertNotIn("himloco_lab.tasks", sys.modules)
        self.assertNotIn("himloco_lab.assets.unitree", sys.modules)
        self.assertNotIn("isaaclab_tasks", sys.modules)

    def test_reset_observations_populate_all_history_frames(self):
        wrapped = self.wrapper_type(FakeLabEnv(), history_length=2, privileged_history_length=1)
        expected = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        torch.testing.assert_close(wrapped.get_observations(), expected.repeat(1, 3))
        self.assertEqual(wrapped.get_privileged_observations().shape, (2, 12))
        wrapped.step(torch.zeros(2, 2))
        obs, _ = wrapped.reset()
        torch.testing.assert_close(obs, expected.repeat(1, 3))

    def test_done_clears_history_and_keeps_terminal_critic_history(self):
        wrapped = self.wrapper_type(FakeLabEnv(), history_length=2, privileged_history_length=1)
        initial_obs = wrapped.get_observations().clone()
        initial_critic = wrapped.get_privileged_observations().clone()
        obs, critic, _, dones, _, ids, terminal = wrapped.step(torch.zeros(2, 2))
        torch.testing.assert_close(ids, torch.tensor([0]))
        self.assertTrue(dones[0])
        torch.testing.assert_close(obs[0], torch.full((9,), -1.))
        torch.testing.assert_close(critic[0], torch.full((12,), -2.))
        torch.testing.assert_close(obs[1, :3], initial_obs[1, :3] + 10)
        torch.testing.assert_close(obs[1, 3:], initial_obs[1, :-3])
        expected_terminal = torch.cat((initial_critic[0, :6] + 10, initial_critic[0, :6]))
        torch.testing.assert_close(terminal[0], expected_terminal)

    def test_history_length_validation(self):
        with self.assertRaises(ValueError):
            self.wrapper_type(FakeLabEnv(), history_length=-1)

    def test_action_clip_happens_before_lab_scaling(self):
        base = FakeLabEnv()
        wrapped = self.wrapper_type(base, clip_actions=4.0)
        wrapped.step(torch.tensor([[9., -9.], [.5, -1.]]))
        torch.testing.assert_close(base.received_actions, torch.tensor([[4., -4.], [.5, -1.]]))
        with self.assertRaises(ValueError):
            self.wrapper_type(FakeLabEnv(), clip_actions=float("nan"))

    def test_raw_ppo_samples_and_log_probs_survive_environment_clipping(self):
        base = FakeLabEnv()
        wrapped = self.wrapper_type(base, history_length=1, clip_actions=4.0)
        policy = HIMActorCritic(6, 6, 3, 2, actor_hidden_dims=[8], critic_hidden_dims=[8])
        algorithm = HIMPPO(policy, action_clip=4.0)
        algorithm.init_storage(2, 1, [6], [6], [2])
        obs, critic = wrapped.get_observations(), wrapped.get_privileged_observations()
        policy.update_distribution(obs)
        raw = torch.tensor([[9., -9.], [.5, -1.]])
        expected_log_prob = policy.get_actions_log_prob(raw).detach().clone()
        with patch.object(policy, "act", return_value=raw):
            actions = algorithm.act(obs, critic)
        _, next_critic, rewards, done, infos, ids, terminal = wrapped.step(actions)
        next_critic = next_critic.clone()
        next_critic[ids] = terminal
        algorithm.process_env_step(rewards, done, infos, next_critic)
        torch.testing.assert_close(base.received_actions, raw.clamp(-4., 4.))
        torch.testing.assert_close(algorithm.storage.actions[0], raw)
        torch.testing.assert_close(algorithm.storage.actions_log_prob[0, :, 0], expected_log_prob)

    def test_noise_projection_preserves_checkpoint_keys_and_live_gradients(self):
        legacy = HIMActorCritic(12, 9, 6, 2, actor_hidden_dims=[8], critic_hidden_dims=[8])
        legacy.std.data.fill_(4.86)
        bounded = HIMActorCritic(12, 9, 6, 2, actor_hidden_dims=[8], critic_hidden_dims=[8],
                                min_noise_std=.05, max_noise_std=.8)
        self.assertEqual(set(legacy.state_dict()), set(bounded.state_dict()))
        bounded.load_state_dict(legacy.state_dict(), strict=True)
        observations = torch.randn(3, 12)
        torch.testing.assert_close(bounded.act_inference(observations), legacy.act_inference(observations))
        bounded.project_noise_std_()
        torch.testing.assert_close(bounded.std, torch.full((2,), .8))
        bounded.effective_noise_std.sum().backward()
        # Forward-only clamping a loaded 4.86 parameter would give zero here.
        torch.testing.assert_close(bounded.std.grad, torch.ones(2))
        with torch.no_grad():
            bounded.std.fill_(-1.)
        bounded.project_noise_std_()
        torch.testing.assert_close(bounded.effective_noise_std, torch.full((2,), .05))
        legacy.project_noise_std_()
        torch.testing.assert_close(legacy.effective_noise_std, torch.full((2,), 4.86))

    def test_mean_overflow_penalty_has_correct_direction_and_ignores_safe_means(self):
        policy = HIMActorCritic(12, 9, 6, 2, actor_hidden_dims=[8], critic_hidden_dims=[8])
        algorithm = HIMPPO(policy, action_clip=2., action_mean_penalty_coef=.1)
        means = torch.tensor([[-3., 0., 4.], [1., -1., 2.]], requires_grad=True)
        loss = algorithm.action_mean_overflow(means)
        self.assertAlmostEqual(loss.item(), 2.5)
        loss.backward()
        torch.testing.assert_close(means.grad, torch.tensor([[-1., 0., 2.], [0., 0., 0.]]))
        with self.assertRaisesRegex(ValueError, "requires action_clip"):
            HIMPPO(policy, action_mean_penalty_coef=.1)

    def test_noise_and_adaptive_rate_bounds_apply_during_updates_and_checkpoint_load(self):
        cfg = small_runner_cfg()
        cfg["policy"].update(init_noise_std=.1, min_noise_std=.05, max_noise_std=.1)
        cfg["algorithm"].update(entropy_coef=100., learning_rate=.005,
            schedule="adaptive", min_learning_rate=.0002, max_learning_rate=.0003, action_clip=.05)
        runner = HIMOnPolicyRunner(FakeRunnerEnv(), cfg, device="cpu")
        runner.learn(1)
        self.assertTrue(torch.all(runner.alg.actor_critic.std <= .1))
        self.assertTrue(torch.all(runner.alg.actor_critic.std >= .05))
        self.assertGreaterEqual(runner.alg.learning_rate, .0002)
        self.assertLessEqual(runner.alg.learning_rate, .0003)
        self.assertGreater(runner.alg.last_update_metrics["action_clip_fraction"], 0.)
        self.assertLessEqual(runner.alg.last_update_metrics["action_clip_fraction"], 1.)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = str(Path(directory) / "legacy.pt")
            # Emulate an old checkpoint whose exploration std exceeds new bounds.
            runner.alg.actor_critic.std.data.fill_(4.86)
            runner.save(checkpoint)
            restored = HIMOnPolicyRunner(FakeRunnerEnv(), cfg, device="cpu")
            restored.load(checkpoint)
            torch.testing.assert_close(restored.alg.actor_critic.std, torch.full((12,), .1))

    def test_wrapper_and_algorithm_action_limits_must_agree(self):
        cfg = small_runner_cfg()
        env = FakeRunnerEnv()
        env.clip_actions = 4.
        runner = HIMOnPolicyRunner(env, cfg, device="cpu")
        self.assertEqual(runner.alg.action_clip, 4.)
        self.assertNotIn("action_clip", cfg["algorithm"])
        cfg["algorithm"]["action_clip"] = 2.
        with self.assertRaisesRegex(ValueError, "must match"):
            HIMOnPolicyRunner(env, cfg, device="cpu")

    def test_true_terminal_takes_precedence_over_time_limit(self):
        base = FakeLabEnv()
        wrapped = self.wrapper_type(base)
        original_step = base.step

        def simultaneous_time_limit(actions):
            result = list(original_step(actions))
            result[4] = torch.ones(2, dtype=torch.bool)
            return tuple(result)

        with patch.object(base, "step", side_effect=simultaneous_time_limit):
            *_, infos, ids, terminal = wrapped.step(torch.zeros(2, 2))
        torch.testing.assert_close(infos["time_outs"], torch.tensor([False, True]))

    def test_nonfinite_terminal_critic_is_not_hidden_by_auto_reset(self):
        base = FakeLabEnv()
        wrapped = self.wrapper_type(base)
        original_step = base.step

        def corrupted_terminal(actions):
            result = original_step(actions)
            result[1]["critic"][0, -1] = float("nan")
            return result

        with patch.object(base, "step", side_effect=corrupted_terminal):
            with self.assertRaisesRegex(ValueError, "terminal critic"):
                wrapped.step(torch.zeros(2, 2))

    def test_estimator_uses_policy_prefix_and_velocity_suffix(self):
        estimator = HIMEstimator(temporal_steps=2, num_one_step_obs=6)
        history = torch.randn(8, 12)
        critic = torch.randn(8, 11)
        critic[:, :3] = 7.0  # Commands must remain in the future observation.
        critic[:, 6:9] = .4  # Privileged velocity must not enter its policy prefix.
        expected_velocity_loss = torch.nn.functional.mse_loss(
            estimator.encode(history)[0], critic[:, 6:9]
        ).item()
        seen = []
        hook = estimator.target.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
        velocity_loss, swap_loss = estimator.update(history, critic)
        hook.remove()
        torch.testing.assert_close(seen[0], critic[:, :6])
        self.assertAlmostEqual(velocity_loss, expected_velocity_loss, places=6)
        self.assertTrue(torch.isfinite(torch.tensor(swap_loss)))

    def test_timeout_bootstrap_uses_terminal_next_value(self):
        policy = HIMActorCritic(12, 9, 6, 2, actor_hidden_dims=[8], critic_hidden_dims=[8])
        algorithm = HIMPPO(policy, gamma=.9)
        algorithm.init_storage(2, 1, [12], [9], [2])
        algorithm.act(torch.zeros(2, 12), torch.zeros(2, 9))
        algorithm.transition.values.fill_(1.)
        with patch.object(policy, "evaluate", return_value=torch.tensor([[10.], [20.]])):
            algorithm.process_env_step(torch.tensor([2., 3.]), torch.ones(2, dtype=torch.bool),
                                       {"time_outs": torch.tensor([True, False])}, torch.ones(2, 9))
        torch.testing.assert_close(algorithm.storage.rewards[0, :, 0], torch.tensor([11., 3.]))

    def test_minibatches_cover_remainder_samples(self):
        storage = HIMRolloutStorage(5, 1, [1], [4], [1])
        storage.observations[0, :, 0] = torch.arange(5)
        samples = [batch[0].flatten() for batch in storage.mini_batch_generator(2, 1)]
        torch.testing.assert_close(torch.cat(samples).sort().values, torch.arange(5).float())
        with self.assertRaises(ValueError):
            list(storage.mini_batch_generator(6, 1))

    def test_single_sample_advantage_is_finite(self):
        storage = HIMRolloutStorage(1, 1, [1], [4], [1])
        storage.compute_returns(torch.zeros(1, 1), .99, .95)
        self.assertTrue(torch.isfinite(storage.advantages).all())

    def test_runner_updates_estimator_and_policy_and_resumes_next_iteration(self):
        runner = HIMOnPolicyRunner(FakeRunnerEnv(), small_runner_cfg(), device="cpu")
        initial_actor = runner.alg.actor_critic.actor[0].weight.detach().clone()
        initial_estimator = runner.alg.actor_critic.estimator.encoder[0].weight.detach().clone()
        runner.learn(2)
        self.assertEqual(runner.current_learning_iteration, 2)
        self.assertFalse(torch.equal(initial_actor, runner.alg.actor_critic.actor[0].weight))
        self.assertFalse(torch.equal(initial_estimator, runner.alg.actor_critic.estimator.encoder[0].weight))
        self.assertTrue(all(torch.isfinite(p).all() for p in runner.alg.actor_critic.parameters()))
        runner.alg.learning_rate = 2.e-4
        for group in runner.alg.optimizer.param_groups:
            group["lr"] = 2.e-4
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = str(Path(directory) / "model_2.pt")
            runner.save(checkpoint)
            restored = HIMOnPolicyRunner(FakeRunnerEnv(), small_runner_cfg(), device="cpu")
            restored.load(checkpoint)
            self.assertEqual(restored.current_learning_iteration, 2)
            self.assertEqual(restored.alg.learning_rate, 2.e-4)
            torch.testing.assert_close(restored.alg.actor_critic.actor[0].weight,
                                       runner.alg.actor_critic.actor[0].weight)
            restored.learn(1)
            self.assertEqual(restored.current_learning_iteration, 3)


if __name__ == "__main__":
    unittest.main()
