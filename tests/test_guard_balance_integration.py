"""Use the actual vendored step and reward path, with fake physics/managers."""
from pathlib import Path
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None

from tests import test_joint_integration as integration


@unittest.skipIf(torch is None, "PyTorch required")
class GuardBalanceIntegrationTests(unittest.TestCase):
    def test_final_step_cost_and_diagnostics_survive_reset_log_replacement(self):
        from go2_him.settings import load_settings
        helper = integration.JointIntegrationTests()
        helper.lab, helper.mdp = integration._load_integration()
        env = helper.make_env(spike=13.313)
        env.cfg.go2_settings = load_settings(Path(__file__).resolve().parents[1] /
                                             "configs/stairs_guard_balanced.yaml")
        # The goal is reached on a step with a final-substep overspeed. There
        # must be a cost and raw violation, but no guard-induced early reset.
        env.reward_manager = SimpleNamespace(compute=lambda dt: -helper.mdp.guard_balance_penalty(env) * dt)
        original_reset = env._reset_idx

        def reset_with_new_log(ids):
            original_reset(ids)
            env.extras["log"] = {"Reset/new_log": torch.tensor(1.)}

        env._reset_idx = reset_with_new_log
        result = env.step(torch.zeros(2, 12))
        self.assertEqual(env.success_before_reset.tolist(), [True, True])
        self.assertLess(float(result[2][0]), 0.)
        self.assertEqual(float(result[2][1]), 0.)
        logs = env.extras["log"]
        self.assertIn("Reset/new_log", logs)
        self.assertIn("GuardBalance/total_cost", logs)
        self.assertGreater(float(logs["GuardBalance/total_cost"]), 0.)
        self.assertGreater(float(logs["JointLimits/velocity_peak_ratio"]), 1.)
        torch.testing.assert_close(result[2].mean(), -logs["GuardBalance/total_cost"] * env.step_dt)
        self.assertEqual(float(env.scene["robot"].data.joint_vel.abs().max()), 0.)


if __name__ == "__main__":
    unittest.main()
