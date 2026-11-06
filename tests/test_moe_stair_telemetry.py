"""Climb-speed evidence follows the actual top platform after stair extension."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_moe_finetune import Telemetry, gait_summary


class StairTelemetryTest(unittest.TestCase):
    def test_six_steps_only_finish_climb_at_sixth_platform(self):
        data = SimpleNamespace(qpos=np.r_[-.1, 0., 1.2, np.zeros(16)],
                               qvel=np.zeros(18), xmat=np.eye(3)[None].reshape(1, 9),
                               geom_xpos=np.zeros((4, 3)))
        exp = SimpleNamespace(data=data, base=0, feet=np.arange(4), qadr=np.arange(7, 19),
                              vadr=np.arange(6, 18), platform_start=1.5,
                              contacts=lambda: ({0, 1, 2, 3}, set(), False, False))
        env = SimpleNamespace(exp=exp, elapsed=0.)
        telemetry = Telemetry(env)
        for t, x in ((.2, .1), (.4, .4), (1.4, 1.4)):
            env.elapsed = t
            data.qpos[0] = x
            telemetry(env)
            self.assertIsNone(telemetry.exit_time)
        env.elapsed = 1.6
        data.qpos[0] = 1.6
        telemetry(env)
        self.assertAlmostEqual(telemetry.entry_time, .1)
        self.assertAlmostEqual(telemetry.exit_time, 1.5)
        result = telemetry.result()
        self.assertEqual(result["climb_distance_m"], 1.5)
        self.assertAlmostEqual(gait_summary([result])["climb_horizontal_speed_m_s"], 1.5 / 1.4)

    def test_legacy_integrals_keep_two_step_distance(self):
        result = dict(seconds=3., physics_samples=1500, roll_sq_seconds=0.,
                      hip_positive_rad_joint_seconds=0., hip_joint_seconds=12.,
                      stance_slip_sq_contact_seconds=0., stance_contact_seconds=3.,
                      forward_displacement_m=1., climb_seconds=1.)
        self.assertEqual(gait_summary([result])["climb_horizontal_speed_m_s"], .3)


if __name__ == "__main__":
    unittest.main()
