"""Behavioral checks for the MoE stair task reward, with no physics dependency."""
import copy
import unittest

import numpy as np

from go2_him.moe_stair_rewards import compute_stair_rewards


def state(speed=0.6, dt=0.02):
    return {
        "body_vxy": [speed, 0.], "commands": [.6, 0., 0.],
        "body_w": [0., 0., 0.], "gravity": [0., 0., -1.],
        "foot_clearance": [0., 0., 0., 0.],
        "foot_velocity": np.zeros((4, 3)), "foot_contact": [True] * 4,
        "actions": np.zeros(12), "prev_actions": np.zeros(12),
        "base_height": .3, "body_vz": 0., "dx": speed * dt, "dt": dt,
    }


class MoEStairRewardTests(unittest.TestCase):
    def test_progress_outweighs_full_guard_budget_and_stalling_has_no_positive_reward(self):
        forward = compute_stair_rewards(state())
        stalled = compute_stair_rewards(state(speed=0.))
        self.assertGreater(forward["rate_total"] - 3., 5.)
        self.assertLess(stalled["rate_total"], 0.)
        for name in ("progress", "tracking", "yaw_tracking"):
            self.assertEqual(stalled["rate_terms"][name], 0.)

    def test_progress_cap_does_not_reward_rushing_toward_wall(self):
        normal = compute_stair_rewards(state())
        rushing = compute_stair_rewards(state(speed=2.))
        reverse = compute_stair_rewards(state(speed=-.6))
        self.assertEqual(normal["rate_terms"]["progress"], rushing["rate_terms"]["progress"])
        self.assertLess(rushing["rate_total"], normal["rate_total"])
        self.assertEqual(reverse["rate_terms"]["progress"], 0.)
        self.assertLess(reverse["rate_total"], 0.)

    def test_pitch_is_allowed_but_roll_is_penalized(self):
        level = compute_stair_rewards(state())
        pitch_state = state()
        pitch_state["gravity"] = [-.5, 0., -np.sqrt(.75)]
        roll_state = state()
        roll_state["gravity"] = [0., -.5, -np.sqrt(.75)]
        self.assertEqual(compute_stair_rewards(pitch_state)["rate_total"], level["rate_total"])
        self.assertLess(compute_stair_rewards(roll_state)["rate_total"], level["rate_total"] - 1.)

    def test_dt_integrates_continuous_rates_once(self):
        fast = compute_stair_rewards(state(dt=.01))
        slow = compute_stair_rewards(state(dt=.02))
        self.assertAlmostEqual(fast["rate_total"], slow["rate_total"])
        self.assertAlmostEqual(fast["step_reward"] * 100, slow["step_reward"] * 50)
        self.assertAlmostEqual(fast["rate_terms"]["progress"], 5.)

    def test_stop_phase_disables_progress_and_rewards_grounded_stop(self):
        stop = state(speed=0.)
        stop.update(commands=[0., 0., 0.], stop_phase=True)
        stationary = compute_stair_rewards(stop)
        stop.update(body_vxy=[.6, 0.], dx=.012)
        drifting = compute_stair_rewards(stop)
        self.assertGreater(stationary["rate_total"], drifting["rate_total"])
        self.assertEqual(drifting["rate_terms"]["progress"], 0.)
        self.assertEqual(drifting["rate_terms"]["stop_hold"], 0.)
        stop.update(body_vxy=[.051, 0.], dx=.00102)
        self.assertEqual(compute_stair_rewards(stop)["rate_terms"]["stop_hold"], 0.)
        stop["foot_contact"] = [True, True, False, False]
        self.assertEqual(compute_stair_rewards(stop)["rate_terms"]["stop_hold"], 0.)

    def test_foot_clearance_is_terrain_relative_and_only_swing_feet_count(self):
        good = state()
        good["foot_contact"] = [False, True, True, True]
        good["foot_velocity"][0, 0] = .5
        good["foot_clearance"][0] = .17
        bad = copy.deepcopy(good)
        bad["foot_clearance"][0] = .02
        self.assertEqual(compute_stair_rewards(good)["rate_terms"]["foot_clearance"], 0.)
        self.assertLess(compute_stair_rewards(bad)["rate_terms"]["foot_clearance"], 0.)

    def test_events_do_not_scale_with_dt_and_caller_owns_one_shot_flags(self):
        first = state()
        first.update(stop_phase=True, commands=[0., 0., 0.],
                     new_platform_event=True, success_event=True)
        event = compute_stair_rewards(first)
        first["dt"] = .01
        self.assertEqual(compute_stair_rewards(first)["event_total"], event["event_total"])
        first.update(new_platform_event=False, success_event=False)
        self.assertEqual(compute_stair_rewards(first)["event_total"], 0.)
        first.update(new_platform_event=True, success_event=True, failure_event=True)
        self.assertEqual(compute_stair_rewards(first)["event_total"], -10.)

    def test_joint_pose_is_diagnostic_only_not_full_range_posture_penalty(self):
        sample = state()
        baseline = compute_stair_rewards(sample)
        sample.update(jointq=np.full(12, 2.), q0=np.zeros(12))
        result = compute_stair_rewards(sample)
        self.assertEqual(result["step_reward"], baseline["step_reward"])
        self.assertEqual(result["diagnostics"]["default_pose_deviation_l1_diagnostic_only"], 24.)
        self.assertFalse(result["diagnostics"]["joint_guard_cost_included"])

    def test_invalid_state_is_not_silently_rewarded(self):
        for key, value in (("dt", 0.), ("dx", np.nan), ("foot_contact", [0, 1, 2, 1])):
            sample = state()
            sample[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                compute_stair_rewards(sample)


if __name__ == "__main__":
    unittest.main()
