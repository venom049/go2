"""Six-level geometry, curriculum and final-platform checks for offline MoE RL."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_moe_strict_guard import environment
from go2_him.moe_stair_env import MoEStairEnv, env_worker, stair_environment_contract


class StairCountTest(unittest.TestCase):
    def test_only_step_count_and_derived_geometry_change(self):
        old = stair_environment_contract()
        six = stair_environment_contract(6)
        changed = {key for key in old if old[key] != six[key]}
        self.assertEqual(changed, {'steps', 'platform_start', 'platform_end', 'top_height', 'wall_front'})
        self.assertEqual(six['platform_start'], 1.5)
        self.assertEqual(six['platform_end'], 2.4)
        self.assertAlmostEqual(six['top_height'], .9)
        self.assertEqual(six['wall_front'], 2.4)
        self.assertEqual(six['episode_timeout_seconds'], 15.)
        self.assertEqual(six['stable_stop_seconds'], 2.)
        for invalid in (0, 1, True, 2.5, '6'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                stair_environment_contract(invalid)

    def test_two_step_terrain_is_identical_to_original(self):
        env = environment(mode='audit')
        xy = np.array([[x, y] for x in (-.1, 0., .1, .299, .3, .6, 1.2, 1.201)
                       for y in (-.451, -.45, 0., .45, .451)])
        x, y = xy[:, 0], xy[:, 1]
        original = np.where(np.abs(y) <= .45,
                            np.where((x >= .30) & (x <= 1.20), .30,
                                     np.where((x >= 0) & (x < .30), .15, 0.)), 0.)
        np.testing.assert_array_equal(env.terrain_height(xy), original)

    def test_six_step_terrain_has_all_six_levels_and_landing_edges(self):
        env = environment(mode='audit')
        env.stair_steps, env.top = 6, .9
        points = np.array([[.15, 0.], [.45, 0.], [.75, 0.], [1.05, 0.], [1.35, 0.],
                           [1.5, 0.], [2.4, 0.], [2.401, 0.], [1.6, .451], [-.001, 0.]])
        np.testing.assert_allclose(env.terrain_height(points), [.15, .3, .45, .6, .75, .9, .9, 0., 0., 0.], atol=1e-15)

    def test_stop_command_requires_sixth_landing_not_old_second_level(self):
        env = environment(mode='audit', decimation=1)
        env.stair_steps, env.top = 6, .9
        env.exp.contacts = lambda: ({0, 1, 2, 3}, {0, 1, 2, 3}, False, False)
        # A pose on the former top landing must continue its climb command.
        env.exp.data.qpos[:3] = [.75, 0., .6]
        env.exp.data.geom_xpos[:] = [[.95, .12, .32], [.95, -.12, .32],
                                    [.55, .12, .32], [.55, -.12, .32]]
        env.last_x = .75
        env.step(np.zeros(12))
        self.assertFalse(env.stop_phase)
        self.assertFalse(env.platform_seen)
        # Translate that same supported pose onto the final landing.
        env.exp.data.qpos[:3] += [1.2, 0., .6]
        env.exp.data.geom_xpos[:] += [1.2, 0., .6]
        env.last_x = 1.95
        _, _, done, info = env.step(np.zeros(12))
        self.assertFalse(done)
        self.assertTrue(env.stop_phase)
        self.assertTrue(env.platform_seen)
        self.assertEqual(info['reward_terms']['event_terms']['platform'], 5.)
        self.assertAlmostEqual(env.stable_time, .002)
        _, _, _, info = env.step(np.zeros(12))
        self.assertEqual(info['reward_terms']['event_terms']['platform'], 0.)
        self.assertAlmostEqual(env.stable_time, .004)

    def test_worker_passes_requested_count_to_every_environment(self):
        class Connection:
            def __init__(self): self.sent = []; self.closed = False
            def send(self, item): self.sent.append(item)
            def recv(self): return 'close', None
            def close(self): self.closed = True

        conn = Connection()
        with patch('go2_him.moe_stair_env.MoEStairEnv') as constructor:
            constructor.return_value.observation.return_value = np.zeros(225)
            env_worker(conn, 41, 3, stair_steps=6)
            self.assertEqual([call.args for call in constructor.call_args_list], [(41,), (42,), (43,)])
            self.assertEqual([call.kwargs for call in constructor.call_args_list], [{'stair_steps': 6}] * 3)
        self.assertEqual(conn.sent[0].shape, (3, 225))
        self.assertTrue(conn.closed)


class StairPhysicsTest(unittest.TestCase):
    def test_real_geometry_raycast_and_start_distance_across_curriculum(self):
        for steps in (2, 6):
            env = MoEStairEnv(seed=42, evaluation=True, guard_objective='target_clipped', stair_steps=steps)
            try:
                e, m, d, mj = env.exp, env.exp.model, env.exp.data, env.exp.mj
                ids = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, f'stair_{level}')
                       for level in range(1, steps)] + [e.platform_id]
                self.assertTrue(all(gid >= 0 for gid in ids))
                self.assertEqual(mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, f'stair_{steps}'), -1)
                self.assertEqual(len(ids), steps)
                self.assertAlmostEqual(m.geom_pos[e.wall_id, 0] - m.geom_size[e.wall_id, 0], env.platform_end)
                self.assertAlmostEqual(m.geom_pos[e.platform_id, 0] - m.geom_size[e.platform_id, 0], env.platform_start)
                self.assertAlmostEqual(m.geom_pos[e.platform_id, 0] + m.geom_size[e.platform_id, 0], env.platform_end)
                # Descending and then returning to the maximum verifies that
                # physical surfaces follow each reset, not stale compiled size.
                for height in (.15, .067, .1245, .15):
                    with self.subTest(steps=steps, height=height):
                        obs = env.reset(sample={'yaw_rad': 0., 'y_m': 0.}, height=height, vx=.6)
                        self.assertEqual(obs.shape, (225,))
                        np.testing.assert_array_equal(obs[-60:], np.zeros(60))
                        self.assertAlmostEqual(float((d.geom_xpos[e.feet[:2], 0] + e.radii[:2]).max()), -.5)
                        self.assertAlmostEqual(env.top, steps * height)
                        self.assertAlmostEqual(e.top_z, env.top)
                        for level, gid in enumerate(ids, 1):
                            x = (level - .5) * .3 if level < steps else env.platform_start + .45
                            hit = np.array([-1], dtype=np.int32)
                            distance = mj.mj_ray(m, d, np.array([x, 0., 2.]), np.array([0., 0., -1.]),
                                                 None, 1, -1, hit)
                            self.assertEqual(int(hit[0]), gid)
                            self.assertAlmostEqual(2. - distance, level * height, places=12)
                            self.assertAlmostEqual(float(env.terrain_height([x, 0.])), 2. - distance, places=12)
                raw = np.zeros(12, dtype=np.float32)
                raw[2] = 5.4441
                _, _, done, info = env.step(raw)
                self.assertFalse(done)
                self.assertGreater(info['target_violation_rad'][2], 0.)
                self.assertLessEqual(info['target_step']['sent_target_rad'][2], env.profile.upper[2])
                self.assertEqual(env.target_commands.summary()['sent']['excess_count'], 0)
            finally:
                env.exp.close()


if __name__ == '__main__':
    unittest.main()
