"""Validate user-editable experiment knobs without requiring a simulator."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from go2_him.settings import load_settings


class SettingsTests(unittest.TestCase):
    def test_default_config_is_valid(self):
        settings = load_settings()
        self.assertEqual(settings["num_envs"], 1024)
        self.assertEqual(settings["terrain"]["num_steps"], 8)

    def test_joint_constraints_preserve_old_experiments_and_enforce_guard_bounds(self):
        old = load_settings("configs/stairs_torque_limited.yaml")
        new = load_settings("configs/stairs_joint_limited.yaml")
        self.assertFalse(old["joint_constraints"]["enabled"])
        self.assertEqual(old["rewards"]["velocity_excess"], 0)
        self.assertTrue(new["joint_constraints"]["enabled"])
        self.assertTrue(new["torque"]["enabled"])
        self.assertEqual(new["joint_constraints"]["velocity_limit"], 13.)
        for override in ("joint_constraints.velocity_limit=13.1", "joint_constraints.velocity_limit=0",
                         "joint_constraints.tracking_error_limit=0.91", "joint_constraints.position_soft_ratio=1",
                         "joint_constraints.velocity_soft_ratio=0", "joint_constraints.tracking_soft_ratio=-1",
                         "joint_constraints.startup_seconds=30", "joint_constraints.startup_multiplier=0.5",
                         "joint_constraints.enabled=1", "joint_constraints.terminate_on_violation=0",
                         "rewards.velocity_margin=1", "rewards.velocity_excess=1",
                         "rewards.position_margin=1", "rewards.position_excess=1",
                         "rewards.target_position_excess=1", "rewards.tracking_error=1",
                         "joint_constraints.enabled=false"):
            with self.subTest(override=override), self.assertRaises(ValueError):
                load_settings("configs/stairs_joint_limited.yaml", [override])
        for kind in ("velocity_margin", "velocity_excess", "position_margin", "position_excess",
                     "target_position_excess", "tracking_error"):
            with self.subTest(disabled_penalty=kind), self.assertRaises(ValueError):
                load_settings(overrides=[f"rewards.{kind}=-1"])

    def test_nested_overrides_preserve_other_defaults(self):
        settings = load_settings(overrides=["rewards.foot_clearance=-0.3", "training.learning_rate=0.0005"])
        self.assertEqual(settings["rewards"]["foot_clearance"], -.3)
        self.assertEqual(settings["training"]["learning_rate"], .0005)
        self.assertEqual(settings["terrain"]["num_steps"], 8)
        self.assertEqual(load_settings()["rewards"]["foot_clearance"], -.1)

    def test_partial_yaml_merges_and_cli_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.yaml"
            path.write_text("num_envs: 32\nrewards:\n  foot_slip: -0.4\n")
            settings = load_settings(path, ["num_envs=64"])
        self.assertEqual(settings["num_envs"], 64)
        self.assertEqual(settings["rewards"]["foot_slip"], -.4)

    def test_typo_and_wrong_mapping_are_rejected(self):
        for override in ("reward.foot_slip=-1", "rewards.foot_sliip=-1", "terrain=3", "num_envs", "=2", "control.action_scale.foo=1"):
            with self.subTest(override=override), self.assertRaises(ValueError):
                load_settings(overrides=[override])

    def test_non_mapping_yaml_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            for content in ("- 1\n- 2\n", "[]", "false", "0"):
                path.write_text(content)
                with self.subTest(content=content), self.assertRaises(ValueError):
                    load_settings(path)

    def test_boolean_flags_are_strict(self):
        settings = load_settings(overrides=["terrain.curriculum=false", "randomization.enabled=false"])
        self.assertFalse(settings["terrain"]["curriculum"])
        for override in ("terrain.curriculum=1", "randomization.enabled=0", "num_envs=true", "rewards.success=true"):
            with self.subTest(override=override), self.assertRaises(ValueError):
                load_settings(overrides=[override])

    def test_nonfinite_and_nonnumeric_values_are_rejected(self):
        for override in ("rewards.success=.nan", "control.stiffness=.inf", "training.learning_rate=null", "targets.base_height=hello"):
            with self.subTest(override=override), self.assertRaises(ValueError):
                load_settings(overrides=[override])

    def test_integer_and_terrain_boundaries_are_rejected(self):
        for override in ("num_envs=0", "max_iterations=-1", "seed=-1", "seed=4294967296", "terrain.rows=1.5",
                         "terrain.initial_level=0.5", "terrain.initial_level=10", "terrain.height_max=0.31",
                         "terrain.height_min=0.2", "terrain.tread=0", "terrain.num_steps=20"):
            with self.subTest(override=override), self.assertRaises(ValueError):
                load_settings(overrides=[override])

    def test_control_training_and_target_ranges_are_rejected(self):
        for override in ("control.action_clip=0", "control.action_scale=-1", "control.damping=0",
                         "commands.forward_min=0.7", "episode_length_s=0", "training.learning_rate=-0.01",
                         "training.entropy_coef=-1", "training.save_interval=0", "targets.base_height=0",
                         "targets.foot_clearance=-0.1"):
            with self.subTest(override=override), self.assertRaises(ValueError):
                load_settings(overrides=[override])

    def test_randomization_ranges_are_ordered(self):
        for override in ("randomization.friction_min=-0.1", "randomization.friction_max=0.5",
                         "randomization.payload_min=2"):
            with self.subTest(override=override), self.assertRaises(ValueError):
                load_settings(overrides=[override])

    def test_minibatch_count_cannot_exceed_samples(self):
        with self.assertRaises(ValueError):
            load_settings(overrides=["num_envs=1", "training.num_steps_per_env=1", "training.num_mini_batches=2"])
        settings = load_settings(overrides=["num_envs=1", "training.num_steps_per_env=1", "training.num_mini_batches=1"])
        self.assertEqual(settings["num_envs"], 1)

    def test_gait_refinement_is_opt_in_and_noise_lr_bounds_are_consistent(self):
        baseline = load_settings()
        natural = load_settings(Path(__file__).resolve().parents[1] / "configs/stairs_natural.yaml")
        self.assertEqual(baseline["rewards"]["action_smoothness"], 0.)
        self.assertEqual(baseline["training"]["max_noise_std"], 0.)
        self.assertLess(natural["rewards"]["action_smoothness"], 0.)
        self.assertEqual(natural["training"]["max_noise_std"], .8)
        for override in ("training.min_noise_std=0", "training.max_noise_std=0.2",
                         "training.max_noise_std=-1", "training.min_learning_rate=0.1",
                         "training.max_learning_rate=0.0001", "training.action_mean_penalty_coef=-1",
                         "targets.air_time=0.5", "targets.max_air_time=0"):
            with self.subTest(override=override), self.assertRaises(ValueError):
                load_settings(overrides=[override])

    def test_lateral_penalty_bounds_and_signs(self):
        baseline = load_settings()
        lateral = load_settings(Path(__file__).resolve().parents[1] / "configs/stairs_lateral.yaml")
        self.assertEqual(baseline["rewards"]["hip_abduction"], 0.)
        self.assertEqual(baseline["rewards"]["feet_lateral"], 0.)
        self.assertLess(lateral["rewards"]["hip_abduction"], 0.)
        self.assertLess(lateral["rewards"]["feet_lateral"], 0.)
        for override in ("targets.hip_abduction_max=0", "targets.hip_abduction_max=1.1",
                         "targets.feet_lateral_min=-0.1", "targets.feet_lateral_max=0.1",
                         "rewards.hip_abduction=1", "rewards.feet_lateral=1"):
            with self.subTest(override=override), self.assertRaises(ValueError):
                load_settings(overrides=[override])

    def test_stability_speed_refinement_is_opt_in_and_accepts_fixed_eval_commands(self):
        baseline = load_settings()
        path = Path(__file__).resolve().parents[1] / "configs/stairs_stable_fast.yaml"
        stable = load_settings(path)
        self.assertEqual(baseline["rewards"]["roll_rate"], 0.)
        self.assertEqual(baseline["rewards"]["lateral_velocity"], 0.)
        self.assertEqual(baseline["commands"]["speed_ramp_updates"], 0)
        self.assertEqual(baseline["targets"]["feet_lateral_inner_weight"], 1.)
        self.assertEqual(stable["commands"]["speed_ramp_updates"], 300)
        self.assertEqual(stable["max_iterations"], 600)
        self.assertLess(stable["rewards"]["roll_rate"], 0.)
        self.assertLess(stable["rewards"]["lateral_velocity"], 0.)
        self.assertGreater(stable["targets"]["feet_lateral_inner_weight"], 1.)
        fixed = load_settings(path, ["commands.forward_min=0.4", "commands.forward_max=0.4"])
        self.assertEqual(fixed["commands"]["forward_max"], .4)
        for override in ("commands.speed_ramp_updates=-1", "commands.speed_ramp_updates=3.5",
                         "commands.start_forward_min=0", "commands.start_forward_max=0.1",
                         "rewards.roll_rate=0.1", "rewards.lateral_velocity=1",
                         "targets.feet_lateral_inner_weight=0", "targets.feet_lateral_inner_weight=-1"):
            with self.subTest(override=override), self.assertRaises(ValueError):
                load_settings(overrides=[override])


if __name__ == "__main__":
    unittest.main()
