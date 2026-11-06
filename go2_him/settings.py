"""Simulator-free experiment loading; reject typos instead of silently ignoring them."""
import copy
import math
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "stairs.yaml"


def _merge(dst, src, path=""):
    if not isinstance(src, dict):
        raise ValueError((path.rstrip(".") or "Settings") + " must be a mapping")
    for key, value in src.items():
        if key not in dst:
            raise ValueError("Unknown setting: " + path + key)
        if isinstance(dst[key], dict):
            if not isinstance(value, dict):
                raise ValueError(path + key + " must be a mapping")
            _merge(dst[key], value, path + key + ".")
        else:
            if isinstance(value, dict):
                raise ValueError(path + key + " must be a scalar value")
            dst[key] = value


def load_settings(path=None, overrides=()):
    data = yaml.safe_load(DEFAULT_CONFIG.read_text())
    if path:
        custom = yaml.safe_load(Path(path).read_text())
        _merge(data, {} if custom is None else custom)
    for assignment in overrides:
        if "=" not in assignment:
            raise ValueError("Use --set key=value: " + assignment)
        key, value = assignment.split("=", 1)
        patch = yaml.safe_load(value)
        for part in reversed(key.split(".")):
            patch = {part: patch}
        _merge(data, patch)
    validate(data)
    return copy.deepcopy(data)


def validate(s):
    boolean_settings = {"terrain.curriculum", "randomization.enabled", "torque.enabled",
                        "joint_constraints.enabled", "joint_constraints.terminate_on_violation",
                        "guard_balance.enabled"}

    def finite_tree(obj, path=""):
        for key, value in obj.items():
            full_key = path + key
            if isinstance(value, dict):
                finite_tree(value, full_key + ".")
            elif full_key in boolean_settings:
                if type(value) is not bool:
                    raise ValueError(full_key + " must be true or false")
            elif type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(full_key + " must be a finite number")
    finite_tree(s)
    for key in ("num_envs", "max_iterations", "seed"):
        if type(s[key]) is not int or s[key] < (0 if key == "seed" else 1):
            raise ValueError("Invalid " + key)
    if s["seed"] >= 2**32:
        raise ValueError("seed must be in [0, 2**32) for NumPy/Isaac Lab seeding")
    t = s["terrain"]
    for key in ("rows", "cols", "num_steps"):
        if type(t[key]) is not int or t[key] < 1:
            raise ValueError("terrain." + key + " must be positive integer")
    if type(t["initial_level"]) is not int or not 0 <= t["initial_level"] < t["rows"]:
        raise ValueError("terrain.initial_level must be within rows")
    if not 0 <= t["height_min"] <= t["height_max"] <= 0.3:
        raise ValueError("Require 0 <= stair heights <= 0.3 m")
    if t["tread"] <= 0 or t["num_steps"] * t["tread"] > 4.0:
        raise ValueError("Stairs must fit 8 m tile with approach and landing")
    if not 0 < s["commands"]["forward_min"] <= s["commands"]["forward_max"]:
        raise ValueError("Require positive forward command range")
    commands = s["commands"]
    if not 0 < commands["start_forward_min"] <= commands["start_forward_max"]:
        raise ValueError("Require positive initial forward command range")
    if type(commands["speed_ramp_updates"]) is not int or commands["speed_ramp_updates"] < 0:
        raise ValueError("commands.speed_ramp_updates must be a nonnegative integer")
    if s["episode_length_s"] <= 0:
        raise ValueError("episode_length_s must be positive")
    for key in ("stiffness", "damping", "action_scale", "action_clip"):
        if s["control"][key] <= 0:
            raise ValueError("control." + key + " must be positive")
    if s["targets"]["base_height"] <= 0 or s["targets"]["foot_clearance"] < 0:
        raise ValueError("targets.base_height must be positive and foot_clearance nonnegative")
    randomization = s["randomization"]
    if not 0 <= randomization["friction_min"] <= randomization["friction_max"]:
        raise ValueError("Require 0 <= randomization.friction_min <= friction_max")
    if randomization["payload_min"] > randomization["payload_max"]:
        raise ValueError("randomization.payload_min must not exceed payload_max")
    if s["training"]["learning_rate"] <= 0:
        raise ValueError("training.learning_rate must be positive")
    if s["training"]["entropy_coef"] < 0:
        raise ValueError("training.entropy_coef must be nonnegative")
    train = s["training"]
    if not 0 < train["min_noise_std"] <= train["init_noise_std"]:
        raise ValueError("Require 0 < min_noise_std <= init_noise_std")
    if train["max_noise_std"] < 0 or (train["max_noise_std"] > 0 and train["max_noise_std"] < train["init_noise_std"]):
        raise ValueError("max_noise_std must be 0 (unbounded) or >= init_noise_std")
    if not 0 < train["min_learning_rate"] <= train["learning_rate"] <= train["max_learning_rate"]:
        raise ValueError("Require min_learning_rate <= learning_rate <= max_learning_rate")
    if train["action_mean_penalty_coef"] < 0:
        raise ValueError("action_mean_penalty_coef must be nonnegative")
    if not 0 <= s["targets"]["air_time"] < s["targets"]["max_air_time"]:
        raise ValueError("Require 0 <= targets.air_time < targets.max_air_time")
    if not 0 < s["targets"]["hip_abduction_max"] < 1.0472:
        raise ValueError("targets.hip_abduction_max must lie inside Go2's physical hip limit")
    if not 0 < s["targets"]["feet_lateral_min"] < s["targets"]["feet_lateral_max"]:
        raise ValueError("Require 0 < targets.feet_lateral_min < targets.feet_lateral_max")
    if s["targets"]["feet_lateral_inner_weight"] <= 0:
        raise ValueError("targets.feet_lateral_inner_weight must be positive")
    torque = s["torque"]
    if not 0 < torque["soft_ratio"] < 1:
        raise ValueError("torque.soft_ratio must be in (0, 1)")
    if not 0 < torque["startup_seconds"] <= s["episode_length_s"]:
        raise ValueError("torque.startup_seconds must be positive and within episode duration")
    if torque["startup_multiplier"] < 1:
        raise ValueError("torque.startup_multiplier must be >= 1")
    if not torque["enabled"] and any(s["rewards"][k] != 0 for k in ("torque_margin", "torque_excess")):
        raise ValueError("Torque penalties require torque.enabled=true")
    joint = s["joint_constraints"]
    # Keep experiment overrides at or below the existing real-controller guards.
    if not 0 < joint["velocity_limit"] <= 13.0:
        raise ValueError("joint_constraints.velocity_limit must be in (0, 13] rad/s")
    if not 0 < joint["tracking_error_limit"] <= 0.9:
        raise ValueError("joint_constraints.tracking_error_limit must be in (0, 0.9] rad")
    for key in ("velocity_soft_ratio", "position_soft_ratio", "tracking_soft_ratio"):
        if not 0 < joint[key] < 1:
            raise ValueError("joint_constraints." + key + " must be in (0, 1)")
    if not 0 < joint["startup_seconds"] <= s["episode_length_s"] or joint["startup_multiplier"] < 1:
        raise ValueError("Invalid joint_constraints startup duration/multiplier")
    joint_penalties = ("velocity_margin", "velocity_excess", "position_margin", "position_excess",
                       "target_position_excess", "tracking_error")
    if not joint["enabled"] and any(s["rewards"][k] != 0 for k in joint_penalties):
        raise ValueError("Joint constraint penalties require joint_constraints.enabled=true")
    for name in ("torque_margin", "torque_excess", "hip_abduction", "feet_lateral", "roll_rate", "lateral_velocity", *joint_penalties):
        if s["rewards"][name] > 0:
            raise ValueError("rewards." + name + " is a penalty and must be nonpositive")
    guard = s["guard_balance"]
    if type(guard["ramp_updates"]) is not int or guard["ramp_updates"] < 0:
        raise ValueError("guard_balance.ramp_updates must be a nonnegative integer")
    if not 0 < guard["initial_scale"] <= 1:
        raise ValueError("guard_balance.initial_scale must be in (0, 1]")
    for kind in ("velocity", "torque", "tracking"):
        if not 0 < guard[kind + "_soft_ratio"] < 1:
            raise ValueError("guard_balance soft ratios must be in (0, 1)")
    for key in ("position_overshoot_scale_rad", "target_position_overshoot_scale_rad"):
        if guard[key] <= 0:
            raise ValueError("guard_balance position scales must be positive")
    budgets = [guard["budget_" + kind] for kind in ("velocity", "torque", "tracking", "target_position", "position")]
    if any(budget <= 0 for budget in budgets) or sum(budgets) > 3.0 + 1e-9:
        raise ValueError("Guard budgets must be positive and total at most 3.0 per second")
    guard_weight = s["rewards"]["guard_balance"]
    if guard_weight not in (0.0, -1.0) or guard["enabled"] != (guard_weight == -1.0):
        raise ValueError("guard_balance.enabled requires rewards.guard_balance=-1; disabled requires 0")
    if guard["enabled"]:
        if not torque["enabled"] or not joint["enabled"] or joint["terminate_on_violation"]:
            raise ValueError("Balanced guard reward requires both audits and no joint-violation termination")
        duplicate = ("torque_margin", "torque_excess", "joint_limits", "joint_vel", *joint_penalties)
        if any(s["rewards"][key] != 0 for key in duplicate):
            raise ValueError("Balanced guard reward cannot be stacked with duplicate limit penalties")
    for key in ("num_steps_per_env", "save_interval", "num_learning_epochs", "num_mini_batches"):
        if type(s["training"][key]) is not int or s["training"][key] < 1:
            raise ValueError("training." + key + " must be positive integer")
    if s["num_envs"] * s["training"]["num_steps_per_env"] < s["training"]["num_mini_batches"]:
        raise ValueError("Not enough rollout samples for PPO minibatches")
