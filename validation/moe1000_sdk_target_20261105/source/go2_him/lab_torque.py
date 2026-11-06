"""Isaac Lab torque instrumentation. Import only after AppLauncher."""
import torch
from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.utils import configclass
from himloco_lab.envs import HimlocoManagerBasedRLEnv
from himloco_lab.rsl_rl.runners.him_on_policy_runner import HIMOnPolicyRunner
from go2_him.torque import TorqueWindow
from go2_him.joint_constraints import JointConstraintWindow


class MonitoredPDActuator(IdealPDActuator):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.torque_window = TorqueWindow(self.stiffness.shape[0], self.joint_names,
                                         self.stiffness.device, cfg.torque_soft_ratio)
        self.joint_window = None
        if cfg.joint_constraints_enabled:
            self.joint_window = JointConstraintWindow(
                self.stiffness.shape[0], self.joint_names, self.stiffness.device,
                velocity_limit=cfg.joint_velocity_limit,
                velocity_soft_ratio=cfg.joint_velocity_soft_ratio,
                position_soft_ratio=cfg.joint_position_soft_ratio,
                tracking_error_limit=cfg.joint_tracking_error_limit,
                tracking_soft_ratio=cfg.joint_tracking_soft_ratio)

    def compute(self, control_action, joint_pos, joint_vel):
        if self.joint_window is not None:
            # Audit the actual PD target before physics, without silently
            # projecting it into a different policy/deployment action contract.
            self.joint_window.record_command(control_action.joint_positions, joint_pos)
        result = super().compute(control_action, joint_pos, joint_vel)
        # Keep raw computed_effort intact for penalties; additionally cap output
        # at the URDF bounds even if a hybrid controller changes actuator limits.
        if not bool(torch.isfinite(self.computed_effort).all()):
            raise FloatingPointError("Nonfinite requested PD torque; simulation stopped")
        hard = self.torque_window.limits
        self.applied_effort = self.applied_effort.clamp(-hard, hard)
        if not bool(torch.isfinite(self.applied_effort).all()):
            raise FloatingPointError("Nonfinite applied PD torque; simulation stopped")
        result.joint_efforts = self.applied_effort
        self.torque_window.record(self.computed_effort, self.applied_effort)
        return result


@configclass
class MonitoredPDActuatorCfg(IdealPDActuatorCfg):
    class_type = MonitoredPDActuator
    torque_soft_ratio: float = .8
    joint_constraints_enabled: bool = False
    joint_velocity_limit: float = 13.0
    joint_velocity_soft_ratio: float = .8
    joint_position_soft_ratio: float = .9
    joint_tracking_error_limit: float = .9
    joint_tracking_soft_ratio: float = .8


class TorqueHimlocoEnv(HimlocoManagerBasedRLEnv):
    def _post_physics_substep(self):
        robot = self.scene["robot"]
        actuator = robot.actuators["legs"]
        joint_window = getattr(actuator, "joint_window", None)
        if joint_window is not None:
            if not hasattr(self, "_joint_audit_ids"):
                self._joint_audit_ids = [robot.joint_names.index(n) for n in actuator.joint_names]
            # Runs after scene.update at *every* 5 ms integration, including
            # the last substep of a terminal interval, before automatic reset.
            joint_window.record_state(robot.data.joint_pos[:, self._joint_audit_ids],
                                      robot.data.joint_vel[:, self._joint_audit_ids])

    def step(self, action):
        # A zero-weight/disabled term must never leak a previous step's costs.
        self._guard_balance_result = None
        window = getattr(self.scene["robot"].actuators["legs"], "torque_window", None)
        if window is not None:
            startup_s = self.cfg.go2_settings["torque"]["startup_seconds"]
            window.begin(self.cfg.decimation, self.episode_length_buf * self.step_dt < startup_s)
        joint_window = getattr(self.scene["robot"].actuators["legs"], "joint_window", None)
        if joint_window is not None:
            startup_s = self.cfg.go2_settings["joint_constraints"]["startup_seconds"]
            joint_window.begin(self.cfg.decimation, self.episode_length_buf * self.step_dt < startup_s)
        result = super().step(action)
        if window is not None:
            # reset() and the extra post-reset actuator write cannot clear the
            # frozen window. Log every interval even when no episode finishes.
            self.extras["log"] = {**self.extras.get("log", {}), **window.log_metrics()}
        if joint_window is not None:
            self.extras["log"] = {**self.extras.get("log", {}), **joint_window.log_metrics()}
        if self._guard_balance_result is not None:
            self.extras["log"] = {**self.extras.get("log", {}), **self._guard_balance_result.log_metrics()}
        return result


class TorqueHIMRunner(HIMOnPolicyRunner):
    def log(self, locs, width=80, pad=35):
        # Parent averages all metrics. Peak torque must use MAX over rollout
        # steps, or a brief startup spike would be hidden in an average.
        peak_keys = {k for row in locs['ep_infos'] for k in row
                     if k.startswith(('Torque/', 'JointLimits/')) and 'peak' in k}
        filtered = dict(locs)
        filtered['ep_infos'] = [{k:v for k,v in row.items() if k not in peak_keys} for row in locs['ep_infos']]
        super().log(filtered, width=width, pad=pad)
        for key in sorted(peak_keys):
            values = [torch.as_tensor(row[key], device=self.device).reshape(-1) for row in locs['ep_infos'] if key in row]
            value = torch.cat(values).amax()
            self.writer.add_scalar(key, value, locs['it'])
            if key == 'Torque/requested_peak_ratio':
                print(f"Maximum requested torque / limit this update: {float(value):.4f}", flush=True)
            elif key == 'JointLimits/velocity_peak_ratio':
                print(f"Maximum measured joint speed / limit this update: {float(value):.4f}", flush=True)
