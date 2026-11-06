"""Ideal, inelastic joint stops for the offline MuJoCo Euler integrator.

This is our simulation model, NOT a Unitree SDK/firmware implementation.
MuJoCo first predicts one step. A mass-metric velocity QP bounds displacement,
then the ENTIRE configuration is re-integrated from the previous state. A
second mass-metric impact QP removes outward velocity at reached stops. This
includes floating-base coupling; it is not independent qpos clipping.
The predictor can violate limits and is audited separately. Only validated
endpoints are exposed to the policy. Contacts remain MuJoCo's soft model;
operator splitting is an approximation, not a calibrated mechanical stop.
"""
from __future__ import annotations
import numpy as np

MODES = ('hard', 'native')
ROUNDING_MARGIN = 1e-10


def position_constraint_contract(mode='hard'):
    if mode not in MODES:
        raise ValueError('Unknown actual-position constraint mode')
    if mode == 'native':
        return {'mode': mode, 'kind': 'original MJCF soft limits', 'hard_position_guarantee': False}
    from go2_him.official_guard import load_official_profile
    p = load_official_profile()
    return {'mode': mode, 'version': 1, 'kind': 'mass-metric displacement QP and inelastic boundary impact',
            'joint_names': list(p.names), 'lower_rad': p.lower.tolist(), 'upper_rad': p.upper.tolist(),
            'profile_sha256': p.source_sha256, 'restitution': 0., 'rounding_margin_rad': ROUNDING_MARGIN,
            'integrator': 'Euler only', 'scope': 'Every accepted 2 ms endpoint and its linear hinge path, including settling',
            'failure': 'restore prior integration state and raise; never silently accept invalid q',
            'predictor_audit': True, 'motor_effort_and_stop_impulse_separate': True,
            'physical_fidelity': 'Ideal joint stops; split from soft contact solve; not calibrated hardware or firmware'}


def mass_project_velocity(mass, velocity, indices, lower, upper):
    """min .5*(v-v0)' M (v-v0), lower <= v[indices] <= upper.

    Primal active-set box QP on the Schur complement. Raises on failure;
    never substitutes componentwise velocity clipping for coupled dynamics.
    Returns v, generalized impulse M(v-v0).
    """
    mass, velocity = np.asarray(mass, float), np.asarray(velocity, float)
    indices = np.asarray(indices, int)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if (mass.shape != (velocity.size, velocity.size) or lower.shape != indices.shape
            or upper.shape != indices.shape or len(set(indices)) != len(indices)
            or np.any(indices < 0) or np.any(indices >= velocity.size)
            or np.any(lower > upper) or not np.isfinite(mass).all()
            or not np.isfinite(velocity).all() or np.isnan(lower).any() or np.isnan(upper).any()):
        raise ValueError('Invalid mass projection inputs')
    if np.all(velocity[indices] >= lower) and np.all(velocity[indices] <= upper):
        return velocity.copy(), np.zeros_like(velocity)
    np.linalg.cholesky(mass)  # Reject non-positive inertia, rather than regularizing silently.
    columns = np.linalg.solve(mass, np.eye(velocity.size)[:, indices])
    schur = columns[indices]
    hessian = np.linalg.inv(schur)
    center = velocity[indices]
    x = np.clip(center, lower, upper)
    active = np.where(x == lower, -1, np.where(x == upper, 1, 0))
    equal = lower == upper
    for _ in range(256):
        free, fixed = np.flatnonzero(active == 0), np.flatnonzero(active != 0)
        proposal = x.copy()
        if free.size:
            proposal[free] = center[free] - np.linalg.solve(hessian[np.ix_(free, free)],
                hessian[np.ix_(free, fixed)] @ (x[fixed] - center[fixed]))
        direction = proposal - x
        alpha, hit, side = 1., None, 0
        for i in free:
            if proposal[i] < lower[i]:
                a = (lower[i] - x[i]) / direction[i]
                if a < alpha: alpha, hit, side = a, i, -1
            elif proposal[i] > upper[i]:
                a = (upper[i] - x[i]) / direction[i]
                if a < alpha: alpha, hit, side = a, i, 1
        x += alpha * direction
        if hit is not None:
            x[hit] = lower[hit] if side == -1 else upper[hit]
            active[hit] = side
            continue
        gradient = hessian @ (x - center)
        wrong = np.where((active == -1) & ~equal, -gradient,
                         np.where((active == 1) & ~equal, gradient, 0.))
        if wrong.max(initial=0.) > 1e-10:
            active[int(np.argmax(wrong))] = 0
            continue
        # Active coordinates are explicit QP equalities; assign them exactly to
        # avoid roundoff in the dense solve, then re-solve the free velocities.
        impulse_j = np.linalg.solve(schur, x - center)
        result = velocity + columns @ impulse_j
        result[indices] = x
        if np.any(x < lower - 1e-10) or np.any(x > upper + 1e-10):
            raise FloatingPointError('Hard-limit QP infeasible result')
        return result, mass @ (result - velocity)
    raise FloatingPointError('Hard-limit QP did not converge')


class HardJointLimits:
    def __init__(self, mj, model, qadr, vadr, jids, profile):
        if model.opt.integrator != mj.mjtIntegrator.mjINT_EULER:
            raise ValueError('Hard joint stops require Euler integration')
        if np.any(model.jnt_type[jids] != mj.mjtJoint.mjJNT_HINGE):
            raise ValueError('Hard limits require named hinge joints')
        names=tuple(mj.mj_id2name(model,mj.mjtObj.mjOBJ_JOINT,int(j)).removesuffix('_joint') for j in jids)
        if (names != profile.names or not np.array_equal(model.jnt_qposadr[jids],qadr)
                or not np.array_equal(model.jnt_dofadr[jids],vadr)):
            raise ValueError('Hard-limit names or state addresses do not match the profile')
        self.mj, self.model = mj, model
        self.qadr, self.vadr = np.asarray(qadr), np.asarray(vadr)
        self.profile = profile
        self.lower, self.upper = profile.lower, profile.upper
        self.original_ranges = model.jnt_range[jids].copy()
        # Replace the old inconsistent soft ranges in this model instance only.
        model.jnt_range[jids, 0] = self.lower
        model.jnt_range[jids, 1] = self.upper
        model.jnt_limited[jids] = 1
        self.signature = mj.mjtState.mjSTATE_INTEGRATION
        self.state = np.empty(mj.mj_stateSize(model, self.signature))
        self.mass = np.empty((model.nv, model.nv))
        self.reset_statistics()

    def reset_statistics(self):
        self.steps = self.interventions = self.predictor_count = self.endpoint_count = 0
        self.predictor_max = self.endpoint_max = self.impulse_max = self.impulse_l1 = 0.
        self.energy_removed = 0.

    def summary(self):
        return {'mode': 'hard', 'physics_steps': self.steps, 'joint_endpoint_samples': self.steps * 12,
                'intervention_steps': self.interventions,
                'predictor_excess_count': self.predictor_count, 'predictor_max_excess_rad': self.predictor_max,
                'accepted_excess_count': self.endpoint_count, 'accepted_max_excess_rad': self.endpoint_max,
                'stop_generalized_impulse_max_abs': self.impulse_max,
                'stop_generalized_impulse_sum_abs': self.impulse_l1,
                'impulse_units': 'base translation N*s, all rotations N*m*s; not motor torque',
                'kinetic_energy_removed_j': self.energy_removed,
                'scope': 'one accepted endpoint per physics step; predictor diagnostics retained separately'}

    def check_position(self, q):
        if not np.isfinite(q).all() or np.any(q < self.lower) or np.any(q > self.upper):
            raise FloatingPointError('Actual joint position outside hard bounds')

    def step(self, data):
        mj, model = self.mj, self.model
        q0 = data.qpos.copy()
        self.check_position(q0[self.qadr])
        if not np.isfinite(data.qvel).all() or not np.isfinite(data.ctrl).all():
            raise FloatingPointError('Nonfinite hard-step input')
        mj.mj_getState(model, data, self.state, self.signature)
        oldtime = float(data.time)
        warnings = np.array([w.number for w in data.warning])
        dt = float(model.opt.timestep)
        if not np.isfinite(dt) or dt <= 0 or model.opt.integrator != mj.mjtIntegrator.mjINT_EULER:
            raise ValueError('Hard step requires a positive timestep and Euler integration')
        try:
            mj.mj_step(model, data)
            if (not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
                    or abs(data.time - oldtime - dt) > 1e-10
                    or np.any(np.array([w.number for w in data.warning]) > warnings)):
                raise FloatingPointError('Invalid MuJoCo hard-step predictor')
            raw_q = data.qpos[self.qadr].copy()
            excess = np.maximum(np.maximum(self.lower - raw_q, raw_q - self.upper), 0.)
            v0 = data.qvel.copy()
            near_lower, near_upper = raw_q <= self.lower + ROUNDING_MARGIN, raw_q >= self.upper - ROUNDING_MARGIN
            intervene = bool(np.any(excess) or np.any(near_lower & (v0[self.vadr] < 0))
                             or np.any(near_upper & (v0[self.vadr] > 0)))
            impulse = np.zeros(model.nv)
            energy_removed = 0.
            if intervene:
                # qM still belongs to the pre-integration configuration.
                mj.mj_fullM(model, self.mass, data.qM)
                old_q = q0[self.qadr]
                inner_lower = np.minimum(self.lower + ROUNDING_MARGIN, old_q)
                inner_upper = np.maximum(self.upper - ROUNDING_MARGIN, old_q)
                v_path, impulse_path = mass_project_velocity(self.mass, v0, self.vadr,
                    (inner_lower - old_q) / dt, (inner_upper - old_q) / dt)
                candidate = q0.copy()
                mj.mj_integratePos(model, candidate, v_path, dt)
                self.check_position(candidate[self.qadr])
                at_lower = candidate[self.qadr] <= self.lower + 2 * ROUNDING_MARGIN
                at_upper = candidate[self.qadr] >= self.upper - 2 * ROUNDING_MARGIN
                v_final, impulse_stop = mass_project_velocity(self.mass, v_path, self.vadr,
                    np.where(at_lower, 0., -np.inf), np.where(at_upper, 0., np.inf))
                if np.any(at_lower & (v_final[self.vadr] < -1e-10)) or np.any(at_upper & (v_final[self.vadr] > 1e-10)):
                    raise FloatingPointError('Outward boundary velocity after hard stop')
                energy_removed = float(.5 * (v0 @ self.mass @ v0 - v_final @ self.mass @ v_final))
                if energy_removed < -1e-8 * max(1., float(v0 @ self.mass @ v0)):
                    raise FloatingPointError('Hard stop added kinetic energy')
                data.qpos[:] = candidate
                data.qvel[:] = v_final
                data.qacc_warmstart[:] = 0.
                impulse = impulse_path + impulse_stop
            self.check_position(data.qpos[self.qadr])
            # The caller refreshes forward kinematics before observation/reward.
            # actuator_force remains the predictor's applied MOTOR force, not
            # the additional ideal-stop impulse (which is recorded separately).
        except BaseException:
            mj.mj_setState(model, data, self.state, self.signature)
            mj.mj_forward(model, data)
            raise
        self.steps += 1
        self.interventions += int(intervene)
        self.predictor_count += int(np.count_nonzero(excess))
        self.predictor_max = max(self.predictor_max, float(excess.max(initial=0.)))
        self.impulse_max = max(self.impulse_max, float(np.abs(impulse).max(initial=0.)))
        self.impulse_l1 += float(np.abs(impulse).sum())
        self.energy_removed += max(0., energy_removed)
