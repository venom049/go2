"""Regression tests for substep spikes, reset boundaries and acceptance."""
import unittest
try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    from go2_him.torque import TorqueWindow, TorqueAudit, joint_effort_limits
from go2_him.settings import load_settings

NAMES = [f'{leg}_{kind}_joint' for leg in ('FL', 'FR', 'RL', 'RR') for kind in ('hip', 'thigh', 'calf')]

@unittest.skipIf(torch is None, "PyTorch required for torque tensor tests")
class TorqueTests(unittest.TestCase):
    def window(self, n=2, steps=4):
        w = TorqueWindow(n, NAMES)
        w.begin(steps, torch.tensor([True] + [False]*(n-1)))
        return w

    def test_first_substep_spike_survives_clipping_and_reset_write(self):
        w = self.window()
        raw = torch.zeros(2, 12); raw[0, 1] = -33.74
        w.record(raw, raw.clamp(-w.limits, w.limits))
        for _ in range(3): w.record(torch.zeros_like(raw), torch.zeros_like(raw))
        # Auto-reset writes must not overwrite the terminal interval.
        w.record(torch.full_like(raw, 999), torch.full_like(raw, 999))
        self.assertAlmostEqual(float(w.requested_peak[0, 1]), 33.74, places=4)
        self.assertGreater(float(w.penalty('excess', 2)[0]), 0)
        audit = TorqueAudit(w); audit.update(w, torch.tensor([True, False]))
        r = audit.report()
        self.assertFalse(r['torque_limits_passed'])
        self.assertEqual(r['requested_excess_joint_samples'], 1)
        self.assertEqual(r['applied_excess_joint_samples'], 0)
        self.assertEqual(r['startup_excess_joint_samples'], 1)
        self.assertEqual(r['physics_samples_per_joint'], 4)

    def test_soft_margin_sign_startup_and_limit_equality(self):
        w = self.window(2, 1)
        raw = torch.zeros(2, 12); raw[:,0] = .9*w.limits[0]; raw[:,1] = -w.limits[1]
        w.record(raw, raw)
        self.assertTrue(torch.allclose(w.penalty('margin', 2), torch.tensor([.1,.05]), atol=1e-6))
        self.assertEqual(float(w.penalty('excess').sum()), 0)
        a = TorqueAudit(w); a.update(w, torch.ones(2, dtype=torch.bool))
        self.assertTrue(a.report()['torque_limits_passed'])

    def test_joint_order_and_distinct_calf_limit(self):
        names = NAMES[::-1]; lim = joint_effort_limits(names)
        self.assertAlmostEqual(float(lim[0]),45.43,places=4)
        self.assertAlmostEqual(float(lim[1]),23.7,places=4)
        with self.assertRaises(ValueError): joint_effort_limits(['unknown']*12)

    def test_incomplete_or_nonfinite_cannot_pass(self):
        w = self.window()
        with self.assertRaises(RuntimeError): w.penalty('margin')
        with self.assertRaises(RuntimeError): TorqueAudit(w).update(w, torch.ones(2,dtype=torch.bool))
        raw = torch.zeros(2,12); raw[0,0] = float('nan')
        with self.assertRaises(FloatingPointError): w.record(raw, torch.zeros_like(raw))
        self.assertFalse(TorqueAudit(w).report()['torque_limits_passed'])

    def test_active_episode_mask_and_log_snapshots(self):
        w = self.window(2,1); raw = torch.zeros(2,12); raw[1,0] = 40
        w.record(raw, raw.clamp(-w.limits, w.limits))
        logs = w.log_metrics(); a = TorqueAudit(w)
        a.update(w, torch.tensor([True,False]))
        self.assertTrue(a.report()['torque_limits_passed'])
        w.begin(1, torch.zeros(2,dtype=torch.bool)); w.record(torch.zeros_like(raw),torch.zeros_like(raw))
        self.assertEqual(float(logs['Torque/requested_peak_nm/FL_hip_joint']),40)
        self.assertEqual(float(w.requested_peak.sum()),0)

    def test_applied_violation_is_independent_failure(self):
        w = self.window(1,1); raw = torch.zeros(1,12); applied=raw.clone(); applied[0,2]=-46
        w.record(raw,applied); a=TorqueAudit(w); a.update(w,torch.ones(1,dtype=torch.bool))
        self.assertFalse(a.report()['torque_limits_passed'])
        self.assertEqual(a.report()['requested_excess_joint_samples'],0)
        self.assertEqual(a.report()['applied_excess_joint_samples'],1)

    def test_limits_match_actual_urdf(self):
        from pathlib import Path
        import xml.etree.ElementTree as ET
        joints = {j.attrib['name']: float(j.find('limit').attrib['effort'])
                  for j in ET.parse(Path(__file__).resolve().parents[1] / 'assets/go2/urdf/go2.urdf').getroot().findall('joint')
                  if j.attrib.get('type') == 'revolute'}
        limits = joint_effort_limits(NAMES)
        for i, name in enumerate(NAMES): self.assertAlmostEqual(float(limits[i]), joints[name], places=4)

    def test_config_opt_in_and_rejections(self):
        self.assertFalse(load_settings('configs/stairs_stable_fast.yaml')['torque']['enabled'])
        self.assertTrue(load_settings('configs/stairs_torque_limited.yaml')['torque']['enabled'])
        for patch in ('torque.enabled=1','torque.soft_ratio=1','torque.soft_ratio=0',
                      'torque.startup_multiplier=0.5','torque.startup_seconds=0',
                      'rewards.torque_margin=1','rewards.torque_excess=-1'):
            with self.subTest(patch=patch), self.assertRaises(ValueError): load_settings(overrides=[patch])

if __name__ == '__main__': unittest.main()
