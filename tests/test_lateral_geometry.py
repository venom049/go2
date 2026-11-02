"""Body-frame foot geometry checks, independent of Isaac Sim."""

import math
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required; use the training environment Python")
class LateralGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from go2_him.lateral_geometry import signed_foot_lateral_positions
        cls.calculate = staticmethod(signed_foot_lateral_positions)

    def feet(self):
        return torch.tensor([[[.3, .15, -.25], [.3, -.18, -.30], [-.3, .16, -.20], [-.3, -.20, -.35]]], dtype=torch.float64)

    def test_signed_distances_and_left_right_mirror(self):
        feet = self.feet()
        root = torch.zeros(1, 3, dtype=feet.dtype)
        quat = torch.tensor([[1., 0., 0., 0.]], dtype=feet.dtype)
        expected = torch.tensor([[.15, .18, .16, .20]], dtype=feet.dtype)
        torch.testing.assert_close(self.calculate(root, quat, feet), expected)
        reflected = feet[:, [1, 0, 3, 2], :].clone()
        reflected[:, :, 1] *= -1
        torch.testing.assert_close(self.calculate(root, quat, reflected), expected[:, [1, 0, 3, 2]])

    def test_full_yaw_pitch_roll_and_translation_are_removed(self):
        feet = self.feet()
        yaw, pitch, roll = .93, -.61, .23
        c, s = math.cos, math.sin
        rz = torch.tensor([[c(yaw), -s(yaw), 0.], [s(yaw), c(yaw), 0.], [0., 0., 1.]], dtype=feet.dtype)
        ry = torch.tensor([[c(pitch), 0., s(pitch)], [0., 1., 0.], [-s(pitch), 0., c(pitch)]], dtype=feet.dtype)
        rx = torch.tensor([[1., 0., 0.], [0., c(roll), -s(roll)], [0., s(roll), c(roll)]], dtype=feet.dtype)
        rotation = rz @ ry @ rx
        cy, sy, cp, sp, cr, sr = c(yaw / 2), s(yaw / 2), c(pitch / 2), s(pitch / 2), c(roll / 2), s(roll / 2)
        quat = torch.tensor([[cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
                              cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy]], dtype=feet.dtype)
        root = torch.tensor([[3.2, -8.7, 1.4]], dtype=feet.dtype)
        world = feet @ rotation.T + root[:, None, :]
        torch.testing.assert_close(self.calculate(root, quat, world), torch.tensor([[.15, .18, .16, .20]], dtype=feet.dtype))
        # q and -q represent exactly the same orientation.
        torch.testing.assert_close(self.calculate(root, -quat, world), self.calculate(root, quat, world))

    def test_shape_and_static_sign_validation(self):
        feet = self.feet()
        root = torch.zeros(1, 3, dtype=feet.dtype)
        quat = torch.tensor([[1., 0., 0., 0.]], dtype=feet.dtype)
        bad_inputs = [(root[0], quat, feet), (root, quat[:, :3], feet), (root, quat, feet[:, :3]),
                      (root.repeat(2, 1), quat, feet)]
        for args in bad_inputs:
            with self.subTest(shapes=[arg.shape for arg in args]), self.assertRaises(ValueError):
                self.calculate(*args)
        for signs in ((1, -1), (1, 0, 1, -1)):
            with self.assertRaises(ValueError):
                self.calculate(root, quat, feet, signs)
        with self.assertRaises(TypeError):
            self.calculate(root, quat, feet, torch.tensor([1, -1, 1, -1]))


if __name__ == "__main__":
    unittest.main()
