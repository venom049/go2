"""CPU-only checks of the actual converted model, not simulator smoke tests."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "assets" / "go2"
spec = importlib.util.spec_from_file_location("prepare_go2_asset", ROOT / "scripts" / "prepare_go2_asset.py")
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


def urdf_inertia(link):
    element = link.find("inertial")
    mass = float(element.find("mass").get("value"))
    position = converter.numbers(element.find("origin").get("xyz"))
    values = {key: float(value) for key, value in element.find("inertia").attrib.items()}
    tensor = [[values["ixx"], values["ixy"], values["ixz"]],
              [values["ixy"], values["iyy"], values["iyz"]],
              [values["ixz"], values["iyz"], values["izz"]]]
    return mass, position, tensor


class Go2AssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = ET.parse(ASSET / "urdf" / "go2.urdf").getroot()
        cls.source = ET.parse(ASSET / "mjcf" / "go2.xml").getroot()
        cls.links = {link.get("name"): link for link in cls.robot.findall("link")}

    def test_joint_names_axes_ranges_and_effort_match_source(self):
        source_defaults = converter.defaults_for(self.source)
        joints = {j.get("name"): j for j in self.robot.findall("joint") if j.get("type") == "revolute"}
        expected = {f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR")
                    for joint in ("hip", "thigh", "calf")}
        self.assertEqual(set(joints), expected)
        for source_joint in self.source.findall(".//worldbody//joint"):
            attributes = converter.attributes(source_joint, source_defaults)
            joint = joints[attributes["name"]]
            limits = joint.find("limit")
            self.assertEqual(converter.numbers(joint.find("axis").get("xyz")), converter.numbers(attributes["axis"]))
            self.assertEqual([float(limits.get(k)) for k in ("lower", "upper")], converter.numbers(attributes["range"]))
            self.assertEqual(float(limits.get("effort")), 45.43 if "calf" in joint.get("name") else 23.7)

    def test_each_limb_transform_and_floating_root(self):
        joints = {joint.find("child").get("link"): joint for joint in self.robot.findall("joint")}
        self.assertNotIn("base_link", joints)
        for body in self.source.findall(".//worldbody//body"):
            if body.get("name") == "base_link":
                continue
            joint = joints[body.get("name")]
            self.assertEqual(converter.numbers(joint.find("origin").get("xyz")), converter.numbers(body.get("pos")))

    def test_total_mass_and_positive_inertia(self):
        self.assertEqual(len(self.links), 17)
        self.assertAlmostEqual(sum(urdf_inertia(link)[0] for link in self.links.values()), 15.206408, places=10)
        for name, link in self.links.items():
            mass, _, i = urdf_inertia(link)
            self.assertGreater(mass, 0, name)
            # Sylvester criterion checks positive definite symmetric inertia.
            self.assertGreater(i[0][0], 0, name)
            self.assertGreater(i[0][0] * i[1][1] - i[0][1] ** 2, 0, name)
            det = (i[0][0] * (i[1][1] * i[2][2] - i[1][2] ** 2)
                   - i[0][1] * (i[0][1] * i[2][2] - i[1][2] * i[0][2])
                   + i[0][2] * (i[0][1] * i[1][2] - i[1][1] * i[0][2]))
            self.assertGreater(det, 0, name)

    def test_rigid_body_mass_center_and_inertia_conserved(self):
        for body in self.source.findall(".//worldbody//body"):
            source = body.find("inertial")
            if source is None:
                continue
            name = body.get("name")
            expected_mass, expected_com, expected_tensor = converter.source_inertia(source)
            mass, com, tensor = urdf_inertia(self.links[name])
            components = [(mass, com, tensor)]
            if name.endswith("_calf"):
                foot = body.find("body")
                fm, fc, fi = urdf_inertia(self.links[foot.get("name")])
                fc = [a + b for a, b in zip(fc, converter.numbers(foot.get("pos")))]
                components.append((fm, fc, fi))
            total = sum(component[0] for component in components)
            self.assertAlmostEqual(total, expected_mass, places=12)
            combined_com = [sum(m * c[k] for m, c, _ in components) / total for k in range(3)]
            for a, b in zip(combined_com, expected_com):
                self.assertAlmostEqual(a, b, places=12, msg=name)
            for row in range(3):
                for col in range(3):
                    combined = sum(i[row][col] + m * converter.parallel_axis(c)[row][col]
                                   for m, c, i in components)
                    expected = expected_tensor[row][col] + expected_mass * converter.parallel_axis(expected_com)[row][col]
                    self.assertAlmostEqual(combined, expected, places=12, msg=name)

    def test_collision_count_and_isolated_feet(self):
        self.assertEqual(len(self.robot.findall(".//collision")), 23)
        self.assertEqual(len(self.robot.findall(".//visual")), 33)
        for leg in ("FL", "FR", "RL", "RR"):
            foot = self.links[leg + "_foot"]
            collisions = foot.findall("collision")
            self.assertEqual(len(collisions), 1)
            self.assertEqual(float(collisions[0].find("geometry/sphere").get("radius")), 0.022)
            self.assertEqual(converter.numbers(collisions[0].find("origin").get("xyz")), [-0.002, 0, 0])
            self.assertEqual(len(self.links[leg + "_calf"].findall("collision")), 2)

    def test_meshes_exist_and_are_original(self):
        provenance = json.loads((ASSET / "PROVENANCE.json").read_text())
        for mesh in self.robot.findall(".//mesh"):
            self.assertTrue((ASSET / "urdf" / mesh.get("filename")).resolve().is_file())
        for path, digest in provenance["source_sha256"].items():
            self.assertEqual(hashlib.sha256((ASSET / path).read_bytes()).hexdigest(), digest)
        self.assertEqual(hashlib.sha256((ASSET / "urdf" / "go2.urdf").read_bytes()).hexdigest(), provenance["urdf_sha256"])
        self.assertEqual(len(provenance["source_sha256"]), 17)

    def test_offline_rebuild_is_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            generated = converter.prepare(ASSET / "mjcf", Path(directory), ASSET / "LICENSE")
            self.assertEqual(generated.read_bytes(), (ASSET / "urdf" / "go2.urdf").read_bytes())


if __name__ == "__main__":
    unittest.main()
