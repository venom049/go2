#!/usr/bin/env python3
"""Prepare the exact Go2 model referenced by go2-stairs-sim2real for URDF import.

This intentionally does not import training code or load/download policy weights.
The converter is narrowly scoped to the pinned MJCF, not a general MJCF converter.
Only the Python standard library is required; the delivered model rebuilds offline.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_URL = "https://github.com/wxz-521/go2-stairs-sim2real"
UPSTREAM_COMMIT = "530a10fee4232317aef4343f2b85b6bad9cf7905"
MODEL_URL = "https://github.com/unitreerobotics/unitree_mujoco.git"
MODEL_COMMIT = "ae6a8403e272733e9996ef59990880330496177f"
MODEL_SUBDIR = "unitree_robots/go2"
MODEL_CONTENT_SHA256 = "f8ff4b63d3ccec08a4cc4627ac44d44eaa279fb3a0d55ef3756f4896a36127aa"
FOOT_MASS = 0.01
JOINT_VELOCITY_LIMIT = 30.0  # MJCF has no velocity limit; explicitly documented assumption.


def numbers(value):
    return [float(x) for x in value.split()]


def fmt(values):
    if isinstance(values, (float, int)):
        return format(values, ".14g")
    return " ".join(format(x, ".14g") for x in values)


def rotation(quaternion):
    """MJCF wxyz quaternion to a rotation matrix, including non-unit inputs."""
    w, x, y, z = quaternion
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = [v / norm for v in (w, x, y, z)]
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def quaternion_rpy(quaternion):
    r = rotation(quaternion)
    pitch = math.asin(max(-1.0, min(1.0, -r[2][0])))
    if abs(math.cos(pitch)) > 1e-9:
        return [math.atan2(r[2][1], r[2][2]), pitch, math.atan2(r[1][0], r[0][0])]
    return [math.atan2(-r[1][2], r[1][1]), pitch, 0.0]


def parallel_axis(position):
    norm = sum(x * x for x in position)
    return [[(norm if i == j else 0.0) - position[i] * position[j]
             for j in range(3)] for i in range(3)]


def source_inertia(inertial):
    mass = float(inertial.attrib["mass"])
    position = numbers(inertial.attrib["pos"])
    r = rotation(numbers(inertial.get("quat", "1 0 0 0")))
    diagonal = numbers(inertial.attrib["diaginertia"])
    tensor = [[sum(r[i][k] * diagonal[k] * r[j][k] for k in range(3))
               for j in range(3)] for i in range(3)]
    return mass, position, tensor


def split_foot_inertia(mass, position, tensor, foot_position, foot_radius):
    """Partition one rigid body without changing its combined inertia or COM."""
    remaining_mass = mass - FOOT_MASS
    remaining_position = [(mass * p - FOOT_MASS * f) / remaining_mass
                          for p, f in zip(position, foot_position)]
    foot_scalar = 0.4 * FOOT_MASS * foot_radius ** 2
    original_shift = parallel_axis(position)
    foot_shift = parallel_axis(foot_position)
    remaining_shift = parallel_axis(remaining_position)
    remaining_tensor = [
        [tensor[i][j] + mass * original_shift[i][j]
         - FOOT_MASS * foot_shift[i][j] - (foot_scalar if i == j else 0.0)
         - remaining_mass * remaining_shift[i][j]
         for j in range(3)] for i in range(3)
    ]
    foot_tensor = [[foot_scalar if i == j else 0.0 for j in range(3)] for i in range(3)]
    return (remaining_mass, remaining_position, remaining_tensor), foot_tensor


def add_inertial(link, mass, position, tensor):
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz=fmt(position), rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=fmt(mass))
    ET.SubElement(inertial, "inertia", **{
        "ixx": fmt(tensor[0][0]), "ixy": fmt(tensor[0][1]), "ixz": fmt(tensor[0][2]),
        "iyy": fmt(tensor[1][1]), "iyz": fmt(tensor[1][2]), "izz": fmt(tensor[2][2]),
    })


def defaults_for(root):
    defaults = {}

    def visit(node, inherited):
        current = copy.deepcopy(inherited)
        for child in node:
            if child.tag != "default":
                current.setdefault(child.tag, {}).update(child.attrib)
        defaults[node.get("class", "")] = current
        for child in node.findall("default"):
            visit(child, current)

    visit(root.find("default"), {})
    return defaults


def attributes(element, defaults, fallback="go2"):
    result = dict(defaults[element.get("class", fallback)].get(element.tag, {}))
    result.update(element.attrib)
    return result


def add_geometry(link, source_geom, defaults, meshes, offset=None):
    attr = attributes(source_geom, defaults)
    visual = attr.get("class") == "visual"
    tag = "visual" if visual else "collision"
    element = ET.SubElement(link, tag)
    if "name" in attr:
        element.set("name", attr["name"])
    position = numbers(attr.get("pos", "0 0 0"))
    if offset is not None:
        position = [a - b for a, b in zip(position, offset)]
    ET.SubElement(element, "origin", xyz=fmt(position),
                  rpy=fmt(quaternion_rpy(numbers(attr.get("quat", "1 0 0 0")))))
    geometry = ET.SubElement(element, "geometry")
    shape = attr.get("type", "sphere")
    size = numbers(attr.get("size", "0"))
    if shape == "mesh":
        ET.SubElement(geometry, "mesh", filename="../mjcf/assets/" + meshes[attr["mesh"]])
    elif shape == "box":
        ET.SubElement(geometry, "box", size=fmt([2 * x for x in size]))
    elif shape == "cylinder":
        ET.SubElement(geometry, "cylinder", radius=fmt(size[0]), length=fmt(2 * size[1]))
    elif shape == "sphere":
        ET.SubElement(geometry, "sphere", radius=fmt(size[0]))
    else:
        raise ValueError("Unsupported source geometry: " + shape)
    if visual and "material" in attr:
        ET.SubElement(element, "material", name=attr["material"])


def convert_model(source_xml):
    root = ET.parse(source_xml).getroot()
    defaults = defaults_for(root)
    meshes = {Path(mesh.attrib["file"]).stem: mesh.attrib["file"]
              for mesh in root.findall("asset/mesh")}
    motors = {motor.attrib["joint"]: attributes(motor, defaults)
              for motor in root.findall("actuator/motor")}
    robot = ET.Element("robot", name="go2")
    robot.append(ET.Comment("Generated by scripts/prepare_go2_asset.py; see ../PROVENANCE.json"))
    for material in root.findall("asset/material"):
        element = ET.SubElement(robot, "material", name=material.attrib["name"])
        ET.SubElement(element, "color", rgba=material.attrib["rgba"])

    def body_to_link(body, parent=None):
        name = body.attrib["name"]
        # The pinned model has body translations only. Refuse silent mis-conversions.
        if any(key in body.attrib for key in ("quat", "euler", "axisangle", "xyaxes", "zaxis")):
            raise ValueError("Unexpected body rotation in pinned model: " + name)
        link = ET.SubElement(robot, "link", name=name)
        foot_body = next((b for b in body.findall("body") if b.attrib["name"].endswith("_foot")), None)
        geoms = body.findall("geom")
        inertial = body.find("inertial")
        if inertial is None:
            raise ValueError("Unexpected inertial-free body: " + name)
        inertia = source_inertia(inertial)
        if foot_body is not None:
            foot_offset = numbers(foot_body.attrib["pos"])
            foot_geoms = [g for g in geoms if g.get("class") == "foot" or g.get("mesh") == "foot"]
            foot_collision = next(g for g in foot_geoms if g.get("class") == "foot")
            foot_attr = attributes(foot_collision, defaults)
            foot_center = numbers(foot_attr["pos"])
            inertia, foot_tensor = split_foot_inertia(*inertia, foot_center, float(foot_attr["size"]))
            foot_link = ET.SubElement(robot, "link", name=foot_body.attrib["name"])
            add_inertial(foot_link, FOOT_MASS,
                         [a - b for a, b in zip(foot_center, foot_offset)], foot_tensor)
            for geom in foot_geoms:
                add_geometry(foot_link, geom, defaults, meshes, offset=foot_offset)
            joint = ET.SubElement(robot, "joint", name=foot_body.attrib["name"] + "_joint", type="fixed")
            ET.SubElement(joint, "parent", link=name)
            ET.SubElement(joint, "child", link=foot_body.attrib["name"])
            ET.SubElement(joint, "origin", xyz=fmt(foot_offset), rpy="0 0 0")
            geoms = [g for g in geoms if g not in foot_geoms]
        add_inertial(link, *inertia)
        for geom in geoms:
            add_geometry(link, geom, defaults, meshes)
        if parent is not None:
            source_joint = body.find("joint")
            if source_joint is None:
                raise ValueError("Unexpected joint-free body: " + name)
            attr = attributes(source_joint, defaults)
            joint = ET.SubElement(robot, "joint", name=attr["name"], type="revolute")
            ET.SubElement(joint, "parent", link=parent)
            ET.SubElement(joint, "child", link=name)
            ET.SubElement(joint, "origin", xyz=body.get("pos", "0 0 0"), rpy="0 0 0")
            ET.SubElement(joint, "axis", xyz=attr["axis"])
            limits = numbers(attr["range"])
            effort = max(abs(x) for x in numbers(motors[attr["name"]]["ctrlrange"]))
            ET.SubElement(joint, "limit", lower=fmt(limits[0]), upper=fmt(limits[1]),
                          effort=fmt(effort), velocity=fmt(JOINT_VELOCITY_LIMIT))
            ET.SubElement(joint, "dynamics", damping=attr["damping"], friction=attr["frictionloss"])
        for child in body.findall("body"):
            if child is not foot_body:
                body_to_link(child, name)

    body_to_link(root.find("worldbody/body"))
    return robot, meshes


def indent(element, level=0):
    """ET.indent equivalent also usable with Isaac Gym's Python 3.8."""
    space = "\n" + level * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = space + "  "
        for child in element:
            indent(child, level + 1)
        child.tail = space
    if level and (not element.tail or not element.tail.strip()):
        element.tail = space


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source_dir, output_dir, license_path=None):
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    robot, meshes = convert_model(source_dir / "go2.xml")
    source_digest = hashlib.sha256()
    model_files = [source_dir / "go2.xml"]
    model_files.extend(source_dir / "assets" / name for name in meshes.values())
    for path in sorted(model_files):
        source_digest.update(str(path.relative_to(source_dir)).encode("utf-8"))
        source_digest.update(b"\0")
        source_digest.update(path.read_bytes())
        source_digest.update(b"\0")
    if source_digest.hexdigest() != MODEL_CONTENT_SHA256:
        raise ValueError("Source MJCF/meshes differ from the pinned Unitree model. "
                         "Use --fetch to recover the expected source.")
    mjcf_dir = output_dir / "mjcf"
    (mjcf_dir / "assets").mkdir(parents=True, exist_ok=True)
    (output_dir / "urdf").mkdir(parents=True, exist_ok=True)
    source_files = [(source_dir / "go2.xml", mjcf_dir / "go2.xml")]
    source_files.extend((source_dir / "assets" / name, mjcf_dir / "assets" / name)
                        for name in meshes.values())
    for source, destination in source_files:
        if source != destination:
            shutil.copyfile(str(source), str(destination))
    if license_path is not None:
        if Path(license_path).resolve() != output_dir / "LICENSE":
            shutil.copyfile(str(license_path), str(output_dir / "LICENSE"))
    if not (output_dir / "LICENSE").is_file():
        raise FileNotFoundError("Unitree LICENSE is required. Pass --license or use --fetch.")
    indent(robot)
    urdf_path = output_dir / "urdf" / "go2.urdf"
    ET.ElementTree(robot).write(str(urdf_path), encoding="utf-8", xml_declaration=True)
    manifest = {
        "requested_repository": UPSTREAM_URL,
        "requested_repository_commit": UPSTREAM_COMMIT,
        "model_dependency_declared_in": "mujoco_sim/setup.sh",
        "model_repository": MODEL_URL,
        "model_commit": MODEL_COMMIT,
        "model_subdirectory": MODEL_SUBDIR,
        "model_content_sha256": MODEL_CONTENT_SHA256,
        "total_mass_kg": 15.206408,
        "source_sha256": {str(dst.relative_to(output_dir)): sha256(dst)
                          for _, dst in sorted(source_files)},
        "urdf_sha256": sha256(urdf_path),
        "conversion": [
            "Original MJCF and all 16 OBJ meshes are copied byte-for-byte; no policy/checkpoint is used.",
            "MJCF quaternion-rotated diagonal inertias become full URDF inertia tensors in link axes.",
            "Primitive dimensions, collision poses, visual poses, joint axes/ranges and actuator effort limits are preserved.",
            "Each existing empty fixed foot body receives its calf's foot collision and visual geometry.",
            "Each foot is assigned 0.01 kg and solid-sphere inertia; calf mass, COM and inertia are adjusted to exactly preserve combined rigid-body inertia.",
            "MJCF base world translation (z=0.445), freejoint, sensors and home keyframe are omitted; the simulator supplies the floating root state.",
            "URDF joint velocity limit is 30 rad/s, an explicit training assumption absent from source MJCF.",
            "MJCF joint damping=0.1 and frictionloss=0.2 are copied to URDF dynamics; actuator config may override them.",
            "MJCF armature=0.01 is not representable in standard URDF; configure armature=0.01 in the simulator actuator.",
            "MuJoCo friction, contact dimensionality, solver and margin parameters have no exact URDF/PhysX equivalent; configure PhysX contact material and solver separately.",
            "Keep fixed foot bodies when importing (merge_fixed_joints=False) for isolated foot contact observations.",
        ],
    }
    (output_dir / "PROVENANCE.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return urdf_path


def fetch_and_prepare(output_dir):
    with tempfile.TemporaryDirectory(prefix="go2-model-") as temporary:
        checkout = Path(temporary) / "unitree_mujoco"
        commands = [
            ["git", "clone", "--filter=blob:none", "--no-checkout", MODEL_URL, str(checkout)],
            ["git", "-C", str(checkout), "sparse-checkout", "init", "--cone"],
            ["git", "-C", str(checkout), "sparse-checkout", "set", MODEL_SUBDIR],
            ["git", "-C", str(checkout), "checkout", "--detach", MODEL_COMMIT],
        ]
        for command in commands:
            subprocess.run(command, check=True)
        return prepare(checkout / MODEL_SUBDIR, output_dir, checkout / "LICENSE")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "assets" / "go2")
    parser.add_argument("--source-dir", type=Path, help="Directory containing original go2.xml and assets/")
    parser.add_argument("--license", type=Path, help="Original Unitree LICENSE when importing a local source")
    parser.add_argument("--fetch", action="store_true", help="Fetch pinned model dependency instead of rebuilding offline")
    args = parser.parse_args()
    if args.fetch and args.source_dir:
        parser.error("--fetch and --source-dir are mutually exclusive")
    if args.fetch:
        result = fetch_and_prepare(args.output_dir)
    else:
        source = args.source_dir or args.output_dir / "mjcf"
        if not (source / "go2.xml").is_file():
            parser.error("Model source is missing; pass --fetch or --source-dir.")
        result = prepare(source, args.output_dir, args.license)
    print("Prepared robot asset: " + str(result))


if __name__ == "__main__":
    main()
