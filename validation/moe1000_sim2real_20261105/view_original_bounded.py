"""Visualize frozen RoboGauge weights with the audited SDK target projection."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "source"), str(ROOT / "source/scripts")]
import test_robogauge_stairs as original
import test_robogauge_two_steps_wall as wall
from go2_him.official_guard import load_official_profile
from go2_him.target_clip import transform_target_action

PROFILE = load_official_profile(joint_names=original.JOINTS)
STATS = {"control_steps": 0, "clipped_joint_targets": 0, "sent_target_violations": 0}
ORIGINAL_SHA = "bc08253fd68f661bf6be5ce5ca70cec555c3fde2c397d23fbe9814a998e056f5"


class BoundedPolicy(original.Policy):
    def __init__(self, np, ort, cfg, path):
        if original.digest(path) != ORIGINAL_SHA:
            raise ValueError("This viewer requires the unchanged original RoboGauge weights")
        super().__init__(np, ort, cfg, path)

    def act(self, values):
        super().act(values)
        result = transform_target_action(self.last_action, self.scale, self.offset,
                                         PROFILE.lower, PROFILE.upper)
        if self.np.any(result.sent_excess):
            raise RuntimeError("SDK target projection violated its bounds")
        self.last_action = result.sent_action.copy()
        STATS["control_steps"] += 1
        STATS["clipped_joint_targets"] += int(self.np.count_nonzero(result.raw_excess))
        return result.sent_target


write_json = original.write_json


def write_with_adapter_metadata(path, data):
    if Path(path).name == "manifest.json":
        data["observation_contract"] = "225 floats; term-major 5-frame history; inverse-mapped SDK-clipped last action"
        data["action_contract"] = "Unchanged original ONNX; scale/offset then SDK target hard clipping before PD"
        data["sdk_target_profile"] = PROFILE.metadata()
        data["target_adapter_sha256"] = original.digest(__file__)
    if Path(path).name == "summary.json":
        data["sdk_target_adapter"] = dict(STATS)
    write_json(path, data)


if __name__ == "__main__":
    original.Policy = BoundedPolicy
    original.write_json = write_with_adapter_metadata
    wall.main()
