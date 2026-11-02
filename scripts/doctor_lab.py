#!/usr/bin/env python3
"""Check the actual training interpreter without launching Isaac Sim."""
import importlib.metadata as metadata
import json
import platform
import sys
import _bootstrap
from go2_him.settings import ROOT


def main():
    report = {"python":sys.version, "platform":platform.platform(), "packages":{}, "errors":[]}
    for name in ("isaacsim", "isaaclab", "torch", "numpy", "trimesh", "gymnasium", "tensorboard", "PyYAML"):
        try: report["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError: report["errors"].append("Missing package " + name)
    if sys.version_info[:2] != (3, 11): report["errors"].append("Use Python 3.11 for Isaac Sim 5.1")
    for name, prefix in (("isaacsim", "5.1."), ("isaaclab", "2.3.")):
        if not report["packages"].get(name, "").startswith(prefix): report["errors"].append("Expected " + name + " " + prefix + "x")
    try:
        import torch
        report["cuda_available"] = torch.cuda.is_available()
        report["torch_cuda"] = torch.version.cuda
        if not torch.cuda.is_available(): report["errors"].append("CUDA unavailable")
        else:
            report["gpu"] = torch.cuda.get_device_name(0)
            report["vram_gib"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
            sample = torch.randn(128,128,device="cuda")
            report["cuda_matmul_finite"] = bool(torch.isfinite(sample @ sample).all())
    except Exception as exc: report["errors"].append(str(exc))
    for name in ("assets/go2/urdf/go2.urdf", "upstream/himloco_lab/source/himloco_lab/himloco_lab/rsl_rl/algorithms/him_ppo.py"):
        if not (ROOT / name).is_file(): report["errors"].append("Missing file " + name)
    print(json.dumps(report, indent=2))
    return bool(report["errors"])


if __name__ == "__main__": sys.exit(main())
