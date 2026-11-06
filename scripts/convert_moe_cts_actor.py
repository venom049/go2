#!/usr/bin/env python3
"""Reconstruct the pinned deployment actor and audit all computed ONNX nodes.

No training, robot connection, critic fabrication, or optimizer restoration.
The output is a fresh actor warm-start artifact, never an original CTS resume.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import _bootstrap
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import torch

from go2_him.moe_cts_actor import (CHECKPOINT_FORMAT, JOINT_ORDER, MoECTSActor,
                                  SOURCE_ONNX_SHA256, TERM_DIMS)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "upstream/robogauge/unitree_cpp_deploy/logs/rsl_rl/go2_moe_cts_expert_goal_137000_0.6745/exported/policy.onnx"


def reconstruct(source):
    raw = Path(source).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_ONNX_SHA256:
        raise ValueError("Source SHA256 differs from the audited RoboGauge best ONNX")
    graph = onnx.load_model_from_string(raw)
    onnx.checker.check_model(graph, full_check=True)
    if [(op.domain, op.version) for op in graph.opset_import] != [("", 11)]:
        raise ValueError("Expected the audited opset-11 graph")
    model = MoECTSActor()
    expected = model.state_dict()
    tensors = graph.graph.initializer
    names = [value.name for value in tensors]
    if len(tensors) != 22 or len(set(names)) != 22 or set(names) != set(expected):
        raise ValueError("Expected exactly the 22 actor/encoder tensors, with no extra or missing keys")
    converted = {}
    for value in tensors:
        array = numpy_helper.to_array(value)
        if array.dtype != np.float32 or tuple(array.shape) != tuple(expected[value.name].shape):
            raise ValueError(f"Wrong dtype or dimensions: {value.name}")
        if not np.isfinite(array).all():
            raise ValueError(f"Nonfinite source parameter: {value.name}")
        converted[value.name] = torch.from_numpy(array.copy())
    model.load_state_dict(converted, strict=True)
    return model.eval(), graph


def test_inputs(seed=42, samples=32):
    """Network-space inputs: normal motion, arbitrary values, and clip edges."""
    rng = np.random.default_rng(seed)
    yield "zero", np.zeros((1, 225), dtype=np.float32)
    standing = [np.zeros((5, d), dtype=np.float32) for d in TERM_DIMS]
    standing[1][:, 2] = -1.0
    standing[2][:, 0] = 1.2  # 0.6 m/s command, scale 2
    baseline = np.concatenate([x.reshape(-1) for x in standing])[None, :]
    yield "standing_forward", baseline
    yield "history_ramp", (np.arange(225, dtype=np.float32)[None, :] - 112) / 20
    for i in range(samples):
        terms = [x.copy() for x in standing]
        terms[0] = rng.normal(0, 0.25, (5, 3))
        gravity = rng.normal(0, 0.12, (5, 3)); gravity[:, 2] -= 1
        terms[1] = gravity / np.linalg.norm(gravity, axis=1, keepdims=True)
        terms[2][:, 0] = rng.uniform(-0.4, 1.6)
        terms[2][:, 1] = rng.uniform(-0.2, 0.2)
        terms[2][:, 2] = rng.uniform(-0.1, 0.1)
        terms[3] = rng.normal(0, 0.3, (5, 12))
        terms[4] = rng.uniform(-0.65, 0.65, (5, 12))  # ±13 rad/s at scale .05
        terms[5] = rng.normal(0, 1, (5, 12))
        yield f"normal_{i}", np.concatenate([x.reshape(-1) for x in terms])[None, :].astype(np.float32)
        yield f"random_{i}", rng.normal(0, 3, (1, 225)).astype(np.float32)
    # Observation clipping precedes scaling in deploy.yaml. This deliberately
    # tests extreme graph inputs; it does not claim these are physical states.
    limits = np.concatenate([np.tile(np.asarray(scale, dtype=np.float32), 5)
                             for scale in ([25]*3, [100]*3, [200, 200, 25],
                                           [100]*12, [5]*12, [100]*12)])[None, :]
    yield "clip_upper", limits
    yield "clip_lower", -limits
    yield "clip_alternating", limits * np.where(np.arange(225)[None, :] % 2, -1, 1).astype(np.float32)
    yield "tiny", np.full((1, 225), 1e-12, dtype=np.float32)
    for i in range(samples):
        yield f"boundary_{i}", (limits * rng.choice([-1, 1], size=(1, 225))).astype(np.float32)


def check_parity(model, graph, samples=32, atol=2e-4, rtol=2e-4,
                 intermediate_scale_rtol=1e-4, stress_action_scale_rtol=1e-5):
    inferred = onnx.shape_inference.infer_shapes(graph, strict_mode=True)
    values = {v.name: v for v in [*inferred.graph.input, *inferred.graph.value_info,
                                *inferred.graph.output]}
    computed = [output for node in inferred.graph.node if node.op_type != "Constant" for output in node.output]
    del inferred.graph.output[:]
    for name in computed:
        if name not in values:
            raise ValueError(f"Missing inferred type for intermediate {name}")
        inferred.graph.output.append(values[name])
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(inferred.SerializeToString(), opts, providers=["CPUExecutionProvider"])
    stats = {name: dict(max_absolute_error=0.0, max_tolerance_ratio=0.0,
                       max_error_over_tensor_peak=0.0) for name in computed}
    cases = []
    family_outputs = {}
    largest_action_error = 0.0
    with torch.no_grad():
        for name, inputs in test_inputs(samples=samples):
            family = ("clip_boundary" if name.startswith(("clip_", "boundary_")) else
                      "normal" if name.startswith("normal_") else
                      "random" if name.startswith("random_") else "deterministic")
            family_outputs.setdefault(family, {n: 0.0 for n in ("actions", "weights", "latent")})
            reference = session.run(computed, {"obs": inputs})
            trace = model.forward_intermediates(torch.from_numpy(inputs))
            if set(trace) != set(computed):
                raise ValueError(f"Intermediate coverage mismatch: missing={set(computed)-set(trace)}, extra={set(trace)-set(computed)}")
            for output_name, expected in zip(computed, reference):
                actual = trace[output_name].numpy()
                if expected.shape != actual.shape or expected.dtype != actual.dtype:
                    raise AssertionError(f"{name}: shape/dtype mismatch at {output_name}")
                if not np.isfinite(expected).all() or not np.isfinite(actual).all():
                    raise AssertionError(f"{name}: nonfinite value at {output_name}")
                error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
                if expected.dtype.kind in "iu":
                    if not np.array_equal(actual, expected):
                        raise AssertionError(f"{name}: integer mismatch at {output_name}")
                    ratio = 0.0
                else:
                    # Float32 GEMM/Conv reductions have backend-dependent
                    # ordering. This released model has hidden values above
                    # 1e8 and severe cancellation before L2 normalization.
                    # Intermediate checks therefore include a documented
                    # tensor-scale term; the three public outputs do not.
                    scale_term = 0.0 if output_name in ("actions", "weights", "latent") else (
                        intermediate_scale_rtol * float(np.abs(expected).max()))
                    if family == "clip_boundary" and output_name == "actions":
                        # Extreme clip-corner inputs yield actions in the
                        # thousands, well outside physical policy operation.
                        # Account for cancellation there separately, and
                        # report their errors apart from normal observations.
                        scale_term = stress_action_scale_rtol * float(np.abs(expected).max())
                    ratio = float(np.max(error / (atol + rtol * np.abs(expected) + scale_term)))
                    if ratio > 1.0:
                        raise AssertionError(f"{name}: parity failure at {output_name}: abs={error.max()}, tolerance_ratio={ratio}")
                stats[output_name]["max_absolute_error"] = max(stats[output_name]["max_absolute_error"], float(error.max()))
                stats[output_name]["max_tolerance_ratio"] = max(stats[output_name]["max_tolerance_ratio"], ratio)
                stats[output_name]["max_error_over_tensor_peak"] = max(
                    stats[output_name]["max_error_over_tensor_peak"],
                    float(error.max()) / max(float(np.abs(expected).max()), 1e-30))
                if output_name in family_outputs[family]:
                    family_outputs[family][output_name] = max(family_outputs[family][output_name], float(error.max()))
            # The same runtime forward path must expose exactly the audited outputs.
            actual_outputs = model.forward_with_aux(torch.from_numpy(inputs))
            for output_name, actual in zip(("actions", "weights", "latent"), actual_outputs):
                torch.testing.assert_close(actual, trace[output_name], rtol=0, atol=0)
            largest_action_error = max(largest_action_error, stats["actions"]["max_absolute_error"])
            cases.append(name)
        # ONNX was exported at batch one; validate our batched extension against
        # its independently audited per-row results (important for PPO rollout).
        batch = np.concatenate([x for _, x in list(test_inputs(samples=2))[:8]], axis=0)
        batched = model.forward_with_aux(torch.from_numpy(batch))
        for i, inputs in enumerate(batch):
            single = model.forward_with_aux(torch.from_numpy(inputs[None, :]))
            for actual, expected in zip(batched, single):
                torch.testing.assert_close(actual[i:i+1], expected, rtol=rtol, atol=atol)
    return dict(passed=True, input_case_count=len(cases), input_cases=cases,
                computed_node_outputs_checked=len(computed),
                constant_nodes_pinned_by_source_sha256=sum(n.op_type == "Constant" for n in graph.graph.node),
                operator_counts=dict(Counter(n.op_type for n in graph.graph.node)),
                all_computed_operators_covered=True, batch_extension_rows=len(batch),
                atol=atol, rtol=rtol, intermediate_tensor_scale_rtol=intermediate_scale_rtol,
                stress_action_tensor_scale_rtol=stress_action_scale_rtol,
                tolerance_note="All computed node outputs: atol + rtol*abs(reference); intermediate tensors add intermediate_tensor_scale_rtol*max(abs(reference)). Only actions at extreme clip-boundary stress cases add stress_action_tensor_scale_rtol*max(abs(reference)). Normal/random/deterministic public outputs and all weights/latent use the elementwise tolerance only. Float32 numerical equivalence, not bitwise equality.",
                max_public_output_absolute_error_by_input_family=family_outputs,
                max_action_absolute_error=largest_action_error,
                outputs=stats, torch_version=str(torch.__version__),
                onnxruntime_version=ort.__version__, onnx_version=onnx.__version__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "validation/moe_cts_actor_conversion_20261104")
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args()
    if args.samples < 8:
        parser.error("At least 8 samples per generated family are required")
    checkpoint_path = args.out_dir / "actor.pt"
    report_path = args.out_dir / "conversion.json"
    if checkpoint_path.exists() or report_path.exists():
        raise FileExistsError("Conversion outputs already exist; choose a new out-dir")
    torch.set_num_threads(1)
    model, graph = reconstruct(args.source)
    report = check_parity(model, graph, samples=args.samples)
    provenance = dict(format=CHECKPOINT_FORMAT, source_onnx_sha256=SOURCE_ONNX_SHA256,
                      source_onnx=str(args.source.resolve()), parity_passed=True,
                      source_repository="https://github.com/robogauge/code",
                      source_revision="f116b00a934262348f41cbfd0d07cc4cb6fbf8d8",
                      observation_layout="term-major; gyro3,gravity3,command3,q_rel12,dq12,last_action12; 5 oldest-to-newest samples each",
                      inputs_already_scaled=True, joint_order=list(JOINT_ORDER),
                      absent_training_state=["teacher_encoder", "critic", "action_noise_std", "optimizer1", "optimizer2", "training_iteration"],
                      semantics="Deployment actor reconstruction for a new warm-start; not original CTS training resume")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids replacing any prior policy artifact.
    with checkpoint_path.open("xb") as stream:
        torch.save({**provenance, "actor_state_dict": model.state_dict()}, stream)
    report.update(provenance)
    report.update(checkpoint_path=str(checkpoint_path.resolve()),
                  checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                  checkpoint_bytes=checkpoint_path.stat().st_size,
                  parameter_tensor_count=len(model.state_dict()),
                  parameter_count=sum(x.numel() for x in model.parameters()))
    with report_path.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("passed", "input_case_count", "computed_node_outputs_checked",
                                                  "max_action_absolute_error", "checkpoint_path", "checkpoint_bytes")}, indent=2))


if __name__ == "__main__":
    main()
