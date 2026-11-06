"""Trainable reconstruction of the audited RoboGauge deployment actor.

This is the deployed student MoE and action network, not a CTS training
checkpoint.  It contains no teacher, critic, action distribution or optimizer.
Inputs are already scaled term-major observations (five oldest-to-newest
samples per term).  The caller owns history, reset semantics and joint mapping.
"""
from pathlib import Path

import torch
from torch import nn


SOURCE_ONNX_SHA256 = "bc08253fd68f661bf6be5ce5ca70cec555c3fde2c397d23fbe9814a998e056f5"
CHECKPOINT_FORMAT = "robogauge_moe_cts_deployment_actor_v1"
TERM_DIMS = (3, 3, 3, 12, 12, 12)
HISTORY_LENGTH = 5
OBSERVATION_DIM = 225
JOINT_ORDER = tuple(f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR")
                    for joint in ("hip", "thigh", "calf"))


def _record(trace, name, value):
    if trace is not None:
        trace[name] = value
    return value


class StudentMoEEncoder(nn.Module):
    """Eight 32-dimensional experts, with the ONNX parameter names intact."""

    def __init__(self):
        super().__init__()
        self.experts_backbone = nn.Sequential(nn.Linear(225, 512), nn.ELU(),
                                              nn.Linear(512, 256), nn.ELU())
        self.experts_hidden = nn.Sequential(nn.Linear(256, 2048), nn.ELU())
        self.experts_out = nn.Conv1d(2048, 256, kernel_size=1, groups=8)
        self.gating_network = nn.Sequential(nn.Linear(225, 512), nn.ELU(),
                                            nn.Linear(512, 256), nn.ELU(),
                                            nn.Linear(256, 8), nn.Softmax(dim=1))

    def forward(self, history):
        return self._forward(history, None)

    def _forward(self, history, trace):
        prefix = "/student_moe_encoder/"
        gate_names = (
            "gating_network/gating_network.0/Gemm_output_0",
            "gating_network/experts_backbone.1/Elu_output_0",
            "gating_network/gating_network.2/Gemm_output_0",
            "gating_network/experts_backbone.1_1/Elu_output_0",
            "gating_network/gating_network.4/Gemm_output_0",
        )
        weights = history
        for layer, name in zip(self.gating_network[:5], gate_names):
            weights = _record(trace, prefix + name, layer(weights))
        weights = _record(trace, "weights", self.gating_network[5](weights))
        backbone_names = (
            "experts_backbone/experts_backbone.0/Gemm_output_0",
            "experts_backbone/experts_backbone.1/Elu_output_0",
            "experts_backbone/experts_backbone.2/Gemm_output_0",
            "experts_backbone/experts_backbone.1_1/Elu_output_0",
        )
        hidden = history
        for layer, name in zip(self.experts_backbone, backbone_names):
            hidden = _record(trace, prefix + name, layer(hidden))
        hidden = _record(trace, prefix + "experts_hidden/experts_hidden.0/Gemm_output_0",
                         self.experts_hidden[0](hidden))
        hidden = _record(trace, prefix + "experts_hidden/experts_backbone.1/Elu_output_0",
                         self.experts_hidden[1](hidden))
        hidden = _record(trace, prefix + "Unsqueeze_output_0", hidden.unsqueeze(-1))
        experts = _record(trace, prefix + "experts_out/Conv_output_0", self.experts_out(hidden))
        experts = _record(trace, prefix + "Reshape_output_0", experts.reshape(-1, 8, 32))
        expanded_weights = _record(trace, prefix + "Unsqueeze_1_output_0", weights.unsqueeze(-1))
        weighted = _record(trace, prefix + "Mul_output_0", expanded_weights * experts)
        mixture = _record(trace, prefix + "ReduceSum_output_0", weighted.sum(dim=1))
        absolute = _record(trace, prefix + "norm_layer/Abs_output_0", mixture.abs())
        square = _record(trace, prefix + "norm_layer/Pow_output_0", absolute.pow(2.0))
        total = _record(trace, prefix + "norm_layer/ReduceSum_output_0", square.sum(-1, keepdim=True))
        norm = _record(trace, prefix + "norm_layer/Pow_1_output_0", total.pow(0.5))
        clipped = _record(trace, prefix + "norm_layer/Clip_output_0", norm.clamp_min(1e-12))
        if trace is not None:
            trace[prefix + "norm_layer/Shape_output_0"] = torch.tensor(
                mixture.shape, dtype=torch.int64, device=mixture.device)
        denominator = _record(trace, prefix + "norm_layer/Expand_output_0", clipped.expand_as(mixture))
        latent = _record(trace, "latent", mixture / denominator)
        return latent, weights


class MoECTSActor(nn.Module):
    """225 scaled term-major inputs to 12 unscaled joint-position actions.

    ``forward_with_aux`` also returns the eight mixture weights and latent32.
    Both subnetworks are differentiable.  No action clipping or PD conversion
    is performed here, matching the released ONNX network.
    """

    def __init__(self):
        super().__init__()
        self.student_moe_encoder = StudentMoEEncoder()
        self.actor = nn.Sequential(nn.Linear(77, 512), nn.ELU(),
                                   nn.Linear(512, 256), nn.ELU(),
                                   nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 12))

    @staticmethod
    def _history(obs, trace):
        if obs.ndim != 2 or obs.shape[1] != OBSERVATION_DIM:
            raise ValueError("Expected scaled term-major observation shape [batch, 225]")
        if obs.dtype != torch.float32:
            raise ValueError("The audited actor uses float32 observations and parameters")
        chunks = torch.split(obs, [dim * HISTORY_LENGTH for dim in TERM_DIMS], dim=1)
        terms = []
        for i, (chunk, dim) in enumerate(zip(chunks, TERM_DIMS)):
            suffix = "" if i == 0 else f"_{i}"
            _record(trace, f"/Slice{suffix}_output_0", chunk)
            terms.append(_record(trace, f"/Reshape{suffix}_output_0", chunk.reshape(-1, 5, dim)))
        flat_parts = []
        for frame in range(5):
            parts = []
            for term, values in enumerate(terms):
                index = frame * 6 + term
                suffix = "" if index == 0 else f"_{index}"
                parts.append(_record(trace, f"/Gather{suffix}_output_0", values[:, frame, :]))
            if frame < 2:
                flat_parts.extend(parts)
            else:
                suffix = "" if frame == 2 else f"_{frame - 2}"
                flat_parts.append(_record(trace, f"/Concat{suffix}_output_0", torch.cat(parts, dim=1)))
        history = _record(trace, "/Concat_3_output_0", torch.cat(flat_parts, dim=1))
        latest = _record(trace, "/Slice_6_output_0", history[:, -45:])
        return history, latest

    def _forward(self, obs, trace):
        history, latest = self._history(obs, trace)
        latent, weights = self.student_moe_encoder._forward(history, trace)
        actions = _record(trace, "/Concat_4_output_0", torch.cat((latent, latest), dim=1))
        names = ("/actor/actor.0/Gemm_output_0", "/actor/actor.1/Elu_output_0",
                 "/actor/actor.2/Gemm_output_0", "/actor/actor.1_1/Elu_output_0",
                 "/actor/actor.4/Gemm_output_0", "/actor/actor.1_2/Elu_output_0", "actions")
        for layer, name in zip(self.actor, names):
            actions = _record(trace, name, layer(actions))
        return actions, weights, latent

    def forward(self, obs):
        return self._forward(obs, None)[0]

    def forward_with_aux(self, obs):
        return self._forward(obs, None)

    def forward_intermediates(self, obs):
        """Named non-Constant ONNX outputs, for conversion audit only."""
        trace = {}
        self._forward(obs, trace)
        return trace


def load_actor_checkpoint(path: str | Path, device="cpu") -> MoECTSActor:
    """Load only this converter's actor artifact, never silently ignore keys."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or saved.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Not a converted RoboGauge deployment-actor checkpoint")
    if saved.get("source_onnx_sha256") != SOURCE_ONNX_SHA256:
        raise ValueError("Unexpected source ONNX hash")
    if saved.get("parity_passed") is not True:
        raise ValueError("The actor artifact has not passed conversion parity")
    state = saved.get("actor_state_dict")
    if not isinstance(state, dict) or len(state) != 22:
        raise ValueError("Expected all 22 actor/encoder parameter tensors")
    if any(not isinstance(value, torch.Tensor) or value.dtype != torch.float32
           or not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Actor parameters must be finite float32")
    model = MoECTSActor()
    model.load_state_dict(state, strict=True)
    return model.to(device)
