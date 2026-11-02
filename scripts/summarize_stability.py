#!/usr/bin/env python3
"""Pool independent stability evaluations without ranking or selecting models.

Example: python3 scripts/summarize_stability.py --print-table
Labels default to evaluate_stability.py filenames; --label 'after*=candidate600'
can override them. Fixed height and command speed come from report settings.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import fnmatch
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
FILENAME = re.compile(rf"(?P<label>.+)_(?P<controller>stairs|walk|hybrid)_(?P<height>{NUMBER})m_(?P<speed>{NUMBER})mps_seed(?P<seed>\d+)\.json")
RMS = re.compile(r"(?:^|_)rms(?:_|$)")
TRAVERSAL_METRICS = ("stairs_traversal_time_mean_s", "stairs_traversal_speed_mean_m_s")
METRIC_TOTALS = ("valid_env_samples", "stairs_valid_env_samples", "nonfinite_metric_elements",
                 "selected_env_count", "stairs_completed_traversals")


def count(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def parse_label_rules(items):
    rules = []
    for item in items:
        pattern, sep, label = item.partition("=")
        if not sep or not pattern or not label:
            raise ValueError("--label must have the form FILE_GLOB=LABEL")
        rules.append((pattern, label))
    return rules


def report_identity(path, report, label_rules=()):
    match = FILENAME.fullmatch(path.name)
    settings = report["settings"]
    labels = {label for pattern, label in label_rules
              if fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(str(path), pattern)}
    if len(labels) > 1:
        raise ValueError(f"Conflicting --label rules for {path.name}")
    label = next(iter(labels), None) or report.get("label") or (match["label"] if match else None)
    if not isinstance(label, str) or not label:
        raise ValueError(f"Cannot determine label for {path.name}; supply --label")
    controller = report.get("controller") or (match["controller"] if match else None)
    if controller not in ("stairs", "walk", "hybrid"):
        raise ValueError(f"Cannot determine controller for {path.name}")
    height = finite(settings["terrain"]["height_min"], "terrain.height_min")
    height_max = finite(settings["terrain"]["height_max"], "terrain.height_max")
    speed = finite(settings["commands"]["forward_min"], "commands.forward_min")
    speed_max = finite(settings["commands"]["forward_max"], "commands.forward_max")
    if height != height_max or speed != speed_max:
        raise ValueError(f"{path.name}: grouping requires fixed evaluation height and speed")
    seed = count(settings["seed"], "settings.seed")
    return (label, controller, height, speed), seed


def pool_metrics(reports):
    """Merge sufficient statistics, never average seed means unweighted."""
    reports = [r for r in reports if r is not None]
    if not reports:
        return None
    names = sorted({name for r in reports for name in r["metric_element_counts"]})
    result = {"source_report_count": len(reports), "metric_element_counts": {}}
    for name in names:
        numerator = 0.
        denominator = 0
        for r in reports:
            n = count(r["metric_element_counts"].get(name, 0), f"{name}.count")
            if name in TRAVERSAL_METRICS and name in r["metric_element_counts"]:
                completed = count(r["stairs_completed_traversals"], "stairs_completed_traversals")
                if n != completed:
                    raise ValueError(f"{name} count differs from complete traversals")
            if not n:
                continue
            value = finite(r.get(name), name)
            if RMS.search(name) and value < 0:
                raise ValueError(f"{name}: RMS cannot be negative")
            numerator += (value * value if RMS.search(name) else value) * n
            denominator += n
        mean = numerator / denominator if denominator else None
        result[name] = math.sqrt(mean) if mean is not None and RMS.search(name) else mean
        result["metric_element_counts"][name] = denominator
    for name in METRIC_TOTALS:
        if any(name in r for r in reports):
            result[name] = sum(count(r.get(name, 0), name) for r in reports)
    maxima = [finite(r["raw_action_abs_max"], "raw_action_abs_max")
              for r in reports if r.get("raw_action_abs_max") is not None]
    result["raw_action_abs_max"] = max(maxima) if maxima else None
    # Different bounds or sampling definitions must not silently be averaged.
    metadata = {key for r in reports for key in r} - set(names) - set(METRIC_TOTALS)
    metadata -= {"metric_element_counts", "raw_action_abs_max", "source_report_count"}
    for name in sorted(metadata):
        variants = {json.dumps(r[name], sort_keys=True, allow_nan=False) for r in reports if name in r}
        if len(variants) != 1:
            raise ValueError(f"Incompatible metric metadata: {name}")
        result[name] = json.loads(variants.pop())
    return result


def pool_switch(records):
    present = [(seed, r["policy_switch"]) for _, seed, r in records if "policy_switch" in r]
    if not present:
        return None
    mode_counts = [0, 0, 0]
    per_env = []
    maxima = []
    for seed, switch in present:
        values = switch["walk_transition_stairs_samples"]
        if len(values) != 3:
            raise ValueError("Expected walking/blending/stairs sample counts")
        mode_counts = [a + count(b, "mode samples") for a, b in zip(mode_counts, values)]
        maxima.append(finite(switch["max_joint_target_delta_rad"], "max_joint_target_delta_rad"))
        transitions = switch["mode_transitions_per_env"]
        completed = switch.get("flat_stairs_flat_completed_per_env")
        completions = switch.get("flat_stairs_flat_completion_counts_per_env")
        for values in (completed, completions):
            if values is not None and len(values) != len(transitions):
                raise ValueError("Policy-switch arrays have different environment counts")
        for env_id, n in enumerate(transitions):
            item = {"seed": seed, "env_id": env_id, "mode_transitions": count(n, "mode transitions")}
            if completed is not None:
                if not isinstance(completed[env_id], bool):
                    raise ValueError("flat_stairs_flat_completed_per_env must contain booleans")
                item["flat_stairs_flat_completed"] = completed[env_id]
            if completions is not None:
                item["flat_stairs_flat_completions"] = count(completions[env_id], "transition completions")
                if completed is not None and completed[env_id] != (completions[env_id] > 0):
                    raise ValueError("Policy-switch completion boolean/count disagree")
            per_env.append(item)
    known = [r for r in per_env if "flat_stairs_flat_completed" in r]
    success = sum(r["flat_stairs_flat_completed"] for r in known)
    return {
        "source_report_count": len(present),
        "walk_transition_stairs_samples": mode_counts,
        "mode_transitions_total": sum(r["mode_transitions"] for r in per_env),
        "environments_with_sequence_report": len(known),
        "flat_stairs_flat_completed_envs": success if known else None,
        "flat_stairs_flat_completed_fraction": success / len(known) if known else None,
        "flat_stairs_flat_completions_total": sum(r["flat_stairs_flat_completions"] for r in per_env
            if "flat_stairs_flat_completions" in r) if any("flat_stairs_flat_completions" in r for r in per_env) else None,
        "max_joint_target_delta_rad": max(maxima),
        "per_environment": per_env,
        "definition": "counts retain each seed/environment; sequence completion denotes controller handover, not staircase success",
    }


def summarize(files, label_rules=()):
    groups = defaultdict(list)
    skipped = []
    for path in sorted({Path(p).resolve() for p in files}):
        report = json.loads(path.read_text())
        if not isinstance(report, dict) or not {"settings", "episodes", "successes"}.issubset(report):
            skipped.append(str(path))
            continue  # Existing comparison.json or other provenance metadata.
        key, seed = report_identity(path, report, label_rules)
        if any(old_seed == seed for _, old_seed, _ in groups[key]):
            raise ValueError(f"Duplicate seed {seed} within condition {key}; refusing double counting")
        if report.get("episodes_per_env_target") != 1:
            raise ValueError(f"{path.name}: expected first-episode evaluation quota of 1")
        total = count(report["episodes"], "episodes")
        parts = sum(count(report[name], name) for name in ("successes", "failures", "timeouts"))
        if total != parts:
            raise ValueError(f"{path.name}: successes/failures/timeouts do not sum to episodes")
        groups[key].append((path, seed, report))
    if not groups:
        raise ValueError("No first-episode evaluation reports found")
    pooled = []
    for (label, controller, height, speed), records in sorted(groups.items()):
        records.sort(key=lambda record: record[1])
        result = {"label": label, "controller": controller, "height_m": height,
                  "command_speed_m_s": speed, "seeds": [seed for _, seed, _ in records],
                  "seed_count": len(records), "checkpoint_paths": sorted({r["checkpoint"] for _, _, r in records})}
        for name in ("episodes", "successes", "failures", "timeouts", "envs_incomplete"):
            result[name] = sum(count(r[name], name) for _, _, r in records)
        result["success_rate"] = result["successes"] / result["episodes"] if result["episodes"] else None
        for name in ("gait_metrics", "gait_metrics_successful_episodes"):
            result[name] = pool_metrics([r.get(name) for _, _, r in records])
        result["policy_switch"] = pool_switch(records)
        # Keep raw per-seed records so missing diagnostics, stop reasons,
        # incomplete rows, source settings and controller descriptions survive.
        result["per_seed"] = [{"source_file": str(path), "seed": seed, **r} for path, seed, r in records]
        pooled.append(result)
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "aggregation": {
            "grouping": "label, controller, fixed settings terrain height, fixed settings command speed",
            "success_rate": "sum(successes) / sum(completed episodes), not mean of seed percentages",
            "means": "sum(metric mean * metric_element_count) / sum(metric_element_count)",
            "rms": "sqrt(sum(seed RMS squared * metric_element_count) / sum(metric_element_count))",
            "traversals": "time and speed weighted by complete geometric traversal count, not samples or successful-episode count; mean speed is not inverse mean time",
            "maxima": "maximum across source reports",
            "successful_subset": "independently pooled completed successful first episodes; no failing-episode samples are included",
            "selection": "No model ranking, significance claim, or automatic selection is performed",
        },
        "groups": pooled,
        "skipped_non_evaluation_files": skipped,
    }


def print_table(summary):
    print("label | controller | height m | command m/s | success | segment m/s | segment s | roll deg | outward deg")
    for group in summary["groups"]:
        metrics = group["gait_metrics_successful_episodes"] or {}
        def number(key, degrees=False):
            value = metrics.get(key)
            return "-" if value is None else f"{math.degrees(value) if degrees else value:.3f}"
        print(f"{group['label']} | {group['controller']} | {group['height_m']:g} | {group['command_speed_m_s']:g} | "
              f"{group['successes']}/{group['episodes']} | "
              f"{number('stairs_traversal_speed_mean_m_s')} | {number('stairs_traversal_time_mean_s')} | "
              f"{number('stairs_body_roll_rms_rad', True)} | {number('stairs_hip_outward_component_mean_rad', True)}")
    print("Segment/pose columns use successful first episodes; zero-height terrain uses the same horizontal segment. Missing metrics are '-'.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "validation/stable_fast")
    parser.add_argument("--output", type=Path, help="Default: INPUT_DIR/comparison.json")
    parser.add_argument("--label", action="append", default=[], metavar="FILE_GLOB=LABEL")
    parser.add_argument("--print-table", action="store_true")
    args = parser.parse_args()
    try:
        summary = summarize(args.input_dir.glob("*.json"), parse_label_rules(args.label))
    except (ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    output = args.output or args.input_dir / "comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if args.print_table:
        print_table(summary)
    print(f"Wrote {output.resolve()} ({len(summary['groups'])} conditions)")


if __name__ == "__main__":
    main()
