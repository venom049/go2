"""Authorized torque-constrained warm start and fixed evaluations; no retries."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PYTHON = "/opt/conda/envs/isaaclab/bin/python"
SOURCE = ROOT / "validation/torque_constraints/reference1100/model_1100.pt"
CONSOLE = OUT / "training.console.log"
EVAL_CONSOLE = OUT / "evaluation.console.log"
JOB = OUT / "job.json"


def write_job(job, **updates):
    job.update(updates, updated_at_unix=time.time())
    tmp = JOB.with_suffix(".tmp")
    tmp.write_text(json.dumps(job, indent=2) + "\n")
    tmp.replace(JOB)


def main():
    if JOB.exists():
        raise FileExistsError("Refusing to replace an existing continuation job")
    command = ["bash", "scripts/run_lab.sh", "scripts/train_lab.py", "--headless",
        "--config", "configs/stairs_torque_limited.yaml", "--warm-start", str(SOURCE),
        "--num-envs", "1024",
        "--max-iterations", "1000", "--run-name", "stairs_torque_limited"]
    job = dict(pid=os.getpid(), train_pid=None, start_iteration=0,
        additional_updates=1000, target_iteration=1000, num_envs=1024,
        source_checkpoint=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        command=command, output_dir=str(OUT), console_log=str(CONSOLE),
        evaluation_console=str(EVAL_CONSOLE), run_directory=None,
        pipeline_status="starting", started_at_unix=time.time(),
        eval_summary_path=str(OUT / "comparison.json"),
        evaluation_plan={"iterations":[500,1000], "reference_source_iteration":1100, "seeds":[42,43], "num_envs":64,
            "conditions":[{"height":.15,"tread":.30,"controller":"stairs"},{"height":.18,"tread":.32,"controller":"hybrid"}], "command_m_s":0.6, "gate_lookahead_m":1.1})
    write_job(job)
    try:
        assert job["source_sha256"] == "877ff67ce7619516fa0e9350eccdd6537e7170640e5ef0a8b095a3624b2c27f5"
        env = dict(os.environ, GO2_PYTHON=PYTHON)
        snapshot = OUT / "training_source_snapshot.tar.gz"
        with tarfile.open(snapshot, "w:gz") as tar:
            for name in ["go2_him", "configs", "scripts", "tests", "upstream/himloco_lab/source", "upstream/SOURCES.json"]:
                path = ROOT / name
                candidates = sorted(path.rglob("*")) if path.is_dir() else [path]
                for f in candidates:
                    if f.is_file() and not any(x in f.parts for x in ["__pycache__", ".git"]):
                        tar.add(f, arcname=str(f.relative_to(ROOT)))
        with CONSOLE.open("x") as log:
            train = subprocess.Popen(command, cwd=ROOT, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            write_job(job, train_pid=train.pid, pipeline_status="training")
            code = train.wait()
        write_job(job, train_returncode=code)
        text = CONSOLE.read_text(errors="replace")
        match = re.search(r"^GO2_RUN_DIR=(.+)$", text, re.M)
        if match:
            run = Path(match[1].strip())
            write_job(job, run_directory=str(run))
            shutil.copy2(snapshot, run / snapshot.name)
        if code or not match:
            raise RuntimeError(f"Training exited with code {code}; see {CONSOLE}")
        completion = json.loads((run / "completed.json").read_text())
        if completion.get("iterations") != 1000 or completion.get("all_model_tensors_finite") is not True:
            raise RuntimeError("Training completion/model finiteness check failed")
        if not all((run / f"model_{i}.pt").is_file() for i in [500,1000]):
            raise FileNotFoundError("Expected evaluation checkpoints are missing")
        write_job(job, pipeline_status="evaluating", training_completed_at_unix=time.time())
        # Re-evaluate the reference with identical torque instrumentation.
        # Violations are model results, not pipeline errors: save them all.
        rows = []
        with EVAL_CONSOLE.open("x") as log:
            for label, checkpoint in [("reference1100", SOURCE),
                                       ("torque500", run / "model_500.pt"),
                                       ("torque1000", run / "model_1000.pt")]:
                for height, tread, controller in [(.15,.30,"stairs"),(.18,.32,"hybrid")]:
                    for seed in [42,43]:
                        stem = f"{label}_{controller}_{height}m_seed{seed}"
                        output = OUT / (stem + ".json")
                        if output.exists(): raise FileExistsError(output)
                        cmd = ["bash", "scripts/run_lab.sh", "scripts/play_lab.py", "--headless",
                            "--checkpoint", str(checkpoint), "--controller", controller,
                            "--num-envs", "64", "--steps", "1500", "--episodes-per-env", "1",
                            "--gait-metrics", "--torque-metrics", "--gate-lookahead", "1.1", "--output", str(output)]
                        for key,value in {"seed":seed,"terrain.rows":1,"terrain.cols":4,"terrain.initial_level":0,
                            "terrain.height_min":height,"terrain.height_max":height,"terrain.tread":tread,
                            "commands.forward_min":.6,"commands.forward_max":.6,
                            "targets.hip_abduction_max":.22,"targets.feet_lateral_min":.12,
                            "targets.feet_lateral_max":.205}.items():
                            cmd += ["--set",f"{key}={value}"]
                        print("START " + stem, file=log, flush=True)
                        with output.with_suffix(".console.log").open("x") as condition_log:
                            proc = subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                stdout=condition_log,stderr=subprocess.STDOUT)
                            write_job(job, eval_pid=proc.pid, evaluating_condition=stem,
                                      evaluation_console=str(output.with_suffix(".console.log")))
                            code = proc.wait()
                        if code: raise RuntimeError(f"Evaluation process failed {stem}: {code}")
                        result = json.loads(output.read_text())
                        if result["envs_incomplete"] or result["episodes"] != 64:
                            raise RuntimeError(f"Incomplete evaluation: {stem}")
                        rows.append(dict(label=label, condition=stem, height_m=height,tread_m=tread,
                            controller=controller,seed=seed,command_m_s=.6,
                            successes=result["successes"],episodes=result["episodes"],
                            torque_acceptance_passed=result["torque_acceptance_passed"],
                            torque_audit=result["torque_audit"],
                            successful_episode_gait=result.get("gait_metrics_successful_episodes",{}),
                            report_path=str(output)))
                        print(json.dumps({"finished":stem,"successes":result["successes"],
                                          "torque_acceptance_passed":result["torque_acceptance_passed"]}),file=log,flush=True)
            summary = {"source_checkpoint":str(SOURCE),"rows":rows,
                       "note":"0.6 m/s is commanded speed; gait statistics select successful first episodes; torque includes failures. No automatic model replacement or real-robot deployment.",
                       "torque_eligible_candidates":[label for label in ("torque500","torque1000")
                           if all(row["torque_acceptance_passed"] for row in rows if row["label"]==label)]}
            (OUT / "comparison.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
        write_job(job, pipeline_status="completed", finished_at_unix=time.time())
    except BaseException as exc:
        write_job(job, pipeline_status="failed", error=f"{type(exc).__name__}: {exc}", finished_at_unix=time.time())
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
