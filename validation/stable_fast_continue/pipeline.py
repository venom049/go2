"""One authorized continuation, followed by fixed-stair comparisons; no retries."""
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
SOURCE = ROOT / "logs/20261102_183927_stairs_stable_fast/model_600.pt"
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
        "--config", str(SOURCE.parent / "settings.yaml"), "--resume", str(SOURCE),
        "--set", "commands.speed_ramp_updates=0", "--num-envs", "1024",
        "--max-iterations", "1000", "--run-name", "stairs_stable_fast_continue"]
    job = dict(pid=os.getpid(), train_pid=None, start_iteration=600,
        additional_updates=1000, target_iteration=1600, num_envs=1024,
        source_checkpoint=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        command=command, output_dir=str(OUT), console_log=str(CONSOLE),
        evaluation_console=str(EVAL_CONSOLE), run_directory=None,
        pipeline_status="starting", started_at_unix=time.time(),
        eval_summary_path=str(OUT / "comparison.json"),
        evaluation_plan={"iterations":[1100,1600], "seeds":[42,43], "num_envs":64,
            "height_m":0.18, "command_m_s":0.6, "controller":"hybrid", "gate_lookahead_m":1.1})
    write_job(job)
    try:
        assert job["source_sha256"] == "45133df9bbc5439d67656607988403229d2558656ef848c6ca4f93b02e484124"
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
        if completion.get("iterations") != 1600 or completion.get("all_model_tensors_finite") is not True:
            raise RuntimeError("Training completion/model finiteness check failed")
        if not all((run / f"model_{i}.pt").is_file() for i in [1100,1600]):
            raise FileNotFoundError("Expected evaluation checkpoints are missing")
        write_job(job, pipeline_status="evaluating", training_completed_at_unix=time.time())
        # Preserve the established model600/gate1.1 reference data, using distinct labels.
        for seed in [42,43]:
            src = ROOT / "validation/stable_fast" / f"gate11_hybrid_0.18m_0.6mps_seed{seed}.json"
            shutil.copy2(src, OUT / f"reference600_hybrid_0.18m_0.6mps_seed{seed}.json")
        with EVAL_CONSOLE.open("x") as log:
            for iteration in [1100,1600]:
                cmd = [PYTHON, "scripts/evaluate_stability.py", "--checkpoint", str(run / f"model_{iteration}.pt"),
                    "--label", f"continued{iteration}", "--controller", "hybrid", "--speeds", "0.6",
                    "--seeds", "42", "43", "--gate-lookahead", "1.1", "--height", "0.18",
                    "--num-envs", "64", "--output-dir", str(OUT)]
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT)
                write_job(job, eval_pid=proc.pid, evaluating_iteration=iteration)
                if proc.wait():
                    raise RuntimeError(f"Independent evaluation failed for model_{iteration}; see {EVAL_CONSOLE}")
            subprocess.run([PYTHON, "scripts/summarize_stability.py", "--input-dir", str(OUT), "--print-table"],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        write_job(job, pipeline_status="completed", finished_at_unix=time.time())
    except BaseException as exc:
        write_job(job, pipeline_status="failed", error=f"{type(exc).__name__}: {exc}", finished_at_unix=time.time())
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
