#!/usr/bin/env python3
"""Read-only joint-limit training monitor; acknowledgements remain local."""
import sys
from pathlib import Path
import check_continuation as monitor


if __name__ == "__main__":
    monitor.ACK_MILESTONES = (500, 1000)
    monitor.main(["--config", str(Path(__file__).resolve().parents[1] / ".monitor/joint_job.json"), *sys.argv[1:]])
