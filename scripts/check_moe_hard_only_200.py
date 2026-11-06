#!/usr/bin/env python3
"""Independent acknowledgements for the hard-actual-position-only experiment."""
import shutil
from pathlib import Path
import check_moe_remote_200 as monitor

monitor.STATE = monitor.ROOT/'validation/moe_hard_only_20261105/monitor_state.json'
monitor.STATE.parent.mkdir(parents=True,exist_ok=True)
monitor.SNAPSHOT = monitor.STATE.with_name('monitor_snapshot.json')
monitor.REMOTE = monitor.REMOTE.replace(
    '/moe_hard_20261105/formal_run', '/moe_hard_only_20261105/formal_run').replace(
    'run_moe_target_clip_training.py', 'run_moe_hard_only_training.py')
# Keep the verified public host-key record across temporary-directory cleanup.
cached = monitor.STATE.with_name('known_hosts')
temporary = Path('/tmp/moe-train-30916-known-hosts')
if cached.exists() and not temporary.exists():
    shutil.copy2(cached,temporary)

if __name__=='__main__':
    monitor.main()
