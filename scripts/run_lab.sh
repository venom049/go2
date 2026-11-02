#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
if [[ -f "$PROJECT_DIR/.nvidia_eula_accepted" ]]; then
    export OMNI_KIT_ACCEPT_EULA=YES
fi
if [[ "$(id -u)" == 0 ]]; then
    export OMNI_KIT_ALLOW_ROOT=1
fi
PYTHON_BIN="${GO2_PYTHON:-python}"
# The NVIDIA pip bundle nests the normal source tree; use it as a conventional
# Isaac Lab checkout instead of the thin outer bootstrap package.
LAB_SOURCE="$($PYTHON_BIN -c 'import importlib.util,pathlib; s=importlib.util.find_spec("isaaclab"); print(pathlib.Path(s.origin).parent/"source" if s else "")')"
if [[ -d "$LAB_SOURCE/isaaclab/isaaclab" ]]; then
    export PYTHONPATH="$LAB_SOURCE/isaaclab:$LAB_SOURCE/isaaclab_assets:$LAB_SOURCE/isaaclab_tasks:$LAB_SOURCE/isaaclab_rl:${PYTHONPATH:-}"
fi
export PYTHONUNBUFFERED=1
exec "$PYTHON_BIN" "$@"
