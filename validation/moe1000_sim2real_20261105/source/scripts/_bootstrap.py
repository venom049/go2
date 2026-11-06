"""Resolve project and vendored algorithm without installing global packages."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream/himloco_lab/source/himloco_lab"))
