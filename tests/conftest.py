from __future__ import annotations

import sys
from pathlib import Path

# Tests are allowed to import implementation modules directly. Users should
# only use main.py / pybf.compile_source.
ROOT = Path(__file__).resolve().parents[1]
INTERNAL = ROOT / "pybf"
if str(INTERNAL) not in sys.path:
    sys.path.insert(0, str(INTERNAL))
