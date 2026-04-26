"""Benchmark package."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python -m benchmark.*` from a source checkout without an editable install.
_src = Path(__file__).resolve().parents[1] / "src"
_src_str = str(_src)
if _src_str not in sys.path:
    sys.path.insert(0, _src_str)