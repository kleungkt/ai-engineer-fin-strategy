"""conftest.py – shared fixtures and sys.path setup for all tests."""

import sys
from pathlib import Path

# Ensure the src/ directory is importable (screener.py does `from parser import ...`)
_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
