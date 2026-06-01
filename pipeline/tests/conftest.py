"""conftest.py — shared fixtures for pipeline tests."""

import sys
from pathlib import Path

_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root / "projects" / "01-nl-stock-query" / "src"))
sys.path.insert(0, str(_root / "projects" / "03-ai-strategy-generator" / "src"))
sys.path.insert(0, str(_root / "projects" / "04-strategy-diagnostics"))