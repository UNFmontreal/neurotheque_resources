"""Compat wrapper for tools/dev/export_repo_to_xml.py.

Allows calling from repo root:
  python export_repo_to_xml.py --out repo.xml [--args]
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    script = Path(__file__).parent / "tools" / "dev" / "export_repo_to_xml.py"
    runpy.run_path(str(script), run_name="__main__")
