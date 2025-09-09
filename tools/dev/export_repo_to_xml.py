"""Export repository file inventory to XML for audit purposes.

Usage:
  python tools/dev/export_repo_to_xml.py > repo.xml
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path


def build_xml(root_dir: Path) -> ET.Element:
    root = ET.Element("repository")
    for dirpath, _, filenames in os.walk(root_dir):
        rel_dir = os.path.relpath(dirpath, root_dir)
        dir_el = ET.SubElement(root, "dir", path=rel_dir)
        for fname in sorted(filenames):
            ET.SubElement(dir_el, "file", name=fname)
    return root


if __name__ == "__main__":
    here = Path(__file__).resolve().parents[2]
    tree = ET.ElementTree(build_xml(here))
    ET.indent(tree, space="  ")  # Python 3.9+
    tree.write("-", encoding="unicode")

