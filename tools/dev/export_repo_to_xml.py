#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Repo→XML packer: serialize an entire repository into a single, LLM-friendly XML file.

Features
- Respects .gitignore (via `git ls-files`) when available
- Skips binaries; truncates large files (configurable)
- Converts .ipynb notebooks into ordered <cell> entries (outputs stripped)
- Optional Python symbol index (<function>, <class>) via AST
- Adds repo metadata (remote URL, HEAD commit), file tree, and per-file hashes

Recommended for pasting into models that parse XML well (e.g., Claude/GPT).
"""

import argparse
import os
import sys
import json
import time
import hashlib
import subprocess
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

# ------------------------- Helpers -------------------------

TEXT_EXTS = {
    ".py": "python", ".md": "markdown", ".txt": "text",
    ".yml": "yaml", ".yaml": "yaml", ".toml": "toml",
    ".cfg": "ini", ".ini": "ini", ".json": "json",
    ".csv": "csv", ".tsv": "tsv", ".xml": "xml",
    ".html": "html", ".css": "css", ".js": "javascript",
    ".sh": "bash", ".bat": "batch", ".ps1": "powershell",
    ".ipynb": "jupyter-notebook",
}

DEFAULT_EXCLUDES = {
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".ipynb_checkpoints", ".DS_Store", "node_modules", "build", "dist",
    ".venv", "venv", ".vscode", ".idea", ".ruff_cache", ".cache",
    "figures", "logs", "data"
}

BINARY_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".pdf",
    ".zip", ".gz", ".xz", ".7z", ".tar", ".whl",
    ".so", ".dll", ".dylib", ".exe", ".bin",
    ".fif", ".edf", ".bdf", ".set", ".npy", ".npz"
}

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def is_likely_binary(path: Path, first_bytes: bytes) -> bool:
    if path.suffix.lower() in BINARY_EXTS:
        return True
    # Heuristic: NUL bytes or a lot of decode errors -> binary
    if b"\x00" in first_bytes:
        return True
    try:
        first_bytes.decode("utf-8")
        return False
    except UnicodeDecodeError:
        return True

def to_cdata(s: str) -> str:
    # Prevent ending the CDATA section accidentally
    return "<![CDATA[" + s.replace("]]>", "]]]]><![CDATA[>") + "]]>"

def git_cmd(repo_root: Path, args: List[str]) -> Optional[str]:
    try:
        res = subprocess.run(
            ["git"] + args, cwd=str(repo_root),
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True, text=True
        )
        return res.stdout.strip()
    except Exception:
        return None

def list_git_tracked_files(repo_root: Path) -> Optional[List[Path]]:
    out = git_cmd(repo_root, ["ls-files", "-z"])
    if out is None:
        return None
    items = [Path(p) for p in out.split("\x00") if p]
    return items

def get_repo_metadata(repo_root: Path):
    name = repo_root.name
    remote = git_cmd(repo_root, ["config", "--get", "remote.origin.url"]) or ""
    commit = git_cmd(repo_root, ["rev-parse", "HEAD"]) or ""
    return name, remote, commit

def within_excluded(path: Path, repo_root: Path, excluded_dirs: set) -> bool:
    # Check any ancestor directory is excluded
    rel = path.relative_to(repo_root)
    parts = rel.parts
    for p in parts:
        if p in excluded_dirs:
            return True
    return False

def safe_rel(repo_root: Path, p: Path) -> str:
    try:
        return str(p.relative_to(repo_root).as_posix())
    except Exception:
        return str(p.as_posix())

def load_text_file(p: Path, max_bytes: int) -> Tuple[str, int, str]:
    data = p.read_bytes()
    size = len(data)
    if max_bytes and size > max_bytes:
        data = data[:max_bytes]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        # Lossy fallback but stays text; we already filter binaries
        text = data.decode("utf-8", errors="replace")
    return text, size, sha256_bytes(p.read_bytes())

def parse_ipynb_cells(p: Path, max_bytes: int) -> Tuple[List[Tuple[str, str]], int, str]:
    raw = p.read_bytes()
    h = sha256_bytes(raw)
    size = len(raw)
    if max_bytes and size > max_bytes:
        # Even for notebooks, avoid loading >max_bytes; parse anyway if possible
        # We still try to parse to keep structure.
        raw_trunc = raw[:max_bytes]
        try:
            nb = json.loads(raw_trunc.decode("utf-8", errors="ignore"))
        except Exception:
            # Fall back to one big truncated code cell of raw text
            return [("markdown", f"(Truncated notebook {p.name} at {max_bytes} bytes)")], size, h
    try:
        nb = json.loads(raw.decode("utf-8"))
    except Exception:
        # Not a valid json? Treat as text file
        text = raw.decode("utf-8", errors="replace")
        return [("markdown", text)], size, h

    cells_out: List[Tuple[str, str]] = []
    for cell in nb.get("cells", []):
        cell_type = cell.get("cell_type", "raw")
        src = cell.get("source", [])
        if isinstance(src, list):
            content = "".join(src)
        else:
            content = str(src)
        if cell_type not in ("code", "markdown"):
            cell_type = "markdown"
        cells_out.append((cell_type, content))
    return cells_out, size, h

def build_tree_index(files_rel: List[str]) -> dict:
    """
    Build a nested dict representing the directory tree.
    Example:
    {
      "src": {"steps": {"file.py": None}, "other.py": None},
      "README.md": None
    }
    """
    root = {}
    for f in files_rel:
        parts = f.split("/")
        cur = root
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                cur[part] = None
            else:
                cur = cur.setdefault(part, {})
    return root

def write_tree_as_xml(out, tree: dict, indent: int = 2):
    pad = " " * indent
    for name, child in sorted(tree.items(), key=lambda kv: kv[0].lower()):
        if child is None:
            out.write(f"{pad}<file name=\"{name}\" />\n")
        else:
            out.write(f"{pad}<dir name=\"{name}\">\n")
            write_tree_as_xml(out, child, indent + 2)
            out.write(f"{pad}</dir>\n")

def format_py_symbols(src_text: str) -> List[str]:
    """
    Return a list of <function .../> and <class .../> tags (as strings)
    extracted from top-level Python AST. No external deps.
    """
    import ast
    tags = []
    try:
        tree = ast.parse(src_text)
    except Exception:
        return tags

    def fmt_args(a: ast.arguments) -> str:
        def name(arg):
            return arg.arg if hasattr(arg, "arg") else str(arg)
        parts = []
        # positional
        parts += [name(x) for x in a.posonlyargs]  # py3.8+
        if a.posonlyargs:
            parts[-1] += "/"
        # args
        parts += [name(x) for x in a.args]
        # vararg
        if a.vararg:
            parts.append("*" + name(a.vararg))
        # kwonly
        if a.kwonlyargs:
            if not a.vararg:
                parts.append("*")
            parts += [name(x) for x in a.kwonlyargs]
        # kw vararg
        if a.kwarg:
            parts.append("**" + name(a.kwarg))
        return "(" + ", ".join(parts) + ")"

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            sig = fmt_args(node.args)
            tags.append(f"<function name=\"{node.name}\" signature=\"{sig}\" />")
        elif isinstance(node, ast.AsyncFunctionDef):
            sig = fmt_args(node.args)
            tags.append(f"<function name=\"{node.name}\" signature=\"async {sig}\" />")
        elif isinstance(node, ast.ClassDef):
            bases = [getattr(b, "id", getattr(getattr(b, "attr", None), "attr", None)) or "base"
                     for b in node.bases]
            base_attr = f" bases=\"{', '.join(bases)}\"" if bases else ""
            tags.append(f"<class name=\"{node.name}\"{base_attr} />")
    return tags

# ------------------------- Main -------------------------

def main():
    ap = argparse.ArgumentParser(description="Serialize a repository into a single XML file for LLMs.")
    ap.add_argument("--repo", type=str, default=".", help="Path to repo root (default: .)")
    ap.add_argument("--out", type=str, required=True, help="Output XML path")
    ap.add_argument("--include-ext", type=str,
                    default=".py,.md,.txt,.yml,.yaml,.toml,.cfg,.ini,.json,.csv,.tsv,.ipynb",
                    help="Comma-separated list of file extensions to include")
    ap.add_argument("--exclude-dir", type=str,
                    default=",".join(sorted(DEFAULT_EXCLUDES)),
                    help="Comma-separated dir names to exclude anywhere in the path")
    ap.add_argument("--max-file-bytes", type=int, default=300_000,
                    help="Truncate any single file above this size (0 = unlimited)")
    ap.add_argument("--max-total-bytes", type=int, default=0,
                    help="Stop after this many cumulative bytes (0 = unlimited)")
    ap.add_argument("--index-symbols", action="store_true",
                    help="Extract Python functions/classes into <symbols>")
    args = ap.parse_args()

    repo_root = Path(args.repo).resolve()
    include_exts = {e.strip().lower() for e in args.include_ext.split(",") if e.strip()}
    excluded_dirs = {e.strip() for e in args.exclude_dir.split(",") if e.strip()}

    # Gather files
    git_files = list_git_tracked_files(repo_root)
    files: List[Path] = []
    if git_files:
        files = [repo_root / f for f in git_files]
    else:
        for root, dirs, fns in os.walk(repo_root):
            # Prune excluded dirs early
            dirs[:] = [d for d in dirs if d not in excluded_dirs]
            for fn in fns:
                files.append(Path(root) / fn)

    # Filter by ext and excludes
    filtered: List[Path] = []
    for p in files:
        if not p.is_file():
            continue
        if within_excluded(p, repo_root, excluded_dirs):
            continue
        ext = p.suffix.lower()
        if ext and ext not in include_exts:
            # allow README, LICENSE, Makefile etc with no ext
            if p.name.upper() not in {"README", "LICENSE", "MAKEFILE"}:
                continue
        filtered.append(p)

    # Build list of relative paths (strings) for tree
    files_rel_str = [safe_rel(repo_root, p) for p in filtered]
    tree = build_tree_index(files_rel_str)

    name, remote, commit = get_repo_metadata(repo_root)
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Write XML
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_bytes_written = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        out.write('<?xml version="1.0" encoding="utf-8"?>\n')
        out.write(f'<repository name="{name}" root="{repo_root.as_posix()}" created="{created}" commit="{commit}" remote="{remote}">\n')

        # Tree
        out.write("  <tree>\n")
        write_tree_as_xml(out, tree, indent=4)
        out.write("  </tree>\n")

        out.write("  <files>\n")
        for p in sorted(filtered, key=lambda x: safe_rel(repo_root, x).lower()):
            rel = safe_rel(repo_root, p)
            ext = p.suffix.lower()
            # Peek first bytes for binary detection
            try:
                first = p.read_bytes()[:4096]
            except Exception:
                continue

            if is_likely_binary(p, first):
                # Skip binaries but keep a stub to be explicit
                size = p.stat().st_size
                h = sha256_bytes(p.read_bytes())
                out.write(f'    <skipped_file path="{rel}" reason="binary" size_bytes="{size}" sha256="{h}" />\n')
                continue

            language = TEXT_EXTS.get(ext, "text")

            if ext == ".ipynb":
                cells, size, h = parse_ipynb_cells(p, args.max_file_bytes)
                out.write(f'    <notebook path="{rel}" language="jupyter-notebook" size_bytes="{size}" sha256="{h}">\n')
                for (cell_type, content) in cells:
                    out.write(f'      <cell type="{cell_type}">{to_cdata(content)}</cell>\n')
                    total_bytes_written += len(content.encode("utf-8"))
                    if args.max_total_bytes and total_bytes_written > args.max_total_bytes:
                        out.write('      <truncated reason="max_total_bytes"/>\n')
                        out.write("    </notebook>\n")
                        out.write("  </files>\n</repository>\n")
                        return
                # Optional symbols for notebooks: skip (no AST)
                out.write("    </notebook>\n")
                continue

            # Regular text file
            try:
                text, size, h = load_text_file(p, args.max_file_bytes)
            except Exception:
                # Fallback: skip unreadable
                out.write(f'    <skipped_file path="{rel}" reason="read_error" />\n')
                continue

            out.write(f'    <file path="{rel}" language="{language}" size_bytes="{size}" sha256="{h}">\n')

            # Optional symbol index for Python
            if args.index_symbols and ext == ".py":
                try:
                    symbol_tags = format_py_symbols(text)
                except Exception:
                    symbol_tags = []
                if symbol_tags:
                    out.write("      <symbols>\n")
                    for t in symbol_tags:
                        out.write("        " + t + "\n")
                    out.write("      </symbols>\n")

            out.write("      <content>")
            out.write(to_cdata(text))
            out.write("</content>\n")
            out.write("    </file>\n")

            total_bytes_written += len(text.encode("utf-8"))
            if args.max_total_bytes and total_bytes_written > args.max_total_bytes:
                out.write('    <truncated reason="max_total_bytes"/>\n')
                out.write("  </files>\n</repository>\n")
                return

        out.write("  </files>\n")
        out.write("</repository>\n")

    print(f"Wrote {out_path} (approx_payload_bytes={total_bytes_written})")

if __name__ == "__main__":
    main()
