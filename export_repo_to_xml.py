#!/usr/bin/env python3
"""
Repo→XML packer: serialize an entire repository into a single, LLM-friendly XML file.

Highlights:
- Git-aware listing (respects .gitignore via `git ls-files` when available; otherwise walk + optional pathspec)
- Skips binaries by default; per-file truncation (configurable)
- Converts .ipynb notebooks into ordered <cell> entries (outputs stripped)
- Optional Python symbol index (<function>, <class>) via AST
- Repo metadata (remote URL, branch, HEAD commit), hierarchical <tree>, per-file hashes
- Optional gzip compression

Recommended for pasting into models that parse XML well.
"""

import argparse
import hashlib
import json
import os
import sys
import gzip
import time
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Set, Tuple, Dict, Iterable
from xml.sax.saxutils import quoteattr

# Optional pathspec for fallback .gitignore-style filtering during os.walk
try:
    import pathspec  # type: ignore
    HAS_PATHSPEC = True
except Exception:
    HAS_PATHSPEC = False


# ------------------------- Config -------------------------

# Languages by extension
LANG_BY_EXT = {
    "py": "python", "pyi": "python-stubs",
    "md": "markdown", "mdx": "markdown", "txt": "text", "rst": "rst",
    "json": "json", "yml": "yaml", "yaml": "yaml",
    "toml": "toml", "ini": "ini", "cfg": "config",
    "csv": "csv", "tsv": "tsv",
    "xml": "xml", "html": "html", "css": "css",
    "js": "javascript", "mjs": "javascript", "cjs": "javascript",
    "ts": "typescript", "tsx": "tsx",
    "sh": "shell", "bash": "shell", "zsh": "shell", "fish": "shell",
    "bat": "batch", "ps1": "powershell",
    "gitignore": "gitignore", "gitattributes": "gitattributes", "editorconfig": "editorconfig",
    "ipynb": "jupyter-notebook",
}

# Default include extensions (no dot)
DEFAULT_EXTS = {
    "py", "pyi", "md", "mdx", "txt", "rst",
    "json", "yml", "yaml", "toml", "ini", "cfg",
    "csv", "tsv",
    "xml", "html", "css",
    "js", "mjs", "cjs", "ts", "tsx",
    "sh", "bash", "zsh", "fish", "bat", "ps1",
    "gitignore", "gitattributes", "editorconfig",
    "ipynb",
}

# Exclude directories anywhere in path
DEFAULT_EXCLUDE_DIRS = {
    ".git", ".hg", ".svn",
    "__pycache__", ".mypy_cache", ".pytest_cache",
    ".ipynb_checkpoints", ".DS_Store",
    "node_modules", "build", "dist",
    ".venv", "venv", "env",
    ".vscode", ".idea", ".ruff_cache", ".cache",
    "figures", "logs", "reports", "test_output",
    "data", "derivatives",
}

# Binary hints (extensions)
BINARY_EXTS = {
    "png", "jpg", "jpeg", "gif", "bmp", "ico", "svg", "svgz",
    "pdf",
    "zip", "gz", "xz", "7z", "rar", "tar",
    "whl",
    "mp3", "wav", "ogg", "flac",
    "mp4", "mov", "avi", "mkv",
    "so", "dll", "dylib", "o", "a", "class", "jar", "exe", "bin",
    "pkl", "npz", "npy",
    "fif", "edf", "bdf", "set",
}


# ------------------------- Helpers -------------------------

def to_cdata(s: str) -> str:
    # Prevent ending CDATA accidentally by splitting occurrences of "]]>"
    return "<![CDATA[" + s.replace("]]>", "]]]]><![CDATA[>") + "]]>"

def sha256_stream(path: Path, chunk_size: int = 1 << 20) -> Tuple[int, str]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    return size, h.hexdigest()

def guess_language(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    return LANG_BY_EXT.get(ext, "text" if ext == "" else ext)

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
    return [repo_root / p for p in out.split("\x00") if p]

def get_repo_metadata(repo_root: Path) -> Dict[str, Optional[str]]:
    return {
        "name": repo_root.name,
        "remote": git_cmd(repo_root, ["config", "--get", "remote.origin.url"]) or None,
        "commit": git_cmd(repo_root, ["rev-parse", "HEAD"]) or None,
        "branch": git_cmd(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"]) or None,
    }

def within_excluded(path: Path, repo_root: Path, excluded_dirs: Set[str]) -> bool:
    try:
        rel = path.relative_to(repo_root)
    except Exception:
        return False
    for part in rel.parts:
        if part in excluded_dirs:
            return True
    return False

def rel_posix(root: Path, p: Path) -> str:
    try:
        return p.relative_to(root).as_posix()
    except Exception:
        return p.as_posix()

def is_likely_binary(path: Path, sample: bytes) -> bool:
    ext = path.suffix.lower().lstrip(".")
    if ext in BINARY_EXTS:
        return True
    if b"\x00" in sample:
        return True
    try:
        sample.decode("utf-8")
        return False
    except UnicodeDecodeError:
        return True

def load_text_file(p: Path, max_bytes: int) -> Tuple[str, int, str]:
    raw = p.read_bytes()
    size = len(raw)
    data = raw if max_bytes <= 0 or size <= max_bytes else raw[:max_bytes]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
    sha = hashlib.sha256(raw).hexdigest()
    return text, size, sha

def parse_ipynb_cells(p: Path, max_bytes: int) -> Tuple[List[Tuple[str, str]], int, str]:
    raw = p.read_bytes()
    h = hashlib.sha256(raw).hexdigest()
    size = len(raw)
    if max_bytes and size > max_bytes:
        raw_trunc = raw[:max_bytes]
        try:
            nb = json.loads(raw_trunc.decode("utf-8", errors="ignore"))
        except Exception:
            return [("markdown", f"(Truncated notebook {p.name} at {max_bytes} bytes)")], size, h
    try:
        nb = json.loads(raw.decode("utf-8"))
    except Exception:
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

def format_py_symbols(src_text: str) -> List[str]:
    import ast
    tags: List[str] = []
    try:
        tree = ast.parse(src_text)
    except Exception:
        return tags

    def fmt_args(a: "ast.arguments") -> str:
        def name(arg):
            return arg.arg if hasattr(arg, "arg") else str(arg)
        parts: List[str] = []
        parts += [name(x) for x in getattr(a, "posonlyargs", [])]
        if getattr(a, "posonlyargs", []):
            parts[-1] += "/"
        parts += [name(x) for x in a.args]
        if a.vararg:
            parts.append("*" + name(a.vararg))
        if a.kwonlyargs:
            if not a.vararg:
                parts.append("*")
            parts += [name(x) for x in a.kwonlyargs]
        if a.kwarg:
            parts.append("**" + name(a.kwarg))
        return "(" + ", ".join(parts) + ")"

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            tags.append(f"<function name=\"{node.name}\" signature=\"{fmt_args(node.args)}\" />")
        elif isinstance(node, ast.AsyncFunctionDef):
            tags.append(f"<function name=\"{node.name}\" signature=\"async {fmt_args(node.args)}\" />")
        elif isinstance(node, ast.ClassDef):
            bases = []
            for b in node.bases:
                try:
                    bases.append(getattr(b, "id", None) or getattr(b, "attr", None) or "base")
                except Exception:
                    bases.append("base")
            base_attr = f" bases=\"{', '.join(bases)}\"" if bases else ""
            tags.append(f"<class name=\"{node.name}\"{base_attr} />")
    return tags

def build_tree_index(files_rel: List[str]) -> dict:
    root: dict = {}
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
            out.write(f"{pad}<file name={quoteattr(name)} />\n")
        else:
            out.write(f"{pad}<dir name={quoteattr(name)}>\n")
            write_tree_as_xml(out, child, indent + 2)
            out.write(f"{pad}</dir>\n")


# ------------------------- Core -------------------------

def gather_files(repo_root: Path,
                 respect_gitignore: bool,
                 exclude_dirs: Set[str]) -> List[Path]:
    # Prefer git ls-files (respects .gitignore)
    if respect_gitignore:
        git_files = list_git_tracked_files(repo_root)
        if git_files:
            return [p for p in git_files if p.is_file()]
    # Fallback: walk filesystem; optionally use pathspec to mimic .gitignore
    spec = None
    if respect_gitignore and HAS_PATHSPEC:
        gi = repo_root / ".gitignore"
        if gi.exists():
            try:
                spec = pathspec.PathSpec.from_lines("gitwildmatch", gi.read_text(encoding="utf-8", errors="ignore").splitlines())
            except Exception:
                spec = None
    out: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dpath = Path(dirpath)
        rel = dpath.relative_to(repo_root).as_posix() if dpath != repo_root else ""
        # prune dirs
        keep = []
        for d in dirnames:
            sub = "/".join(x for x in (rel, d) if x)
            skip = False
            if spec is not None and spec.match_file(sub.rstrip("/") + "/"):
                skip = True
            if not skip and d in exclude_dirs:
                skip = True
            if not skip:
                keep.append(d)
        dirnames[:] = keep
        for fn in filenames:
            sub = "/".join(x for x in (rel, fn) if x)
            if spec is not None and spec.match_file(sub):
                continue
            out.append(dpath / fn)
    return out

def write_xml(repo_root: Path,
              output: Path,
              gzip_output: bool,
              max_bytes_per_file: int,
              max_total_bytes: int,
              include_binaries_stub: bool,
              exclude_dirs: Set[str],
              exts: Optional[Set[str]],
              respect_gitignore: bool,
              index_symbols: bool):
    files_all = gather_files(repo_root, respect_gitignore, exclude_dirs)

    # Filter by ext and excludes
    filtered: List[Path] = []
    for p in files_all:
        if not p.is_file():
            continue
        if within_excluded(p, repo_root, exclude_dirs):
            continue
        ext = p.suffix.lower().lstrip(".")
        if exts is None:
            pass
        else:
            if ext and ext not in exts:
                # allow README, LICENSE, MAKEFILE without extension
                if p.name.upper() not in {"README", "LICENSE", "MAKEFILE"}:
                    continue
        filtered.append(p)

    files_rel = [rel_posix(repo_root, p) for p in filtered]
    tree = build_tree_index(files_rel)
    git_info = get_repo_metadata(repo_root)
    generated_at = datetime.now(timezone.utc).isoformat()
    repo_name = repo_root.name

    total_files_scanned = 0
    files_included = 0
    files_skipped = 0
    total_bytes_original = 0
    total_bytes_payload = 0

    opener = gzip.open if gzip_output else open
    with opener(output, "wt", encoding="utf-8", newline="\n") as out:
        out.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        out.write(f"<repository name={quoteattr(repo_name)} root={quoteattr(str(repo_root))} generated_at={quoteattr(generated_at)} generator={quoteattr('repo2xml 2.0')}>\n")

        # config
        out.write("  <config>\n")
        out.write(f"    <respect_gitignore>{'true' if respect_gitignore else 'false'}</respect_gitignore>\n")
        out.write(f"    <include_binaries_stub>{'true' if include_binaries_stub else 'false'}</include_binaries_stub>\n")
        out.write(f"    <max_bytes_per_file>{max_bytes_per_file}</max_bytes_per_file>\n")
        out.write(f"    <max_total_bytes>{max_total_bytes}</max_total_bytes>\n")
        out.write("    <exclude_dirs>")
        out.write(",".join(sorted(exclude_dirs)))
        out.write("</exclude_dirs>\n")
        if exts is None:
            out.write("    <exts>ALL</exts>\n")
        else:
            out.write("    <exts>")
            out.write(",".join(sorted(exts)))
            out.write("</exts>\n")
        out.write("  </config>\n")

        # git
        out.write("  <git>\n")
        if git_info.get("remote"):
            out.write(f"    <remote>{git_info['remote']}</remote>\n")
        if git_info.get("branch"):
            out.write(f"    <branch>{git_info['branch']}</branch>\n")
        if git_info.get("commit"):
            out.write(f"    <commit>{git_info['commit']}</commit>\n")
        out.write("  </git>\n")

        # tree
        out.write("  <tree>\n")
        write_tree_as_xml(out, tree, indent=4)
        out.write("  </tree>\n")

        # files
        out.write("  <files>\n")
        for p in sorted(filtered, key=lambda x: rel_posix(repo_root, x).lower()):
            total_files_scanned += 1
            rel = rel_posix(repo_root, p)
            # Peek for binary detection
            try:
                first = p.read_bytes()[:4096]
            except Exception:
                files_skipped += 1
                out.write(f'    <skipped_file path={quoteattr(rel)} reason="read_error" />\n')
                continue

            # Compute size and sha once (streaming)
            try:
                size_bytes, sha_full = sha256_stream(p)
            except Exception:
                files_skipped += 1
                out.write(f'    <skipped_file path={quoteattr(rel)} reason="hash_error" />\n')
                continue
            total_bytes_original += size_bytes

            if is_likely_binary(p, first):
                if include_binaries_stub:
                    out.write(f'    <skipped_file path={quoteattr(rel)} reason="binary" size_bytes="{size_bytes}" sha256="{sha_full}" />\n')
                else:
                    files_skipped += 1
                continue

            ext = p.suffix.lower().lstrip(".")
            language = guess_language(p)

            if ext == "ipynb":
                try:
                    cells, size, sha = parse_ipynb_cells(p, max_bytes_per_file)
                except Exception:
                    files_skipped += 1
                    out.write(f'    <skipped_file path={quoteattr(rel)} reason="ipynb_parse_error" />\n')
                    continue
                out.write(f'    <notebook path={quoteattr(rel)} language={quoteattr("jupyter-notebook")} size_bytes="{size}" sha256="{sha}">\n')
                for (cell_type, content) in cells:
                    out.write(f'      <cell type={quoteattr(cell_type)}>{to_cdata(content)}</cell>\n')
                    total_bytes_payload += len(content.encode("utf-8"))
                    if max_total_bytes and total_bytes_payload > max_total_bytes:
                        out.write('      <truncated reason="max_total_bytes"/>\n')
                        out.write("    </notebook>\n")
                        out.write("  </files>\n")
                        out.write("  <summary>\n")
                        out.write(f"    <total_files_scanned>{total_files_scanned}</total_files_scanned>\n")
                        out.write(f"    <files_included>{files_included}</files_included>\n")
                        out.write(f"    <files_skipped>{files_skipped}</files_skipped>\n")
                        out.write(f"    <total_bytes_original>{total_bytes_original}</total_bytes_original>\n")
                        out.write(f"    <total_bytes_included>{total_bytes_payload}</total_bytes_included>\n")
                        out.write("  </summary>\n")
                        out.write("</repository>\n")
                        return
                out.write("    </notebook>\n")
                files_included += 1
                continue

            # Regular text file
            try:
                text, size, sha = load_text_file(p, max_bytes_per_file)
            except Exception:
                files_skipped += 1
                out.write(f'    <skipped_file path={quoteattr(rel)} reason="read_error" />\n')
                continue

            mtime = ""
            try:
                st = p.stat()
                mtime = datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()
            except Exception:
                pass

            out.write(f'    <file path={quoteattr(rel)} language={quoteattr(language)} size_bytes="{size}" sha256="{sha}" mtime={quoteattr(mtime)}>\n')

            if index_symbols and ext == "py":
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

            files_included += 1
            total_bytes_payload += len(text.encode("utf-8"))
            if max_total_bytes and total_bytes_payload > max_total_bytes:
                out.write('    <truncated reason="max_total_bytes"/>\n')
                out.write("  </files>\n")
                out.write("  <summary>\n")
                out.write(f"    <total_files_scanned>{total_files_scanned}</total_files_scanned>\n")
                out.write(f"    <files_included>{files_included}</files_included>\n")
                out.write(f"    <files_skipped>{files_skipped}</files_skipped>\n")
                out.write(f"    <total_bytes_original>{total_bytes_original}</total_bytes_original>\n")
                out.write(f"    <total_bytes_included>{total_bytes_payload}</total_bytes_included>\n")
                out.write("  </summary>\n")
                out.write("</repository>\n")
                return

        out.write("  </files>\n")

        # summary
        out.write("  <summary>\n")
        out.write(f"    <total_files_scanned>{total_files_scanned}</total_files_scanned>\n")
        out.write(f"    <files_included>{files_included}</files_included>\n")
        out.write(f"    <files_skipped>{files_skipped}</files_skipped>\n")
        out.write(f"    <total_bytes_original>{total_bytes_original}</total_bytes_original>\n")
        out.write(f"    <total_bytes_included>{total_bytes_payload}</total_bytes_included>\n")
        out.write("  </summary>\n")

        out.write("</repository>\n")


# ------------------------- CLI -------------------------

def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="Export a repository to a single XML file for LLM ingestion.")
    p.add_argument("--root", type=str, default=".", help="Path to repository root.")
    p.add_argument("--output", type=str, default="repo.xml", help="Output XML file path (use .gz with --gzip).")
    p.add_argument("--gzip", action="store_true", help="Compress output with gzip (writes text XML inside .gz).")

    p.add_argument("--max-bytes", type=int, default=300_000, help="Max bytes per file to include (0 = no limit).")
    p.add_argument("--max-total-bytes", type=int, default=0, help="Stop after this many cumulative content bytes (0 = unlimited).")

    p.add_argument("--include-binaries", action="store_true",
                   help="Include a <skipped_file ... reason=\"binary\"/> stub for binaries. If absent, binaries are fully omitted.")
    p.add_argument("--no-gitignore", action="store_true", help="Do not respect .gitignore (skip git ls-files and pathspec).")

    p.add_argument("--exclude-dirs", nargs="*", default=sorted(DEFAULT_EXCLUDE_DIRS),
                   help="Directory names to exclude anywhere in the tree.")
    p.add_argument("--ext", nargs="*", default=sorted(DEFAULT_EXTS),
                   help="Extensions to include (no dot). Use 'ALL' to include all.")

    p.add_argument("--index-symbols", action="store_true", help="Extract Python functions/classes into <symbols>.")
    return p.parse_args(argv)

def main():
    args = parse_args()
    repo_root = Path(args.root).resolve()
    if not repo_root.exists() or not repo_root.is_dir():
        print(f"Error: root directory not found: {repo_root}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output).resolve()
    gzip_out = bool(args.gzip)
    respect_gitignore = not args.no_gitignore

    exclude_dirs = set(args.exclude_dirs) if args.exclude_dirs else set()
    exts: Optional[Set[str]]
    if args.ext and any(x.upper() == "ALL" for x in args.ext):
        exts = None
    else:
        exts = set(x.lower().lstrip(".") for x in args.ext) if args.ext else None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_xml(
        repo_root=repo_root,
        output=out_path,
        gzip_output=gzip_out,
        max_bytes_per_file=max(0, int(args.max_bytes)),
        max_total_bytes=max(0, int(args.max_total_bytes)),
        include_binaries_stub=bool(args.include_binaries),
        exclude_dirs=exclude_dirs,
        exts=exts,
        respect_gitignore=respect_gitignore,
        index_symbols=bool(args.index_symbols),
    )
    print(f"Wrote XML to: {out_path}{' (gzipped)' if gzip_out else ''}")

if __name__ == "__main__":
    main()