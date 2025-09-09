#!/usr/bin/env python3
"""
Batch preprocessing across a BIDS dataset (JSON-driven).
Discovers runs under BIDS root and calls the single-run preprocessor.

Python 3.10+
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

from mne_bids import get_entity_vals

from preprocess_single import preprocess_one


def _load_cfg(path: Path) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _iter_runs(bids_root: Path,
               task: Optional[str] = None) -> Iterable[Dict[str, Optional[str]]]:
    subs = get_entity_vals(bids_root, entity_key="subject")
    for sub in subs:
        sessions = get_entity_vals(bids_root, entity_key="session", subject=sub) or [None]
        for ses in sessions:
            tasks = get_entity_vals(bids_root, entity_key="task", subject=sub, session=ses)
            for t in (tasks or []):
                if task and t != task:
                    continue
                runs = get_entity_vals(bids_root, entity_key="run", subject=sub, session=ses, task=t) or [None]
                for run in runs:
                    yield {"sub": sub, "ses": ses, "task": t, "run": run}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Batch preprocess a BIDS dataset (JSON-driven).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--bids-root", required=True)
    ap.add_argument("--task", help="Filter by task (optional)")
    ap.add_argument("--n-jobs", type=int, default=1, help="Reserved for future parallel execution")
    args = ap.parse_args(argv)

    cfg = _load_cfg(Path(args.config))
    bids_root = Path(args.bids_root).resolve()

    for run in _iter_runs(bids_root, task=args.task):
        print(f"[batch] {run}")
        preprocess_one(cfg, bids_root, run["sub"], run["ses"], run["task"], run["run"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

