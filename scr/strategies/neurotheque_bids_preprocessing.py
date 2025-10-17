#!/usr/bin/env python
"""
General BIDS preprocessing launcher for Neurotheque EEG datasets.

This script scans a BIDS directory for EEG recordings (BrainVision, FIF, EDF, etc.),
filters them according to the subject/session/run/task arguments, and invokes the
standard Neurotheque preprocessing pipeline for each matching recording.

Examples
--------
Process every run in the dataset (uses defaults from the config):
    python -m scr.strategies.neurotheque_bids_preprocessing

Process a single subject across all sessions/runs:
    python -m scr.strategies.neurotheque_bids_preprocessing --subject 01

Process one session for a subject:
    python -m scr.strategies.neurotheque_bids_preprocessing --subject 01 --session 001

Process a specific run and override BIDS root:
    python -m scr.strategies.neurotheque_bids_preprocessing \\
        --bids-root D:/data/5pt/bids --subject 01 --session 001 --run 01
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set

import yaml

# ---------------------------------------------------------------------------
# Repository bootstrap: make sure we can import the pipeline package
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PIPELINE_LABEL = "neurotheque-preproc"

try:
    from scr.pipeline import Pipeline  # noqa: E402
except ModuleNotFoundError as exc:  # pragma: no cover - informative failure path
    missing = exc.name
    if missing == "autoreject":
        raise SystemExit(
            "The 'autoreject' package is required for Neurotheque preprocessing.\n"
            "Install it with `pip install autoreject` in your environment and retry."
        ) from exc
    raise

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------
SUPPORTED_DATA_EXTENSIONS = {".vhdr", ".edf", ".fif", ".fif.gz", ".bdf", ".set"}
ENTITY_PATTERNS = {
    "subject": re.compile(r"sub-([A-Za-z0-9]+)"),
    "session": re.compile(r"ses-([A-Za-z0-9]+)"),
    "task": re.compile(r"task-([A-Za-z0-9]+)"),
    "run": re.compile(r"run-([A-Za-z0-9]+)"),
}


@dataclass(frozen=True)
class RunDescriptor:
    """Lightweight container describing one BIDS recording."""

    path: Path
    subject: str
    session: Optional[str]
    run: Optional[str]
    task: Optional[str]

    def label(self) -> str:
        """Human-readable summary for logging."""
        parts = [f"sub-{self.subject}"]
        parts.append(f"ses-{self.session}" if self.session else "ses-NA")
        parts.append(f"task-{self.task}" if self.task else "task-NA")
        parts.append(f"run-{self.run}" if self.run else "run-NA")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Argument parsing & normalisation
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    default_config = REPO_ROOT / "configs" / "neurotheque_bids_pipeline.json"
    parser = argparse.ArgumentParser(
        description="Run the Neurotheque preprocessing pipeline across BIDS EEG datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help="Path to a pipeline configuration file (JSON preferred).",
    )
    parser.add_argument(
        "--bids-root",
        type=Path,
        help="Override the BIDS root directory specified in the config.",
    )
    parser.add_argument(
        "--subject",
        default=None,
        help="Subject label (e.g., 01). Omit or use 'all' to process every subject.",
    )
    parser.add_argument(
        "--session",
        help="Session label (e.g., 001). Use 'all' or omit to include every session.",
    )
    parser.add_argument(
        "--run",
        help="Run label (e.g., 01). Use 'all' or omit to include every run.",
    )
    parser.add_argument(
        "--task",
        help="Task label (e.g., mario). Use 'all' or omit to include every task.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching recordings without running the preprocessing pipeline.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip JSON schema validation when instantiating the pipeline.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep processing remaining runs even if one of them fails.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging verbosity (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args()


def _normalise_filter(value: Optional[str], prefix: Optional[str] = None) -> Optional[str]:
    """Convert CLI filters to canonical values (None => wildcard)."""
    if value is None:
        return None
    value = value.strip()
    if not value or value.lower() == "all":
        return None
    if prefix and value.lower().startswith(prefix):
        value = value[len(prefix) :]
    return value


def _normalise_filter_collection(
    values: Optional[Iterable[str]], prefix: Optional[str] = None
) -> Optional[Set[str]]:
    """Normalise list-based filters to a set of canonical labels."""
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        values = [values]
    normalised: Set[str] = set()
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text.lower() == "all":
            continue
        if prefix and text.lower().startswith(prefix):
            text = text[len(prefix) :]
        normalised.add(text)
    return normalised or None


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------
def load_config(path: Path) -> dict:
    """Load a YAML or JSON pipeline configuration file."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix in {".yml", ".yaml"}:
        return yaml.safe_load(text)
    if suffix == ".json":
        import json

        return json.loads(text)
    raise ValueError(f"Unsupported config format '{suffix}'. Use YAML or JSON.")


def resolve_bids_root(config: dict, override: Optional[Path]) -> Path:
    """Determine the BIDS root to use for discovery."""
    if override:
        return override.expanduser().resolve()

    directory = config.get("directory", {})
    root = Path(directory.get("root", REPO_ROOT)).expanduser()
    if not root.is_absolute():
        root = (REPO_ROOT / root).resolve()
    raw_dir = directory.get("raw_data_dir")
    if not raw_dir:
        raise ValueError("The config must define 'directory.raw_data_dir'.")
    bids_root = (root / raw_dir).expanduser().resolve()
    return bids_root


# ---------------------------------------------------------------------------
# BIDS traversal utilities
# ---------------------------------------------------------------------------
def extract_entities(path: Path) -> dict:
    """Parse subject/session/run/task labels from a BIDS filename."""
    name = path.name
    entities = {}
    for key, pattern in ENTITY_PATTERNS.items():
        match = pattern.search(name)
        if match:
            entities[key] = match.group(1)
    return entities


def iter_bids_runs(
    bids_root: Path,
    subject_filters: Optional[Set[str]],
    session_filters: Optional[Set[str]],
    run_filters: Optional[Set[str]],
    task_filters: Optional[Set[str]],
) -> List[RunDescriptor]:
    """Return all recordings that satisfy the provided filters."""
    runs: List[RunDescriptor] = []
    lower_parts_cache = {}

    def allowed_path(path: Path) -> bool:
        if path.suffixes:
            suffix_combo = "".join(s.lower() for s in path.suffixes[-2:])
            if suffix_combo == ".fif.gz":
                return True
        ext = path.suffix.lower()
        return ext in SUPPORTED_DATA_EXTENSIONS

    for candidate in sorted(bids_root.rglob("*_eeg.*")):
        # Skip derivatives or non-EEG folders
        key = candidate.parent
        lower_parts = lower_parts_cache.get(key)
        if lower_parts is None:
            lower_parts = {part.lower() for part in candidate.parts}
            lower_parts_cache[key] = lower_parts

        if "derivatives" in lower_parts:
            continue
        if candidate.parent.name.lower() != "eeg":
            continue
        if not allowed_path(candidate):
            continue

        entities = extract_entities(candidate)
        subject = entities.get("subject")
        if not subject:
            continue
        session = entities.get("session")
        run = entities.get("run")
        task = entities.get("task")

        if subject_filters and subject not in subject_filters:
            continue
        if session_filters and session not in session_filters:
            continue
        if run_filters and run not in run_filters:
            continue
        if task_filters and task not in task_filters:
            continue

        runs.append(
            RunDescriptor(
                path=candidate.resolve(),
                subject=subject,
                session=session,
                run=run,
                task=task,
            )
        )

    runs.sort(key=lambda r: (r.subject, r.session or "", r.run or "", r.path.name))
    return runs


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------
def prepare_run_config(base_config: dict, bids_root: Path, run: RunDescriptor) -> dict:
    """Create a run-specific pipeline config."""
    config = deepcopy(base_config)

    directory = config.setdefault("directory", {})
    pipeline_label = config.get("pipeline_label", PIPELINE_LABEL)
    directory["root"] = str(bids_root)
    directory["raw_data_dir"] = "."
    directory["processed_dir"] = f"derivatives/{pipeline_label}"
    directory["derivatives_dir"] = f"derivatives/{pipeline_label}"
    directory["reports_dir"] = f"derivatives/{pipeline_label}/reports"

    config["default_subject"] = run.subject
    config["default_session"] = run.session or base_config.get("default_session", "001")
    config["default_run"] = run.run or base_config.get("default_run", "01")
    if run.task:
        config["default_task"] = run.task
    else:
        config.pop("default_task", None)

    steps = config.get("pipeline", {}).get("steps", [])
    for step in steps:
        if step.get("name") == "LoadData":
            params = step.setdefault("params", {})
            params["input_file"] = str(run.path)
            break

    return config


def setup_logging(level: str) -> None:
    """Init logging with a sensible format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    base_config = load_config(args.config)
    config_filters = base_config.get("bids_filters", {}) or {}
    subject_filters_cfg = _normalise_filter_collection(config_filters.get("subjects"), "sub-")
    session_filters_cfg = _normalise_filter_collection(config_filters.get("sessions"), "ses-")
    run_filters_cfg = _normalise_filter_collection(config_filters.get("runs"), "run-")
    task_filters_cfg = _normalise_filter_collection(config_filters.get("tasks"), "task-")

    if args.subject is not None:
        subject_value = _normalise_filter(args.subject, "sub-")
        subject_filters = {subject_value} if subject_value else None
    else:
        subject_filters = subject_filters_cfg

    if args.session is not None:
        session_value = _normalise_filter(args.session, "ses-")
        session_filters = {session_value} if session_value else None
    else:
        session_filters = session_filters_cfg

    if args.run is not None:
        run_value = _normalise_filter(args.run, "run-")
        run_filters = {run_value} if run_value else None
    else:
        run_filters = run_filters_cfg

    if args.task is not None:
        task_value = _normalise_filter(args.task, "task-")
        task_filters = {task_value} if task_value else None
    else:
        task_filters = task_filters_cfg

    bids_root = resolve_bids_root(base_config, args.bids_root)

    if not bids_root.exists():
        raise FileNotFoundError(f"BIDS root not found: {bids_root}")

    logging.info("Using config: %s", args.config.resolve())
    logging.info("Scanning BIDS root: %s", bids_root)

    runs = iter_bids_runs(bids_root, subject_filters, session_filters, run_filters, task_filters)
    if not runs:
        logging.warning("No recordings matched the provided filters.")
        return 0

    logging.info("Found %d recording(s) to process.", len(runs))
    for idx, run in enumerate(runs, start=1):
        logging.info("[%d/%d] %s -> %s", idx, len(runs), run.label(), run.path)

    if args.dry_run:
        logging.info("Dry-run requested; no preprocessing executed.")
        return 0

    failures: List[RunDescriptor] = []
    for idx, run in enumerate(runs, start=1):
        logging.info("=== [%d/%d] Starting %s ===", idx, len(runs), run.label())
        run_config = prepare_run_config(base_config, bids_root, run)
        try:
            pipeline = Pipeline(config_dict=run_config, validate_config=not args.no_validate)
            pipeline.run()
            logging.info("=== [%d/%d] Completed %s ===", idx, len(runs), run.label())
        except Exception as exc:  # pragma: no cover - runtime failure path
            logging.exception("Pipeline failed for %s: %s", run.label(), exc)
            failures.append(run)
            if not args.continue_on_error:
                return 1

    if failures:
        logging.warning("Completed with %d failure(s).", len(failures))
        for failed in failures:
            logging.warning(" - %s (%s)", failed.label(), failed.path)
        return 1

    logging.info("All recordings processed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
