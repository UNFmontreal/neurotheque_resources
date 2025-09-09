#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generalized BIDSification for DSI‑24 EDF folders.

- Recursively finds *.edf under --source
- Infers BIDS entities via regex "patterns" and "fallbacks" from a JSON config
- Loads EDF, finds events from a configured trigger channel
- Maps events purely from JSON (no per-task branches)
- Writes BIDS with mne-bids and saves a per-file JSON log under derivatives

Python 3.10+
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import mne
import numpy as np
from mne.channels import make_standard_montage
from mne.io import read_raw_edf

# hard dependency for this script
from mne_bids import BIDSPath, write_raw_bids, make_dataset_description

from event_mapping import map_events_from_config


# ------------------------------ Config dataclasses ------------------------------

@dataclass
class FilenameParsingConfig:
    patterns: List[str]
    fallbacks: Dict[str, str]


@dataclass
class BidsConfig:
    manufacturer: str
    manufacturersModelName: str
    line_freq: Optional[int]
    montage: object
    eeg_reference: Optional[str]
    dataset_description: Dict[str, object]


@dataclass
class PathsConfig:
    source_root: Path
    bids_root: Path
    derivatives_root: Path
    overwrite: bool


@dataclass
class EventsGlobal:
    trigger_channel: str
    default_task: str
    tasks: Dict[str, object]  # task -> dict(event_map, onset_shift_sec, epoching)


@dataclass
class ProjectConfig:
    name: str
    version: str
    random_seed: Optional[int] = None
    parallel_jobs: Optional[int] = None


# ------------------------------ Helpers ------------------------------

def _compile_patterns(patterns: Iterable[str]) -> List[re.Pattern]:
    out = []
    for pat in patterns:
        out.append(re.compile(pat, flags=re.IGNORECASE))
    return out


def _zeropad_numeric(value: str, width: int) -> str:
    try:
        return f"{int(value):0{width}d}"
    except Exception:
        return value


def parse_entities_from_name(path: Path,
                             patterns: List[str],
                             fallbacks: Dict[str, str]) -> Dict[str, Optional[str]]:
    """
    Parse {subject, session, task, run} from filename using a list of regex
    patterns with named groups; if missing values remain, fill from fallbacks.

    Examples of accepted groups in regex:
      (?P<subject>...), (?P<session>...), (?P<task>...), (?P<run>...)

    Returns dict with keys: subject, session, task, run (all strings or None).
    """
    name = path.name
    compiled = _compile_patterns(patterns)
    out: Dict[str, Optional[str]] = dict(subject=None, session=None, task=None, run=None)

    for rgx in compiled:
        m = rgx.search(name)
        if not m:
            continue
        for key in out.keys():
            if key in m.groupdict():
                out[key] = m.group(key)

        if all(out[k] is not None for k in out):
            break

    # normalize & pad
    if out["subject"]:
        out["subject"] = _zeropad_numeric(out["subject"], 2)
    if out["session"]:
        out["session"] = _zeropad_numeric(out["session"], 2)
    if out["run"]:
        out["run"] = _zeropad_numeric(out["run"], 2)

    # apply fallbacks
    for k, v in fallbacks.items():
        if out.get(k) in (None, "", "unknown"):
            out[k] = v

    return out


def build_bids_path(entities: Dict[str, Optional[str]], bids_root: Path) -> BIDSPath:
    """
    Build a BIDSPath from entity dict. Session and run are optional.
    """
    bp = BIDSPath(
        subject=str(entities["subject"]),
        session=str(entities["session"]) if entities.get("session") else None,
        task=str(entities["task"]),
        run=str(entities["run"]) if entities.get("run") else None,
        datatype="eeg",
        root=bids_root,
    )
    return bp


def _resolve_montage(montage_cfg: object):
    """
    Resolve montage from config:
      - "standard_1020" string
      - or {"kind":"dig_montage","path":"..."} (SFP/els etc.)
    """
    if isinstance(montage_cfg, str):
        if montage_cfg.lower() == "standard_1020":
            return make_standard_montage("standard_1020")
        raise ValueError(f"Unsupported montage string: {montage_cfg}")
    if isinstance(montage_cfg, dict):
        if montage_cfg.get("kind") == "dig_montage":
            path = Path(montage_cfg["path"])
            if not path.exists():
                raise FileNotFoundError(path)
            return mne.channels.read_custom_montage(str(path))
    return None


def load_raw_edf(path: Path, montage_cfg: object, line_freq: Optional[int]) -> mne.io.BaseRaw:
    raw = read_raw_edf(str(path), preload=True, verbose=False)
    # standardize channel names/casing lightly (avoid changing IDs in EDF header)
    raw.rename_channels({ch: ch.strip() for ch in raw.ch_names})
    # apply montage if requested
    montage = _resolve_montage(montage_cfg)
    if montage is not None:
        with mne.utils.use_log_level("ERROR"):
            try:
                raw.set_montage(montage, match_case=False, on_missing="warn")
            except Exception:
                pass
    # annotate powerline frequency (added later in sidecar too)
    if line_freq and isinstance(line_freq, int):
        raw.info["line_freq"] = line_freq
    return raw


def _detect_stim_channel(raw: mne.io.BaseRaw, preferred: str) -> Optional[str]:
    # prioritize configured name if present; otherwise pick first stim channel
    chs = {c.lower(): c for c in raw.ch_names}
    if preferred and preferred.lower() in chs:
        return chs[preferred.lower()]
    # fallback to any stim channel
    types = dict(zip(raw.ch_names, raw.get_channel_types()))
    cand = [ch for ch, t in types.items() if t == "stim"]
    if cand:
        return cand[0]
    # last resort: common vendor names
    for guess in ("Trigger", "TRIG", "STATUS", "STI 014", "STI014"):
        if guess in raw.ch_names:
            return guess
    return None


def find_events(raw: mne.io.BaseRaw, trigger_channel: str) -> np.ndarray:
    stim = _detect_stim_channel(raw, trigger_channel)
    if not stim:
        return np.empty((0, 3), dtype=int)
    ev = mne.find_events(
        raw,
        stim_channel=stim,
        shortest_event=1,
        consecutive=False,
        uint_cast=True,
        verbose=False,
    )
    return ev if ev is not None else np.empty((0, 3), dtype=int)


def _patch_eeg_sidecar(bids_path: BIDSPath,
                       eeg_reference_text: Optional[str],
                       power_line_freq: Optional[int]) -> None:
    """
    Ensure required/recommended EEG sidecar fields are present.
    """
    bp = bids_path.copy().update(suffix="eeg", extension=".json")
    json_path = Path(bp.fpath)
    if not json_path.exists():
        return
    meta = json.loads(json_path.read_text(encoding="utf-8"))

    # REQUIRED
    if eeg_reference_text:
        meta["EEGReference"] = eeg_reference_text
    if power_line_freq is not None:
        meta["PowerLineFrequency"] = int(power_line_freq)
    if "SoftwareFilters" not in meta or not meta["SoftwareFilters"]:
        meta["SoftwareFilters"] = "n/a"

    # RECOMMENDED
    meta.setdefault("Manufacturer", "Wearable Sensing")
    meta.setdefault("ManufacturersModelName", "DSI-24")
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _discover_edfs(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*.edf") if p.is_file()])


# ------------------------------ Main worker ------------------------------

def bidsify_one(
    edf_path: Path,
    cfg_paths: PathsConfig,
    cfg_bids: BidsConfig,
    cfg_filename: FilenameParsingConfig,
    cfg_events: EventsGlobal,
    cfg_project: ProjectConfig,
    warnings_as_list: Optional[List[str]] = None,
) -> Tuple[Optional[BIDSPath], dict]:
    """
    BIDSify a single EDF file according to the provided config objects.
    Returns (BIDSPath or None, log_dict).
    """
    t0 = time.time()
    warnings_list: List[str] = [] if warnings_as_list is None else warnings_as_list

    # 1) Entities
    entities = parse_entities_from_name(edf_path, cfg_filename.patterns, cfg_filename.fallbacks)
    task = entities.get("task") or cfg_events.default_task

    # 2) Build BIDS path
    bp = build_bids_path(entities, cfg_paths.bids_root)

    # 3) Load EDF
    raw = load_raw_edf(edf_path, cfg_bids.montage, cfg_bids.line_freq)

    # 4) Events discovery
    ev = find_events(raw, cfg_events.trigger_channel)

    # 5) Map events per JSON task section (or default task if not found)
    task_key = str(task).lower()
    task_cfg = cfg_events.tasks.get(task_key, cfg_events.tasks.get(cfg_events.default_task, {}))
    mapped_events, event_id = map_events_from_config(
        raw=raw,
        events=ev,
        task_config=task_cfg,
    )

    # 6) Write BIDS
    bp = BIDSPath(  # ensure fields are present
        subject=bp.subject, session=bp.session, task=bp.task, run=bp.run,
        datatype="eeg", root=cfg_paths.bids_root
    )

    # ensure dataset_description.json exists/minimal
    try:
        make_dataset_description(
            path=cfg_paths.bids_root,
            name=cfg_bids.dataset_description.get("Name", "DSI‑24 EEG"),
            dataset_type="raw",
            authors=cfg_bids.dataset_description.get("Authors"),
            overwrite=False,
        )
    except Exception as e:
        warnings_list.append(f"dataset_description: {e!r}")

    # mne-bids write
    write_raw_bids(
        raw=raw,
        bids_path=bp,
        events_data=mapped_events if mapped_events.size else None,
        event_id=event_id if event_id else None,
        overwrite=cfg_paths.overwrite,
        format="EDF",  # keep original extension
        verbose=False,
    )

    # 7) Patch EEG sidecar with required fields
    _patch_eeg_sidecar(bp, cfg_bids.eeg_reference, cfg_bids.line_freq)

    # 8) Build JSON log and write to derivatives
    dur = time.time() - t0
    log = {
        "input_file": str(edf_path),
        "bids_path": {
            "subject": bp.subject, "session": bp.session,
            "task": bp.task, "run": bp.run
        },
        "events_found": int(ev.shape[0]),
        "events_mapped": int(mapped_events.shape[0]) if mapped_events.size else 0,
        "event_id": event_id,
        "duration_sec": round(dur, 3),
        "warnings": warnings_list,
        "version": cfg_project.version,
    }

    log_name = f"sub-{bp.subject}"
    if bp.session:
        log_name += f"_ses-{bp.session}"
    log_name += f"_task-{bp.task}"
    if bp.run:
        log_name += f"_run-{bp.run}"
    log_name += "_desc-bidsify_log.json"

    logs_dir = cfg_paths.derivatives_root / "logs"
    _save_json(log, logs_dir / log_name)
    return bp, log


# ------------------------------ CLI ------------------------------

def _load_pipeline_config(path: Path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    return cfg


def _coerce_bids_config(cfg: dict) -> Tuple[PathsConfig, BidsConfig, FilenameParsingConfig, EventsGlobal, ProjectConfig]:
    paths = cfg["paths"]
    bids = cfg["bids"]
    fnp = cfg["filename_parsing"]
    evs = cfg["events"]
    prj = cfg.get("project", {"name": "neurotheque", "version": "0.1.0"})

    # resolve roots
    source_root = Path(paths["source_root"]).expanduser().resolve()
    bids_root = Path(paths["bids_root"]).expanduser().resolve()
    derivatives_root = Path(paths.get("derivatives_root") or (bids_root / "derivatives" / "neurotheque-preproc" / prj["version"]))
    derivatives_root = derivatives_root.resolve()
    overwrite = bool(paths.get("overwrite", False))

    # montage
    montage_cfg = bids.get("montage", "standard_1020")

    return (
        PathsConfig(source_root, bids_root, derivatives_root, overwrite),
        BidsConfig(
            manufacturer=bids.get("manufacturer", "Wearable Sensing"),
            manufacturersModelName=bids.get("manufacturersModelName", "DSI-24"),
            line_freq=bids.get("line_freq"),
            montage=montage_cfg,
            eeg_reference=bids.get("eeg_reference"),
            dataset_description=bids.get("dataset_description", {"Name": "Neurothèque DSI‑24 Dataset", "BIDSVersion": "1.8.0"}),
        ),
        FilenameParsingConfig(
            patterns=list(fnp.get("patterns", [])),
            fallbacks=dict(fnp.get("fallbacks", {})),
        ),
        EventsGlobal(
            trigger_channel=evs.get("trigger_channel", "Trigger"),
            default_task=str(evs.get("default_task", "gonogo")).lower(),
            tasks=dict(evs.get("tasks", {})),
        ),
        ProjectConfig(name=prj.get("name", "neurotheque"), version=prj.get("version", "0.1.0"),
                      random_seed=prj.get("random_seed"), parallel_jobs=prj.get("parallel_jobs")),
    )


def _iter_edfs(root: Path) -> Iterable[Path]:
    yield from _discover_edfs(root)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="BIDSify a folder of DSI‑24 EDF files (JSON‑driven).")
    p.add_argument("--config", required=True, help="Path to JSON config (see config_schema.json).")
    p.add_argument("--source", required=True, help="Folder containing .edf files (recursed).")
    p.add_argument("--bids-root", required=True, help="Output BIDS root.")
    args = p.parse_args(argv)

    cfg = _load_pipeline_config(Path(args.config))
    cfg["paths"]["source_root"] = args.source
    cfg["paths"]["bids_root"] = args.bids_root

    paths_cfg, bids_cfg, fn_cfg, ev_cfg, prj_cfg = _coerce_bids_config(cfg)

    # ensure derivatives root exists
    paths_cfg.derivatives_root.mkdir(parents=True, exist_ok=True)

    edfs = list(_iter_edfs(paths_cfg.source_root))
    if not edfs:
        print(f"[dsi_bids] No EDF files found under: {paths_cfg.source_root}", file=sys.stderr)
        return 2

    print(f"[dsi_bids] Found {len(edfs)} EDF files")
    ok, errors = 0, 0
    for edf in edfs:
        try:
            bp, log = bidsify_one(
                edf_path=edf,
                cfg_paths=paths_cfg,
                cfg_bids=bids_cfg,
                cfg_filename=fn_cfg,
                cfg_events=ev_cfg,
                cfg_project=prj_cfg,
            )
            print(f"[dsi_bids] Wrote: {bp.fpath if bp else '(skipped)'}")
            ok += 1
        except Exception as e:
            errors += 1
            print(f"[dsi_bids] ERROR for {edf.name}: {e}", file=sys.stderr)

    print(f"[dsi_bids] Done. ok={ok}, errors={errors}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

