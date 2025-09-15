#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: BIDSify the Mario EEG dataset (DSI-24) and integrate behavior.

- Converts all EDFs under --eeg-root to BIDS using scr.utils.dsi24_bids.bidsify_edf
- Optionally merges behavior events (fixation, game start, keypress) into BIDS events.tsv
- Updates dataset_description.json with a dataset name and optional authors

Usage
  python examples/bidsify_mario.py \
    --eeg-root data/mario/eeg \
    --behav-root data/mario/behav \
    --bids-root derivatives/bids \
    --subject 01

Notes
- Behavior logs are optional; when present, they are appended to events.tsv with
  trial_type labels such as 'fixation_dot', 'gym-retro_game', 'keypress_l/r/b'.
- This script does not remove the original stim-derived events; it only adds behavior rows.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import mne
from mne_bids import BIDSPath, make_dataset_description

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scr.utils.dsi24_bids import bidsify_edf


def _glob_edfs(eeg_root: Path, subject: Optional[str]) -> List[Path]:
    pats = ["*.edf"] if subject is None else [f"sub-{subject}_*.edf"]
    edfs: List[Path] = []
    for pat in pats:
        edfs.extend(sorted(eeg_root.glob(pat)))
    return edfs


def _parse_entities_from_name(name: str) -> Optional[Tuple[str, str, str]]:
    # Expect: sub-01_ses-001_task-*_run-01_raw.edf
    try:
        parts = name.split("_")
        sub = parts[0].split("-")[1]
        ses = parts[1].split("-")[1]
        run = [p for p in parts if p.startswith("run-")][0].split("-")[1]
        return sub, ses, run
    except Exception:
        return None


def _find_behavior_tsv(behav_root: Path, sub: str, ses: str, run: str) -> Optional[Path]:
    folder = behav_root / f"sub-{sub}" / f"ses-{ses}"
    if not folder.exists():
        return None
    # Pattern: sub-01_ses-003_YYYYmmdd-HHMMSS_task-mario_run-01_events.tsv
    cands = sorted(folder.glob(f"sub-{sub}_ses-{ses}_*_task-mario_run-{run}_events.tsv"))
    return cands[0] if cands else None


def _append_behavior_events_to_bids(events_tsv: Path, raw_like: Path, behav_tsv: Path) -> int:
    """
    Append behavior events to an existing BIDS events.tsv. Returns number of rows appended.
    raw_like is a BrainVision .vhdr path used to read sfreq for sample computation.
    """
    # Load sfreq from the written BIDS EEG file
    raw = mne.io.read_raw_brainvision(str(raw_like), preload=False, verbose=False)
    sfreq = float(raw.info["sfreq"]) if raw.info.get("sfreq") else 300.0

    # Read existing events.tsv
    with events_tsv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        fieldnames = reader.fieldnames or ["onset", "duration", "trial_type", "value", "sample"]

    # Prepare behavior rows
    added: List[Dict[str, str]] = []
    with behav_tsv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            tri = (r.get("trial_type") or "").strip()
            onset = float(r.get("onset") or 0.0)
            dur = float(r.get("duration") or 0.0)
            # Keypress specialization
            if tri == "keypress":
                key = (r.get("key") or "").strip().lower()
                if key in {"l", "r", "b"}:
                    tri = f"keypress_{key}"
            # Compute sample (best-effort, aligned to EEG start)
            sample = int(round(onset * sfreq))
            if sample < 0:
                continue
            added.append({
                "onset": f"{onset}",
                "duration": f"{dur}",
                "trial_type": tri or "behavior",
                "value": "n/a",
                "sample": f"{sample}",
            })

    if not added:
        return 0

    # Merge, sort by onset, and write back
    merged = rows + added
    def _onset_key(d: Dict[str, str]) -> float:
        try:
            return float(d.get("onset") or 0.0)
        except Exception:
            return 0.0
    merged.sort(key=_onset_key)

    # Ensure field order
    want = ["onset", "duration", "trial_type", "value", "sample"]
    for w in want:
        if w not in fieldnames:
            fieldnames.append(w)

    with events_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for r in merged:
            writer.writerow({k: r.get(k, "n/a") for k in fieldnames})

    return len(added)


def _standardize_and_thin_events(events_tsv: Path, raw_like: Path, thin_stim2_ms: Optional[float]) -> Tuple[int, int]:
    """
    Ensure standard columns and optionally thin dense stim_2 events by time.
    Returns (kept_rows, removed_rows).
    """
    # Load sfreq
    raw = mne.io.read_raw_brainvision(str(raw_like), preload=False, verbose=False)
    sfreq = float(raw.info["sfreq"]) if raw.info.get("sfreq") else 300.0

    # Read rows permissively
    rows: List[Dict[str, str]] = []
    with events_tsv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(dict(r))

    # Normalize core fields
    def _get_onset(r: Dict[str, str]) -> Optional[float]:
        for key in ("onset", "Onset"):
            try:
                return float(r.get(key))
            except Exception:
                pass
        try:
            samp = int(r.get("sample"))
            return samp / sfreq
        except Exception:
            return None

    std: List[Dict[str, str]] = []
    for r in rows:
        onset = _get_onset(r)
        if onset is None:
            continue
        duration = r.get("duration") or "0.0"
        trial_type = (r.get("trial_type") or "").strip()
        value = r.get("value") or "n/a"
        try:
            sample = int(r.get("sample"))
        except Exception:
            sample = int(round(onset * sfreq))
        std.append({
            "onset": f"{onset}",
            "duration": duration,
            "trial_type": trial_type,
            "value": value,
            "sample": f"{sample}",
        })

    removed = 0
    if thin_stim2_ms and thin_stim2_ms > 0:
        refractory = int(round(thin_stim2_ms * 1e-3 * sfreq))
        last_kept = None
        thinned: List[Dict[str, str]] = []
        for r in std:
            if r["trial_type"] != "stim_2":
                thinned.append(r)
                continue
            s = int(r["sample"]) if r.get("sample") else int(round(float(r["onset"]) * sfreq))
            if last_kept is None or (s - last_kept) >= refractory:
                thinned.append(r)
                last_kept = s
            else:
                removed += 1
        std = thinned

    # De-duplicate by (sample, value)
    seen = set()
    dedup: List[Dict[str, str]] = []
    for r in std:
        key = (int(r["sample"]), r["value"]) if r.get("sample") else (r["onset"], r["value"])
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        dedup.append(r)
    std = dedup

    fieldnames = ["onset", "duration", "trial_type", "value", "sample"]
    with events_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for r in std:
            writer.writerow({k: r.get(k, "n/a") for k in fieldnames})

    return len(std), removed


def _ensure_dataset_description(bids_root: Path, name: str, authors: Optional[List[str]] = None) -> None:
    # Create minimal file if missing
    try:
        make_dataset_description(path=str(bids_root), name=name, dataset_type="raw", authors=authors, overwrite=False)
    except Exception:
        pass
    # Update Name/Authors unconditionally
    dd = bids_root / "dataset_description.json"
    if dd.exists():
        try:
            meta = json.loads(dd.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        meta["Name"] = name
        if authors is not None:
            meta["Authors"] = authors
        dd.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="BIDSify Mario EEG dataset and integrate behavior")
    ap.add_argument("--eeg-root", default="data/mario/eeg", help="Folder with EDF files")
    ap.add_argument("--behav-root", default="data/mario/behav", help="Folder with behavioral TSVs")
    ap.add_argument("--bids-root", default="derivatives/bids", help="Output BIDS root")
    ap.add_argument("--subject", default=None, help="Only process this subject label (e.g., 01)")
    ap.add_argument("--task", default="mario", help="BIDS task label")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    ap.add_argument("--no-behavior", action="store_true", help="Do not integrate behavior events")
    ap.add_argument("--dataset-name", default="Neurotheque Mario EEG", help="dataset_description Name")
    ap.add_argument("--thin-stim2-ms", type=float, default=50.0, help="Keep at most one stim_2 every N milliseconds")
    args = ap.parse_args(list(argv) if argv is not None else None)

    eeg_root = Path(args.eeg_root)
    behav_root = Path(args.behav_root)
    bids_root = Path(args.bids_root)
    bids_root.mkdir(parents=True, exist_ok=True)

    _ensure_dataset_description(bids_root, name=args.dataset_name, authors=None)

    edfs = _glob_edfs(eeg_root, args.subject)
    if not edfs:
        print(f"No EDFs found under {eeg_root}")
        return 1

    ok = 0
    for edf in edfs:
        ents = _parse_entities_from_name(edf.name)
        if not ents:
            print(f"[skip] Unparsable name: {edf.name}")
            continue
        sub, ses, run = ents
        # Convert to BIDS (or overwrite if requested)
        try:
            bps = bidsify_edf(
                edf_path=str(edf),
                bids_root=str(bids_root),
                subject=sub,
                session=ses,
                task=args.task,
                run=run,
                line_freq=60,
                overwrite=bool(args.overwrite),
                apply_montage=True,
                event_map_json=None,
                task_splits_json=None,
            )
        except Exception as e:
            print(f"[err] {edf.name}: {e}")
            continue

        # Append behavior events when available
        if not args.no_behavior and bps:
            bp = bps[0]
            # Validate directory exists
            eeg_dir = Path(bp.directory)
            vhdr = eeg_dir / f"sub-{sub}_ses-{ses}_task-{args.task}_run-{run}_eeg.vhdr"
            evts = eeg_dir / f"sub-{sub}_ses-{ses}_task-{args.task}_run-{run}_events.tsv"
            if vhdr.exists() and evts.exists():
                behav_tsv = _find_behavior_tsv(behav_root, sub, ses, run)
                if behav_tsv and behav_tsv.exists():
                    try:
                        added = _append_behavior_events_to_bids(evts, vhdr, behav_tsv)
                        print(f"[ok] {edf.name}: +{added} behavior events")
                    except Exception as e:
                        print(f"[warn] Failed to append behavior for {edf.name}: {e}")
                else:
                    print(f"[info] No behavior TSV for {edf.name}")

            # Standardize and thin dense stim_2 events
            if vhdr.exists() and evts.exists():
                try:
                    kept, removed = _standardize_and_thin_events(evts, vhdr, args.thin_stim2_ms)
                    if removed:
                        print(f"[ok] {edf.name}: thinned events (kept={kept}, removed={removed})")
                except Exception as e:
                    print(f"[warn] Failed to thin events for {edf.name}: {e}")

        ok += 1
    print(f"Done. Converted {ok} file(s) into BIDS at {bids_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
