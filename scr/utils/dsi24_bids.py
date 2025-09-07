#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSI-24 BIDS helpers and a CLI to BIDSify an EDF recording.

Improvements over original:
- Guarantees REQUIRED EEG sidecar fields: EEGReference, SamplingFrequency (via MNE-BIDS), PowerLineFrequency, SoftwareFilters.
- Encodes DSI-24 reference explicitly: "Pz (Common Mode Follower); ground at Fpz" (override via CLI if needed).
- Smarter channel typing and stim detection with 8-bit mask (TriggerHub).
- Unifies events from both a digital stim channel and EDF annotations (union + de-dup).
- Optional JSON-based event description map and task-based splitting by event markers.
- Safer defaults (e.g., 60 Hz mains for CA/US).
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional, List
import re
import json
from pathlib import Path

import numpy as np
import mne
from mne.io import BaseRaw
from mne.channels import make_standard_montage

# Optional dependency; required for BIDS export
try:
    from mne_bids import BIDSPath, write_raw_bids, make_dataset_description
    _HAS_MNE_BIDS = True
except Exception:
    _HAS_MNE_BIDS = False

# ---------- Constants & vendor quirks ----------
LEGACY_TO_1020 = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

# Common stimulus channel names seen in EDFs / vendor streams
STIM_NAMES = {
    "STI 014", "STI014", "STATUS", "Status", "TRIG", "TRIGGER", "TRG",
    "EVENT", "EVENTS", "MARKER", "DIN", "DIGITAL", "DIGITALIO", "STIM"
    # Note: 'PHOTO' excluded here unless it's clearly digital (see smart detection).
}

EOG_NAMES = {"HEOG", "VEOG", "EOG", "EOGL", "EOGR"}

_VENDOR_PREFIX_RE = re.compile(r"^EEG\s+(?:X\d+:)?", flags=re.IGNORECASE)
_PZ_SUFFIX_RE = re.compile(r"-PZ$", flags=re.IGNORECASE)
_X_PREFIX_RE = re.compile(r"^EEG\s+X(\d+):", flags=re.IGNORECASE)

# When folding annotation-derived events into numeric space, avoid collisions
ANNOT_CODE_OFFSET = 10000

# ---------- Channel name normalization & typing ----------
def rename_dsi_channels(raw: BaseRaw) -> Dict[str, str]:
    """Rename legacy 10-20 codes to the modern names."""
    mapping: Dict[str, str] = {ch: LEGACY_TO_1020[ch] for ch in raw.ch_names if ch in LEGACY_TO_1020}
    if mapping:
        raw.rename_channels(mapping)
    return mapping

def normalize_dsi_channel_names(raw: BaseRaw) -> Dict[str, str]:
    """
    Normalize DSI-24 names:
    - Remove leading 'EEG ' and 'EEG X#:'.
    - Remove trailing '-Pz'.
    - Keep vendor trigger labels intact (no forced 'STATUS').
    - Ensure uniqueness by suffixing duplicates.
    """
    original = list(raw.ch_names)
    taken = set(original)
    mapping: Dict[str, str] = {}

    for ch in original:
        new = ch.strip()
        x_match = _X_PREFIX_RE.match(new)
        x_idx = x_match.group(1) if x_match else None

        new = _VENDOR_PREFIX_RE.sub("", new)
        new = _PZ_SUFFIX_RE.sub("", new)

        if new == "" and x_idx is not None:
            new = f"X{x_idx}"

        if new != ch:
            candidate = new
            if candidate in taken:
                idx = 2
                while f"{candidate}_{idx}" in taken:
                    idx += 1
                candidate = f"{candidate}_{idx}"
            mapping[ch] = candidate
            taken.add(candidate)

    if mapping:
        raw.rename_channels(mapping)
    return mapping

def _looks_digital_like(raw: BaseRaw, ch_name: str, max_levels: int = 32) -> bool:
    """Heuristic: treat channel as digital if it has few discrete integer levels in a short sample."""
    try:
        picks = mne.pick_channels(raw.ch_names, include=[ch_name])
        if len(picks) != 1:
            return False
        # Sample up to ~20k samples to keep light
        n = min(raw.n_times, 20000)
        data = raw.get_data(picks=picks, start=0, stop=n).ravel()
        if data.size == 0:
            return False
        # Rescale to ints if close to integers
        vals = np.unique(np.round(data).astype(int))
        return len(vals) <= max_levels
    except Exception:
        return False

def set_channel_types(raw: BaseRaw) -> Dict[str, str]:
    """
    Set appropriate channel types for DSI-24 data.
    Maps:
      - TRIG/STIM-like labels -> stim
      - EOG/ECG labels -> eog/ecg
      - CM, X# -> misc
      - Detects digital-like channels (incl. 'PHOTO' if thresholded) -> stim
    """
    mapping: Dict[str, str] = {}
    for ch in raw.ch_names:
        cu = ch.upper()
        if cu in STIM_NAMES:
            mapping[ch] = "stim"
        elif cu in EOG_NAMES:
            mapping[ch] = "eog"
        elif "ECG" in cu:
            mapping[ch] = "ecg"
        elif cu == "CM" or cu.startswith("X"):
            mapping[ch] = "misc"
        elif cu == "PHOTO" and _looks_digital_like(raw, ch):
            mapping[ch] = "stim"
    if mapping:
        raw.set_channel_types(mapping)
    return mapping

def find_stim_channel(raw: BaseRaw) -> Optional[str]:
    """Find the primary stimulus/trigger channel, if any."""
    types = dict(zip(raw.ch_names, raw.get_channel_types()))
    # Prefer canonical names if typed as stim
    for pref in ("Trigger", "STATUS"):
        if pref in raw.ch_names and types.get(pref) == "stim":
            return pref
    # Any channel already typed as stim
    for ch, t in types.items():
        if t == "stim":
            return ch
    # Fall back to known names (not typed yet)
    for name in STIM_NAMES:
        if name in raw.ch_names:
            return name
    return None

# ---------- Events extraction & harmonization ----------
def _resolve_event_id_conflicts(
    base_id: Dict[str, int], extra_id: Dict[str, int], used_codes: Optional[set] = None
) -> Tuple[Dict[str, int], Dict[int, int]]:
    """
    Merge two event_id dicts; for conflicts in codes, remap the 'extra' dict by offset.
    Returns merged_event_id, remap_oldcode_to_newcode.
    """
    merged = dict(base_id)
    remap: Dict[int, int] = {}
    used = set(base_id.values()) if used_codes is None else set(used_codes)
    for desc, code in extra_id.items():
        if desc in merged:
            # Same description; if codes differ, prefer existing code, remap extra to it
            if merged[desc] != code:
                remap[code] = merged[desc]
            continue
        if code in used:
            new_code = code + ANNOT_CODE_OFFSET
            while new_code in used:
                new_code += 1
            merged[desc] = new_code
            remap[code] = new_code
            used.add(new_code)
        else:
            merged[desc] = code
            used.add(code)
    return merged, remap

def extract_events_and_id(
    raw: BaseRaw, stim_ch: Optional[str], mask: int = 0xFF
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Extract events from digital trigger channel and annotations, take the union,
    and return a consolidated MNE events array and event_id mapping suitable for BIDS.
    """
    events_list: List[np.ndarray] = []
    event_id: Dict[str, int] = {}

    # 1) Stim channel (bitmasked, unsigned; TriggerHub is 8-bit)
    if stim_ch:
        ev_stim = mne.find_events(
            raw, stim_channel=stim_ch, mask=mask, mask_type='and',
            shortest_event=1, uint_cast=True, consecutive=False, verbose=False
        )
        if ev_stim is not None and len(ev_stim) > 0:
            events_list.append(ev_stim)
            codes = np.unique(ev_stim[:, 2]).astype(int)
            event_id.update({f"stim_{c}": int(c) for c in codes})

    # 2) Annotations
    try:
        ev_ann, ann_id = mne.events_from_annotations(raw, verbose=False)
    except Exception:
        ev_ann, ann_id = np.empty((0, 3), dtype=int), {}
    if ev_ann.size:
        # Merge IDs, remapping annotation codes if they collide
        merged_id, remap = _resolve_event_id_conflicts(event_id, ann_id)
        if remap:
            # remap codes in ev_ann
            remapped = ev_ann.copy()
            for old, new in remap.items():
                remapped[remapped[:, 2] == old, 2] = new
            ev_ann = remapped
        event_id = merged_id
        events_list.append(ev_ann)

    if not events_list:
        return np.empty((0, 3), dtype=int), {}

    # Union & sort by sample
    events = np.vstack(events_list)
    # De-duplicate exact duplicates (sample, prev, code)
    if events.size:
        events = np.unique(events, axis=0)
        events = events[np.argsort(events[:, 0])]

    return events, event_id

def apply_standard_montage(raw: BaseRaw) -> bool:
    try:
        montage = make_standard_montage("standard_1020")
        raw.set_montage(montage)
        return True
    except Exception:
        return False

def prepare_dsi24_raw(raw: BaseRaw, apply_montage: bool = True) -> BaseRaw:
    normalize_dsi_channel_names(raw)
    rename_dsi_channels(raw)
    # set_channel_types may query data (for 'PHOTO'), so call after preload
    set_channel_types(raw)

    # Enforce a single stim channel: prefer 'Trigger' or 'STATUS'
    try:
        types = dict(zip(raw.ch_names, raw.get_channel_types()))
        stim_candidates = [ch for ch, t in types.items() if t == 'stim']
        if len(stim_candidates) > 1:
            primary = 'Trigger' if 'Trigger' in stim_candidates else (
                'STATUS' if 'STATUS' in stim_candidates else stim_candidates[0]
            )
            for ch in stim_candidates:
                if ch != primary:
                    raw.set_channel_types({ch: 'misc'})
    except Exception:
        pass

    if apply_montage:
        apply_standard_montage(raw)
    return raw

# ---------- Sidecar patching ----------
def _patch_eeg_sidecar(
    bids_path: "BIDSPath",
    eeg_reference_text: str,
    power_line_freq: Optional[int],
    recording_type: str = "continuous",
    eeg_ground_text: Optional[str] = "Fpz"
) -> None:
    """
    Ensure REQUIRED and recommended EEG metadata are present and correct.
    Adds/sets:
      - EEGReference (REQUIRED)
      - PowerLineFrequency (REQUIRED)
      - SoftwareFilters (REQUIRED => set to "n/a" if missing)
      - EEGGround (RECOMMENDED)
      - Manufacturer, TriggerChannelCount, EEG/EOG/ECG/EMG/Misc counts (RECOMMENDED)
      - RecordingType (RECOMMENDED)
    """
    bp = bids_path.copy(); bp.update(suffix="eeg", extension=".json")
    eeg_json_path = Path(bp.fpath)
    if not eeg_json_path.exists():
        return

    with eeg_json_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    # REQUIRED
    meta["EEGReference"] = eeg_reference_text
    if power_line_freq is not None:
        meta["PowerLineFrequency"] = int(power_line_freq)
    # Ensure REQUIRED SoftwareFilters is present (string "n/a" or object)
    if "SoftwareFilters" not in meta or meta["SoftwareFilters"] in (None, ""):
        meta["SoftwareFilters"] = "n/a"

    # RECOMMENDED
    if eeg_ground_text:
        meta.setdefault("EEGGround", eeg_ground_text)
    meta.setdefault("RecordingType", recording_type)
    meta.setdefault("Manufacturer", "Wearable Sensing")
    meta.setdefault("ManufacturersModelName", "DSI-24")

    # Populate channel counts from channels.tsv when available
    try:
        bp_ch = bids_path.copy(); bp_ch.update(suffix='channels', extension='.tsv')
        ch_tsv = Path(bp_ch.fpath)
        if ch_tsv.exists():
            import csv
            eeg_cnt = eog_cnt = ecg_cnt = emg_cnt = misc_cnt = trig_cnt = 0
            with ch_tsv.open('r', encoding='utf-8') as ftsv:
                rdr = csv.DictReader(ftsv, delimiter='\t')
                for row in rdr:
                    t = (row.get('type') or row.get('Type') or '').upper()
                    if t == 'EEG': eeg_cnt += 1
                    elif t == 'EOG': eog_cnt += 1
                    elif t == 'ECG': ecg_cnt += 1
                    elif t == 'EMG': emg_cnt += 1
                    elif t in ('TRIG', 'STIM'): trig_cnt += 1
                    else: misc_cnt += 1
            meta["EEGChannelCount"] = eeg_cnt
            meta["EOGChannelCount"] = eog_cnt
            meta["ECGChannelCount"] = ecg_cnt
            meta["EMGChannelCount"] = emg_cnt
            meta["MiscChannelCount"] = misc_cnt
            meta["TriggerChannelCount"] = trig_cnt
    except Exception:
        pass

    with eeg_json_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

# ---------- Task-aware writer ----------
def _write_one_bids_run(
    raw: BaseRaw,
    bids_root: Path,
    subject: str,
    session: str,
    task: str,
    run: Optional[str],
    events: np.ndarray,
    event_id: Dict[str, int],
    line_freq: Optional[int],
    overwrite: bool,
):
    bids_path = BIDSPath(
        subject=subject, session=session, task=task, run=run,
        datatype="eeg", root=str(bids_root)
    )
    raw_to_write = raw.copy()
    try:
        raw_to_write.set_annotations(mne.Annotations([], [], []))
    except Exception:
        pass    
    
    write_raw_bids(
        raw_to_write,
        bids_path,
        overwrite=overwrite,
        allow_preload=True,
        format="BrainVision",
        events=events if events.size else None,
        event_id=event_id if events.size else None,
        verbose=False,
    )
    # Note: DSI-24 typical ref/ground; override with CLI if needed.
    _patch_eeg_sidecar(
        bids_path,
        eeg_reference_text="Pz (Common Mode Follower); ground at Fpz",
        power_line_freq=line_freq,
        recording_type="continuous",
        eeg_ground_text="Fpz"
    )
    return bids_path

def bidsify_edf(
    edf_path: Path | str,
    bids_root: Path | str,
    subject: str,
    session: str,
    task: str,
    run: Optional[str] = None,
    line_freq: Optional[int] = 60,         # default 60 (CA/US); override if 50
    overwrite: bool = True,
    apply_montage: bool = True,
    event_map_json: Optional[Path | str] = None,
    task_splits_json: Optional[Path | str] = None,
) -> List["BIDSPath"]:
    """
    Read an EDF, normalize/annotate channels, extract events, and write BIDS.
    Optionally split the recording into multiple tasks using an event-driven scheme.

    Returns the list of BIDSPath objects written.
    """
    if not _HAS_MNE_BIDS:
        raise RuntimeError("mne-bids is required. Please install it (pip install mne-bids).")

    edf_path = Path(edf_path)
    bids_root = Path(bids_root)

    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)

    # Set mains frequency if known (50 or 60)
    if line_freq is not None:
        raw.info["line_freq"] = int(line_freq)

    # Prepare channels (naming/types/montage)
    prepare_dsi24_raw(raw, apply_montage=apply_montage)

    # Extra normalization pass if vendor labels survived
    upper_names = [n.upper() for n in raw.ch_names]
    if any(n.startswith("EEG ") or n.endswith("-PZ") for n in upper_names):
        normalize_dsi_channel_names(raw)
        set_channel_types(raw)

    # Events (union of stim & annotations)
    stim_ch = find_stim_channel(raw)
    events, event_id = extract_events_and_id(raw, stim_ch, mask=0xFF)

    # Optional event description mapping (JSON): {"123": "go", "124": "nogo", "onset": 801, ...}
    # MNE-BIDS expects event_id: {description -> code}; if provided, enforce it.
    if event_map_json:
        with open(event_map_json, "r", encoding="utf-8") as f:
            user_map = json.load(f)
        # Normalize keys to int codes; values to descriptions (strings)
        norm_map: Dict[str, int] = {}
        for k, v in user_map.items():
            # accept numeric code as string or int; or description->code
            if isinstance(k, str) and k.isdigit():
                norm_map[str(v)] = int(k)
            elif isinstance(v, int):
                norm_map[str(k)] = int(v)
            else:
                raise ValueError("event_map_json must map code<->description (use int codes).")
        # Remap codes in events if user_map changes them
        merged_id, remap = _resolve_event_id_conflicts(norm_map, event_id)
        if events.size and remap:
            ev = events.copy()
            for old, new in remap.items():
                ev[ev[:, 2] == old, 2] = new
            events = ev
        event_id = merged_id

    # Special-case: five-point task (alternate onset / first_touch)
    if task.lower() in {"5pt", "fivepoint", "five_point", "five-point"} and events.size >= 3:
        mid = events[1:-1]
        onset_list, resp_list = [], []
        for i, e in enumerate(mid):
            if i % 2 == 0:
                onset_list.append([int(e[0]), 0, 801])
            else:
                resp_list.append([int(e[0]), 0, 802])
        n = min(len(onset_list), len(resp_list))
        onset_arr = np.array(onset_list[:n], dtype=int)
        resp_arr = np.array(resp_list[:n], dtype=int)
        events = np.vstack([onset_arr, resp_arr]) if n > 0 else np.empty((0, 3), dtype=int)
        if events.size:
            events = events[np.argsort(events[:, 0])]
        event_id = {"onset": 801, "first_touch": 802}

    # Convert events to annotations (union) so that browsing the BIDSified raw in MNE shows them
    if events.size:
        # If event_id keys are descriptions, write them to annotations
        # Build description for each code
        inv = {code: desc for desc, code in event_id.items()}
        descs = {int(code): inv.get(int(code), f"stim_{int(code)}") for code in np.unique(events[:, 2])}
        annots = mne.annotations_from_events(
            events=events,
            sfreq=raw.info["sfreq"],
            event_desc=descs,
        )
        if raw.annotations is not None and len(raw.annotations) > 0:
            raw.set_annotations(raw.annotations + annots)
        else:
            raw.set_annotations(annots)

    # Optional task splitting by event markers, e.g.,
    # task_splits_json = {"gonogo":{"start":101,"stop":102}, "stroop":{"start":201,"stop":202}}
    written: List[BIDSPath] = []
    if task_splits_json:
        with open(task_splits_json, "r", encoding="utf-8") as f:
            splits = json.load(f)
        run_idx = 1
        for tname, markers in splits.items():
            start_code = int(markers["start"]); stop_code = int(markers["stop"])
            # find first occurrence of start and next occurrence of stop after it
            idx_start = np.where(events[:, 2] == start_code)[0]
            idx_stop = np.where(events[:, 2] == stop_code)[0]
            if idx_start.size == 0 or idx_stop.size == 0:
                continue
            s_sample = events[idx_start[0], 0]
            e_sample = events[idx_stop[idx_stop > idx_start[0]][0], 0]
            tmin = s_sample / raw.info["sfreq"]
            tmax = e_sample / raw.info["sfreq"]
            rseg = raw.copy().crop(tmin=tmin, tmax=tmax, include_tmax=False)
            # Keep only events falling into this window and shift times
            mask = (events[:, 0] >= s_sample) & (events[:, 0] < e_sample)
            ev_seg = events[mask].copy()
            ev_seg[:, 0] -= s_sample
            written.append(_write_one_bids_run(
                rseg, bids_root, subject, session, tname, f"{run_idx:02d}",
                ev_seg, event_id, line_freq, overwrite
            ))
            run_idx += 1
    else:
        written.append(_write_one_bids_run(
            raw, bids_root, subject, session, task, run, events, event_id, line_freq, overwrite
        ))

    # Ensure dataset_description.json exists (minimal ok)
    try:
        make_dataset_description(path=bids_root, name="DSI-24 EEG dataset", dataset_type="raw", overwrite=False)
    except Exception:
        pass

    return written

# ---------- CLI ----------
def _build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(description="BIDSify a DSI-24 EDF recording.")
    # Positional style (EDF, BIDS root)
    p.add_argument("edf", nargs="?", help="Path to input EDF file (positional)")
    p.add_argument("bids_root", nargs="?", help="BIDS root directory (positional)")
    # Optional flags (aliases supported)
    p.add_argument("--edf", dest="edf_opt", help="Path to input EDF file")
    p.add_argument("--bids-root", dest="bids_root_opt", help="BIDS root directory")
    # Subject/session/task/run
    p.add_argument("--sub", "--subject", dest="sub", required=True, help="Subject label (e.g., 01)")
    p.add_argument("--ses", "--session", dest="ses", required=True, help="Session label (e.g., 001)")
    p.add_argument("--task", required=True, help="Task name (e.g., gonogo)")
    p.add_argument("--run", default=None, help="Run label (e.g., 01)")
    # Power line frequency
    p.add_argument("--line-freq", "--powerline", dest="line_freq", type=int, default=60,
                   help="Power line frequency (50 or 60)")
    # Overwrite controls
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    p.add_argument("--no-overwrite", action="store_true", help="Do not overwrite existing files")
    p.add_argument("--no-montage", action="store_true", help="Disable applying standard_1020 montage")
    # Event handling
    p.add_argument("--event-map", type=str, default=None,
                   help="JSON mapping event codes <-> descriptions to force trial_type labels")
    # Task splitting
    p.add_argument("--task-splits", type=str, default=None,
                   help="JSON specifying task segments by start/stop codes")
    return p

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    edf_path = args.edf_opt or args.edf
    bids_root = args.bids_root_opt or args.bids_root
    if not edf_path or not bids_root:
        parser.error("Provide EDF and BIDS root either positionally or via --edf/--bids-root")

    overwrite = True if args.overwrite else (not args.no_overwrite)

    bids_paths = bidsify_edf(
        edf_path=edf_path,
        bids_root=bids_root,
        subject=args.sub,
        session=args.ses,
        task=args.task,
        run=args.run,
        line_freq=args.line_freq,
        overwrite=overwrite,
        apply_montage=not args.no_montage,
        event_map_json=args.event_map,
        task_splits_json=args.task_splits,
    )
    for bp in bids_paths:
        print(f"Wrote BIDS to: {bp.directory}")

if __name__ == "__main__":
    main()

__all__ = [
    "rename_dsi_channels",
    "normalize_dsi_channel_names",
    "set_channel_types",
    "find_stim_channel",
    "extract_events_and_id",
    "prepare_dsi24_raw",
    "bidsify_edf",
]
