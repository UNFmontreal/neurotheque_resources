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
from typing import Dict, Tuple, Optional, List, Union
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shutil
import pickle
import importlib.util
import mne
from mne.io import BaseRaw
from mne.channels import make_standard_montage

# Optional dependency; required for BIDS export
try:
    from mne_bids import BIDSPath, write_raw_bids, make_dataset_description
    _HAS_MNE_BIDS = True
except Exception:
    _HAS_MNE_BIDS = False

# Note: Mario-specific behavior alignment step is imported lazily inside
# bidsify_edf to surface any import errors directly when used.
MarioEventAlignmentStep = None  # type: ignore
_HAS_MARIO_ALIGN = None  # determined lazily

# Optional import: JSON-driven event mapping (new unified logic)
try:
    from event_mapping import map_events_from_config  # type: ignore
except Exception:  # pragma: no cover - keep backward compatible without root import
    map_events_from_config = None  # type: ignore

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

# ---------- Filename & behavior helpers ----------


def extract_bids_entities_from_name(name: str) -> Tuple[str, str, Optional[str], Optional[str]]:
    """Parse BIDS entities from a filename or stem."""
    stem = Path(name).name
    if '.' in stem:
        stem = stem.split('.', 1)[0]
    parts = stem.split('_')
    subject = session = task = run = None
    for part in parts:
        if part.startswith('sub-') and subject is None:
            subject = part.split('-', 1)[1]
        elif part.startswith('ses-') and session is None:
            session = part.split('-', 1)[1]
        elif part.startswith('task-') and task is None:
            task = part.split('-', 1)[1]
        elif part.startswith('run-') and run is None:
            run = part.split('-', 1)[1]
    if not subject or not session:
        raise ValueError(f'Cannot parse subject/session from name: {name}')
    return subject, session, task, run


def parse_mario_bk2_name(bk2_name: str) -> Tuple[str, str, str]:
    base = Path(bk2_name).name
    lvl = re.search(r'Level(\d+)-(\d+)', base)
    world, level = (lvl.group(1), lvl.group(2)) if lvl else ('0', '0')
    rep_match = re.search(r'_(\d+)(?:\.[A-Za-z0-9]+)?$', base)
    rep = rep_match.group(1) if rep_match else '000'
    return world, level, rep


def find_mario_behavior_tsv(behav_root: Path, subject: str, session: str, task: Optional[str], run: Optional[str]) -> Optional[Path]:
    sub_dir = behav_root / f'sub-{subject}' / f'ses-{session}'
    if not sub_dir.exists():
        return None
    candidates = []
    for cand in sub_dir.glob('*_events.tsv'):
        name = cand.name
        if f'sub-{subject}' not in name or f'ses-{session}' not in name:
            continue
        if task and f'task-{task}'.lower() not in name.lower():
            continue
        if run and f'run-{run}' not in name:
            continue
        candidates.append(cand)
    if not candidates:
        return None
    return sorted(candidates)[0]


_MARIO_ANNOTATIONS_MODULE = None


def _load_generate_annotations_module(module_path: Optional[Path] = None):
    """Lazily load the legacy generate_annotations helpers if available."""
    global _MARIO_ANNOTATIONS_MODULE
    if _MARIO_ANNOTATIONS_MODULE is not None:
        return _MARIO_ANNOTATIONS_MODULE

    default_path = Path('data/mario_eeg/mario_eeg/code/generate_annotations.py')
    path = Path(module_path) if module_path is not None else default_path
    if not path.exists():
        _MARIO_ANNOTATIONS_MODULE = False
        return None
    try:
        spec = importlib.util.spec_from_file_location('_mario_generate_annotations', str(path))
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
            _MARIO_ANNOTATIONS_MODULE = module
            return module
    except Exception as err:
        print(f"[Mario] Warning: failed to load generate_annotations module: {err}")
    _MARIO_ANNOTATIONS_MODULE = False
    return None


def process_mario_behavior_assets(
    behavior_tsv: Path,
    bids_path: "BIDSPath",
    behavior_source_dir: Optional[Path] = None,
    copy_bk2: bool = True,
) -> Optional[Path]:
    if not behavior_tsv.exists():
        return None
    df = pd.read_csv(behavior_tsv, sep='\t')
    if 'trial_type' not in df.columns:
        return None

    beh_sub, beh_ses, beh_task, beh_run = extract_bids_entities_from_name(behavior_tsv.name)
    subject = bids_path.subject or beh_sub
    session = bids_path.session or beh_ses
    task = bids_path.task or beh_task or 'mario'
    run_label = bids_path.run or beh_run or '01'

    eeg_dir = Path(bids_path.directory)
    gamelog_dir = eeg_dir.parent / 'gamelogs'
    gamelog_dir.mkdir(parents=True, exist_ok=True)

    if 'level' not in df.columns:
        df['level'] = pd.NA
    if 'stim_file' not in df.columns:
        df['stim_file'] = pd.NA
    stim_col = 'stim_file'
    source_dir = Path(behavior_source_dir) if behavior_source_dir is not None else None
    invalid_tokens = {'', 'nan', 'none', 'null', '<na>', 'n/a'}

    runvars: List[dict] = []
    rep_event_rows: List[int] = []

    if copy_bk2 and stim_col:
        for idx, row in df.iterrows():
            trial_type = str(row['trial_type']).strip().lower()
            if trial_type != 'gym-retro_game':
                continue
            stim_value = str(row.get(stim_col) or '').strip()
            if stim_value.lower() in invalid_tokens:
                runvars.append({})
                rep_event_rows.append(idx)
                continue
            bk2_name = Path(stim_value).name
            source_candidates = []
            stim_path = Path(stim_value)
            if stim_path.exists():
                source_candidates.append(stim_path)
            if source_dir is not None:
                source_candidates.append(source_dir / bk2_name)
            source_candidates.append(behavior_tsv.parent / bk2_name)
            source_path = next((cand for cand in source_candidates if cand and cand.exists()), None)
            if source_path is None:
                print(f'[Mario] BK2 not found for {bk2_name}; leaving stim_file untouched')
                runvars.append({})
                rep_event_rows.append(idx)
                continue
            world, level, rep = parse_mario_bk2_name(bk2_name)
            dest_base = f'sub-{subject}_ses-{session}_task-{task}_run-{run_label}_level-w{world}l{level}_rep-{rep}'
            dest_path = gamelog_dir / f'{dest_base}.bk2'
            shutil.copy2(source_path, dest_path)
            rel_parts = dest_path.parts[-4:] if len(dest_path.parts) >= 4 else dest_path.parts
            df.at[idx, stim_col] = '/'.join(rel_parts)

            for ext in ('.json', '.pkl', '.npz', '.mp4'):
                sidecar_src = source_path.with_suffix(ext)
                if sidecar_src.exists():
                    try:
                        shutil.copy2(sidecar_src, dest_path.with_suffix(ext))
                    except Exception as err:
                        print(f'[Mario] Warning: failed to copy {sidecar_src.name}: {err}')

            repvars = {}
            pkl_source = source_path.with_suffix('.pkl')
            if pkl_source.exists():
                try:
                    with open(pkl_source, 'rb') as f:
                        repvars = pickle.load(f)
                    df.at[idx, 'level'] = repvars.get('level', f'w{world}l{level}')
                except Exception as err:
                    print(f'[Mario] Warning: failed to load {pkl_source.name}: {err}')
                    repvars = {}
            else:
                df.at[idx, 'level'] = f'w{world}l{level}'

            runvars.append(repvars)
            rep_event_rows.append(idx)

    beh_events_bp = bids_path.copy()
    beh_events_bp.update(suffix='events', extension='.tsv', description='behavior', check=False)
    df.to_csv(beh_events_bp.fpath, sep='\t', index=False)

    module = _load_generate_annotations_module()
    if module and runvars and stim_col in df.columns:
        try:
            base_events = df.loc[rep_event_rows, ['trial_type', 'onset', 'level', stim_col]].reset_index(drop=True)
            annotated_df = module.create_runevents(runvars, base_events)
            if annotated_df is not None and not annotated_df.empty:
                annot_bp = bids_path.copy()
                annot_bp.update(suffix='events', extension='.tsv', description='annotated', check=False)
                annotated_df.to_csv(annot_bp.fpath, sep='\t', index=False)
        except Exception as err:
            print(f'[Mario] Warning: failed to build annotated events: {err}')

    return Path(beh_events_bp.fpath)



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

def _patch_coordsystem_to_template(bids_path: "BIDSPath", space_label: str = "standard_1020") -> None:
    """
    Ensure coordsystem/electrodes describe template 10-20 positions (not digitized) and
    rename any CapTrak-labeled files in this subject/session EEG directory to the given space label.

    This handles both session-level files (no task/run) and run-level files.
    """
    try:
        eeg_dir = Path(bids_path.directory)
        if not eeg_dir.exists():
            return

        # Update any CapTrak coordsystem JSONs found in this directory
        for cs_path in eeg_dir.glob("*_space-CapTrak_coordsystem.json"):
            try:
                with cs_path.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["EEGCoordinateSystem"] = "Other"
                meta["EEGCoordinateUnits"] = "m"
                desc = meta.get("EEGCoordinateSystemDescription", "")
                note = (
                    "Template MNE 'standard_1020' 10-20 positions (not digitized). "
                    "RAS head axes: LPA-RPA = +X, nasion = +Y, vertex = +Z."
                )
                if not desc:
                    meta["EEGCoordinateSystemDescription"] = note
                elif "standard_1020" not in desc:
                    meta["EEGCoordinateSystemDescription"] = f"{desc} Template: {note}"
                meta["AnatomicalLandmarkCoordinateSystem"] = "Other"
                meta["AnatomicalLandmarkCoordinateUnits"] = "m"
                with cs_path.open("w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
            except Exception:
                continue

            # Rename to requested space label
            new_cs_path = cs_path.with_name(cs_path.name.replace("space-CapTrak", f"space-{space_label}"))
            try:
                if not new_cs_path.exists():
                    cs_path.replace(new_cs_path)
            except Exception:
                pass

        # Rename electrodes TSVs similarly
        for elec_path in eeg_dir.glob("*_space-CapTrak_electrodes.tsv"):
            new_elec_path = elec_path.with_name(elec_path.name.replace("space-CapTrak", f"space-{space_label}"))
            try:
                if not new_elec_path.exists():
                    elec_path.replace(new_elec_path)
            except Exception:
                pass
    except Exception:
        # Non-fatal; continue without raising
        pass

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
    
    # Guard against any out-of-range event onsets (can occur with EDF annotations)
    if events is not None and getattr(events, 'size', 0):
        mask = (events[:, 0] >= 0) & (events[:, 0] < raw.n_times)
        if not np.all(mask):
            events = events[mask]

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
    # Ensure coordsystem/electrodes reflect template 10-20 positions (not CapTrak)
    try:
        _patch_coordsystem_to_template(bids_path, space_label="standard_1020")
    except Exception:
        pass
    return bids_path

def bidsify_edf(
    edf_path: Union[Path, str],
    bids_root: Union[Path, str],
    subject: str,
    session: str,
    task: str,
    run: Optional[str] = None,
    line_freq: Optional[int] = 60,         # default 60 (CA/US); override if 50
    overwrite: bool = True,
    apply_montage: bool = True,
    event_map_json: Optional[Union[Path, str]] = None,
    task_splits_json: Optional[Union[Path, str]] = None,
    # Mario-specific optional inputs: if provided, build behavior-aligned events
    mario_behav_tsv: Optional[Union[Path, str]] = None,
    mario_align_params: Optional[dict] = None,
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

    # Events: Prefer Mario behavior-aligned events when a behavior TSV is provided; otherwise use
    # union of stim & annotations (dropping annotation-derived duplicates).
    stim_ch = find_stim_channel(raw)
    events: np.ndarray
    event_id: Dict[str, int]

    used_mario = False
    if mario_behav_tsv is not None:
        # Lazy import with helpful error if it fails
        global MarioEventAlignmentStep, _HAS_MARIO_ALIGN
        if MarioEventAlignmentStep is None or _HAS_MARIO_ALIGN in (None, False):
            try:
                from scr.steps.triggers_mario import MarioEventAlignmentStep as _Mario  # type: ignore
                MarioEventAlignmentStep = _Mario  # cache
                _HAS_MARIO_ALIGN = True
            except Exception as e:  # pragma: no cover
                _HAS_MARIO_ALIGN = False
                raise RuntimeError(f"MarioEventAlignmentStep import failed: {e}") from e
        # Build alignment params with sensible defaults, allowing overrides
        align_params = {
            'behav_tsv_path': str(mario_behav_tsv),
            'stim_channel': stim_ch or 'Trigger',
            'frame_bit_mask': 2,
            'gap_threshold_s': 0.5,
            'max_match_dist_s': 3.0,
            'drift_threshold_ms_per_min': 1.0,
            'prefer_meta_fps': True,
            'plot': False,
            'summary_dir': str(Path(bids_root) / 'mario_alignment_summary' / f'sub-{subject}' / f'ses-{session}' / (f'run-{run}' if run else 'run-01')),
        }
        if mario_align_params:
            align_params.update(mario_align_params)
            # Ensure stim channel stays consistent with the data
            align_params['stim_channel'] = align_params.get('stim_channel') or (stim_ch or 'Trigger')
        # Run alignment; step writes results into raw.info['temp']
        MarioEventAlignmentStep(params=align_params).run(raw)
        temp = raw.info.get('temp', {}) if hasattr(raw, 'info') else {}
        evb = temp.get('behavior_events')
        eidb = temp.get('behavior_event_id') or {}
        if isinstance(evb, np.ndarray) and evb.size:
            events = evb.astype(int)
            event_id = {str(k): int(v) for k, v in eidb.items()}
            used_mario = True
        else:
            # Fallback to generic extraction if behavior alignment produced nothing
            events = np.empty((0, 3), dtype=int)
            event_id = {}

    if not used_mario:
        events, event_id = extract_events_and_id(raw, stim_ch, mask=0xFF)
        # Prefer digital stim channel codes; remove annotation-offset codes (>= ANNOT_CODE_OFFSET)
        if events.size:
            keep = events[:, 2] < ANNOT_CODE_OFFSET
            if not np.all(keep):
                events = events[keep]
            # De-duplicate by (sample, code) ignoring the 'previous' column
            if events.size:
                key = events[:, [0, 2]]
                _, uniq_idx = np.unique(key, axis=0, return_index=True)
                events = events[np.sort(uniq_idx)]
        if event_id:
            present_codes = set(map(int, np.unique(events[:, 2]).tolist())) if events.size else set()
            event_id = {k: v for k, v in event_id.items() if int(v) < ANNOT_CODE_OFFSET and (not present_codes or int(v) in present_codes)}

    # Optional JSON-driven mapping using the shared event_mapping module.
    # Supports either a minimal "task_config" object with an "event_map" key
    # or the legacy simple code<->description dict.
    if event_map_json and (not used_mario):
        with open(event_map_json, "r", encoding="utf-8") as f:
            user_map = json.load(f)

        # Preferred: if new-style task config is provided, use it to map events.
        if isinstance(user_map, dict) and ("event_map" in user_map or "debounce_ms" in user_map or "onset_shift_sec" in user_map or "events" in user_map or "tasks" in user_map):
            if map_events_from_config is None:
                raise RuntimeError("event_mapping module not available to apply JSON-driven mapping.")
            task_cfg = None
            # Full pipeline config style: {"events": {"default_task":..., "tasks": {<task>: {event_map...}}}}
            if "events" in user_map and isinstance(user_map["events"], dict):
                evs = user_map["events"]
                tasks = evs.get("tasks") or {}
                # Prefer the requested task; fallback to default_task
                task_cfg = tasks.get(task.lower()) or tasks.get(str(evs.get("default_task", "")).lower())
            # Or direct task-config dict with event_map at top-level
            if task_cfg is None and "event_map" in user_map:
                task_cfg = user_map

            if task_cfg is None or "event_map" not in task_cfg:
                raise ValueError("Provided JSON does not contain a usable 'event_map' configuration.")

            mapped_events, new_event_id = map_events_from_config(raw, events, task_cfg)
            events = mapped_events
            event_id = new_event_id
        else:
            # Legacy: {code->desc} or {desc->code}. Coerce to {desc->code}.
            norm_map: Dict[str, int] = {}
            for k, v in user_map.items():
                if isinstance(k, str) and k.isdigit():
                    norm_map[str(v)] = int(k)
                elif isinstance(v, int):
                    norm_map[str(k)] = int(v)
                else:
                    raise ValueError("event_map_json must map code<->description (use int codes).")
            merged_id, remap = _resolve_event_id_conflicts(norm_map, event_id)
            if events.size and remap:
                ev = events.copy()
                for old, new in remap.items():
                    ev[ev[:, 2] == old, 2] = new
                events = ev
            event_id = merged_id

    # Remove per-task special-casing when JSON mapping is provided; otherwise keep legacy fallback
    if (not event_map_json) and task.lower() in {"5pt", "fivepoint", "five_point", "five-point"} and events.size >= 3:
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
        # Some EDFs carry existing annotations with a non-None orig_time.
        # Adding annotations requires identical orig_time; if they differ, fall back to replacing.
        try:
            if raw.annotations is not None and len(raw.annotations) > 0:
                raw.set_annotations(raw.annotations + annots)
            else:
                raw.set_annotations(annots)
        except ValueError as e:
            if "orig_time should be the same" in str(e):
                # Replace with new annotations instead of union to avoid failure
                raw.set_annotations(annots)
            else:
                raise

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

def bidsify_sourcedata(
    sourcedata_root: Union[Path, str],
    bids_root: Union[Path, str],
    mario_mode: bool = False,
    behavior_root: Optional[Union[Path, str]] = None,
    eeg_root: Optional[Union[Path, str]] = None,
    line_freq: Optional[int] = 60,
    apply_montage: bool = True,
    overwrite: bool = True,
    dry_run: bool = False,
    mario_align_params: Optional[dict] = None,
) -> List["BIDSPath"]:
    sourcedata_root = Path(sourcedata_root)
    bids_root = Path(bids_root)
    beh_root = Path(behavior_root) if behavior_root else sourcedata_root / 'behav'
    eeg_root_path = Path(eeg_root) if eeg_root else sourcedata_root / 'eeg'

    edf_paths = sorted(eeg_root_path.rglob('*.edf'))
    written: List[BIDSPath] = []
    if not edf_paths:
        print(f'No EDF files found under {eeg_root_path}')
        return written

    for edf_path in edf_paths:
        try:
            subject, session, task, run = extract_bids_entities_from_name(edf_path.stem)
        except ValueError as exc:
            print(f"[skip] {edf_path.name}: {exc}")
            continue
        if not task:
            print(f"[skip] {edf_path.name}: missing task label")
            continue

        mario_tsv = None
        if mario_mode:
            mario_tsv = find_mario_behavior_tsv(beh_root, subject, session, task, run)
            if mario_tsv is None:
                print(f"[Mario] Warning: no behavior TSV found for sub-{subject} ses-{session} task-{task} run-{run or '01'}")

        if dry_run:
            print(f"[dry-run] {edf_path} -> sub-{subject}, ses-{session}, task-{task}, run-{run or '01'}")
            continue

        align_kwargs = dict(mario_align_params) if (mario_align_params and mario_tsv) else None
        bids_paths = bidsify_edf(
            edf_path=edf_path,
            bids_root=bids_root,
            subject=subject,
            session=session,
            task=task,
            run=run,
            line_freq=line_freq,
            overwrite=overwrite,
            apply_montage=apply_montage,
            mario_behav_tsv=mario_tsv,
            mario_align_params=align_kwargs,
        )
        written.extend(bids_paths)

        if mario_mode and mario_tsv:
            beh_source_dir = beh_root / f'sub-{subject}' / f'ses-{session}'
            for bp in bids_paths:
                try:
                    process_mario_behavior_assets(mario_tsv, bp, behavior_source_dir=beh_source_dir)
                except Exception as err:
                    print(f"[Mario] Failed to copy behavior assets for {bp.fpath.name}: {err}")

    return written



# ---------- CLI ----------
def _build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(
        description="BIDSify DSI-24 recordings from a single EDF or a sourcedata tree."
    )
    # Positional style (EDF, BIDS root)
    p.add_argument("edf", nargs="?", help="Path to input EDF file (positional)")
    p.add_argument("bids_root", nargs="?", help="BIDS root directory (positional)")
    # Optional flags (aliases supported)
    p.add_argument("--edf", dest="edf_opt", help="Path to input EDF file")
    p.add_argument("--bids-root", dest="bids_root_opt", help="BIDS root directory")
    p.add_argument("--sourcedata", help="Root directory containing eeg/ and behav/ folders to convert")
    p.add_argument("--eeg-root", help="Override path to EEG EDF directory (defaults to <sourcedata>/eeg)")
    p.add_argument("--behav-root", dest="behav_root", help="Override behavior/gamelog directory (defaults to <sourcedata>/behav)")
    # Subject/session/task/run (single-run mode)
    p.add_argument("--sub", "--subject", dest="sub", help="Subject label (e.g., 01)")
    p.add_argument("--ses", "--session", dest="ses", help="Session label (e.g., 001)")
    p.add_argument("--task", help="Task name (e.g., gonogo)")
    p.add_argument("--run", help="Run label (e.g., 01)")
    # Mario-specific behavior inputs
    p.add_argument("--mario", action="store_true", help="Enable Mario behavior alignment and gamelog copying")
    p.add_argument("--behavior-tsv", dest="behavior_tsv", help="Mario behavior TSV to align with the EEG (single-run mode)")
    p.add_argument("--mario-align-json", dest="mario_align_json", help="Path to JSON with Mario alignment overrides")
    # Power line frequency
    p.add_argument("--line-freq", "--powerline", dest="line_freq", type=int, default=60,
                   help="Power line frequency (50 or 60)")
    # Overwrite controls
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    p.add_argument("--no-overwrite", action="store_true", help="Do not overwrite existing files")
    p.add_argument("--no-montage", action="store_true", help="Disable applying standard_1020 montage")
    # Event handling
    p.add_argument("--event-map", type=str, default=None,
                   help=(
                       "Path to JSON. Accepts: (1) legacy {code<->description} dict, "
                       "(2) a task-config with 'event_map', or (3) the full BIDS-first config "
                       "(selects events.tasks[--task] or default_task)."
                   ))
    # Task splitting
    p.add_argument("--task-splits", type=str, default=None,
                   help="JSON specifying task segments by start/stop codes")
    p.add_argument("--dry-run", action="store_true", help="List actions without writing output (sourcedata mode)")
    return p

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    overwrite = True if args.overwrite else (not args.no_overwrite)
    mario_align_params = None
    if args.mario_align_json:
        with open(args.mario_align_json, 'r', encoding='utf-8') as f:
            mario_align_params = json.load(f)

    if args.sourcedata:
        bids_root = args.bids_root_opt or args.bids_root
        if not bids_root:
            parser.error('Provide BIDS root via positional arg or --bids-root when using --sourcedata')
        written = bidsify_sourcedata(
            sourcedata_root=args.sourcedata,
            bids_root=bids_root,
            mario_mode=bool(args.mario),
            behavior_root=args.behav_root,
            eeg_root=args.eeg_root,
            line_freq=args.line_freq,
            apply_montage=not args.no_montage,
            overwrite=overwrite,
            dry_run=args.dry_run,
            mario_align_params=mario_align_params,
        )
        for bp in written:
            print(f"Wrote BIDS to: {bp.directory}")
        return

    edf_path = args.edf_opt or args.edf
    bids_root = args.bids_root_opt or args.bids_root
    if not edf_path or not bids_root:
        parser.error('Provide EDF and BIDS root either positionally or via --edf/--bids-root')

    required = [
        ('--sub/--subject', args.sub),
        ('--ses/--session', args.ses),
        ('--task', args.task),
    ]
    missing = [label for label, value in required if not value]
    if missing:
        parser.error('Single-run mode requires: ' + ', '.join(missing))

    mario_mode = bool(args.mario or args.behavior_tsv)
    mario_behav_tsv = args.behavior_tsv if mario_mode else None

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
        mario_behav_tsv=mario_behav_tsv,
        mario_align_params=mario_align_params,
    )

    beh_source_dir = None
    if mario_mode and args.behav_root:
        beh_source_dir = Path(args.behav_root) / f'sub-{args.sub}' / f'ses-{args.ses}'

    for bp in bids_paths:
        print(f"Wrote BIDS to: {bp.directory}")
        if mario_mode and mario_behav_tsv:
            try:
                out = process_mario_behavior_assets(
                    Path(mario_behav_tsv),
                    bp,
                    behavior_source_dir=beh_source_dir or Path(mario_behav_tsv).parent,
                )
                if out is not None:
                    print(f"[Mario] Saved behavior events to: {out}")
            except Exception as err:
                print(f"[Mario] Failed to copy behavior assets for {bp.fpath.name}: {err}")

if __name__ == '__main__':
    main()

__all__ = [
    "extract_bids_entities_from_name",
    "parse_mario_bk2_name",
    "find_mario_behavior_tsv",
    "process_mario_behavior_assets",
    "rename_dsi_channels",
    "normalize_dsi_channel_names",
    "set_channel_types",
    "find_stim_channel",
    "extract_events_and_id",
    "prepare_dsi24_raw",
    "bidsify_edf",
    "bidsify_sourcedata",
]
