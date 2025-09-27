#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mario BIDSify (all-in-one): behavior sidecars + annotations + EEG BIDS export.

What it does
------------
- For each run (sub-XX_ses-YYY_task-mario_run-ZZ):
  1) Replays each .bk2 to produce:
     - .npz (frame-wise info + actions), .mp4 (video), .json (summary info), .pkl (repvars dict)
  2) Builds a single run-level "*_desc-annotated_events.tsv" with actions/kills/coins/hits/bricks/powerups
     using the generated .pkl sidecars.
  3) Reads the EDF, normalizes DSI-24 channels (A1/A2 -> misc; T3/T4/T5/T6 -> T7/T8/P7/P8),
     applies standard_1020 montage, sets stim/eog/..., and writes BIDS (BrainVision) with
     durations via annotations.
  4) Copies the .bk2 files into BIDS /gamelogs and writes a copy of run-level events.tsv into /eeg
     updating 'stim_file' to the BIDS-relative gamelog path and adding 'level' (like the colleague's tool).

Inputs
------
--source       Root folder (or sourcedata folder); script discovers Mario EEG runs automatically.
--sub, --ses   Optional subject/session filters. Missing entities trigger processing of all matches.
--run          Optional run label filter.
--stimuli      Optional override for ROM/state location (defaults to sourcedata/stimuli).
--bids-root    Output BIDS root (defaults to <source>/bids).
--no-video     Skip MP4 video generation; still writes .npz/.json/.pkl.
--line-freq    Power line frequency (50 or 60). Default: 60.
--overwrite    Overwrite BIDS output if exists.

Dependencies
------------
- mne, mne-bids, numpy, pandas, stable-retro (imported as 'retro')

Notes
-----
- This script does NOT depend on any private modules. It reuses the logic from your
  generate_annotations.py and generate_sidecars.py to keep your event semantics identical.
"""

from __future__ import annotations
import argparse
import re
import json
import shutil
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

MARIO_TASK_NAME = "Mario (NES Gym-Retro) gameplay"
DERIVATIVE_NAME = "mario-replay"

import numpy as np
import pandas as pd

try:
    import retro  # stable-retro
    from retro.scripts.playback_movie import playback_movie
    _HAS_RETRO = True
except Exception as e:
    _HAS_RETRO = False

import mne
from mne.io import BaseRaw

try:
    from mne_bids import BIDSPath, write_raw_bids
    _HAS_MNE_BIDS = True
except Exception:
    _HAS_MNE_BIDS = False

# -------------------- DSI-24 helpers (channel normalization & typing) --------------------

LEGACY_TO_1020 = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

STIM_NAMES = {
    "STI 014", "STI014", "STATUS", "Status", "TRIG", "TRIGGER", "TRG",
    "EVENT", "EVENTS", "MARKER", "DIN", "DIGITAL", "DIGITALIO", "STIM"
}

EOG_NAMES = {"HEOG", "VEOG", "EOG", "EOGL", "EOGR"}

_VENDOR_PREFIX_RE = re.compile(r"^EEG\s+(?:X\d+:)?", flags=re.IGNORECASE)
_PZ_SUFFIX_RE = re.compile(r"-PZ$", flags=re.IGNORECASE)
_X_PREFIX_RE = re.compile(r"^EEG\s+X(\d+):", flags=re.IGNORECASE)

def rename_dsi_channels(raw: BaseRaw) -> Dict[str, str]:
    mapping: Dict[str, str] = {ch: LEGACY_TO_1020[ch] for ch in raw.ch_names if ch in LEGACY_TO_1020}
    if mapping:
        raw.rename_channels(mapping)
    return mapping

def normalize_dsi_channel_names(raw: BaseRaw) -> Dict[str, str]:
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
    try:
        picks = mne.pick_channels(raw.ch_names, include=[ch_name])
        if len(picks) != 1:
            return False
        n = min(raw.n_times, 20000)
        data = raw.get_data(picks=picks, start=0, stop=n).ravel()
        if data.size == 0:
            return False
        vals = np.unique(np.round(data).astype(int))
        return len(vals) <= max_levels
    except Exception:
        return False

def set_channel_types(raw: BaseRaw) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for ch in raw.ch_names:
        cu = ch.upper()
        if cu in STIM_NAMES:
            mapping[ch] = "stim"
        elif cu in EOG_NAMES:
            mapping[ch] = "eog"
        elif "ECG" in cu:
            mapping[ch] = "ecg"
        # IMPORTANT: map A1/A2 explicitly to misc (per user request)
        elif cu in {"A1", "A2", "M1", "M2"}:
            mapping[ch] = "misc"
        elif cu == "CM" or cu.startswith("X"):
            mapping[ch] = "misc"
        elif cu == "PHOTO" and _looks_digital_like(raw, ch):
            mapping[ch] = "stim"
    if mapping:
        raw.set_channel_types(mapping)
    return mapping

def find_stim_channel(raw: BaseRaw) -> Optional[str]:
    types = dict(zip(raw.ch_names, raw.get_channel_types()))
    for pref in ("Trigger", "STATUS"):
        if pref in raw.ch_names and types.get(pref) == "stim":
            return pref
    for ch, t in types.items():
        if t == "stim":
            return ch
    for name in STIM_NAMES:
        if name in raw.ch_names:
            return name
    return None

def apply_standard_montage(raw: BaseRaw) -> bool:
    try:
        from mne.channels import make_standard_montage
        montage = make_standard_montage("standard_1020")
        raw.set_montage(montage)
        return True
    except Exception:
        return False

def prepare_dsi24_raw(raw: BaseRaw, apply_montage: bool = True) -> BaseRaw:
    normalize_dsi_channel_names(raw)
    rename_dsi_channels(raw)
    set_channel_types(raw)

    # enforce a single stim channel
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

# -------------------- BIDS sidecar patching --------------------

def _patch_eeg_sidecar(
    bids_path: "BIDSPath",
    eeg_reference_text: str,
    power_line_freq: Optional[int],
    recording_type: str = "continuous",
    eeg_ground_text: Optional[str] = "Fpz"
) -> None:
    bp = bids_path.copy(); bp.update(suffix="eeg", extension=".json")
    eeg_json_path = Path(bp.fpath)
    if not eeg_json_path.exists():
        return
    with eeg_json_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    meta["TaskName"] = MARIO_TASK_NAME
    meta["EEGReference"] = eeg_reference_text
    if power_line_freq is not None:
        meta["PowerLineFrequency"] = int(power_line_freq)
    if "SoftwareFilters" not in meta or meta["SoftwareFilters"] in (None, ""):
        meta["SoftwareFilters"] = "n/a"
    if eeg_ground_text:
        meta["EEGGround"] = eeg_ground_text
    meta.setdefault("RecordingType", recording_type)
    meta["Manufacturer"] = "Wearable Sensing"
    meta.setdefault("ManufacturersModelName", "DSI-24")
    meta.setdefault("EEGPlacementScheme", "DSI-24 (10-20 layout equivalent)")

    # Channel counts (recommended)
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

def _replace_coordsystem_with_template(bids_path: "BIDSPath") -> None:
    eeg_dir = Path(bids_path.directory)
    if not eeg_dir.exists():
        return

    # Remove coordsystem/electrodes written by mne-bids
    for pattern in ("*_coordsystem.json", "*_electrodes.tsv"):
        for path in eeg_dir.glob(pattern):
            try:
                path.unlink()
            except Exception:
                pass

    # Write a template coordsystem JSON referencing the standard 10-20 montage
    bp = bids_path.copy()
    bp.update(suffix="coordsystem", extension=".json")

    meta = {
        "EEGCoordinateSystem": "EEG10-10",
        "EEGCoordinateSystemDescription": (
            "Template MNE 'standard_1020' 10-20 montage (no digitized sensor locations)."
        ),
        "EEGCoordinateUnits": "m",
        "AnatomicalLandmarkCoordinateSystem": "EEG10-10",
        "AnatomicalLandmarkCoordinateUnits": "m"
    }

    coordsystem_path = Path(bp.fpath)
    try:
        coordsystem_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception:
        pass


def _patch_channel_units(bids_path: "BIDSPath") -> None:
    bp = bids_path.copy(); bp.update(suffix="channels", extension=".tsv")
    channels_path = Path(bp.fpath)
    if not channels_path.exists():
        return

    try:
        df = pd.read_csv(channels_path, sep='\t')
    except Exception:
        return

    type_col = None
    for cand in ("type", "Type"):
        if cand in df.columns:
            type_col = cand
            break
    units_col = None
    for cand in ("units", "Units"):
        if cand in df.columns:
            units_col = cand
            break
    if type_col is None or units_col is None:
        return

    trig_mask = df[type_col].astype(str).str.upper() == "TRIG"
    if not trig_mask.any():
        return

    units_series = df[units_col]
    blank_mask = units_series.isna() | units_series.astype(str).str.strip().eq("") | units_series.astype(str).str.lower().eq("nan")
    update_mask = trig_mask & blank_mask
    if not update_mask.any():
        return

    df.loc[update_mask, units_col] = "n/a"
    try:
        df.to_csv(channels_path, sep='\t', index=False)
    except Exception:
        pass


def _ensure_derivative_dataset_description(derivative_root: Path) -> None:
    derivative_root.mkdir(parents=True, exist_ok=True)
    desc_path = derivative_root / "dataset_description.json"
    description = {
        "Name": "Mario EEG mario-replay derivative",
        "BIDSVersion": "1.8.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "mario_bidsify",
                "Description": "Mario EEG all-in-one BIDSifier (behavior annotations + BrainVision export)",
            }
        ],
        "PipelineDescription": {
            "Name": DERIVATIVE_NAME,
            "Version": "unknown"
        },
        "SourceDatasets": [
            {"Description": "See root dataset_description.json"}
        ]
    }
    with desc_path.open("w", encoding="utf-8") as f:
        json.dump(description, f, indent=2)


def _write_dataset_level_documents(bids_root: Path, subjects: Iterable[str]) -> None:
    dataset_description = {
        "Name": "Mario EEG (DSI-24) dataset",
        "BIDSVersion": "1.8.0",
        "DatasetType": "raw",
        "GeneratedBy": [
            {
                "Name": "mario_bidsify",
                "Description": "Mario EEG all-in-one BIDSifier (behavior annotations + BrainVision export)",
            }
        ]
    }
    desc_path = bids_root / "dataset_description.json"
    with desc_path.open("w", encoding="utf-8") as f:
        json.dump(dataset_description, f, indent=2)

    readme_text = textwrap.dedent(
        """
        Mario EEG BIDS dataset
        ----------------------
        This dataset contains DSI-24 EEG recorded while participants played Super Mario Bros
        via the Gym-Retro emulator. Data were converted to BIDS using `mario_bidsify`, which
        generates behavioral sidecars, annotated events, and BrainVision exports compliant with EEG-BIDS.
        """
    ).strip() + "\n"
    with (bids_root / "README").open("w", encoding="utf-8") as f:
        f.write(readme_text)

    participants = sorted({sub if sub.startswith("sub-") else f"sub-{sub}" for sub in subjects})
    participants_path = bids_root / "participants.tsv"
    with participants_path.open("w", encoding="utf-8") as f:
        f.write("participant_id\n")
        for participant in participants:
            f.write(f"{participant}\n")

    participants_json = {
        "participant_id": {
            "Description": "Unique participant identifier following the pattern sub-<label>."
        }
    }
    with (bids_root / "participants.json").open("w", encoding="utf-8") as f:
        json.dump(participants_json, f, indent=2)

# -------------------- Behavior sidecars (bk2 -> npz/mp4/json/pkl) --------------------

def _count_kills(repvars):
    kill_count = 0
    for i in range(6):
        for idx, val in enumerate(repvars.get(f'enemy_kill3{i}', [])[:-1]):
            if val in [4, 34, 132]:
                if repvars[f'enemy_kill3{i}'][idx+1] != val:
                    if i == 5:
                        if repvars.get('powerup_yes_no', 0) == 0:
                            kill_count += 1
                    else:
                        kill_count += 1
    return kill_count

def _count_bricks_destroyed(repvars):
    score_increments = list(np.diff(repvars.get('score', [])))
    bricks_destroyed = 0
    for idx, inc in enumerate(score_increments):
        if inc == 5:
            if repvars.get('jump_airborne', [0]*len(score_increments))[idx] == 1:
                bricks_destroyed += 1
    return bricks_destroyed

def _count_hits_taken(repvars):
    diff_state = list(np.diff(repvars.get('powerstate', [])))
    hits_count = sum(1 for val in diff_state if val < -10000)
    diff_lives = list(np.diff(repvars.get('lives', [])))
    hits_count += sum(1 for val in diff_lives if val < 0)
    return hits_count

def _count_powerups_collected(repvars):
    powerup_count = 0
    ps = repvars.get('player_state', [])
    for idx, val in enumerate(ps[:-1]):
        if val in [9, 12, 13] and ps[idx+1] != val:
            powerup_count += 1
    return powerup_count

def summarize_repvars(repvars):
    info_dict = {}
    # world/level
    lvl = repvars.get('level', 'w?l?')
    info_dict['world'] = lvl[1] if len(lvl) > 1 else '?'
    info_dict['level'] = lvl[-1] if len(lvl) > 0 else '?'
    # duration
    info_dict['duration'] = len(repvars.get('score', [])) / 60 if repvars.get('score') is not None else 0
    # outcomes
    term = repvars.get('terminate', [True])[-1] if repvars.get('terminate') else True
    info_dict['terminated'] = bool(term)
    info_dict['cleared'] = bool(term and (repvars.get('lives', [2])[-1] >= 0))
    # scores & positions
    info_dict['final_score'] = repvars.get('score', [0])[-1] if repvars.get('score') else 0
    xlo = repvars.get('xscrollLo', [0])[-1] if repvars.get('xscrollLo') else 0
    xhi = repvars.get('xscrollHi', [0])[-1] if repvars.get('xscrollHi') else 0
    info_dict['final_position'] = int(xlo) + (256 * int(xhi))
    # deltas
    info_dict['lives_lost'] = 2 - (repvars.get('lives', [2])[-1] if repvars.get('lives') else 2)
    info_dict['hits_taken'] = _count_hits_taken(repvars)
    info_dict['enemies_killed'] = _count_kills(repvars)
    info_dict['powerups_collected'] = _count_powerups_collected(repvars)
    info_dict['bricks_destroyed'] = _count_bricks_destroyed(repvars)
    info_dict['coins'] = repvars.get('coins', [0])[-1] if repvars.get('coins') else 0
    return info_dict

def load_repvars_from_npz(npy_file: str, bk2_file: str, emulator_buttons: List[str]):
    with np.load(npy_file, allow_pickle=True) as data:
        info = data['info']
        actions = data['actions']

    repvars = {}
    # Fill variables from info list of dicts
    keys = list(info[0].keys())
    for k in keys:
        repvars[k] = [frame[k] for frame in info]
    # Actions per button
    for idx_button, button in enumerate(emulator_buttons):
        repvars[button] = [frame[idx_button] for frame in actions]

    repvars["filename"] = bk2_file
    stem_parts = Path(bk2_file).stem.split("_")

    if len(stem_parts) >= 2:
        repvars["subject"] = stem_parts[0]
        repvars["session"] = stem_parts[1]
    if stem_parts:
        repvars["repetition"] = stem_parts[-1]
    if len(stem_parts) >= 2:
        level_token = stem_parts[-2]
        level_parts = level_token.split('-', 1)
        if len(level_parts) == 2 and level_parts[1]:
            repvars["level"] = level_parts[1]
        else:
            repvars["level"] = repvars.get("level", "w?l?")
    else:
        repvars["level"] = repvars.get("level", "w?l?")
    repvars["terminate"] = [True]
    return repvars

def playback_bk2_sidecars(bk2_file: Path, stimuli_path: Path, make_video: bool = True, skip_first_step: bool = False) -> Tuple[Optional[dict], Optional[str]]:
    """
    Replay a bk2; write npz/mp4 and json/pkl sidecars.
    Returns (repvars_dict, video_file_path or None).
    """
    if not _HAS_RETRO:
        raise RuntimeError("stable-retro is required. Please install 'stable-retro' to replay .bk2 files.")

    retro.data.Integrations.add_custom_path(str(stimuli_path))

    movie = retro.Movie(str(bk2_file))
    if skip_first_step:
        # Some recordings require skipping the first step for proper alignment
        movie.step()
    game = movie.get_game()
    emulator = retro.make(game, scenario=None, inttype=retro.data.Integrations.CUSTOM_ONLY, render_mode=False)
    emulator.initial_state = movie.get_state()
    emulator.reset()

    # In many datasets, the first repetition requires skipping the first step.
    # We can't know run order here, so don't skip by default.
    npy_file = str(bk2_file).replace(".bk2", ".npz")
    video_file = str(bk2_file).replace(".bk2", ".mp4") if make_video else None

    playback_movie(emulator, movie, npy_file=npy_file, video_file=video_file, lossless='mp4' if make_video else None, info_file=True)
    buttons = emulator.buttons.copy()
    emulator.close()

    repvars = load_repvars_from_npz(npy_file, str(bk2_file), buttons)
    info_dict = summarize_repvars(repvars)

    # write sidecars
    json_sidecar = str(bk2_file).replace(".bk2", ".json")
    with open(json_sidecar, "w") as f:
        json.dump(info_dict, f)
    pkl_sidecar = str(bk2_file).replace(".bk2", ".pkl")
    import pickle
    with open(pkl_sidecar, "wb") as f:
        pickle.dump(repvars, f)

    return repvars, video_file

# -------------------- Annotated events generation (repvars -> run TSV) --------------------

def _build_button_press_events(repvars, key, FS=60):
    var = np.array(repvars.get(key, []), dtype=int)
    if var.size == 0:
        return []
    var[0] = 0
    var[-1] = 0
    diffs = np.diff(var)
    presses = [round(i / FS, 3) for i, x in enumerate(diffs) if x == 1]
    releases = [round(i / FS, 3) for i, x in enumerate(diffs) if x == -1]
    duration = [round(releases[i] - presses[i], 3) for i in range(min(len(presses), len(releases)))]
    onset = presses[: len(duration)]
    level = repvars.get('level', 'w?l?')
    return [
        {'onset': on, 'duration': dur, 'trial_type': key, 'level': level}
        for on, dur in zip(onset, duration)
    ]

def _build_kill_events(repvars, FS=60):
    n_frames = len(repvars.get('START', []))
    killvals_dict = {4: 'stomp', 34: 'impact', 132: 'kick'}
    level = repvars.get('level', 'w?l?')
    rows = []
    for frame_idx in range(max(0, n_frames - 1)):
        for ii in range(6):
            key = f'enemy_kill3{ii}'
            arr = repvars.get(key, [])
            if not arr or frame_idx + 1 >= len(arr):
                continue
            curr_val = arr[frame_idx]
            next_val = arr[frame_idx + 1]
            if curr_val in killvals_dict and curr_val != next_val:
                if ii == 5 and repvars.get('powerup_yes_no', 0) != 0:
                    continue
                rows.append({
                    'onset': frame_idx / FS,
                    'duration': 0,
                    'trial_type': f"Kill/{killvals_dict[curr_val]}",
                    'level': level,
                })
    return rows

def _build_hits_taken_events(repvars, FS=60):
    level = repvars.get('level', 'w?l?')
    rows = []
    diff_state = np.diff(repvars.get('powerstate', []))
    for idx_val, val in enumerate(diff_state):
        if val < -10000:
            rows.append({
                'onset': idx_val / FS,
                'duration': 0,
                'trial_type': 'Hit/powerup_lost',
                'level': level,
            })
    diff_lives = np.diff(repvars.get('lives', []))
    for idx_val, val in enumerate(diff_lives):
        if val < 0:
            rows.append({
                'onset': idx_val / FS,
                'duration': 0,
                'trial_type': 'Hit/life_lost',
                'level': level,
            })
    return rows

def _build_brick_events(repvars, FS=60):
    level = repvars.get('level', 'w?l?')
    score_increments = np.diff(repvars.get('score', []))
    jump = repvars.get('jump_airborne', [0] * len(score_increments))
    return [
        {
            'onset': idx / FS,
            'duration': 0,
            'trial_type': 'Brick_smashed',
            'level': level,
        }
        for idx, inc in enumerate(score_increments)
        if inc == 5 and idx < len(jump) and jump[idx] == 1
    ]

def _build_coin_events(repvars, FS=60):
    level = repvars.get('level', 'w?l?')
    diff_coins = np.diff(repvars.get('coins', []))
    return [
        {
            'onset': idx / FS,
            'duration': 0,
            'trial_type': 'Coin_collected',
            'level': level,
        }
        for idx, val in enumerate(diff_coins)
        if val > 0
    ]

def _build_powerup_events(repvars, FS=60):
    level = repvars.get('level', 'w?l?')
    ps = repvars.get('player_state', [])
    return [
        {
            'onset': idx / FS,
            'duration': 0,
            'trial_type': 'Powerup_collected',
            'level': level,
        }
        for idx, val in enumerate(ps[:-1])
        if val in [9, 12, 13] and ps[idx + 1] != val
    ]

def build_frame_events_from_trigger(raw: BaseRaw, run_events: pd.DataFrame) -> pd.DataFrame:
    columns = ["onset", "duration", "trial_type", "value", "level", "stim_file", "game_log"]
    stim_name = find_stim_channel(raw)
    if stim_name is None:
        return pd.DataFrame(columns=columns)

    try:
        picks = mne.pick_channels(raw.ch_names, include=[stim_name])
        if len(picks) != 1:
            return pd.DataFrame(columns=columns)
        stim_data = raw.get_data(picks=picks)[0]
    except Exception:
        return pd.DataFrame(columns=columns)

    sfreq = float(raw.info.get("sfreq", 1.0))
    binary = (stim_data > 0.5).astype(int)
    edges = np.diff(binary)
    edge_indices = np.where(edges != 0)[0] + 1  # capture rising and falling edges

    if edge_indices.size == 0 or run_events.empty:
        return pd.DataFrame(columns=columns)

    windows = [
        (
            float(row.get("onset", 0.0)),
            float(row.get("onset", 0.0)) + float(row.get("duration", 0.0)),
            str(row.get("level", "n/a")),
        )
        for _, row in run_events.iterrows()
    ]

    if not windows:
        return pd.DataFrame(columns=columns)

    starts = np.array([w[0] for w in windows], dtype=float)
    stops = np.array([w[1] for w in windows], dtype=float)
    levels = np.array([w[2] for w in windows], dtype=object)

    onset_values = edge_indices / sfreq
    lookup_idx = np.searchsorted(starts, onset_values, side='right') - 1
    clamped_idx = lookup_idx.clip(0, len(starts) - 1)
    valid_mask = (
        (lookup_idx >= 0)
        & (lookup_idx < len(starts))
        & (onset_values >= starts[clamped_idx])
        & (onset_values <= stops[clamped_idx])
    )

    if not np.any(valid_mask):
        return pd.DataFrame(columns=columns)

    selected_indices = lookup_idx[valid_mask]
    frame_rows = pd.DataFrame({
        "onset": onset_values[valid_mask],
        "duration": 0.0,
        "trial_type": "frame",
        "value": binary[edge_indices[valid_mask]].astype(int),
        "level": levels[selected_indices],
        "stim_file": "n/a",
        "game_log": "n/a",
    }, columns=columns)

    return frame_rows



def build_run_events_dataframe(runvars: List[dict], events_dataframe: pd.DataFrame, FS=60) -> pd.DataFrame:
    base_df = events_dataframe.copy()
    event_rows: List[dict] = []

    for idx, repvars in enumerate(runvars):
        n_frames_total = len(repvars.get('START', []))
        rep_onset = float(events_dataframe.loc[idx, 'onset']) if idx < len(events_dataframe) else 0.0
        rep_duration = (n_frames_total / FS) if n_frames_total > 0 else 0.0

        if idx < len(base_df) and base_df.loc[idx, 'trial_type'] == 'gym-retro_game' and rep_duration > 0:
            base_df.loc[idx, 'duration'] = rep_duration

        if not repvars:
            continue

        def _add_rows(rows: Iterable[dict]) -> None:
            for row in rows:
                new_row = row.copy()
                new_row['onset'] = round(new_row.get('onset', 0.0) + rep_onset, 3)
                event_rows.append(new_row)

        for act in ['UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'START', 'SELECT']:
            _add_rows(_build_button_press_events(repvars, act, FS=FS))

        for generator in (
            _build_kill_events,
            _build_hits_taken_events,
            _build_brick_events,
            _build_coin_events,
            _build_powerup_events,
        ):
            _add_rows(generator(repvars, FS=FS))

    base_records = base_df.to_dict('records') if not base_df.empty else []
    all_records = base_records + event_rows
    if not all_records:
        return pd.DataFrame()

    events_df = pd.DataFrame.from_records(all_records)
    if 'onset' in events_df.columns:
        events_df = events_df.sort_values('onset').reset_index(drop=True)
    return events_df

# -------------------- BIDS writing (Mario) --------------------

def events_to_annotations(raw: BaseRaw, mario_df: pd.DataFrame) -> mne.Annotations:
    onsets = mario_df["onset"].astype(float).to_numpy()
    durations = mario_df["duration"].astype(float).to_numpy()
    desc = mario_df["trial_type"].astype(str).to_list()
    ann = mne.Annotations(onset=onsets, duration=durations, description=desc)
    return ann

def _parse_world_level_from_bk2name(bk2name: str) -> Tuple[str, str]:
    try:
        base = Path(bk2name).name
        parts = base.split('_')
        chunk = parts[4]
        if 'Level' in chunk:
            lvl = chunk.split('Level')[-1]
            world = f"w{lvl.split('-')[0]}"
            level = f"l{lvl.split('-')[1]}"
        elif 'level-w' in chunk.lower():
            label = chunk.split('level-')[-1]
            world = label[:2]
            level = label[2:]
            if not world.startswith('w'):
                world = f"w{world.lstrip('w')}"
            if not level.startswith('l'):
                level = f"l{level.lstrip('l')}"
        else:
            return "wX", "lY"
        return world, level
    except Exception:
        return "wX", "lY"

def copy_bk2_and_write_events_copy(
    events_tsv: Path,
    bids_root: Path,
    subject: str,
    session: str,
    task: str,
    run: str,
    events_df_override: Optional[pd.DataFrame] = None,
) -> Optional[Path]:
    df = events_df_override.copy() if events_df_override is not None else pd.read_csv(events_tsv, sep='\t')
    if "trial_type" not in df.columns or "stim_file" not in df.columns:
        return None

    if "game_log" not in df.columns:
        df["game_log"] = "n/a"

    reps = df[df["trial_type"] == "gym-retro_game"].copy()
    if reps.empty:
        return None

    sub = f"sub-{subject}"
    ses = f"ses-{session}"
    eeg_dir = Path(bids_root) / sub / ses / "eeg"
    eeg_dir.mkdir(parents=True, exist_ok=True)

    derivative_root = Path(bids_root) / "derivatives" / DERIVATIVE_NAME
    gamelog_dir = derivative_root / sub / ses / "gamelog"
    gamelog_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_bk2_path(bk2_value: str) -> Optional[Path]:
        if not bk2_value or bk2_value in {"n/a", "Missing file"}:
            return None
        path = Path(bk2_value)
        candidates = [path]
        if not path.is_absolute():
            candidates.extend([
                events_tsv.parent / path,
                events_tsv.parent / path.name,
            ])
        candidates.append(Path(bids_root, "sourcedata", "behav", sub, ses, path.name))
        for cand in candidates:
            cand = Path(cand)
            if cand.exists():
                return cand
        return None

    def _copy_if_needed(src: Path, dst: Path) -> None:
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            return
        shutil.copyfile(src, dst)

    for idx, row in reps.iterrows():
        bk2_val = row.get("stim_file") if isinstance(row, pd.Series) else row["stim_file"]
        if pd.isna(bk2_val):
            continue
        bk2_path = _resolve_bk2_path(str(bk2_val))
        if bk2_path is None or not bk2_path.exists():
            continue

        world, level = _parse_world_level_from_bk2name(str(bk2_path))
        rep_tag = bk2_path.stem.split('_')[-1]
        target = gamelog_dir / f"{sub}_{ses}_task-{task}_run-{run}_level-{world}{level}_rep-{rep_tag}.bk2"

        try:
            _copy_if_needed(bk2_path, target)
            for suffix in (".json", ".npz", ".pkl", ".mp4"):
                sidecar = bk2_path.with_suffix(suffix)
                if sidecar.exists():
                    try:
                        _copy_if_needed(sidecar, target.with_suffix(suffix))
                    except Exception:
                        pass
            df.loc[idx, "level"] = f"{world}{level}"
            rel_target = Path("derivatives") / DERIVATIVE_NAME / sub / ses / "gamelog" / target.name
            df.loc[idx, "game_log"] = rel_target.as_posix()
        except Exception:
            continue

    df["stim_file"] = "n/a"

    out_events = eeg_dir / f"{sub}_{ses}_task-{task}_run-{run}_events.tsv"
    try:
        preferred_order = ["onset", "duration", "trial_type", "value", "level", "stim_file", "game_log"]
        column_order = [c for c in preferred_order if c in df.columns]
        df = df[column_order + [c for c in df.columns if c not in column_order]]
        df.to_csv(out_events, sep='\t', index=False)
        events_json = {
            "onset": {"Description": "Event onset in seconds from recording start."},
            "duration": {"Description": "Event duration in seconds."},
            "trial_type": {
                "Description": "Category of event.",
                "Levels": {
                    "gym-retro_game": "A gameplay repetition (bk2) window.",
                    "frame": "Display frame sync transition detected on the DSI TTL channel; 1=rising edge, 0=falling edge.",
                    "UP": "Button press epoch (press to release).",
                    "DOWN": "Button press epoch (press to release).",
                    "LEFT": "Button press epoch (press to release).",
                    "RIGHT": "Button press epoch (press to release).",
                    "A": "Button press epoch (press to release).",
                    "B": "Button press epoch (press to release).",
                    "START": "Button press epoch (press to release).",
                    "SELECT": "Button press epoch (press to release).",
                    "Kill/stomp": "Enemy defeated by stomp.",
                    "Kill/impact": "Enemy killed by impact.",
                    "Kill/kick": "Enemy killed by shell kick.",
                    "Hit/powerup_lost": "Hit that downgrades power state.",
                    "Hit/life_lost": "Life lost.",
                    "Brick_smashed": "Breakable brick destroyed.",
                    "Coin_collected": "Coin collected.",
                    "Powerup_collected": "Mushroom/flower/star collected."
                }
            },
            "value": {
                "Description": "For frame events: TTL level after each digital edge (1=rising, 0=falling).",
                "Units": "a.u."
            },
            "level": {"Description": "Mario world-level label (e.g., w1l1)."},
            "stim_file": {
                "Description": "Reserved for BIDS compatibility. Values are 'n/a' because proprietary stimuli are not distributed in this dataset."
            },
            "game_log": {
                "Description": "Relative path to the derivative mario-replay gamelog (bk2) associated with this repetition.",
                "Units": "n/a"
            }
        }
        with out_events.with_suffix('.json').open('w', encoding='utf-8') as f:
            json.dump(events_json, f, indent=2)
    except Exception:
        out_events = None
    return out_events






def write_bids_run_from_annotated(
    edf_path: Path,
    mario_annotated_tsv: Path,
    bids_root: Path,
    subject: str,
    session: str,
    task: str,
    run: str,
    line_freq: int = 60,
    overwrite: bool = True,
    run_events: Optional[pd.DataFrame] = None,
) -> Tuple["BIDSPath", pd.DataFrame]:
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    raw.info["line_freq"] = int(line_freq)
    prepare_dsi24_raw(raw, apply_montage=True)

    df = pd.read_csv(mario_annotated_tsv, sep='	')
    df.columns = [c.lower() for c in df.columns]
    required = {"onset", "duration", "trial_type"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Annotated events missing required columns {required}. Got: {df.columns}")
    raw.set_annotations(events_to_annotations(raw, df))

    if run_events is None:
        run_events = df[df['trial_type'] == 'gym-retro_game'][['onset', 'duration', 'level']].copy()
        run_events['level'] = run_events['level'].astype(str)
    frame_events = build_frame_events_from_trigger(raw, run_events)

    bids_path = BIDSPath(subject=subject, session=session, task=task, run=run, datatype="eeg", root=str(bids_root))
    write_raw_bids(raw, bids_path, overwrite=overwrite, allow_preload=True, format="BrainVision", verbose=False)

    _patch_eeg_sidecar(
        bids_path,
        eeg_reference_text="Pz (Common Mode Follower); ground at Fpz",
        power_line_freq=line_freq,
        recording_type="continuous",
        eeg_ground_text="Fpz"
    )
    _replace_coordsystem_with_template(bids_path)
    _patch_channel_units(bids_path)
    return bids_path, frame_events



# -------------------- Data discovery --------------------




def _extract_bids_entities(name: str) -> Dict[str, str]:
    """Return the first occurrence of each BIDS entity encoded in a filename stem."""
    entities: Dict[str, str] = {}
    for part in name.split('_'):
        if '-' not in part:
            continue
        key, value = part.split('-', 1)
        key = key.lower()
        if key not in entities:
            entities[key] = value
    return entities


def _normalize_task_label(raw_task: Optional[str]) -> Optional[str]:
    if raw_task is None:
        return None
    task = raw_task.split('-', 1)[0]
    task = task.split('_', 1)[0]
    return task.lower()


def _normalize_run_label(raw_run: Optional[str]) -> Optional[str]:
    if raw_run is None:
        return None
    for sep in ('_', '-', '.'):
        if sep in raw_run:
            raw_run = raw_run.split(sep, 1)[0]
    return raw_run


def _resolve_dataset_root(source_root: Path) -> Path:
    """Allow users to point to dataset root or sourcedata/{behav,eeg}."""
    if (source_root / 'sourcedata').is_dir():
        return source_root
    if source_root.name.lower() == 'sourcedata' and (source_root.parent / 'sourcedata').is_dir():
        return source_root.parent
    if source_root.name.lower() in {'eeg', 'behav'} and source_root.parent.name.lower() == 'sourcedata':
        candidate = source_root.parent.parent
        if (candidate / 'sourcedata').is_dir():
            return candidate
    return source_root


def find_eeg_runs(source_root: Path, sub: str, ses: str, run_filter: Optional[str]) -> List[Tuple[Path, str]]:
    """Return list of (edf_path, run_label) for given subject/session."""
    results: List[Tuple[Path, str]] = []
    search_roots: List[Path] = []
    eeg_root = source_root / "sourcedata" / "eeg"
    if eeg_root.exists():
        search_roots.append(eeg_root)
    else:
        search_roots.append(source_root)

    skip_dirs = {"bids", "derivatives"}
    seen: set[Path] = set()
    for root in search_roots:
        root = root.resolve()
        if root in seen or not root.exists():
            continue
        seen.add(root)

        matches: List[Tuple[Path, str]] = []
        for path in root.rglob("*.edf"):
            if any(part in skip_dirs for part in path.parts):
                continue
            entities = _extract_bids_entities(path.stem)
            if not entities:
                continue

            if entities.get("sub") != sub or entities.get("ses") != ses:
                continue

            task_label = _normalize_task_label(entities.get("task"))
            if task_label != "mario":
                continue

            run_label = _normalize_run_label(entities.get("run"))
            if not run_label:
                continue
            if run_filter and run_label != run_filter:
                continue
            matches.append((path, run_label))

        if matches:
            matches.sort(key=lambda x: x[0].name)
            return matches

    return results



def _iter_mario_eeg_files(source_root: Path) -> Iterable[Tuple[Path, Dict[str, str]]]:
    """Yield (edf_path, entities) for Mario EEG runs under the dataset."""
    search_roots: List[Path] = []
    eeg_root = source_root / "sourcedata" / "eeg"
    if eeg_root.exists():
        search_roots.append(eeg_root)
    else:
        search_roots.append(source_root)

    skip_dirs = {"bids", "derivatives"}
    seen_dirs: set[Path] = set()
    yielded: set[Path] = set()
    for root in search_roots:
        root = root.resolve()
        if root in seen_dirs or not root.exists():
            continue
        seen_dirs.add(root)

        for path in root.rglob("*.edf"):
            path = path.resolve()
            if path in yielded:
                continue
            if any(part in skip_dirs for part in path.parts):
                continue
            entities = _extract_bids_entities(path.stem)
            if not entities:
                continue
            if _normalize_task_label(entities.get("task")) != "mario":
                continue
            yielded.add(path)
            yield path, entities



def _discover_dataset_runs(source_root: Path) -> Dict[str, Dict[str, List[Tuple[Path, str]]]]:
    """Collect available Mario runs grouped by subject/session."""
    dataset: Dict[str, Dict[str, List[Tuple[Path, str]]]] = defaultdict(lambda: defaultdict(list))

    for edf_path, entities in _iter_mario_eeg_files(source_root):
        sub = entities.get("sub")
        ses = entities.get("ses")
        run = _normalize_run_label(entities.get("run"))
        if not sub or not ses or not run:
            continue
        dataset[sub][ses].append((edf_path, run))

    for sub_map in dataset.values():
        for ses_label, runs in sub_map.items():
            runs.sort(key=lambda item: item[1])
            sub_map[ses_label] = runs

    return dataset


def ensure_stimuli_assets(source_root: Path, game: str = "SuperMarioBros-Nes") -> Path:
    """Ensure ROM/state assets exist under sourcedata/stimuli; return the root path."""
    stim_root = source_root / "sourcedata" / "stimuli"
    stim_root.mkdir(parents=True, exist_ok=True)

    if not _HAS_RETRO:
        return stim_root

    from retro import data as retro_data

    stable_root = Path(retro_data.path()) / "stable" / game
    if not stable_root.exists():
        fallback = Path(retro_data.path()) / game
        if fallback.exists():
            stable_root = fallback

    if stable_root.exists():
        game_dest = stim_root / game
        game_dest.mkdir(parents=True, exist_ok=True)
        for item in stable_root.iterdir():
            target = game_dest / item.name
            try:
                if item.is_file():
                    if not target.exists() or target.stat().st_size != item.stat().st_size:
                        shutil.copy2(item, target)
                elif item.is_dir():
                    if not target.exists():
                        shutil.copytree(item, target)
            except Exception:
                continue

    return stim_root


def _build_events_index(source_root: Path) -> Dict[Tuple[str, str, str], Path]:
    """Scan once and index events TSVs by (sub, ses, run)."""

    index: Dict[Tuple[str, str, str], Path] = {}
    search_roots: List[Path] = []
    behav_root = source_root / "sourcedata" / "behav"
    if behav_root.exists():
        search_roots.append(behav_root)
    search_roots.append(source_root)

    seen: set[Path] = set()
    for root in search_roots:
        root = root.resolve()
        if root in seen or not root.exists():
            continue
        seen.add(root)

        for path in root.rglob("*_events.tsv"):
            if "desc-annotated" in path.stem:
                continue
            entities = _extract_bids_entities(path.stem)
            if not entities:
                continue

            task_label = _normalize_task_label(entities.get("task"))
            if task_label and task_label != "mario":
                continue

            sub = entities.get("sub")
            ses = entities.get("ses")
            run = _normalize_run_label(entities.get("run"))
            if not sub or not ses or not run:
                continue

            index.setdefault((sub, ses, run), path)

    return index

# -------------------- Pipeline --------------------


@dataclass(frozen=True)
class MarioRunContext:
    subject: str
    session: str
    run: str
    edf_path: Path
    events_path: Path


def _collect_requested_runs(
    dataset_runs: Dict[str, Dict[str, List[Tuple[Path, str]]]],
    events_index: Dict[Tuple[str, str, str], Path],
    sub_filter: Optional[str],
    ses_filter: Optional[str],
    run_filter: Optional[str],
) -> List[MarioRunContext]:
    contexts: List[MarioRunContext] = []

    if sub_filter:
        if sub_filter not in dataset_runs:
            raise FileNotFoundError(f"Subject sub-{sub_filter} not found")
        subjects = [sub_filter]
    else:
        subjects = sorted(dataset_runs.keys())

    for sub in subjects:
        session_map = dataset_runs[sub]
        if ses_filter:
            if ses_filter not in session_map:
                if sub_filter is not None:
                    raise FileNotFoundError(f"Session ses-{ses_filter} not found for sub-{sub}")
                continue
            sessions = [ses_filter]
        else:
            sessions = sorted(session_map.keys())

        for ses in sessions:
            runs = session_map.get(ses, [])
            if run_filter:
                runs = [(edf, run) for edf, run in runs if run == run_filter]
            for edf_path, run_label in runs:
                events_key = (sub, ses, run_label)
                events_path = events_index.get(events_key)
                if events_path is None:
                    raise FileNotFoundError(
                        f"Run events.tsv not found for sub-{sub} ses-{ses} run-{run_label}"
                    )
                contexts.append(MarioRunContext(sub, ses, run_label, edf_path, events_path))

    return contexts


def _load_run_events_dataframe(events_tsv: Path) -> pd.DataFrame:
    events_df = pd.read_table(events_tsv)
    base_cols = ["trial_type", "onset", "duration", "level", "stim_file"]
    keep_cols = [col for col in base_cols if col in events_df.columns]
    events_df = events_df[events_df['trial_type'] == 'gym-retro_game'][keep_cols].reset_index(drop=True)
    events_df = events_df.reindex(columns=base_cols)
    if 'duration' not in keep_cols:
        events_df['duration'] = 0.0
    events_df['duration'] = pd.to_numeric(events_df['duration'], errors='coerce').fillna(0.0)
    return events_df


def _locate_bk2_source(bk2_value: str, source_root: Path, subject: str, session: str) -> Path:
    bk2_path = Path(bk2_value)
    if bk2_path.exists():
        return bk2_path
    candidate = source_root / "sourcedata" / "behav" / f"sub-{subject}" / f"ses-{session}" / bk2_path.name
    return candidate if candidate.exists() else bk2_path


def _prepare_run_repvars(
    events_df: pd.DataFrame,
    run_ctx: MarioRunContext,
    source_root: Path,
    stimuli_path: Path,
    make_video: bool,
) -> List[dict]:
    runvars: List[dict] = []

    for rep_idx, bk2_value in enumerate(events_df['stim_file'].values.tolist()):
        if isinstance(bk2_value, float) or bk2_value == "Missing file":
            print(f"  - rep {rep_idx:02d}: missing .bk2; will leave empty")
            runvars.append({})
            continue

        bk2_path = _locate_bk2_source(str(bk2_value), source_root, run_ctx.subject, run_ctx.session)
        print(f"  - rep {rep_idx:02d}: {bk2_path.name}")

        npz_path = bk2_path.with_suffix(".npz")
        pkl_path = bk2_path.with_suffix(".pkl")
        json_path = bk2_path.with_suffix(".json")
        video_path = bk2_path.with_suffix(".mp4") if make_video else None

        have_sidecars = npz_path.exists() and pkl_path.exists() and json_path.exists()
        have_video = True if video_path is None else video_path.exists()
        if not (have_sidecars and have_video):
            if _HAS_RETRO:
                print("    • generating sidecars via playback_movie()")
                playback_bk2_sidecars(
                    bk2_path,
                    stimuli_path=stimuli_path,
                    make_video=make_video,
                    skip_first_step=(rep_idx == 0),
                )
            else:
                print("    • sidecars missing but stable-retro unavailable; skipping generation.")

        repvars: dict = {}
        if pkl_path.exists():
            import pickle

            with open(pkl_path, "rb") as f:
                repvars = pickle.load(f)
            events_df.loc[rep_idx, 'level'] = repvars.get('level', events_df.loc[rep_idx, 'level'])
        else:
            if _HAS_RETRO:
                print(f"    ! sidecar load failed for {bk2_path.name}; proceeding without repvars")
            else:
                print(f"    ! no sidecars for {bk2_path.name}; proceeding without repvars")

        runvars.append(repvars)

    return runvars


def _write_or_load_annotated_events(
    annotated_path: Path,
    runvars: List[dict],
    events_df: pd.DataFrame,
    overwrite: bool,
) -> pd.DataFrame:
    required_cols = {"onset", "duration", "trial_type"}
    if annotated_path.exists() and not overwrite:
        print("  - annotated events already exist:", annotated_path.name)
        annotated_df = pd.read_csv(annotated_path, sep='\t')
        lower_cols = {c.lower() for c in annotated_df.columns}
        if not required_cols.issubset(lower_cols):
            print("    • existing annotated file missing columns; regenerating")
            annotated_df = build_run_events_dataframe(runvars, events_df, FS=60)
            annotated_df.to_csv(annotated_path, sep='\t', index=False)
        return annotated_df

    if annotated_path.exists():
        print("  - overwriting annotated events:", annotated_path.name)
    else:
        print("  - writing annotated events:", annotated_path.name)
    annotated_df = build_run_events_dataframe(runvars, events_df, FS=60)
    annotated_df.to_csv(annotated_path, sep='\t', index=False)
    return annotated_df

def mario_bidsify_all_in_one(
    source_root: Path,
    sub: Optional[str] = None,
    ses: Optional[str] = None,
    bids_root: Optional[Path] = None,
    stimuli_path: Optional[Path] = None,
    run_filter: Optional[str] = None,
    make_video: bool = True,
    line_freq: int = 60,
    overwrite: bool = True,
    include_behavior_events: bool = False,
) -> List["BIDSPath"]:
    if not _HAS_MNE_BIDS:
        raise RuntimeError("mne-bids is required. Please 'pip install mne-bids'.")
    if not _HAS_RETRO:
        print("[Mario] stable-retro not installed; skipping .bk2 sidecar generation.")

    source_root = _resolve_dataset_root(source_root.resolve())
    dataset_runs = _discover_dataset_runs(source_root)
    if not dataset_runs:
        raise FileNotFoundError(f"No Mario EDF runs found under {source_root}")
    events_index = _build_events_index(source_root)
    contexts = _collect_requested_runs(dataset_runs, events_index, sub, ses, run_filter)
    if not contexts:
        raise FileNotFoundError("No runs matched the requested filters.")

    if stimuli_path is None:
        stimuli_path = ensure_stimuli_assets(source_root)
    else:
        stimuli_path = Path(stimuli_path)

    if bids_root is None:
        bids_root = source_root / "bids"
    bids_root.mkdir(parents=True, exist_ok=True)
    derivative_root = bids_root / "derivatives" / DERIVATIVE_NAME
    derivative_root.mkdir(parents=True, exist_ok=True)

    written_paths: List[BIDSPath] = []
    processed_subjects: set[str] = set()
    task_label = "mario"

    for ctx in contexts:
        processed_subjects.add(ctx.subject)
        print(
            f"[Mario] Processing: {ctx.edf_path.name} (sub-{ctx.subject}, "
            f"ses-{ctx.session}, run-{ctx.run})"
        )

        events_df = _load_run_events_dataframe(ctx.events_path)
        runvars = _prepare_run_repvars(events_df, ctx, source_root, stimuli_path, make_video)

        annotated_dir = derivative_root / f"sub-{ctx.subject}" / f"ses-{ctx.session}" / "events"
        annotated_dir.mkdir(parents=True, exist_ok=True)
        annotated_path = annotated_dir / (
            f"sub-{ctx.subject}_ses-{ctx.session}_task-{task_label}_"
            f"run-{ctx.run}_desc-annotated_events.tsv"
        )

        annotated_df = _write_or_load_annotated_events(
            annotated_path=annotated_path,
            runvars=runvars,
            events_df=events_df,
            overwrite=overwrite,
        )

        run_events_df = annotated_df[annotated_df['trial_type'] == 'gym-retro_game'].copy().reset_index(drop=True)
        if not run_events_df.empty:
            level_labels = []
            for _, run_row in run_events_df.iterrows():
                stim_entry = str(run_row.get('stim_file', ''))
                world, level = _parse_world_level_from_bk2name(stim_entry)
                level_labels.append(f"{world}{level}")
            run_events_df['level'] = level_labels
        else:
            run_events_df['level'] = 'n/a'
        run_events_df['value'] = 'n/a'

        trigger_windows = run_events_df[['onset', 'duration', 'level']].copy()

        bids_path, frame_events = write_bids_run_from_annotated(
            edf_path=ctx.edf_path,
            mario_annotated_tsv=annotated_path,
            bids_root=bids_root,
            subject=ctx.subject,
            session=ctx.session,
            task=task_label,
            run=ctx.run,
            line_freq=line_freq,
            overwrite=overwrite,
            run_events=trigger_windows,
        )
        written_paths.append(bids_path)

        combined_events = run_events_df[['onset', 'duration', 'trial_type', 'value', 'level', 'stim_file']].copy()
        combined_events['game_log'] = 'n/a'
        if include_behavior_events:
            behavior_events = annotated_df[annotated_df['trial_type'] != 'gym-retro_game'].copy()
            if not behavior_events.empty:
                behavior_events = behavior_events[behavior_events['trial_type'] != 'frame']
                if 'value' not in behavior_events.columns:
                    behavior_events['value'] = 'n/a'
                else:
                    behavior_events['value'] = behavior_events['value'].fillna('n/a')
                if 'level' not in behavior_events.columns:
                    behavior_events['level'] = 'n/a'
                behavior_events['level'] = behavior_events['level'].fillna('n/a')
                if 'stim_file' not in behavior_events.columns:
                    behavior_events['stim_file'] = 'n/a'
                behavior_events['stim_file'] = behavior_events['stim_file'].fillna('n/a').astype(str)
                if not run_events_df.empty:
                    starts = run_events_df['onset'].to_numpy()
                    stops = (run_events_df['onset'] + run_events_df['duration']).to_numpy()
                    try:
                        intervals = pd.IntervalIndex.from_arrays(starts, stops, closed='both')

                        def _assign_level(onset_val: float) -> str:
                            try:
                                idx = intervals.get_indexer([onset_val])[0]
                            except Exception:
                                idx = -1
                            if idx == -1:
                                return 'n/a'
                            return str(run_events_df['level'].iloc[idx])

                        behavior_events['level'] = behavior_events['onset'].astype(float).apply(_assign_level)
                    except Exception:
                        behavior_events['level'] = behavior_events['level'].astype(str)
                behavior_events['level'] = behavior_events['level'].fillna('n/a').astype(str)
                behavior_events = behavior_events[['onset', 'duration', 'trial_type', 'value', 'level', 'stim_file']]
                behavior_events['game_log'] = 'n/a'
                combined_events = pd.concat([combined_events, behavior_events], ignore_index=True, sort=False)
        if not frame_events.empty:
            combined_events = pd.concat([combined_events, frame_events], ignore_index=True, sort=False)
        combined_events['stim_file'] = combined_events['stim_file'].fillna('n/a').astype(str)
        combined_events['level'] = combined_events['level'].fillna('n/a').astype(str)
        combined_events['value'] = combined_events['value'].fillna('n/a')
        frame_mask = combined_events['trial_type'].eq('frame')
        combined_events.loc[~frame_mask, 'value'] = 'n/a'
        if frame_mask.any():
            combined_events.loc[frame_mask, 'value'] = combined_events.loc[frame_mask, 'value'].astype(int)
        combined_events = combined_events.sort_values('onset').reset_index(drop=True)

        copy_bk2_and_write_events_copy(
            events_tsv=ctx.events_path,
            bids_root=bids_root,
            subject=ctx.subject,
            session=ctx.session,
            task=task_label,
            run=ctx.run,
            events_df_override=combined_events,
        )

    _write_dataset_level_documents(bids_root, processed_subjects)
    _ensure_derivative_dataset_description(derivative_root)

    return written_paths


# -------------------- CLI --------------------

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="Mario BIDSify (all-in-one): behavior sidecars + annotations + EEG BIDS."
    )
    p.add_argument(
        "--source",
        "-s",
        required=True,
        type=str,
        help="Dataset root or sourcedata path containing 'behav' and 'eeg'."
    )
    p.add_argument(
        "--sub",
        help="Optional subject label (e.g., 01). Defaults to all subjects."
    )
    p.add_argument(
        "--ses",
        help="Optional session label (e.g., 001). Defaults to all sessions for the selected subject(s)."
    )
    p.add_argument(
        "--run",
        default=None,
        help="Optional run label filter (e.g., 01)."
    )
    p.add_argument(
        "--bids-root",
        type=str,
        default=None,
        help="Output BIDS root (default: <source>/bids)."
    )
    p.add_argument(
        "--stimuli",
        type=str,
        default=None,
        help="Override ROM/state location. Defaults to auto-populated <source>/sourcedata/stimuli."
    )
    p.add_argument("--no-video", action="store_true", help="Skip MP4 video creation for .bk2 sidecars")
    p.add_argument("--line-freq", type=int, default=60, help="Power line frequency (50 or 60)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing BIDS files")
    p.add_argument(
        "--include-behavior-events",
        action="store_true",
        help="Include button presses and gameplay events from behavioral sidecars in exported events.tsv",
    )
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    source_root = Path(args.source)
    bids_root = Path(args.bids_root) if args.bids_root else None
    stimuli = Path(args.stimuli) if args.stimuli else None

    paths = mario_bidsify_all_in_one(
        source_root=source_root,
        sub=args.sub,
        ses=args.ses,
        bids_root=bids_root,
        stimuli_path=stimuli,
        run_filter=args.run,
        make_video=not args.no_video,
        line_freq=args.line_freq,
        overwrite=args.overwrite,
        include_behavior_events=args.include_behavior_events,
    )
    for bp in paths:
        print(f"Wrote BIDS to: {bp.directory}")

if __name__ == "__main__":
    main()
