#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EEG BIDS converter for DSI-24.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import mne


from mne_bids import BIDSPath, write_raw_bids, make_dataset_description, update_sidecar_json
_HAS_MNE_BIDS = True


CONFIG_ENCODING = "utf-8-sig"

def _log(prefix: str, message: str) -> None:
    print(f"[{prefix}] {message}")

def log_info(message: str) -> None:
    _log("info", message)

def log_warn(message: str) -> None:
    _log("warn", message)

def log_plan(message: str) -> None:
    _log("plan", message)

# ---------------------------------------------------------------------------
# DSI-24 channel normalization
# ---------------------------------------------------------------------------

LEGACY_TO_1020 = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

_RAW_STIM_NAMES = {"TRIGGER","TRIG","TRG","STIM"}

_RAW_EOG_NAMES = {"EOG"}

STIM_NAMES = {name.upper() for name in _RAW_STIM_NAMES}
EOG_NAMES = {name.upper() for name in _RAW_EOG_NAMES}

DEFAULT_EEG_REFERENCE = "Pz (Common Mode Follower)"
DEFAULT_EEG_GROUND = "Fpz"

_VENDOR_PREFIX_RE = re.compile(r"^EEG\s+(?:X\d+:)?", flags=re.IGNORECASE)
_PZ_SUFFIX_RE = re.compile(r"-PZ$", flags=re.IGNORECASE)
_X_PREFIX_RE = re.compile(r"^EEG\s+X(\d+):", flags=re.IGNORECASE)


def rename_dsi_channels(raw: mne.io.BaseRaw) -> Dict[str, str]:
    """Replace legacy DSI electrode labels with their 10-20 equivalents."""
    mapping: Dict[str, str] = {ch: LEGACY_TO_1020[ch] for ch in raw.ch_names if ch in LEGACY_TO_1020}
    if mapping:
        raw.rename_channels(mapping)
    return mapping


def normalize_dsi_channel_names(raw: mne.io.BaseRaw) -> Dict[str, str]:
    """Strip vendor prefixes/suffixes so channel names line up with BIDS expectations."""
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


def set_channel_types(raw: mne.io.BaseRaw, force: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Infer DSI-24 channel types (stim, eog, misc...) for BIDS metadata."""
    mapping: Dict[str, str] = {}
    for ch in raw.ch_names:
        cu = ch.upper()
        if cu in STIM_NAMES:
            mapping[ch] = "stim"
        elif cu in EOG_NAMES:
            mapping[ch] = "eog"
        elif "ECG" in cu:
            mapping[ch] = "ecg"
        elif cu in {"A1", "A2", "M1", "M2"}:
            mapping[ch] = "misc"
        elif cu == "CM" or cu.startswith("X"):
            mapping[ch] = "misc"

    if force:
        for k, v in force.items():
            if k in raw.ch_names:
                mapping[k] = v
    if mapping:
        raw.set_channel_types(mapping)
    return mapping


def _sanitize_channel_units(raw: mne.io.BaseRaw) -> None:
    """Ensure raw._orig_units contains ASCII BIDS-compliant unit labels."""
    orig_units = getattr(raw, "_orig_units", None)
    if orig_units is None:
        raw._orig_units = {}
        orig_units = raw._orig_units
    elif not isinstance(orig_units, dict):
        raw._orig_units = dict(orig_units)  # type: ignore[arg-type]
        orig_units = raw._orig_units

    channel_types = raw.get_channel_types()
    for ch, ch_type in zip(raw.ch_names, channel_types):
        ctype = ch_type.lower()
        unit = orig_units.get(ch)
        if ctype in {"stim", "misc"}:
            orig_units[ch] = "n/a"
            continue

        if unit is None:
            if ctype in {"eeg", "eog", "ecg", "emg"}:
                orig_units[ch] = "uV"
            else:
                orig_units[ch] = "n/a"
            continue

        unit_str = str(unit)
        if unit_str == "µV":
            unit_str = "uV"
        elif unit_str.lower() == "na":
            unit_str = "n/a"
        orig_units[ch] = unit_str

def find_stim_channel(raw: mne.io.BaseRaw) -> Optional[str]:
    """Best-effort search for a stimulus channel using heuristics and set channel types."""
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


def apply_standard_montage(raw: mne.io.BaseRaw) -> bool:
    """Attach a 10-20 montage when possible; ignore failures silently."""
    try:
        from mne.channels import make_standard_montage

        montage = make_standard_montage("standard_1020")
        eeg_names = [ch for ch, kind in zip(raw.ch_names, raw.get_channel_types()) if kind == "eeg"]
        montage = montage.copy().pick_channels([ch for ch in eeg_names if ch in montage.ch_names])
        raw.set_montage(montage, match_case=False, on_missing="ignore")
        return True
    except Exception:
        return False


def prepare_dsi24_raw(
    raw: mne.io.BaseRaw,
    apply_montage: bool = True,
    force_types: Optional[Dict[str, str]] = None,
) -> mne.io.BaseRaw:
    """Run the standard normalization pipeline for DSI-24 EDF recordings."""
    normalize_dsi_channel_names(raw)
    rename_dsi_channels(raw)
    set_channel_types(raw, force=force_types)
    _sanitize_channel_units(raw)

    try:
        types = dict(zip(raw.ch_names, raw.get_channel_types()))
        stim_candidates = [ch for ch, t in types.items() if t == "stim"]
        if len(stim_candidates) > 1:
            primary = "Trigger" if "Trigger" in stim_candidates else (
                "STATUS" if "STATUS" in stim_candidates else stim_candidates[0]
            )
            for ch in stim_candidates:
                if ch != primary:
                    raw.set_channel_types({ch: "misc"})
    except Exception:
        pass

    if apply_montage:
        apply_standard_montage(raw)
    return raw

ALLOWED_SUFFIXES = {".edf", ".bdf"}


@dataclass
class RecordingSpec:
    """Normalized instructions for converting a single raw EEG file into BIDS."""
    raw_path: Path
    subject: str
    session: str
    task: str
    run: str
    stim_channel: Optional[str] = None
    line_freq: Optional[int] = None
    overwrite: bool = False
    trigger_rules: Dict[str, Any] = field(default_factory=dict)
    apply_montage: bool = False
    # Optional per-dataset channel type overrides
    force_channel_types: Dict[str, str] = field(default_factory=dict)
    behavior: Optional['BehaviorConfig'] = None

def _spec_label(spec: RecordingSpec) -> str:
    return f"{spec.subject}-{spec.session}-{spec.run}"


@dataclass
class PipelineConfig:
    """Top-level parameters that apply to the full conversion run."""
    bids_root: Path
    dataset_name: Optional[str]
    dataset_authors: List[str]
    recordings: List[RecordingSpec]
    overwrite_dataset_description: bool = True
    # Optional EEG hardware metadata to set in sidecar
    eeg_reference: Optional[str] = None
    eeg_ground: Optional[str] = None


@dataclass
class BehaviorConfig:
    enabled: bool
    source_path: Path
    source_label: str
    onset_column: str
    duration_column: str
    trial_type_column: str
    anchor_trial_types: Optional[List[str]]
    sync_method: str
    alignment_strategy: str
    write_sidecar: bool
    output_beh: bool


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _iter_raw_files(root: Path) -> Iterable[Path]:
    """Yield raw data files under ``root`` whose suffix is explicitly allowed."""
    for suffix in ALLOWED_SUFFIXES:
        yield from root.rglob(f"*{suffix}")


def _parse_entities(stem: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Extract BIDS-style ``sub``, ``ses``, ``task`` and ``run`` entities from a filename stem."""
    subject = session = task = run = None
    for part in stem.split("_"):
        if "-" not in part:
            continue
        key, value = part.split("-", 1)
        key = key.lower()
        if key == "sub" and subject is None:
            subject = value
        elif key == "ses" and session is None:
            session = value
        elif key == "task" and task is None:
            task = value
        elif key == "run" and run is None:
            run = value
    return subject, session, task, run


def _as_list(value: Any) -> Optional[List[str]]:
    """Normalize scalar or iterable config values into a list of trimmed strings."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    return [text] if text else None

def _as_set(value: Any) -> Optional[set]:
    items = _as_list(value)
    return set(items) if items else None


def _sanitize_task_label(label: Optional[str]) -> Optional[str]:
    """Make sure task labels match BIDS expectations (lowercase, no hyphens)."""
    if label is None:
        return None
    task = label.lower()
    if task.endswith("-eeg"):
        task = task[:-4]
    return task.replace("-", "").strip() or None

def _resolve_path(base_dir: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _coerce_str_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        return [str(value)]
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    return None



# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------


def _validate_config_dict(data: Dict[str, Any], source: Path) -> None:
    """Fail fast when required top-level keys are missing or malformed."""
    missing = [key for key in ("bids_root", "recordings") if key not in data]
    if missing:
        raise ValueError(f"Config {source} missing required keys: {', '.join(missing)}")
    if not isinstance(data["recordings"], list):
        raise ValueError("'recordings' must be a list of configuration blocks.")


def load_config(config_path: Path) -> PipelineConfig:
    """Parse a JSON configuration file into the normalized ``PipelineConfig`` model."""
    data = json.loads(config_path.read_text(encoding=CONFIG_ENCODING))
    _validate_config_dict(data, config_path)
    bids_root = Path(data["bids_root"]).expanduser()
    if not bids_root.is_absolute():
        bids_root = (config_path.parent / bids_root).resolve()

    recordings_cfg = data.get("recordings")
    if not isinstance(recordings_cfg, list) or not recordings_cfg:
        raise ValueError("Config must contain a non-empty 'recordings' list.")

    recordings: List[RecordingSpec] = []
    for idx, entry in enumerate(recordings_cfg):
        recordings.extend(_expand_entry(entry, config_path.parent, idx))

    if not recordings:
        raise ValueError("No recordings matched the configuration request.")

    return PipelineConfig(
        bids_root=bids_root,
        dataset_name=data.get("dataset_name"),
        dataset_authors=[str(a) for a in data.get("dataset_authors", [])],
        recordings=recordings,
        overwrite_dataset_description=bool(data.get("overwrite_dataset_description", True)),
        eeg_reference=(data.get("eeg_reference") or None),
        eeg_ground=(data.get("eeg_ground") or None),
    )


def _expand_entry(entry: Dict[str, Any], base_dir: Path, idx: int) -> List[RecordingSpec]:
    """Expand a single `recordings` block into one `RecordingSpec` per matching file."""
    if "root" not in entry:
        raise ValueError(f"Recording block #{idx} missing 'root'.")

    root = _resolve_path(base_dir, entry["root"])
    if not root.exists():
        raise FileNotFoundError(f"Recording block #{idx}: path does not exist -> {root}")

    subject_filter = _as_set(entry.get("subjects") or entry.get("subject"))
    session_filter = _as_set(entry.get("sessions") or entry.get("session"))
    run_filter = _as_set(entry.get("runs") or entry.get("run"))

    task_override = _sanitize_task_label(entry.get("task"))

    line_freq = entry.get("line_freq")
    stim_channel = entry.get("stim_channel")
    overwrite = bool(entry.get("overwrite", False))
    apply_montage = bool(entry.get("apply_montage", False))
    trigger_rules = dict(entry.get("trigger_rules", {}))
    force_channel_types = dict(entry.get("force_channel_types", {}))
    behavior_spec = _build_behavior_config(entry.get("behavior_events"), base_dir, idx)

    def _matches(value: Optional[str], allowed: Optional[set]) -> bool:
        if value is None:
            return False
        return not allowed or value in allowed

    specs: List[RecordingSpec] = []
    for raw_file in sorted(_iter_raw_files(root)):
        subject, session, raw_task, run = _parse_entities(raw_file.stem)
        if not _matches(subject, subject_filter):
            continue
        if not _matches(session, session_filter):
            continue
        run = run or "01"
        if not _matches(run, run_filter):
            continue
        task = task_override or _sanitize_task_label(raw_task) or "task"
        specs.append(
            RecordingSpec(
                raw_path=raw_file.resolve(),
                subject=subject,
                session=session,
                task=task,
                run=run,
                stim_channel=stim_channel,
                line_freq=line_freq,
                overwrite=overwrite,
                trigger_rules=trigger_rules,
                apply_montage=apply_montage,
                force_channel_types=force_channel_types,
                behavior=behavior_spec,
            )
        )

    if not specs:
        log_warn(f"No recordings matched configuration block #{idx} under {root}")
    return specs




def _build_behavior_config(cfg: Any, base_dir: Path, idx: int) -> Optional[BehaviorConfig]:
    if not cfg:
        return None
    if not isinstance(cfg, dict):
        log_warn(
            f"Recording block #{idx}: 'behavior_events' must be a mapping; got {type(cfg).__name__}."
        )
        return None
    if not bool(cfg.get("enabled", False)):
        return None

    path_value = cfg.get("path")
    if not path_value:
        log_warn(
            f"Recording block #{idx}: 'behavior_events.path' is required when enabled."
        )
        return None

    path = _resolve_path(base_dir, path_value)

    onset_col = str(cfg.get("onset_column", "onset"))
    duration_col = str(cfg.get("duration_column", "duration"))
    trial_type_col = str(cfg.get("trial_type_column", "trial_type"))

    anchors_raw = cfg.get("anchor_trial_types")
    anchors = _coerce_str_list(anchors_raw)
    if anchors_raw is not None and anchors is None:
        log_warn(
            f"Recording block #{idx}: 'behavior_events.anchor_trial_types' must be a string or list."
        )

    sync_method = str(cfg.get("sync_method", "next_frame_on")).lower()
    if sync_method not in {"next_frame_on", "nearest_frame_on"}:
        log_warn(
            f"Recording block #{idx}: unsupported behavior_events.sync_method={sync_method!r}; using 'next_frame_on'."
        )
        sync_method = "next_frame_on"

    strategy_aliases = {
        "mean_offset": "mean_offset",
        "mean": "mean_offset",
        "first_anchor": "first_anchor",
        "first": "first_anchor",
        "linear": "linear",
        "linear_drift": "linear",
        "piecewise": "piecewise",
        "piecewise_linear": "piecewise",
    }

    strategy_input = cfg.get("alignment_strategy")
    if strategy_input is None:
        strategy_input = cfg.get("drift_strategy")
    if isinstance(strategy_input, bytes):
        strategy_input = strategy_input.decode(errors="ignore")
    if strategy_input is not None and not isinstance(strategy_input, str):
        log_warn(
            f"Recording block #{idx}: 'alignment_strategy' should be a string; got {type(strategy_input).__name__}."
        )
        strategy_input = None

    strategy_key = strategy_input.lower() if isinstance(strategy_input, str) else "mean_offset"
    strategy = strategy_aliases.get(strategy_key, strategy_key)

    fit_linear_legacy = bool(cfg.get("fit_linear_drift"))
    valid_strategies = {"mean_offset", "first_anchor", "linear", "piecewise"}

    if strategy not in valid_strategies:
        if strategy_key not in {None, "mean_offset"}:
            log_warn(
                f"Recording block #{idx}: unknown alignment_strategy={strategy!r}; using 'mean_offset'."
            )
        strategy = "linear" if fit_linear_legacy else "mean_offset"
    elif fit_linear_legacy and strategy == "mean_offset":
        strategy = "linear"

    return BehaviorConfig(
        enabled=True,
        source_path=path,
        source_label=str(path_value),
        onset_column=onset_col,
        duration_column=duration_col,
        trial_type_column=trial_type_col,
        anchor_trial_types=anchors,
        sync_method=sync_method,
        alignment_strategy=strategy,
        write_sidecar=bool(cfg.get("write_sidecar", True)),
        output_beh=bool(cfg.get("output_beh", True)),
    )



# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def load_raw_file(raw_path: Path) -> mne.io.BaseRaw:
    """Load a supported raw EEG file into MNE."""
    suffix = raw_path.suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError(f"Unsupported raw format: {raw_path}")
    if suffix == ".edf":
        return mne.io.read_raw_edf(str(raw_path), preload=True, verbose=False)
    if suffix == ".bdf":
        return mne.io.read_raw_bdf(str(raw_path), preload=True, verbose=False)
    # Unreachable due to check above
    raise RuntimeError("Unsupported raw format")


def prepare_raw(
    raw: mne.io.BaseRaw,
    line_freq: Optional[int],
    apply_montage: bool,
    force_types: Optional[Dict[str, str]] = None,
) -> mne.io.BaseRaw:
    """Apply line frequency metadata and DSI-specific cleanups before conversion."""
    if line_freq is not None:
        raw.info["line_freq"] = int(line_freq)
    try:
        raw = prepare_dsi24_raw(raw, apply_montage=apply_montage, force_types=force_types)
    except Exception as exc:
        log_warn(f"prepare_dsi24_raw failed: {exc}")
        # Fallback: attempt basic normalization steps individually
        try:
            normalize_dsi_channel_names(raw)
            rename_dsi_channels(raw)
            set_channel_types(raw, force=force_types)
            _sanitize_channel_units(raw)
        except Exception as sub_exc:
            log_warn(f"DSI-24 fallback normalization failed: {sub_exc}")
    else:
        _sanitize_channel_units(raw)
    return raw


def detect_stim_channel(raw: mne.io.BaseRaw, preferred: Optional[str]) -> Optional[str]:
    """Find the best stimulus channel candidate, respecting explicit preferences."""
    if preferred:
        preferred_lower = preferred.lower()
        for ch in raw.ch_names:
            if ch.lower() == preferred_lower:
                return ch
    stim = find_stim_channel(raw)
    if stim is not None:
        return stim
    types = dict(zip(raw.ch_names, raw.get_channel_types()))
    for ch, kind in types.items():
        if kind == "stim":
            return ch
    return None


def _apply_trigger_rules(values: np.ndarray, rules: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Return possibly remapped trigger values and a boolean keep mask."""
    values = values.astype(int, copy=False)
    value_map = rules.get("value_map") or {}
    if value_map:
        mapped: List[int] = []
        for val in values:
            candidate = value_map.get(str(val), value_map.get(val, val))
            try:
                mapped.append(int(candidate))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"value_map entry for trigger {val!r} must be integer-like (got {candidate!r})."
                ) from exc
        values = np.asarray(mapped, dtype=int)

    keep = np.ones_like(values, dtype=bool)
    if bool(rules.get("drop_zero", False)):
        keep &= values != 0

    allowed = rules.get("keep_values") or rules.get("allowed_values")
    if allowed:
        try:
            allowed_set = {int(v) for v in allowed}
        except (TypeError, ValueError) as exc:
            raise ValueError("keep_values entries must be integer-like.") from exc
        keep &= np.array([int(val) in allowed_set for val in values], dtype=bool)

    return values, keep


def _trial_types_from_rules(values: np.ndarray, rules: Dict[str, Any]) -> List[str]:
    """Translate trigger values into trial_type strings using ``trial_type_map`` rules."""
    default_tt = rules.get("trial_type")
    tt_map = rules.get("trial_type_map", {})
    trial_types = []
    for val in values:
        key = str(int(val))
        trial_types.append(tt_map.get(key, default_tt or f"trigger_{int(val)}"))
    return trial_types


def _binary_edge_samples(
    raw: mne.io.BaseRaw,
    stim_name: str,
    threshold: float,
    direction: str = "rise",
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect transitions in a binary trigger channel based on a voltage threshold."""
    picks = mne.pick_channels(raw.ch_names, include=[stim_name])
    if len(picks) != 1:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    data = raw.get_data(picks=picks)[0]
    binary = (data > threshold).astype(int)
    if binary.size <= 1:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    edges = np.diff(binary)
    edge_indices = np.flatnonzero(edges != 0) + 1
    if edge_indices.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    edge_values = binary[edge_indices]
    # Filter by direction
    if direction == "rise":
        keep = edge_values == 1
    elif direction == "fall":
        keep = edge_values == 0
    else:  # both
        keep = np.ones_like(edge_values, dtype=bool)
    return edge_indices[keep].astype(int), edge_values[keep].astype(int)

def _extract_trigger_samples(raw: mne.io.BaseRaw, stim: str, spec: RecordingSpec) -> Tuple[np.ndarray, np.ndarray]:
    rules = spec.trigger_rules
    binary_edges = bool(rules.get("binary_edges", False))
    threshold = float(rules.get("binary_threshold", 0.5))
    direction = str(rules.get("edge_direction", "rise")).lower()

    if binary_edges:
        try:
            samples, values = _binary_edge_samples(raw, stim, threshold, direction=direction)
        except Exception as exc:
            log_warn(f"binary edge detection failed on {stim}: {exc}")
        else:
            if samples.size:
                return samples, values

    try:
        events = mne.find_events(
            raw,
            stim_channel=stim,
            shortest_event=1,
            consecutive=True,
            verbose=False,
        )
    except Exception as exc:
        log_warn(f"Failed to extract events from {stim}: {exc}")
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    if events.size == 0:
        log_warn(f"No discrete events detected on {stim} for {spec.raw_path.name}.")
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    return events[:, 0].astype(int), events[:, 2].astype(int)


def extract_trigger_events(raw: mne.io.BaseRaw, spec: RecordingSpec) -> pd.DataFrame:
    """Build an events table for downstream ``write_raw_bids`` consumption."""
    columns = ["onset", "duration", "trial_type", "value", "sample", "stim_channel"]
    stim_hint = spec.trigger_rules.get("stim_channel") or spec.stim_channel
    stim = detect_stim_channel(raw, stim_hint)
    if stim is None:
        log_warn(f"No stimulus channel detected for {spec.raw_path.name}; events.tsv will be empty.")
        return pd.DataFrame(columns=columns)

    samples, values = _extract_trigger_samples(raw, stim, spec)
    if samples.size == 0:
        return pd.DataFrame(columns=columns)

    sfreq = float(raw.info.get("sfreq", 1.0))
    onsets = samples.astype(float) / sfreq

    values, keep = _apply_trigger_rules(values, spec.trigger_rules)
    if not np.any(keep):
        return pd.DataFrame(columns=columns)

    onsets = onsets[keep]
    values = values[keep]
    samples = samples[keep]
    trial_types = _trial_types_from_rules(values, spec.trigger_rules)

    return pd.DataFrame(
        {
            "onset": onsets,
            "duration": np.zeros_like(onsets),
            "trial_type": trial_types,
            "value": values,
            "sample": samples,
            "stim_channel": stim,
        },
        columns=columns,
    )


def make_mne_events_and_metadata(
    events_df: pd.DataFrame,
) -> Tuple[np.ndarray, Dict[str, int], pd.DataFrame, Dict[str, str]]:
    """Return (events_array, event_id_map, event_metadata_df, extra_cols_desc).

    - events_array: shape (n, 3) with columns [sample, 0, event_code]
    - event_id_map: {trial_type -> event_code}
    - event_metadata_df: extra columns to add to events.tsv (not onset/duration/trial_type)
    - extra_cols_desc: {column_name -> description string} required by mne-bids
    """
    if events_df.empty:
        return (
            np.empty((0, 3), dtype=int),
            {},
            pd.DataFrame(),
            {},
        )

    df = events_df.sort_values("sample", kind="stable").reset_index(drop=True)

    unique_tt = list(dict.fromkeys(df["trial_type"].astype(str).tolist()))

    # Build desired code per trial_type using raw TTL values if available, with mapping 0->1 and others unchanged
    desired_code_by_tt: Dict[str, int] = {}
    used_codes: set[int] = set()
    if "value" in df.columns:
        for tt, raw_val in zip(df["trial_type"].astype(str), df["value" ].astype(int)):
            if tt in desired_code_by_tt:
                continue
            code = int(raw_val)
            code = 1 if code == 0 else code
            # Ensure uniqueness across trial_types; if collision, bump to next free positive int
            if code in used_codes:
                next_code = max(1, code)
                while next_code in used_codes:
                    next_code += 1
                code = next_code
            desired_code_by_tt[tt] = code
            used_codes.add(code)

    # Fallback if no values column or something went wrong
    if not desired_code_by_tt:
        for i, tt in enumerate(unique_tt, start=1):
            desired_code_by_tt[tt] = i

    event_id = {tt: desired_code_by_tt[tt] for tt in unique_tt}

    samples = df["sample"].astype(int).to_numpy()
    zeros = np.zeros_like(samples, dtype=int)
    codes = np.array([event_id[str(tt)] for tt in df["trial_type"]], dtype=int)
    events = np.column_stack([samples, zeros, codes])

    # No extra metadata columns requested; mne-bids will still write trial_type
    event_metadata = pd.DataFrame(index=df.index)
    extra_desc: Dict[str, str] = {}

    return events, event_id, event_metadata, extra_desc


def maybe_update_eeg_sidecar(bids_path: 'BIDSPath', ref: Optional[str], ground: Optional[str]) -> None:
    """Optionally set EEGReference/EEGGround in *_eeg.json using mne-bids."""
    if not _HAS_MNE_BIDS or (ref is None and ground is None):
        return
    entries: Dict[str, Any] = {}
    if ref:
        entries["EEGReference"] = ref
    if ground:
        entries["EEGGround"] = ground
    try:
        update_sidecar_json(bids_path, entries)  # type: ignore[misc]
    except Exception as exc:
        log_warn(f"Failed to update EEG sidecar fields: {exc}")


def _load_behavior_table(behavior: BehaviorConfig, spec: RecordingSpec) -> Optional[pd.DataFrame]:
    if not behavior.source_path.exists():
        log_warn(
            f"Behavioral file not found for {_spec_label(spec)}: {behavior.source_path}"
        )
        return None
    try:
        return pd.read_csv(behavior.source_path, sep="	")
    except Exception as exc:
        log_warn(f"Failed to read behavioral events TSV ({behavior.source_path}): {exc}")
        return None


def _get_frame_on_times(trigger_events: pd.DataFrame, spec: RecordingSpec) -> np.ndarray:
    frame_on = trigger_events[trigger_events["trial_type"] == "frame_on"]
    if frame_on.empty:
        log_warn(
            f"No 'frame_on' events available for {_spec_label(spec)}; cannot align behavioral file."
        )
        return np.empty(0, dtype=float)
    return frame_on["onset"].to_numpy()


def _select_anchor_types(trial_types: pd.Series, behavior: BehaviorConfig) -> List[str]:
    if behavior.anchor_trial_types:
        anchors = [tt for tt in behavior.anchor_trial_types if tt in trial_types.values]
    else:
        anchors = list(dict.fromkeys(trial_types))
    return anchors

def process_behavioral_events(spec: RecordingSpec, trigger_events: pd.DataFrame, bids_path: 'BIDSPath') -> None:
    behavior = spec.behavior
    if behavior is None or not behavior.enabled:
        return

    beh_df = _load_behavior_table(behavior, spec)
    if beh_df is None:
        return

    required = {
        behavior.onset_column,
        behavior.duration_column,
        behavior.trial_type_column,
    }
    missing_cols = [col for col in required if col not in beh_df.columns]
    if missing_cols:
        log_warn(
            f"Behavioral TSV missing required columns {missing_cols} -> {behavior.source_path}; skipping alignment."
        )
        return

    frame_on_times = _get_frame_on_times(trigger_events, spec)
    if frame_on_times.size == 0:
        return

    beh_df = beh_df.copy()
    try:
        beh_df[behavior.onset_column] = beh_df[behavior.onset_column].astype(float)
    except Exception as exc:
        log_warn(
            f"Behavioral onset column '{behavior.onset_column}' could not be converted to float: {exc}; skipping."
        )
        return

    trial_types_series = beh_df[behavior.trial_type_column].astype(str)
    anchor_types = _select_anchor_types(trial_types_series, behavior)
    if not anchor_types:
        log_warn(
            f"No behavioral anchor trial types available for {_spec_label(spec)}; skipping alignment."
        )
        _maybe_write_behavior_copy(behavior, beh_df, beh_df, bids_path, None, corrected=False)
        return

    anchors_df = beh_df[trial_types_series.isin(anchor_types)].copy()
    if anchors_df.empty:
        log_warn(
            f"Behavioral anchors (trial_types={anchor_types}) not present for {_spec_label(spec)}."
        )
        _maybe_write_behavior_copy(behavior, beh_df, beh_df, bids_path, None, corrected=False)
        return

    align_records = _align_anchors_to_frames(
        anchors_df,
        behavior,
        frame_on_times,
    )
    if not align_records:
        log_warn(
            f"Failed to align behavioral anchors to frame_on events for {_spec_label(spec)}."
        )
        _maybe_write_behavior_copy(behavior, beh_df, beh_df, bids_path, None, corrected=False)
        return

    drift_info = _estimate_behavior_drift(align_records, behavior)
    corrected_df = beh_df.copy()
    corrected_df[behavior.onset_column] = _apply_behavior_correction(
        beh_df[behavior.onset_column].to_numpy(),
        drift_info,
    )

    _log_behavior_summary(spec, behavior, align_records, drift_info)
    _maybe_write_behavior_copy(behavior, beh_df, corrected_df, bids_path, drift_info, corrected=True)



def _align_anchors_to_frames(
    anchors_df: pd.DataFrame,
    behavior: BehaviorConfig,
    frame_on_times: np.ndarray,
) -> List[Dict[str, Any]]:
    frame_on_times = np.asarray(frame_on_times, dtype=float)
    if frame_on_times.size == 0:
        return []

    records: List[Dict[str, Any]] = []
    for idx, row in anchors_df.iterrows():
        onset = float(row[behavior.onset_column])
        trial_type = str(row[behavior.trial_type_column])
        matched_onset = _find_frame_on_onset(onset, frame_on_times, behavior.sync_method)
        if matched_onset is None:
            continue
        delta = matched_onset - onset
        records.append(
            {
                "index": idx,
                "trial_type": trial_type,
                "behavior_onset": onset,
                "frame_on_onset": float(matched_onset),
                "delta": float(delta),
            }
        )
    return records


def _find_frame_on_onset(
    onset: float,
    frame_on_times: np.ndarray,
    method: str,
) -> Optional[float]:
    idx = np.searchsorted(frame_on_times, onset, side="left")
    if method == "next_frame_on":
        if idx >= frame_on_times.size:
            return None
        return float(frame_on_times[idx])

    # nearest_frame_on
    candidates: List[Tuple[float, float]] = []
    if idx < frame_on_times.size:
        cand = float(frame_on_times[idx])
        candidates.append((abs(cand - onset), cand))
    if idx > 0:
        cand = float(frame_on_times[idx - 1])
        candidates.append((abs(cand - onset), cand))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _estimate_behavior_drift(
    align_records: List[Dict[str, Any]],
    behavior: BehaviorConfig,
) -> Dict[str, Any]:
    align_records_sorted = sorted(align_records, key=lambda rec: rec["behavior_onset"])
    deltas = np.array([rec["delta"] for rec in align_records_sorted], dtype=float)
    onset_vals = np.array([rec["behavior_onset"] for rec in align_records_sorted], dtype=float)

    per_label: Dict[str, Dict[str, Any]] = {}
    for rec in align_records_sorted:
        label = rec["trial_type"]
        stats = per_label.setdefault(label, {"count": 0, "deltas": []})
        stats["count"] += 1
        stats["deltas"].append(rec["delta"])

    for label, stats in per_label.items():
        deltas_arr = np.array(stats["deltas"], dtype=float)
        stats["mean"] = float(deltas_arr.mean())
        stats["std"] = float(deltas_arr.std(ddof=1)) if deltas_arr.size > 1 else 0.0
        stats["min"] = float(deltas_arr.min())
        stats["max"] = float(deltas_arr.max())
        del stats["deltas"]

    overall_stats = {
        "count": int(deltas.size),
        "mean": float(deltas.mean()),
        "std": float(deltas.std(ddof=1)) if deltas.size > 1 else 0.0,
        "min": float(deltas.min()),
        "max": float(deltas.max()),
    }

    drift_model = _compute_behavior_model(behavior.alignment_strategy, align_records_sorted)

    return {
        "overall_stats": overall_stats,
        "per_label": per_label,
        "model": drift_model,
        "records": align_records_sorted,
        "strategy": drift_model["type"],
    }


def _build_linear_model(onsets: np.ndarray, deltas: np.ndarray) -> Optional[Dict[str, Any]]:
    if onsets.size < 2:
        return None
    reference = float(onsets.mean())
    centered = onsets - reference
    try:
        slope, intercept = np.polyfit(centered, deltas, 1)
    except Exception as exc:
        log_warn(f"Linear drift fit failed ({exc}); reverting to mean offset.")
        return None
    return {
        "type": "linear",
        "offset": float(intercept),
        "slope": float(slope),
        "reference": reference,
    }


def _build_piecewise_model(align_records_sorted: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if len(align_records_sorted) < 2:
        return None

    segments: List[Dict[str, float]] = []
    for current, nxt in zip(align_records_sorted, align_records_sorted[1:]):
        b_start = float(current["behavior_onset"])
        b_end = float(nxt["behavior_onset"])
        f_start = float(current["frame_on_onset"])
        f_end = float(nxt["frame_on_onset"])
        slope = 1.0 if b_end == b_start else (f_end - f_start) / (b_end - b_start)
        intercept = f_start - slope * b_start
        segments.append(
            {
                "behavior_start": b_start,
                "behavior_end": b_end,
                "frame_start": f_start,
                "frame_end": f_end,
                "slope": slope,
                "intercept": intercept,
            }
        )

    if not segments:
        return None

    pre = {
        "slope": segments[0]["slope"],
        "intercept": segments[0]["intercept"],
        "behavior_end": segments[0]["behavior_start"],
    }
    post = {
        "slope": segments[-1]["slope"],
        "intercept": segments[-1]["intercept"],
        "behavior_start": segments[-1]["behavior_end"],
    }
    return {
        "type": "piecewise",
        "segments": segments,
        "pre": pre,
        "post": post,
    }

def _compute_behavior_model(strategy: str, align_records_sorted: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not align_records_sorted:
        return {"type": "mean_offset", "offset": 0.0}

    deltas = np.array([rec["delta"] for rec in align_records_sorted], dtype=float)
    onsets = np.array([rec["behavior_onset"] for rec in align_records_sorted], dtype=float)

    if strategy == "first_anchor":
        offset = float(align_records_sorted[0]["delta"])
        return {"type": "first_anchor", "offset": offset}

    if strategy == "linear":
        model = _build_linear_model(onsets, deltas)
        if model is not None:
            return model
        offset = float(deltas.mean())
        return {"type": "mean_offset", "offset": offset}

    if strategy == "piecewise":
        model = _build_piecewise_model(align_records_sorted)
        if model is not None:
            return model

    offset = float(deltas.mean())
    return {"type": "mean_offset", "offset": offset}



def _apply_behavior_correction(onsets: np.ndarray, drift_info: Dict[str, Any]) -> np.ndarray:
    model = drift_info["model"]
    strategy = model.get("type", "mean_offset")
    onsets = np.asarray(onsets, dtype=float)

    if strategy == "first_anchor" or strategy == "mean_offset":
        offset = float(model.get("offset", 0.0))
        return onsets + offset

    if strategy == "linear":
        offset = float(model.get("offset", 0.0))
        slope = float(model.get("slope", 0.0))
        reference = float(model.get("reference", 0.0))
        return onsets + offset + slope * (onsets - reference)

    if strategy == "piecewise":
        segments = model.get("segments", [])
        if not segments:
            return onsets
        behavior_breaks = [seg["behavior_end"] for seg in segments]
        behavior_starts = [seg["behavior_start"] for seg in segments]
        pre = model.get("pre", {})
        post = model.get("post", {})

        corrected = np.empty_like(onsets)
        for idx, t in enumerate(onsets):
            if t < behavior_starts[0]:
                slope = float(pre.get("slope", segments[0]["slope"]))
                intercept = float(pre.get("intercept", segments[0]["intercept"]))
            elif t >= behavior_breaks[-1]:
                slope = float(post.get("slope", segments[-1]["slope"]))
                intercept = float(post.get("intercept", segments[-1]["intercept"]))
            else:
                seg_idx = np.searchsorted(behavior_breaks, t, side="right")
                seg = segments[min(seg_idx, len(segments) - 1)]
                slope = seg["slope"]
                intercept = seg["intercept"]
            corrected[idx] = slope * t + intercept
        return corrected

    return onsets


def _maybe_write_behavior_copy(
    behavior: BehaviorConfig,
    original_df: pd.DataFrame,
    corrected_df: pd.DataFrame,
    bids_path: 'BIDSPath',
    drift_info: Optional[Dict[str, Any]],
    corrected: bool,
) -> None:
    if not behavior.output_beh:
        return

    beh_bids_path = bids_path.copy().update(datatype="beh", suffix="beh", extension=".tsv")
    beh_dir = Path(beh_bids_path.directory)
    beh_dir.mkdir(parents=True, exist_ok=True)

    output_df = corrected_df if corrected else original_df
    output_df.to_csv(beh_bids_path.fpath, sep="\t", index=False, float_format="%.9f")

    if behavior.write_sidecar:
        sidecar_path = beh_bids_path.copy().update(extension=".json")
        payload = _build_behavior_sidecar(behavior, drift_info, bids_path, corrected)
        Path(sidecar_path.directory).mkdir(parents=True, exist_ok=True)
        Path(sidecar_path.fpath).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_behavior_sidecar(
    behavior: BehaviorConfig,
    drift_info: Optional[Dict[str, Any]],
    bids_path: 'BIDSPath',
    corrected: bool,
) -> Dict[str, Any]:
    relative_eeg_events = None
    root_path = Path(bids_path.root).resolve() if bids_path.root else None
    try:
        eeg_events_path = Path(bids_path.copy().update(suffix="events", extension=".tsv").fpath).resolve()
        if root_path:
            relative_eeg_events = str(eeg_events_path.relative_to(root_path))
        else:
            relative_eeg_events = str(eeg_events_path)
    except Exception:
        relative_eeg_events = str(bids_path.copy().update(suffix="events", extension=".tsv").fpath)

    payload: Dict[str, Any] = {
        "BehavioralSourceFile": behavior.source_label,
        "SynchronizationMethod": behavior.sync_method,
        "AlignmentStrategy": behavior.alignment_strategy,
        "CorrectionApplied": corrected,
        "EEGEventsFile": relative_eeg_events,
    }

    if drift_info:
        payload["AnchorStatistics"] = drift_info["per_label"]
        payload["OverallDeltaStatistics"] = drift_info["overall_stats"]
        payload["DriftModel"] = drift_info["model"]
        payload["AnchorsCount"] = len(drift_info["records"])
    else:
        payload["AnchorStatistics"] = {}
        payload["OverallDeltaStatistics"] = {}
        payload["DriftModel"] = {"type": "none"}
        payload["AnchorsCount"] = 0

    return payload


def _log_behavior_summary(
    spec: RecordingSpec,
    behavior: BehaviorConfig,
    align_records: List[Dict[str, Any]],
    drift_info: Dict[str, Any],
) -> None:
    overall = drift_info["overall_stats"]
    model = drift_info["model"]
    strategy = drift_info.get("strategy", model.get("type", "mean_offset"))
    if strategy == "linear":
        model_desc = (
            f"linear (offset={model.get('offset', 0.0):.4f}s"
            f" slope={model.get('slope', 0.0):.6f}s/s)"
        )
    elif strategy == "piecewise":
        segments = model.get("segments", [])
        model_desc = f"piecewise (segments={len(segments)})"
    elif strategy == "first_anchor":
        model_desc = f"first_anchor (offset={model.get('offset', 0.0):.4f}s)"
    else:
        model_desc = f"mean_offset (offset={model.get('offset', 0.0):.4f}s)"
    log_info(
        "[behavior] "
        f"sub-{spec.subject}_ses-{spec.session}_run-{spec.run}: anchors={overall['count']} mean_delta={overall['mean']:.4f}s std={overall['std']:.4f}s model={model_desc}"
    )
def ensure_dataset_description(root: Path, name: Optional[str], authors: List[str], overwrite: bool) -> None:
    """Create or update the dataset_description.json required by BIDS."""
    if not _HAS_MNE_BIDS:
        log_warn("mne-bids not available; skipping dataset description update.")
        return
    root.mkdir(parents=True, exist_ok=True)
    try:
        make_dataset_description(
            path=str(root),
            name=name or "EEG dataset",
            authors=authors or None,
            dataset_type="raw",
            overwrite=overwrite,
        )
    except Exception:
        pass


def _write_bids_dataset(raw: mne.io.BaseRaw, bids_path: 'BIDSPath', write_kwargs: Dict[str, Any]) -> None:
    montage = None
    try:
        montage = raw.get_montage()
    except Exception:
        montage = None
    if montage is not None:
        try:
            raw.set_montage(None)
        except Exception:
            montage = None

    try:
        raw.set_annotations(None)
    except Exception:
        pass

    write_raw_bids(
        raw,
        bids_path,
        event_metadata=None,
        extra_columns_descriptions=None,
        **write_kwargs,
    )

    if montage is not None:
        try:
            raw.set_montage(montage)
        except Exception:
            pass

def process_recording(spec: RecordingSpec, cfg: PipelineConfig, dry_run: bool = False) -> Optional['BIDSPath']:
    """Convert one recording to BIDS, returning the resulting ``BIDSPath``."""
    log_plan(
        f"{spec.raw_path} -> sub-{spec.subject}_ses-{spec.session}_task-{spec.task}_run-{spec.run}"
    )
    if dry_run:
        return None
    if not _HAS_MNE_BIDS:
        raise RuntimeError("mne-bids is required to run the conversion. Please install 'mne-bids'.")

    raw = load_raw_file(spec.raw_path)
    raw = prepare_raw(raw, spec.line_freq, spec.apply_montage, spec.force_channel_types)

    trigger_events = extract_trigger_events(raw, spec)
    events_array, event_id_map, _, _ = make_mne_events_and_metadata(trigger_events)

    bids_path = BIDSPath(
        subject=spec.subject,
        session=spec.session,
        task=spec.task,
        run=spec.run,
        datatype="eeg",
        root=str(cfg.bids_root),
    )

    write_kwargs: Dict[str, Any] = {
        "overwrite": spec.overwrite,
        "allow_preload": True,
        "format": "BrainVision",
        "verbose": False,
    }
    if events_array.size:
        write_kwargs["events"] = events_array
        if event_id_map:
            write_kwargs["event_id"] = event_id_map

    _write_bids_dataset(raw, bids_path, write_kwargs)
    maybe_update_eeg_sidecar(bids_path, cfg.eeg_reference, cfg.eeg_ground)
    process_behavioral_events(spec, trigger_events, bids_path)
    log_info(f"wrote {bids_path.fpath}")
    return bids_path



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI entry point definition for running the converter as a script."""
    parser = argparse.ArgumentParser(description="EEG BIDS converter")
    parser.add_argument("--config", required=True, help="Path to JSON configuration file")
    parser.add_argument("--dry-run", action="store_true", help="Plan actions without writing output")
    return parser


def run_from_config(config_path: Path, dry_run: bool = False) -> None:
    """Load configuration and process every requested recording."""
    cfg = load_config(config_path)
    if not _HAS_MNE_BIDS and not dry_run:
        raise RuntimeError("mne-bids is required for conversion. Install 'mne-bids'.")
    if not dry_run:
        ensure_dataset_description(
            cfg.bids_root,
            cfg.dataset_name,
            cfg.dataset_authors,
            overwrite=cfg.overwrite_dataset_description,
        )
        log_info(f"Dataset description ready at {cfg.bids_root}")
    else:
        log_info(f"[dry-run] would create/update dataset at {cfg.bids_root}")
    for spec in cfg.recordings:
        process_recording(spec, cfg, dry_run=dry_run)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_from_config(Path(args.config).resolve(), dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
