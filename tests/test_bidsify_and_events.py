import json
import os
from pathlib import Path

import mne
import numpy as np
import pytest
from mne_bids import BIDSPath

from event_mapping import map_events_from_config
from dsi_bids import parse_entities_from_name, build_bids_path
from dsi_bids import _patch_eeg_sidecar as patch_sidecar  # internal test


def _make_synth_raw(sfreq=250.0, n_sec=2.0):
    ch_names = [f"EEG{i:02d}" for i in range(4)] + ["Trigger"]
    ch_types = ["eeg"] * 4 + ["stim"]
    info = mne.create_info(ch_names, sfreq=sfreq, ch_types=ch_types)
    data = np.zeros((len(ch_names), int(sfreq * n_sec)))
    raw = mne.io.RawArray(data, info)
    return raw


def test_event_mapping_debounce_and_shift(tmp_path):
    raw = _make_synth_raw(sfreq=200.0)
    # events: 100, 101, 102 samples with same code=1 -> should collapse if debounce=10ms (2 samples)
    ev = np.array([[100, 0, 1], [101, 0, 1], [102, 0, 1], [300, 0, 2]], dtype=int)
    cfg = {
        "event_map": {"A": [1], "B": [2]},
        "onset_shift_sec": 0.005,  # +1 sample at 200 Hz
        "debounce_ms": 10.0
    }
    mapped, event_id = map_events_from_config(raw, ev, cfg)
    # collapsed to first '1' at 100 -> shift +1 => 101
    assert mapped.shape[0] == 2
    assert mapped[0, 0] == 101
    assert set(event_id.values()) == {1, 2}


def test_filename_parsing_patterns_and_fallbacks():
    patterns = [
        r"sub-(?P<subject>\d+)_(?:ses-(?P<session>\d+)_)?task-(?P<task>[a-z0-9]+)(?:_run-(?P<run>\d+))?\.edf$",
        r"[sS](?P<subject>\d{2})_(?P<task>gonogo).*run[_-]?(?P<run>\d{2})\.edf$",
    ]
    fallbacks = {"subject": "99", "session": "01", "task": "gonogo", "run": "01"}
    p = Path("S01_gonogo_run1.edf")
    ent = parse_entities_from_name(p, patterns, fallbacks)
    assert ent["subject"] == "01"
    assert ent["task"].lower() == "gonogo"
    assert ent["run"] == "01"
    assert ent["session"] == "01"  # from fallback


def test_bids_write_smoke(tmp_path):
    bids_root = tmp_path / "bids"
    raw = _make_synth_raw()
    bp = BIDSPath(subject="01", session="01", task="demo", run="01", datatype="eeg", root=bids_root)
    # Use mne-bids directly for write smoke. No events for simplicity.
    from mne_bids import write_raw_bids, make_dataset_description

    make_dataset_description(path=bids_root, name="Test", dataset_type="raw", overwrite=True)
    write_raw_bids(raw, bp, overwrite=True)
    # patch sidecar required fields
    patch_sidecar(bp, "average", 60)
    sidecar = bp.copy().update(suffix="eeg", extension=".json")
    meta = json.loads(Path(sidecar.fpath).read_text())
    assert meta["PowerLineFrequency"] == 60

