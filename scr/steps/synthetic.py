from __future__ import annotations

import numpy as np
import mne

from .base import BaseStep


class SyntheticRawStep(BaseStep):
    """
    Generate a tiny synthetic Raw object for smoke testing.

    params:
    - duration_sec: float (default 2.0)
    - sfreq: float (default 100.0)
    - n_eeg: int (default 4)
    - add_stim: bool (default True)
    """

    def run(self, data):
        duration = float(self.params.get("duration_sec", 2.0))
        sfreq = float(self.params.get("sfreq", 100.0))
        n_eeg = int(self.params.get("n_eeg", 4))
        add_stim = bool(self.params.get("add_stim", True))

        n_times = int(duration * sfreq)
        ch_names = [f"EEG{i+1:03d}" for i in range(n_eeg)]
        ch_types = ["eeg"] * n_eeg
        if add_stim:
            ch_names.append("Trigger")
            ch_types.append("stim")

        info = mne.create_info(ch_names, sfreq, ch_types)
        rng = np.random.RandomState(42)
        data = rng.randn(len(ch_names), n_times) * 1e-6
        raw = mne.io.RawArray(data, info)
        return raw

