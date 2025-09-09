"""Filtering step for Raw/Epochs with simple validation and hints."""

from __future__ import annotations

from typing import Iterable, Optional

import mne

from .base import BaseStep


class FilterStep(BaseStep):
    """
    Apply band‑pass and optional notch filtering to an MNE object.

    Params keys:
    - l_freq: float | None (default 1.0)
    - h_freq: float | None (default 40.0)
    - notch_freqs: list[float] | tuple[float, ...] (optional)
    - fir_design: str (default "firwin")
    """

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__()
        self.params = params or {}

    def run(self, data: mne.io.BaseRaw | mne.Epochs) -> mne.io.BaseRaw | mne.Epochs:
        if data is None:
            raise ValueError("[FilterStep] No data provided to filter.")

        l_freq = self.params.get("l_freq", 1.0)
        h_freq = self.params.get("h_freq", 40.0)
        notch_freqs = self.params.get("notch_freqs", [])
        fir_design = self.params.get("fir_design", "firwin")

        # Validate band edges
        if l_freq is not None and not isinstance(l_freq, (int, float)):
            raise ValueError("[FilterStep] l_freq must be a number or None.")
        if h_freq is not None and not isinstance(h_freq, (int, float)):
            raise ValueError("[FilterStep] h_freq must be a number or None.")
        if l_freq is not None and h_freq is not None and l_freq >= h_freq:
            raise ValueError(f"[FilterStep] l_freq ({l_freq}) must be < h_freq ({h_freq}).")

        # Validate notch freqs
        if notch_freqs:
            if not isinstance(notch_freqs, Iterable) or isinstance(notch_freqs, (str, bytes)):
                raise ValueError("[FilterStep] notch_freqs must be a list/tuple of numbers.")
            for f in notch_freqs:
                if not isinstance(f, (int, float)):
                    raise ValueError("[FilterStep] notch_freqs must contain numbers only.")

        data.filter(l_freq=l_freq, h_freq=h_freq, fir_design=fir_design)
        if notch_freqs:
            data.notch_filter(freqs=list(notch_freqs), fir_design=fir_design)
        return data
