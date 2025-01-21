# src/steps/load.py

import mne
from pathlib import Path
from .base import BaseStep

class LoadData(BaseStep):
    """
    Step to load EEG data from a file. Supports EDF, FIF, etc.
    """

    def run(self, data):
        """
        Since this is a loader step, `data` is typically None coming in.
        We return an mne.Raw object.

        Expected params:
        - file_path (str): path to data file
        - stim_channel (str): e.g., 'Trigger'
        """
        file_path = Path(self.params.get("file_path"))
        stim_channel = self.params.get("stim_channel", "Trigger")

        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        if file_path.suffix == ".edf":
            raw = mne.io.read_raw_edf(file_path, preload=True, stim_channel=stim_channel)
        elif file_path.suffix == ".fif":
            raw = mne.io.read_raw_fif(file_path, preload=True)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        # Possibly do some initial channel naming or type setting here
        return raw
