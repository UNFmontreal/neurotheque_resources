import mne
from pathlib import Path
from .base import BaseStep

class LoadData(BaseStep):
    """
        Step to load EEG data from a file. 
        If multi-subject, we rely on subject_id, session_id, 
        but we might also just read input_file directly if it was set.
    """

    def run(self, data):

        sub_id = self.params.get("subject_id", None)
        ses_id = self.params.get("session_id", None)
        paths = self.params.get("paths", None)

        file_path = self.params.get("input_file", None)

        if not file_path:
            # If we wanted to auto-generate the path from ProjectPaths:
            if sub_id and ses_id and paths:
                file_path = paths.get_raw_input_file(sub_id, ses_id)
            else:
                raise ValueError("[LoadData] No input_file or subject/session set.")

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        stim_channel = self.params.get("stim_channel", "Trigger")

        if file_path.suffix == ".edf":
            stim_channel = self.params.get("stim_channel", "Trigger")
            raw = mne.io.read_raw_edf(file_path, preload=True, stim_channel=stim_channel)
        elif file_path.suffix == ".fif":
            raw = mne.io.read_raw_fif(file_path, preload=True)
        else:
            raise ValueError(f"Unsupported file extension: {file_path.suffix}")

        return raw
