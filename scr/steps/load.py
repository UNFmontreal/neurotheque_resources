import mne
from pathlib import Path
from .base import BaseStep

class LoadData(BaseStep):
    """
    Step to load EEG data from a file. Supports EDF, FIF, etc.
    """

    def run(self, data):
        """
        If multi-subject mode is active, pipeline sets self.params['input_file'] 
        for each subject's file. Otherwise, we fallback to file_path (single-subject).
        
        Expected params:
        - input_file (str): actual path from pipeline (multi-subject) 
        - file_path  (str): fallback for single-subject
        - stim_channel (str): e.g., 'Trigger'
        """
        # 1) Resolve the final path to load
        input_file = self.params.get("input_file", None)  # from pipeline in multi-subject
        fallback = self.params.get("file_path", None)     # original single-subject usage

        # We'll build an absolute path from either input_file or fallback
        project_root = Path(__file__).resolve().parent.parent.parent

        if input_file:
            file_path = project_root / input_file
        else:
            if not fallback:
                raise ValueError("[LoadData] No file specified in 'input_file' or 'file_path'.")
            file_path = project_root / fallback

        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        stim_channel = self.params.get("stim_channel", "Trigger")

        # 2) Load based on file suffix
        if file_path.suffix == ".edf":
            raw = mne.io.read_raw_edf(file_path, preload=True, stim_channel=stim_channel)
        elif file_path.suffix == ".fif":
            raw = mne.io.read_raw_fif(file_path, preload=True)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        # 3) Return the loaded Raw
        return raw
