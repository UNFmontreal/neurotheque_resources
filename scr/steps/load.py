import logging
import os
from pathlib import Path

import mne

from .base import BaseStep

logger = logging.getLogger(__name__)


class LoadData(BaseStep):
    """
    Load EEG data from a file.

    Behavior:
    - If `input_file` is provided, it is preferred (subject/session/task ignored, with a warning).
    - Otherwise, if `subject_id`, `session_id`, and `paths` (ProjectPaths) are provided and a
      `task_id` is available, constructs a BIDS-like path via `paths.get_raw_eeg_path`.
    - Expands `~` and environment variables in paths and accepts case-insensitive extensions.

    Notes:
    - In multi-file mode the Pipeline injects `input_file` automatically from the
      resolved `file_path_pattern` for each subject/session/task/run, so you typically
      do not need to set `input_file` in the config.

    Supported extensions: .edf/.EDF, .fif/.FIF, .fif.gz
    """

    def run(self, data):
        sub_id = self.params.get("subject_id")
        ses_id = self.params.get("session_id")
        task_id = self.params.get("task_id")
        run_id = self.params.get("run_id")
        paths = self.params.get("paths")

        input_file = self.params.get("input_file")
        resolved_path: Path | None = None

        if input_file:
            if sub_id or ses_id or task_id or run_id:
                logger.warning("[LoadData] 'input_file' provided; subject/session/task/run parameters will be ignored.")
            resolved_path = Path(os.path.expandvars(os.path.expanduser(str(input_file)))).resolve()
        elif sub_id and ses_id and paths and task_id:
            # Try to resolve using BIDS-like helper
            resolved_path = paths.get_raw_eeg_path(sub_id, ses_id, task_id, run_id)
        else:
            raise ValueError("[LoadData] Provide 'input_file' or (subject_id, session_id, task_id, paths).")

        # Handle gzipped FIF
        if not resolved_path.exists() and str(resolved_path).endswith(".fif") and Path(str(resolved_path) + ".gz").exists():
            resolved_path = Path(str(resolved_path) + ".gz")

        if not resolved_path.exists():
            hint = ""
            if sub_id and ses_id and paths and task_id:
                candidate = paths.get_raw_eeg_path(sub_id, ses_id, task_id, run_id)
                hint = f"\n[LoadData] Candidate BIDS-like path was: {candidate}"
            raise FileNotFoundError(f"[LoadData] File not found: {resolved_path}{hint}")

        ext = resolved_path.suffix.lower()
        stim_channel = self.params.get("stim_channel", "Trigger")

        if ext == ".edf":
            try:
                raw = mne.io.read_raw_edf(resolved_path, preload=True, stim_channel=stim_channel)
            except Exception as e:
                raise ValueError(
                    f"[LoadData] Failed to read EDF at {resolved_path}. "
                    f"Hint: verify stim_channel (e.g., 'Trigger') and channel list via raw.ch_names"
                ) from e
        elif ext == ".fif" or (ext == ".gz" and str(resolved_path).lower().endswith(".fif.gz")):
            raw = mne.io.read_raw_fif(resolved_path, preload=True)
        else:
            raise ValueError(f"[LoadData] Unsupported file extension: {resolved_path.suffix}")

        return raw
