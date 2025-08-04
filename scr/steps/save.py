import logging
from pathlib import Path
from .base import BaseStep
import pickle


class SaveData(BaseStep):
    """
    Step to save the data in a BIDS-compliant format.

    Example YAML usage:
    ------------------
    - name: SaveData
      params:
        processing: "cleaned"  # e.g., 'cleaned', 'ica', 'epoched'
        suffix: "eeg"
        extension: ".fif"
    """

    def run(self, data):
        if data is None:
            raise ValueError("[SaveData] No data to save.")

        paths = self.params.get("paths")
        if not paths:
            raise ValueError("[SaveData] 'paths' object not found in params.")

        # Get BIDS entities from params
        bids_params = {
            "subject": self.params.get("subject_id"),
            "session": self.params.get("session_id"),
            "task": self.params.get("task_id"),
            "run": self.params.get("run_id"),
            "processing": self.params.get("processing"),
            "suffix": self.params.get("suffix", "eeg"),
            "extension": self.params.get("extension", ".fif"),
        }

        # Check for required BIDS entities
        if not all([bids_params["subject"], bids_params["session"]]):
            raise ValueError("[SaveData] Missing subject_id or session_id.")

        # Generate the BIDS-compliant path
        output_bids_path = paths.get_bids_path(**bids_params)

        # Save the data
        data.save(output_bids_path, overwrite=True)
        logging.info(f"Saved data to: {output_bids_path}")

        # If there's an autoreject log, store it with a similar BIDS name
        if hasattr(data.info, "temp") and "autoreject_log" in data.info.get("temp", {}):
            log_bids_path = output_bids_path.copy().update(
                suffix="rejectlog", extension=".pkl"
            )
            with open(log_bids_path, "wb") as f:
                pickle.dump(data.info["temp"]["autoreject_log"], f)
            logging.info(f"Saved autoreject log to: {log_bids_path}")

        return data
