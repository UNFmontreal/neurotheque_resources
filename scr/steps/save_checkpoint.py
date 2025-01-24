# src/steps/save_checkpoint.py
import mne
from pathlib import Path
from .base import BaseStep
import pickle

class SaveCheckpoint(BaseStep):
    def run(self, data):
        output_path = Path(self.params["output_path"])
        project_root = Path(__file__).resolve().parent.parent.parent
        
        # Save Raw data
        raw_path = project_root / output_path
        data.save(raw_path, overwrite=True)
        
        # Save reject_log separately if it exists
        if "autoreject_log" in data.info.get("temp", {}):
            log_path = raw_path.with_name(raw_path.stem + "_rejectlog.pkl")
            with open(log_path, "wb") as f:
                pickle.dump(data.info["temp"]["autoreject_log"], f)
        
        return data