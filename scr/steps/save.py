# File: scr/steps/save.py

import logging
from pathlib import Path
from .base import BaseStep
import pickle
class SaveData(BaseStep):
    """
    Save derivatives to disk (Raw or Epochs) in a standardized location.

    Use this when you want a final derivative file (not a resume checkpoint).
    For resume points during a pipeline run, prefer the dedicated SaveCheckpoint step
    which uses the convention 'after_<step>' and is recognized by the resume logic.

    Example YAML usage:
    - name: SaveData
      params:
        output_path: "data/processed/sub-01/ses-001/sub-01_ses-001_desc-clean-epo.fif"
        overwrite: true
    """

    def run(self, data):
        if data is None:
            raise ValueError("[SaveData] No data to save.")

        sub_id = self.params.get("subject_id")
        ses_id = self.params.get("session_id")
        task_id = self.params.get("task_id")
        run_id = self.params.get("run_id")
        paths = self.params.get("paths")
        
        # Get the output path directly or from a specific checkpoint function
        output_path = self.params.get("output_path")
        
        if output_path:
            # Use specified output path
            ckpt_path = Path(output_path)
        else:
            # Use checkpoint path from paths object
            ckpt_path = paths.get_derivative_path(
                subject_id=sub_id,
                session_id=ses_id,
                task_id=task_id,
                run_id=run_id,
                stage="processed"
            )
        
        # Make sure parent directory exists
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save the data
        data.save(str(ckpt_path), overwrite=True)   
        
        # If there's an autoreject log, store it in a separate .pkl
        if hasattr(data.info, 'temp') and data.info.get('temp') and "autoreject_log" in data.info["temp"]:
            log_path = Path(ckpt_path).with_name(Path(ckpt_path).stem + "_rejectlog.pkl")
            with open(log_path, "wb") as f:
                pickle.dump(data.info["temp"]["autoreject_log"], f)

        logging.info(f"Saved data to: {ckpt_path}")
        return data
