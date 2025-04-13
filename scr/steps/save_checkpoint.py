# src/steps/save_checkpoint.py
import mne
from pathlib import Path
from .base import BaseStep
import pickle
import os
import logging

class SaveCheckpoint(BaseStep):
    def run(self, data):
        sub_id = self.params["subject_id"]
        ses_id = self.params["session_id"]
        task_id = self.params.get("task_id", None)
        run_id = self.params.get("run_id", None)
        checkpoint_name = self.params.get("checkpoint_key", "after_autoreject")
        
        ckpt_path = self.params["paths"].get_checkpoint_path(
            subject_id=sub_id,
            session_id=ses_id,
            task_id=task_id,
            run_id=run_id,
            checkpoint_name=checkpoint_name
        )
        
        logging.info(f"Saving checkpoint to: {ckpt_path}")
        data.save(ckpt_path, overwrite=True)
        
        # Note that autoreject info is saved as annotations directly in the data
        # No need to save additional files
        
        return data