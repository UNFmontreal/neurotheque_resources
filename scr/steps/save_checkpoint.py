# src/steps/save_checkpoint.py
import mne
from pathlib import Path
from .base import BaseStep
import pickle
import os

class SaveCheckpoint(BaseStep):
    def run(self, data):
        sub_id = self.params["subject_id"]
        ses_id = self.params["session_id"]
        checkpoint_name = self.params.get("checkpoint_key", "after_autoreject")
        
        ckpt_path = self.params["paths"].get_checkpoint_path(
            subject_id=sub_id,
            session_id=ses_id,
            checkpoint_name=checkpoint_name
        )
        ckpt_path = os.path.join(ckpt_path, f"sub-{sub_id}_ses-{ses_id}_desc-{checkpoint_name}_eeg.fif")
        data.save(ckpt_path, overwrite=True)