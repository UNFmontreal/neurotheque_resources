# File: scr/steps/auto_save.py
import mne
from pathlib import Path
from .base import BaseStep
import os
import logging

class AutoSave(BaseStep):
    """
    Automatically saves the current data with a step-specific name.
    
    This can be inserted after each preprocessing step to create save points.
    
    Expected params:
    --------------------------------------------------------------------------
    subject_id (str): Subject ID
    session_id (str): Session ID
    task_id (str, optional): Task ID
    run_id (str, optional): Run ID
    step_name (str): Name of the preprocessing step that was just completed (e.g., "filtered", "ica", "autoreject")
    paths (ProjectPaths): Path object for constructing save paths
    """
    
    def run(self, data):
        if data is None:
            logging.warning("[AutoSave] No data to save")
            return None
        
        sub_id = self.params.get("subject_id")
        ses_id = self.params.get("session_id")
        task_id = self.params.get("task_id", None)
        run_id = self.params.get("run_id", None)
        step_name = self.params.get("step_name", "unknown_step")
        
        # Get the paths object
        paths = self.params.get("paths")
        if not paths:
            logging.error("[AutoSave] No paths object provided")
            return data
            
        # Get the save path using the checkpoint path with the step name
        save_path = paths.get_checkpoint_path(
            subject_id=sub_id,
            session_id=ses_id,
            task_id=task_id,
            run_id=run_id,
            checkpoint_name=f"after_{step_name}"
        )
        
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(str(save_path)), exist_ok=True)
        
        # Save the data
        logging.info(f"[AutoSave] Saving data after {step_name} step to: {save_path}")
        data.save(save_path, overwrite=True)
        
        # Return the data unchanged
        return data 
