import logging
from .base import BaseStep

class AutoSave(BaseStep):
    """
    Automatically saves the current data with a step-specific name in a BIDS-compliant format.
    
    Expected params:
    ----------------
    subject_id (str): Subject ID
    session_id (str): Session ID
    step_name (str): Name of the completed step (e.g., "filtered", "ica")
    paths (ProjectPaths): Path object for constructing save paths
    ... and other optional BIDS entities (task_id, run_id)
    """
    
    def run(self, data):
        if data is None:
            logging.warning("[AutoSave] No data to save.")
            return None
        
        paths = self.params.get("paths")
        if not paths:
            logging.error("[AutoSave] 'paths' object not found in params.")
            return data

        step_name = self.params.get("step_name", "checkpoint")

        bids_params = {
            'subject': self.params.get("subject_id"),
            'session': self.params.get("session_id"),
            'task': self.params.get("task_id"),
            'run': self.params.get("run_id"),
            'processing': f"after_{step_name}",
            'suffix': 'eeg',
            'extension': '.fif'
        }

        if not all([bids_params['subject'], bids_params['session']]):
            logging.error("[AutoSave] Missing subject_id or session_id.")
            return data

        save_path = paths.get_bids_path(**bids_params)
        
        logging.info(f"[AutoSave] Saving data after {step_name} step to: {save_path}")
        data.save(save_path, overwrite=True)
        
        return data
