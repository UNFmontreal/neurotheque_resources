# File: src/steps/save.py

import logging
from pathlib import Path
from .base import BaseStep
import pickle
class SaveData(BaseStep):
    """
    Step to save the data as an 'after_autoreject' checkpoint. 
    That means next run can skip 'AutoRejectStep' if the file is found.
    
    Example YAML usage:
    ------------------
    - name: SaveCheckpoint
      params:
        checkpoint_key: "after_autoreject"  # or "post_ica", etc.
    """

    def run(self, data):

        if data is None:
            raise ValueError("[SaveData] No data to save.")

        
        sub_id = self.params["subject_id"]
        ses_id = self.params["session_id"]
        paths = self.params["paths"]
        
        # We'll default to "after_autoreject" if not specified

    # some fallback or raise an error  
        
        # If "after_autoreject", we typically do "paths.get_autoreject_checkpoint(...)"
        # but you might have multiple keys => define more in ProjectPaths if needed
        # if ckpt_key == "after_autoreject":
        #     ckpt_path = paths.get_autoreject_checkpoint(sub_id, ses_id)
        # elif ckpt_key == "after_ica":
        #     ckpt_path = paths.get_ica_checkpoint(sub_id, ses_id)
        # else:
        
        ckpt_path = paths.get_autoreject_checkpoint(sub_id, ses_id)
        paths.ensure_parent(ckpt_path) 
        data.save(str(ckpt_path), overwrite=True)   
        
        
        # If there's an autoreject log, store it in a separate .pkl
        if "temp" in data.info and "autoreject_log" in data.info["temp"]:
            log_path = ckpt_path.with_name(ckpt_path.stem + "_rejectlog.pkl")
            with open(log_path, "wb") as f:
                pickle.dump(data.info["temp"]["autoreject_log"], f)

        print(f"[SaveCheckpoint] Saved => {ckpt_path}")
        return data
