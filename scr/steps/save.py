# File: src/steps/save.py

import mne
import logging
from pathlib import Path
from .base import BaseStep

class SaveData(BaseStep):
    """
    Step to save the current data (Raw or Epochs) to disk.
    The pipeline rewrites the output_path to include subject/session 
    if multi-subject mode is active, so no extra logic is needed here.
    
    YAML params example:
      - name: SaveData
        params:
          output_path: "data/pilot_data/raw_preprocessed.fif"
          overwrite: true
    """

    def run(self, data):
        """
        Expected params:
         - output_path (str): path to save the .fif file (already adjusted for sub/ses by pipeline)
         - overwrite (bool, default True): whether to overwrite existing file
        """
        if data is None:
            raise ValueError("[SaveData] No data to save.")

        output_path_str = self.params.get("output_path", "output.fif")
        overwrite = self.params.get("overwrite", True)

        output_path = Path(output_path_str)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logging.info(f"[SaveData] Saving data => {output_path}")
        if isinstance(data, mne.io.BaseRaw):
            data.save(str(output_path), overwrite=overwrite)
        elif isinstance(data, mne.Epochs):
            data.save(str(output_path), overwrite=overwrite)
        else:
            raise TypeError("[SaveData] Only saving Raw or Epochs is currently supported.")

        return data
