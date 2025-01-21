# src/steps/save.py

import mne
from pathlib import Path
from .base import BaseStep

class SaveData(BaseStep):
    """
    Step to save the current data (Raw or Epochs) to disk.
    """

    def run(self, data):
        """
        Expected params:
        - output_path (str): path to save the .fif file
        - overwrite (bool): default True
        """
        if data is None:
            raise ValueError("No data to save.")

        output_path = Path(self.params.get("output_path", "output.fif"))
        overwrite = self.params.get("overwrite", True)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(data, mne.io.BaseRaw):
            data.save(str(output_path), overwrite=overwrite)
        elif isinstance(data, mne.Epochs):
            data.save(str(output_path), overwrite=overwrite)
        else:
            raise TypeError("Only saving Raw or Epochs is currently supported.")

        print(f"[INFO] Data saved to {output_path}")
        return data
