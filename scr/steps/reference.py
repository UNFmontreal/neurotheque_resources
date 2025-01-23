# File: src/steps/reference.py

import logging
from .base import BaseStep
import mne

class ReferenceStep(BaseStep):
    """
    Re-references EEG data according to user-specified method.

    Expected params (all optional, with defaults):
    --------------------------------------------------------------------------
    method : str
        Either "average" for average reference (default) or "channels" for
        custom reference channels.

    channels : list of str
        Which channel(s) to use if method="channels". If None or empty, we error.

    projection : bool
        If True, add a projection to do the re-reference rather than directly
        modifying data. (default: False)

    Example usage in YAML:
    --------------------------------------------------------------------------
    pipeline:
      steps:
        - name: ReferenceStep
          params:
            method: "channels"
            channels: ["TP9", "TP10"]
            projection: false
    """

    def run(self, data):
        if data is None:
            raise ValueError("[ReferenceStep] No data to re-reference.")

        # Get params
        method = self.params.get("method", "average")
        channels = self.params.get("channels", [])
        projection = self.params.get("projection", False)

        logging.info(f"[ReferenceStep] Re-referencing method={method}, projection={projection}")

        if method == "average":
            # set_eeg_reference(ref_channels="average") => average re-ref
            logging.info("[ReferenceStep] Using average reference for EEG channels.")
            data.set_eeg_reference(ref_channels="average", projection=projection)

        elif method == "channels":
            if not channels:
                raise ValueError("[ReferenceStep] method='channels' requires 'channels' param.")
            logging.info(f"[ReferenceStep] Using custom channels {channels} for reference.")
            data.set_eeg_reference(ref_channels=channels, projection=projection)

        else:
            raise ValueError(f"[ReferenceStep] Unknown re-reference method '{method}'.")

        # MNE might add new reference channels to the data if channels were used.
        # If projection=True, we have an EEG ref projection added but not applied
        # until you do e.g. data.apply_proj().

        logging.info("[ReferenceStep] Re-reference complete.")
        return data
