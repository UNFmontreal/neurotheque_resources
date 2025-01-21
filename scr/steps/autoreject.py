# src/steps/autoreject.py

import mne
from autoreject import AutoReject
from .base import BaseStep

class AutoRejectStep(BaseStep):
    """
    Step to apply AutoReject on data.
    If you want to do it on Raw, we typically create 1s epochs internally.
    """

    def run(self, data):
        """
        Expected params:
        - ar_params (dict): parameters for AutoReject (thresh_func, consensus, etc.)
        """
        if data is None:
            raise ValueError("No data available for AutoReject step.")

        ar_params = self.params.get("ar_params", {})
        # Make short epochs
        events_tmp = mne.make_fixed_length_events(data, duration=1.0)
        epochs_tmp = mne.Epochs(data, events_tmp, tmin=0, tmax=1, baseline=None,
                                preload=True)

        ar = AutoReject(**ar_params)
        ar.fit(epochs_tmp)

        # For demonstration, let's just log a message
        print("[INFO] AutoReject fit complete. Use it later on your condition-specific epochs if needed.")
        return data
