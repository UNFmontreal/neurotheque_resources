# src/steps/ica.py

import mne
from mne.preprocessing import ICA
from .base import BaseStep

class ICAStep(BaseStep):
    """
    Step to perform ICA on raw data, optionally exclude specified components.
    """

    def run(self, data):
        """
        Expected params:
        - n_components (float/int): number of ICA components, e.g. 0.95 or an int
        - random_state (int)
        - decim (int): decimation factor
        - exclude (list): which ICs to exclude after fitting
        """
        if data is None:
            raise ValueError("No data available for ICA.")

        n_components = self.params.get("n_components", 0.95)
        random_state = self.params.get("random_state", 42)
        decim = self.params.get("decim", 3)
        exclude_comps = self.params.get("exclude", [])

        ica = ICA(n_components=n_components, random_state=random_state)
        ica.fit(data, decim=decim)
        ica.exclude = exclude_comps
        ica.apply(data)

        return data
