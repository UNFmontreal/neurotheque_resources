# src/steps/filter.py

from .base import BaseStep

class FilterStep(BaseStep):
    """
    Step to apply bandpass and/or notch filters on an mne.Raw or mne.Epochs object.
    """

    def run(self, data):
        """
        Expected params:
        - l_freq (float): low cutoff frequency
        - h_freq (float): high cutoff frequency
        - notch_freqs (list): list of freqs to notch out
        """
        if data is None:
            raise ValueError("No data available to filter.")

        l_freq = self.params.get("l_freq", 1.0)
        h_freq = self.params.get("h_freq", 40.0)
        notch_freqs = self.params.get("notch_freqs", [])

        data.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin")

        if notch_freqs:
            data.notch_filter(freqs=notch_freqs, fir_design="firwin")

        return data
