# src/steps/filter.py

from .base import BaseStep

class FilterStep(BaseStep):
    """
    Step to apply bandpass and/or notch filters on an mne.Raw or mne.Epochs object.
    """
    
    def __init__(self, params=None):
        """
        Initialize the FilterStep with parameters.
        
        Parameters
        ----------
        params : dict, optional
            Dictionary containing filter parameters:
            - l_freq (float): low cutoff frequency
            - h_freq (float): high cutoff frequency 
            - notch_freqs (list): list of freqs to notch out
        """
        super().__init__()
        self.params = params or {}

    def run(self, data):
        """
        Apply filtering to the provided data.
        
        Parameters
        ----------
        data : mne.Raw or mne.Epochs
            Data to be filtered
            
        Returns
        -------
        data : mne.Raw or mne.Epochs
            Filtered data
        """
        if data is None:
            raise ValueError("No data available to filter.")

        l_freq = self.params.get("l_freq", 1.0)
        h_freq = self.params.get("h_freq", 40.0)
        notch_freqs = self.params.get("notch_freqs", [])
        fir_design = self.params.get("fir_design", "firwin")

        if l_freq is not None and h_freq is not None and l_freq >= h_freq:
            raise ValueError(f"[FilterStep] l_freq ({l_freq}) must be < h_freq ({h_freq}).")

        data.filter(l_freq=l_freq, h_freq=h_freq, fir_design=fir_design)

        if notch_freqs:
            data.notch_filter(freqs=notch_freqs, fir_design=fir_design)

        return data
