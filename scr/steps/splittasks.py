# File: eeg_pipeline/src/steps/prepchannels.py

import logging
import mne
from mne.channels import make_standard_montage
from .base import BaseStep

class PrepChannelsStep(BaseStep):
    """
    Drops non-EEG channels, renames them, sets montage, etc.
    """

    def run(self, data):
        if data is None:
            raise ValueError("No Raw data to prepare channels.")

        # Non-EEG channels to remove
        non_eeg_channels = self.params.get("non_eeg_channels", [
            'EEG X1:ECG-Pz', 'EEG X2:-Pz', 'EEG X3:-Pz',
            'CM', 'EEG A1-Pz', 'EEG A2-Pz'
        ])
        existing_non_eeg = [ch for ch in non_eeg_channels if ch in data.info['ch_names']]
        data.drop_channels(existing_non_eeg)

        # Rename EEG channels
        eeg_channels = [ch for ch in data.info['ch_names'] if 'EEG' in ch]
        rename_mapping = {ch: ch.replace('EEG ', '').replace('-Pz', '') for ch in eeg_channels}
        data.rename_channels(rename_mapping)

        # Set channel types
        for ch in data.info['ch_names']:
            if ch in rename_mapping.values():
                data.set_channel_types({ch: 'eeg'})
            elif 'Trigger' in ch:
                data.set_channel_types({ch: 'stim'})
            else:
                data.set_channel_types({ch: 'misc'})

        # Set montage
        montage_name = self.params.get("montage", "standard_1020")
        montage = make_standard_montage(montage_name)
        data.set_montage(montage)

        logging.info("[PrepChannelsStep] Channel prep done.")
        return data
