# src/steps/prepchannels.py

import logging
import sys
import os
from mne.channels import make_standard_montage
from .base import BaseStep

# Ensure utils is in the path
sys.path.append(os.path.abspath('../../'))
from scr.utils.reference import apply_reference

class PrepChannelsStep(BaseStep):
    """
    Drops non-EEG channels, renames them, sets channel types & montage.
    Mirrors the notebook's channel preparation logic.
    Can also handle re-referencing as part of channel preparation.
    """
    def run(self, data):
        if data is None:
            raise ValueError("[PrepChannelsStep] No data to process.")

        # 1) Drop non-EEG channels
        non_eeg_channels = [
            'X1:ECG', 'EEG X1:-Pz', 'ECG-Pz', 'EEG X2:-Pz', 'EEG X3:-Pz',
            'CM', 'EEG A1-Pz', 'EEG A2-Pz' , 'Event'
        ]
        existing_non_eeg = [ch for ch in non_eeg_channels if ch in data.info['ch_names']]
        data.drop_channels(existing_non_eeg)
        logging.info(f"Dropped non-EEG: {existing_non_eeg}")

        # 2) Rename EEG channels 
        eeg_channels = [ch for ch in data.info['ch_names'] if 'EEG ' in ch]
        rename_mapping = {ch: ch.replace('EEG ', '').replace('-Pz', '') for ch in eeg_channels}
        data.rename_channels(rename_mapping)

        # 3) Set channel types
        for ch in data.info['ch_names']:
            if 'Trigger' in ch:
                data.set_channel_types({ch: 'stim'})
            elif 'CM' in ch:
                data.set_channel_types({ch: 'misc'})
            else:
                data.set_channel_types({ch: 'eeg'})
    
        # 4) Apply re-referencing (our system uses Pz as reference)
        reference_params = self.params.get("reference", {})
        if reference_params:
            logging.info("[PrepChannelsStep] Applying re-referencing")
            # Default reference parameters for our EEG system
            if not reference_params.get("method"):
                reference_params["method"] = "channels"
            if not reference_params.get("channels") and reference_params["method"] == "channels":
                reference_params["channels"] = ["Pz"]  # Our system uses Pz as reference
            
            try:
                data = apply_reference(data, reference_params)
            except Exception as e:
                logging.error(f"[PrepChannelsStep] Error during re-referencing: {str(e)}")
                logging.info("[PrepChannelsStep] Continuing without re-referencing")
        
        # 5) Montage
        montage = make_standard_montage('standard_1020')
        data.set_montage(montage)

        logging.info("[PrepChannelsStep] Channels prepared successfully.")
        return data
