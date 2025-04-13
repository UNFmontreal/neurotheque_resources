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

        logging.info(f"[PrepChannelsStep] Running with params: {self.params}")
        on_missing = self.params.get("on_missing", "raise")  # Default to raising an error
        
        # 1) Drop non-EEG channels
        non_eeg_channels = ['EEG X1:ECG-Pz', 'EEG X1:-Pz', 'ECG-Pz', 'EEG X2:-Pz', 'EEG X3:-Pz',
            'CM', 'EEG A1-Pz', 'EEG A2-Pz' , 'Event']
        existing_non_eeg = [ch for ch in non_eeg_channels if ch in data.info['ch_names']]
        
        if existing_non_eeg:
            data.drop_channels(existing_non_eeg)
            logging.info(f"Dropped non-EEG: {existing_non_eeg}")
        else:
            logging.info("No non-EEG channels found to drop")

        # 2) Rename EEG channels 
        eeg_channels = [ch for ch in data.info['ch_names'] if 'EEG ' in ch]
        if eeg_channels:
            rename_mapping = {ch: ch.replace('EEG ', '').replace('-Pz', '') for ch in eeg_channels}
            data.rename_channels(rename_mapping)
            logging.info(f"Renamed {len(rename_mapping)} EEG channels")
        else:
            logging.info("No EEG channels found with 'EEG ' prefix to rename")

        # 3) Set channel types
        eeg_count = 0
        stim_count = 0
        misc_count = 0
        
        for ch in data.info['ch_names']:
            if 'Trigger' in ch:
                data.set_channel_types({ch: 'stim'})
                stim_count += 1
            elif 'CM' in ch:
                data.set_channel_types({ch: 'misc'})
                misc_count += 1
            else:
                data.set_channel_types({ch: 'eeg'})
                eeg_count += 1
        
        logging.info(f"Channel types set: {eeg_count} EEG, {stim_count} STIM, {misc_count} MISC")
    
        # 4) Apply re-referencing (our system uses Pz as reference)
        reference_params = self.params.get("reference", {})
        if reference_params:
            logging.info(f"[PrepChannelsStep] Applying re-referencing with params: {reference_params}")
            # Default reference parameters for our EEG system
            if not reference_params.get("method"):
                reference_params["method"] = "channels"
            if not reference_params.get("channels") and reference_params["method"] == "channels":
                reference_params["channels"] = ["Pz"]  # Our system uses Pz as reference
            
            try:
                data = apply_reference(data, reference_params)
                logging.info("[PrepChannelsStep] Re-referencing applied successfully")
            except Exception as e:
                logging.error(f"[PrepChannelsStep] Error during re-referencing: {str(e)}")
                if on_missing == "raise":
                    raise
                else:
                    logging.info("[PrepChannelsStep] Continuing without re-referencing (on_missing=ignore)")
        
        # 5) Montage
        try:
            montage = make_standard_montage('standard_1020')
            data.set_montage(montage)
            logging.info("[PrepChannelsStep] Standard 10-20 montage applied")
        except Exception as e:
            logging.error(f"[PrepChannelsStep] Error setting montage: {str(e)}")
            if on_missing == "raise":
                raise
            else:
                logging.info("[PrepChannelsStep] Continuing without montage (on_missing=ignore)")

        logging.info("[PrepChannelsStep] Channels prepared successfully.")
        return data
