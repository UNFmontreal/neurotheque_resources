# src/steps/prepchannels.py

import logging
from mne.channels import make_standard_montage
from .base import BaseStep
from scr.utils.reference import apply_reference
from scr.utils.dsi24_bids import rename_dsi_channels, set_channel_types

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
        
        # 1) Rename EEG channels: strip "EEG " prefix and "-Pz" suffix, then map legacy names
        eeg_channels = [ch for ch in data.info['ch_names'] if 'EEG ' in ch or ch.endswith('-Pz')]
        if eeg_channels:
            rename_mapping = {}
            for ch in data.info['ch_names']:
                new = ch.replace('EEG ', '').replace('-Pz', '')
                if new != ch:
                    rename_mapping[ch] = new
            if rename_mapping:
                data.rename_channels(rename_mapping)
                logging.info(f"Renamed {len(rename_mapping)} channels (stripped 'EEG ' and '-Pz')")
        else:
            logging.info("No channels with 'EEG ' prefix or '-Pz' suffix to rename")

        # Apply DSI legacy→10-20 name fixes (e.g., A1→M1, A2→M2, T3→T7, ...)
        mapped = rename_dsi_channels(data)
        if mapped:
            logging.info(f"Applied legacy→10-20 mapping: {mapped}")

        # 2) Set channel types using DSI-24 helper (stim/eog)
        typed = set_channel_types(data)

        # Add ECG and CM typing if present
        for ch in data.info['ch_names']:
            up = ch.upper()
            if 'ECG' in up or 'EKG' in up:
                data.set_channel_types({ch: 'ecg'})
            elif ch == 'CM':
                data.set_channel_types({ch: 'misc'})

        # Count types for logging
        types = dict(zip(data.info['ch_names'], data.get_channel_types()))
        eeg_count = sum(t == 'eeg' for t in types.values())
        stim_count = sum(t == 'stim' for t in types.values())
        eog_count = sum(t == 'eog' for t in types.values())
        ecg_count = sum(t == 'ecg' for t in types.values())
        misc_count = sum(t == 'misc' for t in types.values())
        logging.info(f"Channel types set: {eeg_count} EEG, {stim_count} STIM, {eog_count} EOG, {ecg_count} ECG, {misc_count} MISC")
    
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
