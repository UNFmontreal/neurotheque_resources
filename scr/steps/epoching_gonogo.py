# File: eeg_pipeline/src/steps/epoching_gonogo.py

import mne
import logging
from .base import BaseStep

class GoNoGoEpochingStep(BaseStep):
    """
    Epochs Go/No-Go task data with flexible parameter configuration.
    
    This step creates epochs based on stimulus events for Go and NoGo trials.
    It supports both basic event-based epoching and can also handle
    custom event classifications if available.
    
    Parameters
    ----------
    tmin : float
        Start time of epoch relative to event in seconds (default: -0.2)
    tmax : float
        End time of epoch relative to event in seconds (default: 1.0)
    baseline : tuple or None
        Baseline correction period, e.g. (None, 0) for pre-stimulus period (default: (None, 0))
    event_id : dict
        Dictionary mapping event names to trigger values
        If None, defaults to {'go': 1, 'nogo': 2}
    stim_channel : str
        Name of the stimulus channel (default: 'Trigger')
    reject_by_annotation : bool
        Whether to reject epochs that overlap with annotations (default: True)
    preload : bool
        Whether to preload the data (default: True)
    """
    
    def run(self, data):
        """
        Apply Go/No-Go epoching to the input data.
        
        Parameters
        ----------
        data : mne.io.Raw
            MNE Raw object to be epoched
            
        Returns
        -------
        epochs : mne.Epochs
            Epoched data
        """
        if data is None:
            raise ValueError("[GoNoGoEpochingStep] No data provided.")
        
        # Get parameters with defaults
        tmin = self.params.get("tmin", -0.2)
        tmax = self.params.get("tmax", 1.0)
        baseline = self.params.get("baseline", (None, 0))
        stim_channel = self.params.get("stim_channel", "Trigger")
        event_id = self.params.get("event_id", {'go': 1, 'nogo': 2})
        preload = self.params.get("preload", True)
        reject_by_annotation = self.params.get("reject_by_annotation", True)
        
        # Log the epoching configuration
        logging.info(f"[GoNoGoEpochingStep] Creating epochs from {tmin}s to {tmax}s around events")
        logging.info(f"[GoNoGoEpochingStep] Event IDs: {event_id}")
        logging.info(f"[GoNoGoEpochingStep] Stimulus channel: {stim_channel}")
        
        # Check if we have pre-processed events from a trigger processing step
        if hasattr(data.info, 'parsed_events') and data.info['parsed_events'] is not None:
            # Use the pre-processed events
            logging.info("[GoNoGoEpochingStep] Using pre-processed events from trigger step")
            events = data.info['parsed_events']
        else:
            # Find events in the data
            logging.info(f"[GoNoGoEpochingStep] Finding events using channel: {stim_channel}")
            events = mne.find_events(data, stim_channel=stim_channel, shortest_event=1)
            
            # Verify that we found events matching our event_id
            event_values = events[:, 2]
            expected_values = list(event_id.values())
            found_values = set(event_values)
            missing_values = set(expected_values) - found_values
            
            if missing_values:
                logging.warning(f"[GoNoGoEpochingStep] Some event values not found: {missing_values}")
                logging.warning(f"[GoNoGoEpochingStep] Found event values: {found_values}")
        
        # Create epochs
        logging.info("[GoNoGoEpochingStep] Creating epochs...")
        epochs = mne.Epochs(
            data, 
            events, 
            event_id=event_id, 
            tmin=tmin, 
            tmax=tmax, 
            baseline=baseline, 
            preload=preload, 
            reject_by_annotation=reject_by_annotation
        )
        
        # Log epoch information
        logging.info(f"[GoNoGoEpochingStep] Created {len(epochs)} epochs")
        
        # Log event counts if available
        for event_name, count in epochs.event_id.items():
            event_count = len(epochs[event_name])
            logging.info(f"[GoNoGoEpochingStep] {event_name}: {event_count} epochs")
        
        return epochs
    
    def estimate_performance(self, epochs, response_id=None, response_window=1.0):
        """
        Estimate Go/No-Go performance metrics based on epochs.
        
        Parameters
        ----------
        epochs : mne.Epochs
            Epoched data
        response_id : int or None
            Event ID for response events, if available
        response_window : float
            Time window in seconds to look for responses after stimulus
            
        Returns
        -------
        metrics : dict
            Dictionary with performance metrics
        """
        # Not implemented yet - for future expansion
        pass
