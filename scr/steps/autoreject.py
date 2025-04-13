# File: src/steps/autoreject.py

import logging
import mne
from autoreject import AutoReject
from .base import BaseStep
import json
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

class AutoRejectStep(BaseStep):
    """
    A streamlined AutoReject step for EEG pipelines that saves bad epochs as annotations.
    
    This step:
    - Creates short (1s) epochs from the incoming data
    - Fits AutoReject to identify bad epochs
    - Adds bad epochs as annotations to the raw data
    - Optionally stores reject_log in data.info["temp"]
    - Optionally saves the AutoReject model state for later use
    
    Expected params:
    --------------------------------------------------------------------------
    ar_params (dict):
        Dictionary of parameters passed directly to AutoReject's constructor
        
    plot_results (bool):
        Whether to generate visualization of rejected epochs (default: True)
        
    mode (str):
        Either "fit" (identify bad epochs only) or "fit_transform" (clean data) (default: "fit_transform")
        
    save_cleaned_data (bool):
        Whether to save the cleaned data to disk (default: False)
        
    file_prefix (str):
        Prefix for saved files (default: "autoreject")
        
    output_dir (str or None):
        Directory to save files to (default: None, uses a default location)
        
    store_reject_log (bool):
        Whether to store the reject_log in data.info["temp"] (default: True)
        
    save_model (bool):
        Whether to save the AutoReject model state to disk (default: False)
        
    model_filename (str or None):
        Filename for saved model (default: None, auto-generates a name)
    """

    def run(self, data):
        if data is None:
            raise ValueError("[AutoRejectStep] No data provided.")

        ar_params = self.params.get("ar_params", {})
        plot_results = self.params.get("plot_results", True)
        mode = self.params.get("mode", "fit_transform")  # Can be "fit" or "fit_transform"
        save_cleaned_data = self.params.get("save_cleaned_data", False)
        file_prefix = self.params.get("file_prefix", "autoreject")
        output_dir = self.params.get("output_dir", None)
        store_reject_log = self.params.get("store_reject_log", True)
        save_model = self.params.get("save_model", False)
        model_filename = self.params.get("model_filename", None)
        
        # Check if data is already epoched
        if isinstance(data, mne.Epochs):
            logging.info("[AutoRejectStep] Using existing epochs for AutoReject.")
            epochs_tmp = data
            already_epoched = True
        else:
            # Create temporary epochs for autoreject
            logging.info("[AutoRejectStep] Creating 1-second epochs for AR fitting.")
            events_tmp = mne.make_fixed_length_events(data, duration=1)
            epochs_tmp = mne.Epochs(
                data,
                events_tmp,
                tmin=0,
                tmax=1,
                baseline=None,
                detrend=0,
                preload=True
            )
            already_epoched = False

        # Initialize and fit AutoReject
        logging.info(f"[AutoRejectStep] Running AutoReject with params: {ar_params}")
        ar = AutoReject(**ar_params)
        ar.fit(epochs_tmp)
        
        # Get the reject log for visualization and statistics
        reject_log = ar.get_reject_log(epochs_tmp)
        
        # Apply AutoReject according to mode
        if mode == "fit":
            logging.info("[AutoRejectStep] Running in 'fit' mode - only identifying bad epochs")
            # In 'fit' mode, we only identify bad epochs and add annotations
            if already_epoched:
                # Return a copy of the original epochs with bad epochs marked
                cleaned_data = data.copy()
            else:
                # Convert reject log to annotations and apply to raw data
                cleaned_data = data.copy()
                self._add_reject_log_as_annotations(cleaned_data, reject_log, events_tmp)
        else:  # fit_transform mode
            logging.info("[AutoRejectStep] Running in 'fit_transform' mode - cleaning the data")
            # Apply AutoReject to clean the data
            cleaned_data = ar.transform(epochs_tmp)
            
            # If input was raw data and we want to return raw data
            if not already_epoched and self.params.get("return_raw", True):
                logging.info("[AutoRejectStep] Converting cleaned epochs back to raw data.")
                # Convert cleaned epochs back to raw
                raw_data = epochs_tmp.get_data()
                cleaned_raw = mne.io.RawArray(
                    raw_data.transpose((1, 0, 2)).reshape(len(epochs_tmp.ch_names), -1),
                    epochs_tmp.info
                )
                # Add annotations for bad epochs
                self._add_reject_log_as_annotations(cleaned_raw, reject_log, events_tmp)
                cleaned_data = cleaned_raw
        
        # Log statistics about rejected epochs
        n_epochs_before = len(epochs_tmp)
        n_bad_epochs = np.sum(reject_log.bad_epochs)
        n_rejected_percent = (n_bad_epochs / n_epochs_before) * 100
        
        logging.info(f"[AutoRejectStep] AutoReject identified {n_bad_epochs} bad epochs out of {n_epochs_before} ({n_rejected_percent:.1f}%)")
        
        # Generate simple visualization if requested
        if plot_results:
            self._create_simple_visualization(reject_log, data, ar)
        
        # Save the cleaned data if requested
        if save_cleaned_data:
            self._save_cleaned_data(cleaned_data, file_prefix, output_dir)
        
        # Store reject_log in data.info["temp"] if requested
        if store_reject_log:
            self._store_reject_log_in_info(cleaned_data, reject_log, epochs_tmp, ar)
        
        # Save the AutoReject model state if requested
        if save_model:
            self._save_model_state(ar, model_filename, output_dir)
        
        # Return the annotated/cleaned data
        return cleaned_data
    
    def _add_reject_log_as_annotations(self, raw, reject_log, events):
        """
        Convert AutoReject rejection log to MNE Annotations and add them to the raw data.
        """
        bad_epochs = reject_log.bad_epochs
        if not any(bad_epochs):
            logging.info("[AutoRejectStep] No bad epochs detected, no annotations added")
            return
        
        # Get the sampling frequency
        sfreq = raw.info['sfreq']
        
        # Create lists to store annotation parameters
        onsets = []
        durations = []
        descriptions = []
        
        # For each bad epoch, create an annotation
        for i, is_bad in enumerate(bad_epochs):
            if is_bad:
                # Get epoch onset and duration in seconds
                onset = events[i, 0] / sfreq
                duration = 1.0  # Since we created 1-second epochs
                
                # Add to lists
                onsets.append(onset)
                durations.append(duration)
                descriptions.append('BAD_autoreject')
        
        # Create annotations object
        annot = mne.Annotations(
            onset=onsets,
            duration=durations,
            description=descriptions,
            orig_time=raw.info['meas_date']
        )
        
        # Add annotations to raw data
        raw.set_annotations(raw.annotations + annot)
        
    def _create_simple_visualization(self, reject_log, raw, ar=None):
        """
        Creates a simple visualization of AutoReject results
        """
        try:
            # Get basic statistics
            bad_epochs = reject_log.bad_epochs
            n_epochs = len(bad_epochs)
            n_bad_epochs = np.sum(bad_epochs)
            bad_epoch_percent = (n_bad_epochs / n_epochs) * 100
            
            # Determine output directory
            output_dir = self.params.get("output_dir", None)
            if output_dir is None:
                sub_id = getattr(self.params, "subject_id", "01")
                ses_id = getattr(self.params, "session_id", "001")
                output_dir = f"./data/processed/sub-{sub_id}/ses-{ses_id}/figures"
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Create figure
            plt.figure(figsize=(10, 6))
            plt.title(f'AutoReject Results: {bad_epoch_percent:.1f}% of epochs marked bad')
            
            # Create bar plot showing bad epochs (1=bad, 0=good)
            plt.bar(range(n_epochs), [1 if x else 0 for x in bad_epochs], color='r', alpha=0.7)
            plt.xlabel('Epoch Index')
            plt.ylabel('Rejected (1=bad, 0=good)')
            plt.ylim(-0.1, 1.1)
            
            # Add text with summary statistics
            stats_text = f"Total: {n_epochs} epochs\nBad: {n_bad_epochs} epochs ({bad_epoch_percent:.1f}%)"
            plt.figtext(0.02, 0.02, stats_text, fontsize=12,
                       bbox=dict(facecolor='white', alpha=0.9))
            
            # Save the plot
            sub_id = getattr(self.params, "subject_id", "unknown")
            ses_id = getattr(self.params, "session_id", "unknown")
            plt.savefig(os.path.join(output_dir, f"autoreject_{sub_id}_{ses_id}_epochs.png"), dpi=150)
            plt.close()
            
            # If ar object is provided, plot additional information
            if ar is not None and hasattr(ar, 'threshes_'):
                try:
                    # Create a second figure for thresholds
                    plt.figure(figsize=(12, 8))
                    plt.title('AutoReject Channel Thresholds')
                    
                    # Plot thresholds for each channel
                    ch_names = raw.ch_names
                    ch_thresholds = []
                    
                    # Extract thresholds for each channel
                    for ch_idx, ch_name in enumerate(ch_names):
                        if ch_idx < len(ar.threshes_) and ar.threshes_[ch_idx] is not None:
                            ch_thresholds.append(ar.threshes_[ch_idx])
                        else:
                            ch_thresholds.append(0)  # Default if no threshold
                    
                    # Create bar plot of thresholds
                    plt.bar(range(len(ch_names)), ch_thresholds, alpha=0.7)
                    plt.xticks(range(len(ch_names)), ch_names, rotation=90)
                    plt.xlabel('Channel')
                    plt.ylabel('Threshold (μV)')
                    plt.tight_layout()
                    
                    # Save the plot
                    plt.savefig(os.path.join(output_dir, f"autoreject_{sub_id}_{ses_id}_thresholds.png"), dpi=150)
                    plt.close()
                    
                except Exception as e:
                    logging.warning(f"[AutoRejectStep] Error creating thresholds visualization: {e}")
            
        except Exception as e:
            logging.warning(f"[AutoRejectStep] Error creating visualization: {e}")

    def _save_cleaned_data(self, data, file_prefix, output_dir):
        """
        Save the cleaned data to a specified location with a custom prefix.
        Works with both Raw and Epochs objects.
        """
        try:
            sub_id = getattr(self.params, "subject_id", "01")
            ses_id = getattr(self.params, "session_id", "001")
            
            # If output_dir is not specified, use a default location
            if output_dir is None:
                output_dir = f"./data/processed/sub-{sub_id}/ses-{ses_id}"
                
            # Create the directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Create a filename with the custom prefix
            if isinstance(data, mne.io.Raw):
                filename = f"sub-{sub_id}_ses-{ses_id}_{file_prefix}_raw.fif"
            else:  # Epochs
                filename = f"sub-{sub_id}_ses-{ses_id}_{file_prefix}_epo.fif"
                
            file_path = os.path.join(output_dir, filename)
            
            # Save the data
            data.save(file_path, overwrite=True)
            logging.info(f"[AutoRejectStep] Saved cleaned {'raw data' if isinstance(data, mne.io.Raw) else 'epochs'} to {file_path}")
            
        except Exception as e:
            logging.error(f"[AutoRejectStep] Error saving cleaned data: {e}")
            logging.error(f"[AutoRejectStep] Attempted to save to: {output_dir}")
            # If there's an error, try an alternative path
            try:
                if isinstance(data, mne.io.Raw):
                    alt_path = f"./data/processed/sub-{sub_id}_ses-{ses_id}_{file_prefix}_raw.fif"
                else:  # Epochs
                    alt_path = f"./data/processed/sub-{sub_id}_ses-{ses_id}_{file_prefix}_epo.fif"
                    
                data.save(alt_path, overwrite=True)
                logging.info(f"[AutoRejectStep] Saved cleaned data to alternative path: {alt_path}")
            except Exception as e2:
                logging.error(f"[AutoRejectStep] Error saving to alternative path: {e2}")
                logging.error("[AutoRejectStep] Unable to save cleaned data")

    def _store_reject_log_in_info(self, data, reject_log, epochs, ar):
        """
        Store the reject_log and other AutoReject information in data.info["temp"]
        """
        try:
            # Convert reject_log to a serializable format
            # We can't directly store the RejectLog object as it's not JSON serializable
            reject_log_dict = {
                "bad_epochs": reject_log.bad_epochs.tolist(),
                "ch_names": epochs.ch_names,
                "n_interpolate": ar.n_interpolate_ if hasattr(ar, "n_interpolate_") else None,
                "consensus": ar.consensus_ if hasattr(ar, "consensus_") else None,
                "thresholds": ar.threshes_ if hasattr(ar, "threshes_") else None,
                "n_epochs_total": len(epochs),
                "n_epochs_bad": np.sum(reject_log.bad_epochs)
            }
            
            # Initialize temp if it doesn't exist
            if not hasattr(data.info, "temp"):
                data.info["temp"] = {}
                
            # Store the reject_log information
            data.info["temp"]["autoreject"] = reject_log_dict
            
            logging.info("[AutoRejectStep] Stored AutoReject information in data.info['temp']")
            
        except Exception as e:
            logging.error(f"[AutoRejectStep] Error storing reject_log in data.info: {e}")
            logging.warning("[AutoRejectStep] Will continue without storing reject_log")

    def _save_model_state(self, ar, model_filename, output_dir):
        """
        Save the AutoReject model state to an HDF5 file for later use
        """
        try:
            # Import h5io
            from mne.externals.h5io import write_hdf5
            
            # Set up output directory
            sub_id = getattr(self.params, "subject_id", "01")
            ses_id = getattr(self.params, "session_id", "001")
            
            if output_dir is None:
                output_dir = f"./data/processed/sub-{sub_id}/ses-{ses_id}/models"
                
            os.makedirs(output_dir, exist_ok=True)
            
            # Set up filename
            if model_filename is None:
                model_filename = f"sub-{sub_id}_ses-{ses_id}_autoreject_model.h5"
                
            file_path = os.path.join(output_dir, model_filename)
            
            # Save the model state
            try:
                write_hdf5(file_path, ar.__getstate__(), title='autoreject', overwrite=True)
                logging.info(f"[AutoRejectStep] Saved AutoReject model state to {file_path}")
            except Exception as e:
                # Fallback to pickle if HDF5 fails
                logging.warning(f"[AutoRejectStep] Error saving model with HDF5: {e}")
                logging.warning("[AutoRejectStep] Falling back to pickle")
                
                pickle_path = os.path.join(output_dir, model_filename.replace('.h5', '.pkl'))
                with open(pickle_path, 'wb') as f:
                    pickle.dump(ar, f)
                logging.info(f"[AutoRejectStep] Saved AutoReject model to {pickle_path} using pickle")
                
        except Exception as e:
            logging.error(f"[AutoRejectStep] Error saving model state: {e}")
            logging.warning("[AutoRejectStep] Will continue without saving model state")

    @staticmethod
    def load_model(model_path):
        """
        Load a previously saved AutoReject model
        
        Parameters
        ----------
        model_path : str
            Path to the saved model file (.h5 or .pkl)
            
        Returns
        -------
        ar : AutoReject
            The loaded AutoReject model
        """
        if model_path.endswith('.h5'):
            try:
                from mne.externals.h5io import read_hdf5
                from autoreject import AutoReject
                
                # Load the state
                state = read_hdf5(model_path, title='autoreject')
                
                # Create a new AutoReject instance and restore its state
                ar = AutoReject()
                ar.__setstate__(state)
                
                return ar
            except Exception as e:
                logging.error(f"Error loading model from HDF5: {e}")
                raise
        elif model_path.endswith('.pkl'):
            try:
                with open(model_path, 'rb') as f:
                    ar = pickle.load(f)
                return ar
            except Exception as e:
                logging.error(f"Error loading model from pickle: {e}")
                raise
        else:
            raise ValueError(f"Unsupported file format: {model_path}. Expected .h5 or .pkl")
