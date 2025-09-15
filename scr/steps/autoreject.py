# File: scr/steps/autoreject.py

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

class AutoRejectStep(BaseStep):
    """
    A streamlined AutoReject step for EEG pipelines that saves bad epochs as annotations.
    
    This step:
    - Creates short (1s) epochs from the incoming data if it's raw data
    - Or uses existing epochs if input data is already epoched
    - Fits AutoReject to identify bad epochs
    - Adds bad epochs as annotations to the raw data (if raw was provided)
    - Optionally stores reject_log in data.info["temp"]
    - Optionally saves the AutoReject model state for later use
    
    Expected params:
    --------------------------------------------------------------------------
    ar_params (dict):
        Dictionary of parameters passed directly to AutoReject's constructor
        
    plot_results (bool):
        Whether to generate visualization of rejected epochs (default: True)
        
    interactive (bool):
        Whether to show plots interactively (default: False)
        
    plot_dir (str):
        Directory to save plot figures (default: None)
        
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

        # DEBUG: Log the exact type of the data object
        logging.info(f"[AutoRejectStep] Data type: {type(data)}")
        logging.info(f"[AutoRejectStep] Is instance of mne.Epochs: {isinstance(data, mne.Epochs)}")
        
        if hasattr(mne, 'EpochsFIF'):
            logging.info(f"[AutoRejectStep] Is instance of mne.EpochsFIF: {isinstance(data, mne.EpochsFIF)}")
        
        ar_params = self.params.get("ar_params", {})
        plot_results = self.params.get("plot_results", True)
        interactive = self.params.get("interactive", False)
        plot_dir = self.params.get("plot_dir", None)
        mode = self.params.get("mode", "fit_transform")  # Can be "fit" or "fit_transform"
        save_cleaned_data = self.params.get("save_cleaned_data", False)
        file_prefix = self.params.get("file_prefix", "autoreject")
        output_dir = self.params.get("output_dir", None)
        store_reject_log = self.params.get("store_reject_log", False)
        save_model = self.params.get("save_model", False)
        model_filename = self.params.get("model_filename", None)
        # Optional: load previously saved model if requested
        load_model_flag = self.params.get("load_model", False) or self.params.get("use_saved_model", False)
        model_path = self.params.get("model_path", None)
        if load_model_flag and (not model_path):
            # Try to infer from output_dir + model_filename
            try:
                if output_dir and model_filename:
                    import os
                    candidate = os.path.join(output_dir, model_filename)
                    if os.path.exists(candidate):
                        model_path = candidate
            except Exception:
                pass
        
        # Setup for interactive plotting if requested
        if not interactive:
            try:
                matplotlib.use('Agg', force=True)
            except Exception:
                pass
            original_backend = None
        else:
            original_backend = matplotlib.get_backend()
            try:
                plt.ion()
            except Exception:
                logging.warning("[AutoRejectStep] Interactive backend not available; running headless.")
        
        # Check if data is already epoched - be more flexible with the check
        if isinstance(data, mne.Epochs) or str(type(data).__name__) == 'EpochsFIF' or hasattr(data, 'epochs'):
            logging.info("[AutoRejectStep] Using existing epochs for AutoReject.")
            epochs_tmp = data
            already_epoched = True
            events_tmp = None  # We don't need events for pre-epoched data
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
        ar = None
        used_saved_model = False
        if load_model_flag and model_path:
            try:
                ar = self.load_model(model_path)
                used_saved_model = True
                logging.info(f"[AutoRejectStep] Loaded AutoReject model from {model_path}")
            except Exception as e:
                logging.warning(f"[AutoRejectStep] Failed to load model from {model_path}: {e}. Will fit from scratch.")
                ar = None
                used_saved_model = False
        if ar is None:
            ar = AutoReject(**ar_params)
            ar.fit(epochs_tmp)
        
        # Get the reject log for visualization and statistics
        try:
            reject_log = ar.get_reject_log(epochs_tmp)
        except Exception as e:
            # Likely model-data mismatch; refit from scratch
            logging.warning(f"[AutoRejectStep] get_reject_log failed ({e}); refitting model from scratch.")
            ar = AutoReject(**ar_params)
            ar.fit(epochs_tmp)
            reject_log = ar.get_reject_log(epochs_tmp)
        
        # Apply AutoReject according to mode
        if mode == "fit":
            logging.info("[AutoRejectStep] Running in 'fit' mode - only identifying bad epochs")
            # In 'fit' mode, we only identify bad epochs and add annotations
            if already_epoched:
                # Return a copy of the original epochs with bad epochs marked
                cleaned_data = data.copy()
                
                # Instead of just warning, we'll store the bad epochs in the info dictionary
                # so they can be accessed later even if not as annotations
                if 'temp' not in cleaned_data.info:
                    cleaned_data.info['temp'] = {}
                
                # Store the bad epoch indices directly
                bad_epoch_indices = np.where(reject_log.bad_epochs)[0].tolist()
                cleaned_data.info['temp']['autoreject_bad_epochs'] = bad_epoch_indices
                
                logging.info("[AutoRejectStep] Data already in epochs format. Bad epochs stored in data.info['temp']['autoreject_bad_epochs']")
                logging.info(f"[AutoRejectStep] Found {len(bad_epoch_indices)} bad epochs out of {len(reject_log.bad_epochs)}")
            else:
                # Convert reject log to annotations and apply to raw data
                cleaned_data = data.copy()
                self._add_reject_log_as_annotations(cleaned_data, reject_log, events_tmp)
        else:  # fit_transform mode
            logging.info("[AutoRejectStep] Running in 'fit_transform' mode - cleaning the data")
            # Apply AutoReject to clean the data
            try:
                cleaned_data = ar.transform(epochs_tmp)
            except Exception as e:
                logging.warning(f"[AutoRejectStep] transform failed ({e}); refitting and retrying.")
                ar = AutoReject(**ar_params)
                ar.fit(epochs_tmp)
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
            self._create_simple_visualization(reject_log, data, ar, plot_dir, interactive)
            self._plot_channel_rejection_rates(reject_log, data, plot_dir, interactive)
        # Save the cleaned data if requested
        if save_cleaned_data:
            self._save_cleaned_data(cleaned_data, file_prefix, output_dir)
        
        # Store reject_log in data.info["temp"] if requested
        if store_reject_log:
            self._store_reject_log_in_info(cleaned_data, reject_log, epochs_tmp, ar)
            
        # Debug: Verify temp dict exists after processing
        if 'temp' in cleaned_data.info:
            if 'autoreject_bad_epochs' in cleaned_data.info['temp']:
                logging.info(f"[AutoRejectStep] VERIFICATION: Found {len(cleaned_data.info['temp']['autoreject_bad_epochs'])} bad epochs in info['temp']")
            else:
                logging.warning("[AutoRejectStep] VERIFICATION: 'autoreject_bad_epochs' key not found in info['temp']")
        else:
            logging.warning("[AutoRejectStep] VERIFICATION: 'temp' dictionary not found in info")
        
        # Save the AutoReject model state if requested
        if save_model and not used_saved_model:
            self._save_model_state(ar, model_filename, output_dir)
        
        # Restore original matplotlib backend if needed
        if interactive and original_backend:
            try:
                if original_backend != 'TkAgg' and original_backend != 'Qt5Agg' and original_backend != 'WXAgg':
                    logging.info(f"[AutoRejectStep] Switching matplotlib backend back to {original_backend}")
                    matplotlib.use(original_backend)
                    plt.ioff()  # Turn off interactive mode
            except Exception as e:
                logging.warning(f"[AutoRejectStep] Error switching back to original backend: {e}")
        
        # Return the annotated/cleaned data
        return cleaned_data
    
    def _add_reject_log_as_annotations(self, raw, reject_log, events):
        """
        Convert AutoReject rejection log to MNE Annotations and add them to the raw data.
        """
        # Skip if events is None (which means we had pre-epoched data)
        if events is None:
            logging.info("[AutoRejectStep] Cannot add annotations because no events were provided (data was already epoched).")
            return
            
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
        
    def _create_simple_visualization(self, reject_log, raw, ar=None, plot_dir=None, interactive=False):
        """
        Creates a simple visualization of AutoReject results
        
        Parameters
        ----------
        reject_log : instance of autoreject.RejectLog
            The rejection log from AutoReject
        raw : instance of mne.io.Raw or mne.Epochs
            The data that was processed
        ar : instance of autoreject.AutoReject or None
            The fitted AutoReject object
        plot_dir : str or None
            Directory to save plots
        interactive : bool
            Whether to show plots interactively
        """
        try:
            # Get basic statistics
            bad_epochs = reject_log.bad_epochs
            n_epochs = len(bad_epochs)
            n_bad_epochs = np.sum(bad_epochs)
            bad_epoch_percent = (n_bad_epochs / n_epochs) * 100
            
            # Determine output directory
            if plot_dir is None:
                sub_id = getattr(self.params, "subject_id", "01")
                ses_id = getattr(self.params, "session_id", "001")
                plot_dir = f"./data/processed/sub-{sub_id}/ses-{ses_id}/figures/autoreject"
            
            os.makedirs(plot_dir, exist_ok=True)
            
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
            
            # Show the plot interactively if requested
            if interactive:
                plt.show()
            
            # Save the plot
            sub_id = getattr(self.params, "subject_id", "unknown")
            ses_id = getattr(self.params, "session_id", "unknown")
            run_id = getattr(self.params, "run_id", "01")
            plt.savefig(os.path.join(plot_dir, f"autoreject_{sub_id}_{ses_id}_run-{run_id}_epochs.png"), dpi=150)
            
            # Only close the plot if not in interactive mode
            if not interactive:
                plt.close()
            
            # If ar object is provided, plot additional information
            if ar is not None and hasattr(ar, 'threshes_'):
                try:
                    # Create a second figure for thresholds
                    plt.figure(figsize=(12, 8))
                    plt.title('AutoReject Channel Thresholds')
                    
                    # Plot thresholds for each channel - with better error handling
                    ch_names = raw.ch_names
                    ch_thresholds = []
                    
                    if ar.threshes_ is None:
                        # If threshes_ is None, show a message and skip this plot
                        logging.warning("[AutoRejectStep] No thresholds available in AutoReject object")
                        plt.close()
                        return
                    
                    # Extract thresholds for each channel safely
                    for ch_idx, ch_name in enumerate(ch_names):
                        try:
                            if ch_idx < len(ar.threshes_) and ar.threshes_[ch_idx] is not None:
                                ch_thresholds.append(float(ar.threshes_[ch_idx]))
                            else:
                                ch_thresholds.append(0.0)  # Default if no threshold
                        except (TypeError, IndexError) as e:
                            logging.warning(f"[AutoRejectStep] Error accessing threshold for channel {ch_name}: {e}")
                            ch_thresholds.append(0.0)
                    
                    # Check if we have any valid thresholds
                    if not any(ch_thresholds):
                        logging.warning("[AutoRejectStep] No valid thresholds found for visualization")
                        plt.close()
                        return
                    
                    # Create bar plot of thresholds
                    plt.bar(range(len(ch_names)), ch_thresholds, alpha=0.7)
                    plt.xticks(range(len(ch_names)), ch_names, rotation=90)
                    plt.xlabel('Channel')
                    plt.ylabel('Threshold (μV)')
                    plt.tight_layout()
                    
                    # Show the plot interactively if requested
                    if interactive:
                        plt.show()
                    
                    # Save the plot
                    plt.savefig(os.path.join(plot_dir, f"autoreject_{sub_id}_{ses_id}_run-{run_id}_thresholds.png"), dpi=150)
                    
                    # Only close the plot if not in interactive mode
                    if not interactive:
                        plt.close()
                    
                except Exception as e:
                    logging.warning(f"[AutoRejectStep] Error creating thresholds visualization: {e}")
                    plt.close()  # Make sure we close the figure
            
        except Exception as e:
            logging.warning(f"[AutoRejectStep] Error creating visualization: {e}")
            if 'plt' in locals() and plt.get_fignums():
                plt.close()  # Close any open figures
    def _plot_channel_rejection_rates(self, reject_log, raw, plot_dir=None, interactive=False):
        """
        Creates a detailed visualization of channel-wise rejection rates
        
        Parameters
        ----------
        reject_log : instance of autoreject.RejectLog
            The rejection log from AutoReject
        raw : instance of mne.io.Raw or mne.Epochs
            The data that was processed
        plot_dir : str or None
            Directory to save plots
        interactive : bool
            Whether to show plots interactively
        """
        try:
            # Get channel names
            ch_names = raw.ch_names
            
            # Filter out trigger/stim channels by name or type
            ch_types = raw.get_channel_types() if hasattr(raw, 'get_channel_types') else []
            excluded_channels = []
            
            # Try to identify trigger channels by name
            trigger_keywords = ['trigger', 'stim', 'sti', 'event', 'status']
            
            # If we have channel types, use that to identify stim channels
            if ch_types:
                for idx, (ch_name, ch_type) in enumerate(zip(ch_names, ch_types)):
                    if ch_type == 'stim' or any(keyword in ch_name.lower() for keyword in trigger_keywords):
                        excluded_channels.append(ch_name)
            else:
                # Otherwise, just use name-based detection
                for ch_name in ch_names:
                    if any(keyword in ch_name.lower() for keyword in trigger_keywords):
                        excluded_channels.append(ch_name)
            
            # Log which channels we're excluding
            if excluded_channels:
                logging.info(f"[AutoRejectStep] Excluding trigger/stim channels from rejection plot: {excluded_channels}")
            
            # Calculate rejection rates per channel
            ch_rejection_rates = {}
            
            # Check if we have channel-wise information
            if hasattr(reject_log, 'labels') and reject_log.labels is not None:
                # Loop through all epochs and channels
                for ch_idx, ch_name in enumerate(ch_names):
                    # Skip excluded channels
                    if ch_name in excluded_channels:
                        continue
                        
                    # Count rejected instances for this channel
                    if ch_idx < reject_log.labels.shape[1]:  # Make sure index is valid
                        # Count epochs where this channel was marked for rejection
                        # Exclude completely rejected epochs to focus on channel-specific issues
                        
                        # Convert arrays to bool type before bitwise operations
                        ch_labels = reject_log.labels[:, ch_idx].astype(bool)
                        bad_epochs = reject_log.bad_epochs.astype(bool)
                        
                        # Now perform the bitwise operations on boolean arrays
                        ch_bad_count = np.sum(ch_labels & ~bad_epochs)
                        total_epochs = len(reject_log.labels)
                   
                        # Calculate rejection percentage
                        if total_epochs > 0:
                            rejection_rate = (ch_bad_count / total_epochs) * 100
                        else:
                            rejection_rate = 0.0
                            
                        ch_rejection_rates[ch_name] = rejection_rate
            
            # If no channel-wise data, try to use the channel thresholds as a proxy
            if not ch_rejection_rates and hasattr(reject_log, 'threshes') and reject_log.threshes is not None:
                for ch_idx, ch_name in enumerate(ch_names):
                    # Skip excluded channels
                    if ch_name in excluded_channels:
                        continue
                        
                    # Higher thresholds might indicate more problematic channels
                    # This is an approximation since we don't have actual rejection counts
                    if ch_idx < len(reject_log.threshes) and reject_log.threshes[ch_idx] is not None:
                        # Normalize thresholds to a 0-100 scale for visualization
                        max_thresh = np.max([t for t in reject_log.threshes if t is not None])
                        norm_thresh = (reject_log.threshes[ch_idx] / max_thresh) * 100 if max_thresh > 0 else 0
                        ch_rejection_rates[ch_name] = norm_thresh
            
            # If we still don't have data, check the AutoReject scores
            if not ch_rejection_rates and hasattr(reject_log, 'scores') and reject_log.scores is not None:
                for ch_idx, ch_name in enumerate(ch_names):
                    # Skip excluded channels
                    if ch_name in excluded_channels:
                        continue
                        
                    if ch_idx < reject_log.scores.shape[1]:
                        # Use mean scores as proxy for rejection rate
                        mean_score = np.mean(reject_log.scores[:, ch_idx]) * 100  # Scale to percentage
                        ch_rejection_rates[ch_name] = mean_score
            
            # If we still don't have channel data, we can't create this plot
            if not ch_rejection_rates:
                logging.warning("[AutoRejectStep] No channel-wise rejection data available for visualization")
                return
            
            # Sort channels by rejection rate (highest to lowest)
            sorted_channels = sorted(ch_rejection_rates.items(), key=lambda x: x[1], reverse=True)
            
            # Create figure
            plt.figure(figsize=(12, max(8, len(ch_rejection_rates)/3)))  # Adjust height based on channel count
            
            # Set up channel names and rejection rates
            channels = [ch[0] for ch in sorted_channels]
            rates = [ch[1] for ch in sorted_channels]
            
            # Create color map based on rejection rates
            colors = []
            for rate in rates:
                if rate < 20:
                    colors.append('green')  # Good channels (low rejection)
                elif rate < 50:
                    colors.append('yellowgreen')  # Warning channels
                else:
                    colors.append('salmon')  # Problematic channels (high rejection)
            
            # Create horizontal bar chart
            y_pos = np.arange(len(channels))
            bars = plt.barh(y_pos, rates, align='center', color=colors, alpha=0.7)
            
            # Add rejection percentages as text labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                if width > 0:
                    plt.text(max(5, min(width + 2, 95)), 
                            bar.get_y() + bar.get_height()/2,
                            f"{width:.1f}%", 
                            va='center',
                            fontsize=9)
            
            # Add 50% threshold line
            plt.axvline(x=50, color='red', linestyle='--', alpha=0.7, 
                    label='50% threshold (potentially problematic channels)')
            
            # Add shaded region for potentially problematic area
            plt.axvspan(50, 100, alpha=0.1, color='red')
            
            # Set labels and title
            plt.yticks(y_pos, channels)
            plt.xlabel('Rejection Rate (%)')
            plt.title('Channel-wise Rejection Rates')
            plt.xlim(0, 100)
            
            # Add a text box with potentially problematic channels
            problematic_chs = [ch for ch, rate in sorted_channels if rate >= 50]
            if problematic_chs:
                problematic_text = "Potentially problematic channels:\n" + "\n".join(problematic_chs)
                plt.text(55, 5, problematic_text, fontsize=10, 
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
            
            plt.tight_layout()
            
            # Show interactively if requested
            if interactive:
                plt.show()
            
            # Save the plot
            sub_id = getattr(self.params, "subject_id", "unknown")
            ses_id = getattr(self.params, "session_id", "unknown")
            run_id = getattr(self.params, "run_id", "01")
            
            if plot_dir is None:
                plot_dir = f"./data/processed/sub-{sub_id}/ses-{ses_id}/figures/autoreject"
            os.makedirs(plot_dir, exist_ok=True)
            
            plt.savefig(os.path.join(plot_dir, f"autoreject_{sub_id}_{ses_id}_run-{run_id}_channel_rejection_rates.png"), dpi=150)
            
            # Only close the plot if not in interactive mode
            if not interactive:
                plt.close()
            
            logging.info(f"[AutoRejectStep] Saved channel rejection rates visualization to {plot_dir}")
            
        except Exception as e:
            logging.warning(f"[AutoRejectStep] Error creating channel rejection rates visualization: {e}")
            import traceback
            logging.warning(traceback.format_exc())
            if 'plt' in locals() and plt.get_fignums():
                plt.close()  # Close any open figures
    def _save_cleaned_data(self, data, file_prefix, output_dir):
        """Save cleaned data to disk."""
        try:
            sub_id = self.params.get("subject_id", "01")
            ses_id = self.params.get("session_id", "001")
            run_id = self.params.get("run_id", "01")
            
            # Determine the save path
            if output_dir is None:
                if hasattr(self.params, "paths") and self.params.get("paths") is not None:
                    paths = self.params.get("paths")
                    if hasattr(paths, "get_checkpoint_path"):
                        # Use standard path from ProjectPaths
                        path = os.path.normpath(str(paths.get_checkpoint_path(
                            subject_id=sub_id,
                            session_id=ses_id,
                            checkpoint_name=file_prefix
                        )))
                    else:
                        # Fallback if get_checkpoint_path is missing
                        base_dir = os.path.join(f"./data/processed/sub-{sub_id}/ses-{ses_id}")
                        os.makedirs(base_dir, exist_ok=True)
                        path = os.path.join(base_dir, f"sub-{sub_id}_ses-{ses_id}_run-{run_id}_{file_prefix}_epo.fif")
                else:
                    # No paths object, use direct local path
                    base_dir = os.path.join(f"./data/processed/sub-{sub_id}/ses-{ses_id}")
                    os.makedirs(base_dir, exist_ok=True)
                    path = os.path.join(base_dir, f"sub-{sub_id}_ses-{ses_id}_run-{run_id}_{file_prefix}_epo.fif")
            else:
                # Use provided output directory
                os.makedirs(output_dir, exist_ok=True)
                path = os.path.join(output_dir, f"sub-{sub_id}_ses-{ses_id}_run-{run_id}_{file_prefix}_epo.fif")
            
            logging.info(f"[AutoRejectStep] Saving cleaned data to {path}")
            
            # Create a copy to avoid modifying the original
            data_to_save = data.copy()
            
            # Sanitize info dict to ensure it's serializable
            if 'temp' in data_to_save.info:
                # Remove complex objects that can't be serialized
                del data_to_save.info['temp']
            
            # Fix string encoding issues that cause the '>a' error
            for key in data_to_save.info.keys():
                if key in ['description', 'experimenter', 'proj_name', 'subject_info']:
                    if isinstance(data_to_save.info[key], dict):
                        # Handle dictionary fields like subject_info
                        for subkey in list(data_to_save.info[key].keys()):
                            val = data_to_save.info[key][subkey]
                            if isinstance(val, (bytes, np.bytes_)):
                                data_to_save.info[key][subkey] = str(val)
                            elif hasattr(val, 'dtype') and val.dtype.kind == 'S':
                                data_to_save.info[key][subkey] = str(val)
                    elif isinstance(data_to_save.info[key], (bytes, np.bytes_)):
                        # Handle direct string fields
                        data_to_save.info[key] = str(data_to_save.info[key])
                    elif hasattr(data_to_save.info[key], 'dtype') and data_to_save.info[key].dtype.kind == 'S':
                        data_to_save.info[key] = str(data_to_save.info[key])
            
            # Make sure subject_info exists and has his_id
            if 'subject_info' not in data_to_save.info:
                data_to_save.info['subject_info'] = {}
            if 'his_id' not in data_to_save.info['subject_info']:
                data_to_save.info['subject_info']['his_id'] = f"sub-{sub_id}"
            
            # Save the data
            data_to_save.save(path, overwrite=True)
            logging.info(f"[AutoRejectStep] Successfully saved cleaned data to {path}")
            
        except Exception as e:
            logging.error(f"[AutoRejectStep] Error saving cleaned data: {e}")
            logging.error(f"[AutoRejectStep] Attempted to save to: {output_dir}")
            
            # Try an alternative path as fallback
            try:
                alt_path = f"./autoreject_{file_prefix}_epo.fif"  # Make sure we use _epo.fif extension
                
                # Create a fresh copy with minimal info
                data_copy = data.copy()
                
                # Remove all temp data from info to avoid serialization issues
                if hasattr(data_copy, 'info') and 'temp' in data_copy.info:
                    del data_copy.info['temp']
                
                # Make sure subject_info exists and has his_id
                if not hasattr(data_copy.info, 'subject_info') or data_copy.info['subject_info'] is None:
                    data_copy.info['subject_info'] = {}
                
                # Create a completely new his_id as a regular Python string
                data_copy.info['subject_info'] = {'his_id': f"sub-{sub_id}"}
                
                # Save with minimal info
                data_copy.save(alt_path, overwrite=True)
                logging.info(f"[AutoRejectStep] Saved cleaned data to alternative path: {alt_path}")
            except Exception as alt_e:
                logging.error(f"[AutoRejectStep] Error saving to alternative path: {alt_e}")
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
            if 'temp' not in data.info:
                data.info["temp"] = {}
                
            # Store the reject_log information
            data.info["temp"]["autoreject"] = reject_log_dict
            
            # Also store the indices directly for easier access
            bad_indices = np.where(reject_log.bad_epochs)[0].tolist()
            data.info["temp"]["autoreject_bad_epochs"] = bad_indices
            
            logging.info(f"[AutoRejectStep] Stored AutoReject information in data.info['temp']")
            logging.info(f"[AutoRejectStep] Bad epochs: {len(bad_indices)} out of {len(reject_log.bad_epochs)}")
            
        except Exception as e:
            logging.error(f"[AutoRejectStep] Error storing reject_log in data.info: {e}")
            logging.warning("[AutoRejectStep] Will continue without storing reject_log")

    def _save_model_state(self, ar, model_filename, output_dir):
        """
        Save the AutoReject model state using pickle (more compatible than HDF5)
        """
        try:
            # First try to import h5io from the newer location in MNE
            h5io_available = False
            try:
                # Try newer MNE versions where h5io is directly in mne
                import mne
                if hasattr(mne, 'io') and hasattr(mne.io, 'h5io'):
                    from mne.io.h5io import write_hdf5
                    h5io_available = True
                # Try older versions where it was in externals
                elif hasattr(mne, 'externals') and hasattr(mne.externals, 'h5io'):
                    from mne.externals.h5io import write_hdf5
                    h5io_available = True
                # Try to import h5io on its own
                else:
                    import h5io
                    write_hdf5 = h5io.write_hdf5
                    h5io_available = True
            except ImportError:
                h5io_available = False
                logging.warning("[AutoRejectStep] h5io module not available. Will use pickle instead.")
            
            # Set up output directory
            sub_id = self.params.get("subject_id", "01")
            ses_id = self.params.get("session_id", "001")
            
            if output_dir is None:
                output_dir = f"./data/processed/sub-{sub_id}/ses-{ses_id}/models"
                
            os.makedirs(output_dir, exist_ok=True)
            
            # Set up filename
            if model_filename is None:
                model_filename = f"sub-{sub_id}_ses-{ses_id}_autoreject_model.pkl"
            
            # If h5io is available and filename is .h5, try to use it
            if h5io_available and model_filename.endswith('.h5'):
                file_path = os.path.join(output_dir, model_filename)
                try:
                    write_hdf5(file_path, ar.__getstate__(), title='autoreject', overwrite=True)
                    logging.info(f"[AutoRejectStep] Saved AutoReject model state to {file_path}")
                    return
                except Exception as e:
                    logging.warning(f"[AutoRejectStep] Error saving model with HDF5: {e}")
                    logging.warning("[AutoRejectStep] Falling back to pickle")
            
            # Use pickle as the default or fallback method
            if model_filename.endswith('.h5'):
                model_filename = model_filename.replace('.h5', '.pkl')
                
            pickle_path = os.path.join(output_dir, model_filename)
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
