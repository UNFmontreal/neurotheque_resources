# File: scr/steps/baseline_correction.py

import mne
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from .base import BaseStep
from sklearn.linear_model import LinearRegression, HuberRegressor, RANSACRegressor
import statsmodels.formula.api as smf
import pandas as pd

class BaselineCorrectionStep(BaseStep):
    """
    A step for applying various baseline correction methods to epoched EEG data.
    
    This step implements multiple baseline correction methods:
    - Classic baseline subtraction
    - Regression-based baseline correction (least squares)
    - Huber robust regression
    - RANSAC robust regression
    - Kalman filter for drift removal
    
    Parameters:
    -----------
    params : dict
        Dictionary containing the parameters for baseline correction:
        - method: str or list, baseline correction method(s) to apply
          Options: 'classic', 'regression', 'huber', 'ransac', 'kalman', 'all'
        - baseline: tuple, baseline period in seconds (start, end)
        - plot_comparison: bool, whether to plot a comparison of methods
        - save_plots: bool, whether to save plots to disk
        - plot_dir: str or None, directory to save plots
        - interactive: bool, whether to show plots interactively
        - channel_for_demo: str or None, channel to use for comparison plots
        - plot_title: str or None, title for the comparison plot
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        
        # Set default parameters if not provided
        if 'method' not in self.params:
            self.params['method'] = 'regression'
        
        if 'baseline' not in self.params:
            self.params['baseline'] = (-0.2, 0)
            
        if 'plot_comparison' not in self.params:
            self.params['plot_comparison'] = True
            
        if 'save_plots' not in self.params:
            self.params['save_plots'] = True
        
        logging.info(f"[BaselineCorrectionStep] Initialized with method: {self.params['method']}")
    
    def run(self, data):
        """
        Apply baseline correction to the provided epochs.
        
        Parameters:
        -----------
        data : mne.Epochs
            The epoched data to correct
            
        Returns:
        --------
        epochs_corrected : mne.Epochs or dict
            The baseline-corrected epochs. If multiple methods are requested,
            returns a dictionary mapping method names to corrected epochs objects.
        """
        if data is None:
            raise ValueError("[BaselineCorrectionStep] No data provided.")
            
        if not isinstance(data, mne.Epochs):
            raise ValueError("[BaselineCorrectionStep] Data must be an mne.Epochs object.")
            
        # Get parameters
        method = self.params.get('method', 'regression')
        baseline = self.params.get('baseline', (-0.2, 0))
        plot_comparison = self.params.get('plot_comparison', True)
        save_plots = self.params.get('save_plots', True)
        plot_dir = self.params.get('plot_dir', None)
        interactive = self.params.get('interactive', False)
        channel_for_demo = self.params.get('channel_for_demo', None)
        plot_title = self.params.get('plot_title', 'Comparison of Baseline Correction Methods')
        
        # Setup for plotting
        if plot_dir is None:
            # Try to get a processed_dir from params
            processed_dir = self.params.get('processed_dir', None)
            if processed_dir is None:
                # Try to get from paths object
                paths = self.params.get('paths', None)
                if paths is not None and hasattr(paths, 'get_derivative_path'):
                    sub_id = self.params.get('subject_id', 'unknown')
                    ses_id = self.params.get('session_id', '001')
                    plot_dir = str(paths.get_derivative_path(
                        subject_id=sub_id,
                        session_id=ses_id,
                        stage="baseline_correction"
                    ))
                else:
                    # Default fallback
                    plot_dir = "./baseline_correction"
            else:
                # Use provided processed_dir
                plot_dir = os.path.join(processed_dir, "baseline_correction")
        
        # Create plot directory if needed
        if save_plots:
            os.makedirs(plot_dir, exist_ok=True)
            logging.info(f"[BaselineCorrectionStep] Saving plots to: {plot_dir}")
            
        # Setup matplotlib backend for interactive plots if needed
        original_backend = None
        if interactive:
            try:
                import matplotlib
                original_backend = matplotlib.get_backend()
                if original_backend != 'TkAgg' and original_backend != 'Qt5Agg' and original_backend != 'WXAgg':
                    logging.info(f"[BaselineCorrectionStep] Switching matplotlib backend from {original_backend} to TkAgg for interactive plotting")
                    matplotlib.use('TkAgg')
                    plt.ion()  # Turn on interactive mode
            except Exception as e:
                logging.warning(f"[BaselineCorrectionStep] Could not switch to interactive backend: {e}")
        
        # Convert single method to list for uniform processing
        methods = [method] if isinstance(method, str) and method != 'all' else \
                  ['classic', 'regression', 'huber', 'ransac', 'kalman'] if method == 'all' else \
                  method
                  
        # Log chosen methods
        logging.info(f"[BaselineCorrectionStep] Applying baseline correction methods: {methods}")
        
        # Extract data 
        epochs = data.copy()  # Work on a copy
        data_array = epochs.get_data()  # (n_epochs, n_ch, n_times)
        times = epochs.times
        n_epochs, n_ch, n_times = data_array.shape
        events = epochs.events.copy()
        tmin = epochs.tmin
        
        # Get baseline mask and calculate baseline means
        bl_mask = (times >= baseline[0]) & (times < baseline[1])
        baseline_vals = data_array[:,:,bl_mask].mean(axis=2)  # (n_epochs, n_ch)
        
        # Dictionary to store results
        results = {}
        
        # === Method A: Classic baseline subtraction ===
        if 'classic' in methods:
            logging.info("[BaselineCorrectionStep] Applying classic baseline subtraction...")
            epochs_A = epochs.copy().apply_baseline(baseline)
            results['classic'] = epochs_A
        
        # === Method B: Mass-univariate regression LS ===
        if 'regression' in methods:
            logging.info("[BaselineCorrectionStep] Applying mass-univariate regression (least squares)...")
            resid_B = np.zeros_like(data_array)
            for ch in range(n_ch):
                X = baseline_vals[:,ch].reshape(-1,1)
                for t in range(n_times):
                    y = data_array[:,ch,t]
                    coef = LinearRegression().fit(X,y).coef_[0]
                    resid_B[:,ch,t] = y - coef * X.ravel()
            epochs_B = mne.EpochsArray(resid_B, epochs.info, events, tmin)
            results['regression'] = epochs_B
        
        # === Method D1: Robust Huber regression ===
        if 'huber' in methods:
            logging.info("[BaselineCorrectionStep] Applying robust Huber regression...")
            resid_D1 = np.zeros_like(data_array)
            for ch in range(n_ch):
                X = baseline_vals[:,ch].reshape(-1,1)
                for t in range(n_times):
                    y = data_array[:,ch,t]
                    try:
                        hub = HuberRegressor().fit(X,y)
                        resid_D1[:,ch,t] = y - hub.coef_[0]*X.ravel()
                    except Exception as e:
                        logging.warning(f"[BaselineCorrectionStep] Error in Huber regression for channel {ch}, time {t}: {e}")
                        # Use standard regression as fallback
                        coef = LinearRegression().fit(X,y).coef_[0]
                        resid_D1[:,ch,t] = y - coef * X.ravel()
            
            epochs_D1 = mne.EpochsArray(resid_D1, epochs.info, events, tmin)
            results['huber'] = epochs_D1
        
        # === Method D2: Robust RANSAC regression ===
        if 'ransac' in methods:
            logging.info("[BaselineCorrectionStep] Applying robust RANSAC regression...")
            resid_D2 = np.zeros_like(data_array)
            for ch in range(n_ch):
                X = baseline_vals[:,ch].reshape(-1,1)
                for t in range(n_times):
                    y = data_array[:,ch,t]
                    try:
                        ras = RANSACRegressor().fit(X,y)
                        coef = ras.estimator_.coef_[0]
                        resid_D2[:,ch,t] = y - coef * X.ravel()
                    except Exception as e:
                        logging.warning(f"[BaselineCorrectionStep] Error in RANSAC regression for channel {ch}, time {t}: {e}")
                        # Use standard regression as fallback
                        coef = LinearRegression().fit(X,y).coef_[0]
                        resid_D2[:,ch,t] = y - coef * X.ravel()
            
            epochs_D2 = mne.EpochsArray(resid_D2, epochs.info, events, tmin)
            results['ransac'] = epochs_D2
        
        # === Method E: Adaptive drift removal (manual Kalman) ===
        if 'kalman' in methods:
            logging.info("[BaselineCorrectionStep] Applying Kalman filter for drift removal...")
            # Try to get the raw data
            raw = None
            
            # Check if raw is attached to the epochs
            if hasattr(epochs, 'raw') and epochs.raw is not None:
                raw = epochs.raw.copy()
                logging.info("[BaselineCorrectionStep] Using raw data attached to epochs.")
            
            # If no raw data, skip this method
            if raw is None:
                logging.warning("[BaselineCorrectionStep] Raw data not available for Kalman method. Skipping.")
                if 'kalman' in methods:
                    methods.remove('kalman')
            else:
                try:
                    # Extract EEG channels
                    eeg_picks = mne.pick_types(raw.info, eeg=True)
                    raw_E = raw.copy().pick(eeg_picks)
                    
                    # Extract channel-average signal
                    eeg_data = raw_E.get_data()
                    avg_sig = eeg_data.mean(axis=0)
                    n_times_raw = avg_sig.size
                    
                    # Kalman initialization
                    x = 0.0; P = 1.0
                    Q = 1e-5; R = 1e-2  # Process and measurement noise
                    drift = np.zeros(n_times_raw)
                    
                    # Apply Kalman filter
                    for t in range(n_times_raw):
                        # predict
                        x_pred = x
                        P_pred = P + Q
                        # update
                        z = avg_sig[t]
                        K = P_pred / (P_pred + R)
                        x = x_pred + K*(z - x_pred)
                        P = (1-K)*P_pred
                        drift[t] = x
                    
                    # Subtract drift from each EEG channel
                    for idx in range(len(eeg_picks)):
                        raw_E._data[idx,:] -= drift
                    
                    # Re-epoch & baseline subtract
                    epochs_E = mne.Epochs(raw_E, events, epochs.event_id, tmin, epochs.tmax,
                                        baseline=baseline, preload=True)
                    results['kalman'] = epochs_E
                except Exception as e:
                    logging.error(f"[BaselineCorrectionStep] Error applying Kalman method: {e}")
                    if 'kalman' in methods:
                        methods.remove('kalman')
        # === Method H: Linear Mixed‑Effects Model (GLMM) ===
        if 'glmm' in methods:
            logging.info("[BaselineCorrectionStep] Applying GLMM baseline correction via MixedLM…")
            # Prepare metadata
            if epochs.metadata is None or 'subject' not in epochs.metadata:
                # you must supply subject IDs per epoch
                raise RuntimeError("epochs.metadata['subject'] must exist for GLMM.")
            subjects = epochs.metadata['subject'].values  # length n_epochs
            
            # preallocate residuals container
            resid_H = np.zeros_like(data_array)  # shape (n_epochs, n_ch, n_times)

            # Loop channels & timepoints (consider parallelizing!)
            for ch in range(n_ch):
                for t in range(n_times):
                    # Build DataFrame for this slice
                    df = pd.DataFrame({
                        'voltage':  data_array[:, ch, t],
                        'baseline': baseline_vals[:, ch],
                        'subject':  subjects
                    })
                    # Fit MixedLM: random intercept & slope on baseline per subject
                    try:
                        model = smf.mixedlm(
                            "voltage ~ baseline", 
                            df,
                            groups=df["subject"],
                            re_formula="~baseline"
                        )
                        fit = model.fit(reml=False, method="lbfgs", warn_convergence=False)
                        # residuals are baseline‑corrected voltage
                        resid_H[:, ch, t] = fit.resid
                    except Exception as e:
                        logging.warning(f"[GLMM] Channel {ch} time {t} failed ({e}); falling back to OLS.")
                        # fallback to simple subtraction
                        resid_H[:, ch, t] = data_array[:, ch, t] - df['baseline'].values

            # Create new Epochs from residuals
            epochs_H = mne.EpochsArray(resid_H, epochs.info, events, tmin)
            results['glmm'] = epochs_H        
        # === Plot comparison of methods ===
        if plot_comparison and len(methods) > 1:
            logging.info("[BaselineCorrectionStep] Generating comparison plot of baseline correction methods...")
            
            # Choose a representative channel to plot
            if channel_for_demo is not None and channel_for_demo in epochs.ch_names:
                ch_name = channel_for_demo
            else:
                ch_name = 'P3' if 'P3' in epochs.ch_names else epochs.ch_names[0]
                
            ch_idx = epochs.ch_names.index(ch_name)
            
            plt.figure(figsize=(10, 6))
            
            # Colors for different methods
            colors = {
                'classic': 'blue',
                'regression': 'red', 
                'huber': 'purple',
                'ransac': 'orange',
                'kalman': 'brown'
            }
            
            # Plot each method's average
            for method_name in methods:
                if method_name in results:
                    evoked = results[method_name].average()
                    plt.plot(evoked.times, evoked.data[ch_idx], 
                             label=f"{method_name.capitalize()}", 
                             color=colors.get(method_name, 'gray'))
            
            # Plot original data without baseline for comparison
            if 'classic' not in methods:  # Only if classic isn't already plotted
                plt.plot(epochs.times, epochs.average().data[ch_idx], 
                         label="No correction", linestyle='--', color='black', alpha=0.5)
            
            plt.axvline(0, color='k', linestyle='--', alpha=0.5)
            plt.axhline(0, color='k', linestyle='-', alpha=0.2)
            plt.axvspan(baseline[0], baseline[1], color='gray', alpha=0.2, label='Baseline period')
            
            plt.legend()
            title = f'{plot_title} ({ch_name})' if plot_title else f'Comparison of Baseline Correction Methods ({ch_name})'
            plt.title(title)
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude (μV)')
            plt.grid(True, linestyle=':', alpha=0.3)
            
            if save_plots:
                plot_filename = os.path.join(plot_dir, "baseline_methods_comparison.png")
                plt.savefig(plot_filename, dpi=300)
                logging.info(f"[BaselineCorrectionStep] Saved comparison plot to {plot_filename}")
            
            plt.tight_layout()
            if interactive:
                plt.show()
            else:
                plt.close()
            
            # Also create a difference plot to highlight the effect of each method
            plt.figure(figsize=(10, 6))
            
            # Get no baseline correction as reference
            ref_data = epochs.average().data[ch_idx]
            
            for method_name in methods:
                if method_name in results:
                    evoked = results[method_name].average()
                    diff = evoked.data[ch_idx] - ref_data
                    plt.plot(evoked.times, diff, 
                             label=f"{method_name.capitalize()} - Original", 
                             color=colors.get(method_name, 'gray'))
            
            plt.axvline(0, color='k', linestyle='--', alpha=0.5)
            plt.axhline(0, color='k', linestyle='-', alpha=0.2)
            plt.axvspan(baseline[0], baseline[1], color='gray', alpha=0.2, label='Baseline period')
            
            plt.legend()
            plt.title(f'Difference: Methods - Original Data ({ch_name})')
            plt.xlabel('Time (s)')
            plt.ylabel('Difference (μV)')
            plt.grid(True, linestyle=':', alpha=0.3)
            
            if save_plots:
                plot_filename = os.path.join(plot_dir, "baseline_methods_difference.png")
                plt.savefig(plot_filename, dpi=300)
                logging.info(f"[BaselineCorrectionStep] Saved difference plot to {plot_filename}")
            
            plt.tight_layout()
            if interactive:
                plt.show()
            else:
                plt.close()
        
        # Restore original matplotlib backend if needed
        if interactive and original_backend:
            try:
                import matplotlib
                if original_backend != 'TkAgg' and original_backend != 'Qt5Agg' and original_backend != 'WXAgg':
                    logging.info(f"[BaselineCorrectionStep] Switching matplotlib backend back to {original_backend}")
                    matplotlib.use(original_backend)
                    plt.ioff()  # Turn off interactive mode
            except Exception as e:
                logging.warning(f"[BaselineCorrectionStep] Error switching back to original backend: {e}")
        
        # Return results
        if len(methods) == 1:
            logging.info(f"[BaselineCorrectionStep] Returning epochs corrected using {methods[0]} method")
            return results[methods[0]]
        else:
            logging.info(f"[BaselineCorrectionStep] Returning dictionary of epochs corrected using multiple methods")
            return results 