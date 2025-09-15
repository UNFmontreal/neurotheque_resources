# File: scr/steps/ica_extraction.py

import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.preprocessing import ICA
from scipy import signal

from .base import BaseStep

class ICAExtractionStep(BaseStep):
    """
    ICA extraction step that computes ICA decomposition and visualizes components.
    This step focuses only on the extraction of ICA components without automatic 
    labeling or artifact removal.
    
    Parameters
    ----------
    n_components : float or int
        Number of components to extract. If float between 0 and 1, it's treated as
        the fraction of variance to be explained. If int, it's the exact number of components.
    method : str
        ICA algorithm to use ('fastica', 'infomax', 'picard')
    max_iter : int
        Maximum number of iterations during fitting
    fit_params : dict
        Additional parameters passed to the ICA algorithm
    decim : int
        Decimation factor to use during fitting to reduce computation time
    plot_components : bool
        Whether to generate plots of component topographies
    plot_sources : bool
        Whether to generate plots of source time courses
    plot_properties : bool
        Whether to generate detailed plots of component properties
    interactive : bool
        Whether to allow interactive plots
    plot_dir : str
        Directory to save plots
    
    References:
      - Chaumon et al. (2015), J Neurosci Methods
      - Winkler et al. (2015), NeuroImage
    """
    
    def run(self, data):
        """Run ICA extraction on the input data."""
        # Configure matplotlib backend (default headless unless interactive)
        import matplotlib
        if not self.params.get("interactive", False):
            try:
                matplotlib.use('Agg', force=True)
            except Exception:
                pass
        
        if data is None:
            raise ValueError("[ICAExtractionStep] No data provided for ICA.")

        # --------------------------
        # 1) Merge Default Params
        # --------------------------
        default_params = {
            "n_components": 0.99,      # Or an int if you prefer a fixed number
            "method": "infomax",
            "max_iter": 2000,
            "fit_params": {"extended": True, "l_rate": 1e-3},
            "decim": 3,
            "picks": "eeg",           # restrict ICA to EEG channels by default
            "use_good_epochs_only": True,
            "random_state": 42,
            "plot_dir": None,          # Will default to a subdirectory of paths
            "interactive": True,
            "plot_components": True,
            "plot_sources": True,
            "plot_properties": True,
            "plot_psd": True,
            "plot_overlay": False,
            "plot_details": False,
            "verbose": True,
            "save_data": False
        }
        params = {**default_params, **self.params}
        sub_id = params.get("subject_id", "unknown")
        ses_id = params.get("session_id", "001")
        task_id = params.get("task_id", None)
        run_id = params.get("run_id", None)
        paths = params.get("paths", None)
        
        # Collection to store figures for display in notebook
        self.figures = []
        
        # --------------------------
        # Configure matplotlib backend for plotting
        # --------------------------
        original_backend = matplotlib.get_backend()
        if params["interactive"]:
            try:
                plt.ion()
            except Exception:
                logging.warning("[ICAExtractionStep] Interactive backend not available; continuing headless.")
                
        # --------------------------
        # 2) Instantiate ICA
        # --------------------------
        # Handle 'all' option for n_components
        n_components = params["n_components"]
        if n_components == 'all':
            logging.info("[ICAExtractionStep] Using all available components for ICA decomposition.")
            n_components = None  # None tells MNE to use all components
            
        ica = ICA(
            n_components=n_components,
            method=params["method"],
            max_iter=params["max_iter"],
            fit_params=params["fit_params"],
            random_state=params["random_state"]
        )
        
        # --------------------------
        # 3) Select Data for ICA
        # --------------------------
        if params["use_good_epochs_only"]:
            logging.info("[ICAExtractionStep] Using data with AutoReject annotations to exclude bad segments.")
            
            # Check if data has BAD_autoreject annotations
            has_autoreject_annotations = False
            if data.annotations is not None and len(data.annotations) > 0:
                bad_annot_count = sum(1 for desc in data.annotations.description if desc == 'BAD_autoreject')
                if bad_annot_count > 0:
                    has_autoreject_annotations = True
                    logging.info(f"[ICAExtractionStep] Found {bad_annot_count} 'BAD_autoreject' annotations")
                    
            if has_autoreject_annotations:
                # Create epochs, automatically rejecting those with annotations
                # MNE will skip epochs that overlap with 'BAD' annotations
                events = mne.make_fixed_length_events(data, duration=1)
                good_epochs = mne.Epochs(
                    data, 
                    events, 
                    tmin=0, 
                    tmax=1, 
                    baseline=None,  # No baseline correction for ICA
                    preload=True,
                    reject_by_annotation=True  # This ensures epochs with BAD_* annotations are excluded
                )
                logging.info(f"[ICAExtractionStep] Created {len(good_epochs)} epochs, excluding segments with bad annotations")

            elif isinstance(data, mne.Epochs) or str(type(data).__name__) == 'EpochsFIF' or hasattr(data, 'epochs'):
                # Handle already epoched data
                logging.info("[ICAExtractionStep] Input is already epoched data")
                
                # Check for autoreject info in data.info["temp"]
                bad_indices = None
                if 'temp' in data.info:
                    if 'autoreject_bad_epochs' in data.info['temp']:
                        # Method 1: Direct indices
                        bad_indices = data.info['temp']['autoreject_bad_epochs']
                        logging.info(f"[ICAExtractionStep] Found {len(bad_indices)} bad epochs from autoreject_bad_epochs")
                    elif 'autoreject' in data.info['temp'] and 'bad_epochs' in data.info['temp']['autoreject']:
                        # Method 2: Boolean array from full autoreject info
                        bad_epochs_bool = data.info['temp']['autoreject']['bad_epochs']
                        bad_indices = [i for i, bad in enumerate(bad_epochs_bool) if bad]
                        logging.info(f"[ICAExtractionStep] Found {len(bad_indices)} bad epochs from autoreject info")
                
                if bad_indices:
                    # Create mask of good epochs
                    good_indices = [i for i in range(len(data)) if i not in bad_indices]
                    good_epochs = data[good_indices]
                    logging.info(f"[ICAExtractionStep] Using {len(good_epochs)}/{len(data)} good epochs for ICA")
                else:
                    # No bad epochs found, use all epochs
                    logging.warning("[ICAExtractionStep] No autoreject information found. Using all epochs for ICA.")
                    good_epochs = data
                    logging.info(f"[ICAExtractionStep] Using all {len(good_epochs)} epochs for ICA")
            else:
                    # No annotations found, use all data
                logging.warning("[ICAExtractionStep] No BAD_autoreject annotations found. Using all data for ICA.")
                events = mne.make_fixed_length_events(data, duration=1)
                good_epochs = mne.Epochs(data, events, tmin=0, tmax=1, baseline=None, preload=True)
                logging.info(f"[ICAExtractionStep] Created {len(good_epochs)} epochs (no annotations to exclude)")
        else:
            logging.info("[ICAExtractionStep] Using all data for ICA (use_good_epochs_only=False).")
            good_epochs = data

        # --------------------------
        # 4) Configure plot directory
        # --------------------------
        if params["plot_dir"] is None:
            # Use paths object if available to get standard plot directory
            if paths is not None:
                plot_dir = paths.get_ica_report_dir(sub_id, ses_id, task_id, run_id)
            else:
                # Default to a subdirectory in the current directory
                plot_dir = f"./ica_plots/sub-{sub_id}/ses-{ses_id}"
                if task_id:
                    plot_dir += f"/task-{task_id}"
                if run_id:
                    plot_dir += f"/run-{run_id}"
        else:
            plot_dir = params["plot_dir"]
        
        os.makedirs(plot_dir, exist_ok=True)
        logging.info(f"[ICAExtractionStep] Saving plots to {plot_dir}")
        
        # --------------------------
        # 5) Fit ICA
        # --------------------------
        logging.info(f"[ICAExtractionStep] Fitting ICA with {ica.n_components} components using {params['method']} method...")
        # Build picks argument
        picks_arg = None
        try:
            picks_param = params.get("picks", "eeg")
            if isinstance(picks_param, str):
                key = picks_param.strip().lower()
                if key in (None, "all", "*"):
                    picks_arg = None
                elif key == "eeg":
                    picks_arg = mne.pick_types(good_epochs.info, eeg=True, meg=False, eog=False, ecg=False, stim=False, misc=False, seeg=False, ecog=False)
                else:
                    # Treat as channel name pattern
                    if hasattr(mne, 'pick_channels_regexp'):
                        picks_arg = mne.pick_channels_regexp(good_epochs.ch_names, picks_param)
                    else:
                        picks_arg = mne.pick_channels(good_epochs.ch_names, include=[picks_param], exclude=[])
            elif isinstance(picks_param, (list, tuple)):
                picks_arg = mne.pick_channels(good_epochs.ch_names, include=list(picks_param), exclude=[])
            else:
                picks_arg = None
        except Exception:
            # Fall back silently
            picks_arg = None
        try:
            ica.fit(
                good_epochs,
                picks=picks_arg,
                decim=params["decim"],
                reject=None,
            )
            logging.info(f"[ICAExtractionStep] ICA fitted successfully. Found {ica.n_components_} components.")
            
            # Store the ICA fitted flag
            self.ica = ica
        except Exception as e:
            logging.error(f"[ICAExtractionStep] Error fitting ICA: {e}")
            logging.error("[ICAExtractionStep] ICA extraction failed")
            return data
        
        # Generate component topography plots
        if params["plot_components"]:
            topo_figs = self._plot_component_topographies(ica, plot_dir, params)
            if topo_figs:
                self.figures.extend(topo_figs)
        
        # Generate source time courses
        if params["plot_sources"]: 
            source_figs = self._plot_source_timecourses(ica, data, plot_dir, params)
            if source_figs:
                self.figures.extend(source_figs)
            
        # Generate detailed component property plots
        if params["plot_properties"]:
            prop_figs = self._plot_component_properties(ica, good_epochs, plot_dir, params)
            if prop_figs:
                self.figures.extend(prop_figs)
            
        # Generate PSD plots for each component
        if params["plot_psd"]:
            psd_figs = self._plot_component_psd(ica, data, plot_dir, params)
            if psd_figs:
                self.figures.extend(psd_figs)
            
        # # Generate overlay plots to compare original and reconstructed data
        # if params["plot_overlay"]:
        #     self._plot_data_overlay(ica, data, plot_dir, params)
            
        # Generate component details page with all plots
        # self._generate_component_details(ica, data, good_epochs, plot_dir, params)
        
        # Only generate component details if specifically requested
        if params.get("plot_details", False):
            try:
                self._generate_component_details(ica, data, good_epochs, plot_dir, params)
            except Exception as e:
                logging.error(f"[ICAExtractionStep] Error generating component details: {e}")
                logging.error("[ICAExtractionStep] Continuing without component details")
        

        # --------------------------
        # 6) Save ICA solution
        # --------------------------
        if paths is not None and hasattr(ica, 'n_components_'):  # Only save if ICA was successfully fitted
            try:
                # Use normalized path handling for Windows compatibility
                ica_dir = os.path.normpath(os.path.dirname(str(paths.get_derivative_path(sub_id, ses_id))))
                ica_filename = f'sub-{sub_id}_ses-{ses_id}_task-{task_id}_run-{run_id}_desc-ica_decomposition.fif'
                ica_file = os.path.join(ica_dir, ica_filename)
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(ica_file), exist_ok=True)
                
                # Clean the ICA object to ensure it can be serialized
                ica_clean = ica.copy()                
                # Save without using paths for the final save
                logging.info(f"[ICAExtractionStep] Saving ICA decomposition to {ica_file}")
                ica_clean.save(ica_file, overwrite=True)
            except Exception as e:
                logging.error(f"[ICAExtractionStep] Error saving ICA decomposition: {e}")
                logging.error("[ICAExtractionStep] Continuing without saving ICA decomposition")
        
        # Save the data with ICA components if requested
        save_data = params.get("save_data", False)
        if save_data and hasattr(ica, 'n_components_'):  # Only save if ICA was successfully fitted
            try:
                if paths is None:
                    logging.error("[ICAExtractionStep] Cannot save data: paths object is None; skipping save but continuing.")
                    # Do not return early; continue to store ICA in info and finish
                    raise RuntimeError("paths is None for save_data")
                
                # Use normalized path handling for Windows compatibility
                output_dir = os.path.normpath(os.path.dirname(str(paths.get_derivative_path(sub_id, ses_id))))

                # Create directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                
                # Create filename
                run_str = f"_run-{run_id}" if run_id else ""
                task_str = f"_task-{task_id}" if task_id else ""
                file_name = f"sub-{sub_id}_ses-{ses_id}{task_str}{run_str}_desc-ica_epo.fif"
                
                file_path = os.path.join(output_dir, file_name)
                
                # Make a clean copy of the data to save
                data_to_save = data.copy()
                
                # Save the data
                logging.info(f"[ICAExtractionStep] Saving data with ICA components to {file_path}")
                try:
                    data_to_save.save(file_path, overwrite=True)
                    logging.info(f"[ICAExtractionStep] Successfully saved data")
                except Exception as save_err:
                    logging.error(f"[ICAExtractionStep] Error in data save operation: {save_err}")
            except Exception as e:
                logging.error(f"[ICAExtractionStep] Error saving data with ICA components: {e}")
                logging.error("[ICAExtractionStep] Continuing without saving data")
        
        # If we switched matplotlib backends, switch back to the original
        if params["interactive"] and original_backend not in ['TkAgg', 'Qt5Agg', 'WXAgg']:
            try:
                logging.info(f"[ICAExtractionStep] Switching matplotlib backend back to {original_backend}")
                matplotlib.use(original_backend)
                plt.ioff()  # Turn off interactive mode
            except Exception as e:
                logging.warning(f"[ICAExtractionStep] Error switching back to original backend: {e}")
        
        # Store ica object in the data's info
        if not 'temp' in data.info:
            data.info['temp'] = {}
        data.info['temp']['ica'] = ica
        
        # Display figures in notebook if running in Jupyter
        if 'ipykernel' in sys.modules and self.figures:
            for fig in self.figures:
                if fig is not None:
                    plt.figure(fig.number)
                    plt.show()
        
        return data
    
    def _plot_component_topographies(self, ica, plot_dir, params):
        """Generate topographic plots of ICA components."""
        logging.info("[ICAExtractionStep] Generating component topography plots")
        
        try:
            # Create a directory for component topographies
            topo_dir = os.path.join(plot_dir, "topographies")
            os.makedirs(topo_dir, exist_ok=True)
            
            # Get number of components, with fallbacks for different ICA implementations
            if hasattr(ica, 'n_components_'):
                n_components = ica.n_components_
            elif hasattr(ica, 'n_components'):
                n_components = ica.n_components
            else:
                # As a last resort, try to determine from the mixing matrix shape
                n_components = ica.mixing_matrix_.shape[1] if hasattr(ica, 'mixing_matrix_') else 20
                logging.warning(f"[ICAExtractionStep] Could not determine n_components, using: {n_components}")
            
            n_per_plot = min(20, n_components)  # Maximum 20 components per plot
            all_figs = []
            
            for start_idx in range(0, n_components, n_per_plot):
                end_idx = min(start_idx + n_per_plot, n_components)
                picks = list(range(start_idx, end_idx))
                
                # Create the plot
                fig = ica.plot_components(picks=picks, show=False)
                
                # Store the figure for display in notebook
                if isinstance(fig, list):
                    all_figs.extend(fig)
                else:
                    all_figs.append(fig)
                
                # Save the figure to file without closing it
                if isinstance(fig, list):
                    for i, f in enumerate(fig):
                        f.savefig(os.path.join(topo_dir, f"components_{start_idx+i:03d}.png"), dpi=300)
                else:
                    fig.savefig(os.path.join(topo_dir, f"components_{start_idx:03d}-{end_idx-1:03d}.png"), dpi=300)
                
                # Don't close the figure so it can be displayed in the notebook
                # In interactive mode, you might want to show it too
                if params["interactive"]:
                    plt.show()
        
        except Exception as e:
            logging.error(f"[ICAExtractionStep] Error in _plot_component_topographies: {e}")
            
        # Return the figures for display in notebook
        return all_figs
    
    def _plot_source_timecourses(self, ica, data, plot_dir, params):
        """Generate plots of ICA source time courses."""
        logging.info("[ICAExtractionStep] Generating source time course plots")
        
        all_figs = []
        
        try:
            # Create a directory for source time courses
            sources_dir = os.path.join(plot_dir, "sources")
            os.makedirs(sources_dir, exist_ok=True)
            
            # Create the sources plot
            fig = ica.plot_sources(data, show=False)
            all_figs.append(fig)
            
            # Save the plot to file when it's a Matplotlib Figure
            try:
                if hasattr(fig, 'savefig'):
                    fig.savefig(os.path.join(sources_dir, "sources_all.png"), dpi=300)
                else:
                    logging.debug("[ICAExtractionStep] plot_sources returned a browser; skipping savefig.")
            except Exception as e:
                logging.error(f"[ICAExtractionStep] Error saving source time course plot: {e}")
                
            # Plot individual source time courses
            try:
                # Use get_sources safely based on data type
                sources = ica.get_sources(data)
                
                # Handle different data types
                if isinstance(sources, mne.Epochs) or str(type(sources).__name__) == 'EpochsFIF':
                    # For epochs, take the first epoch or average across epochs
                    source_data = sources.get_data().mean(axis=0)  # Average across epochs
                    sfreq = sources.info['sfreq']
                    times = np.arange(source_data.shape[1]) / sfreq
                    
                    for comp_idx in range(ica.n_components_):
                        try:
                            fig, ax = plt.subplots(figsize=(10, 3))
                            ax.plot(times, source_data[comp_idx], 'b')
                            ax.set_title(f'Component {comp_idx} Time Course (Averaged)')
                            ax.set_xlabel('Time (s)')
                            ax.set_ylabel('Amplitude')
                            fig.tight_layout()
                            fig.savefig(os.path.join(sources_dir, f"source_{comp_idx:03d}.png"), dpi=300)
                            all_figs.append(fig)
                        except Exception as e:
                            logging.error(f"[ICAExtractionStep] Error plotting individual source {comp_idx}: {e}")
                
                elif isinstance(sources, mne.io.Raw):
                    # For Raw data
                    source_data = sources.get_data()
                    times = np.arange(source_data.shape[1]) / sources.info['sfreq']
                    
                    for comp_idx in range(ica.n_components_):
                        try:
                            fig, ax = plt.subplots(figsize=(10, 3))
                            ax.plot(times, source_data[comp_idx], 'b')
                            ax.set_title(f'Component {comp_idx} Time Course')
                            ax.set_xlabel('Time (s)')
                            ax.set_ylabel('Amplitude')
                            fig.tight_layout()
                            fig.savefig(os.path.join(sources_dir, f"source_{comp_idx:03d}.png"), dpi=300)
                            all_figs.append(fig)
                        except Exception as e:
                            logging.error(f"[ICAExtractionStep] Error plotting individual source {comp_idx}: {e}")
                
                else:
                    logging.warning(f"[ICAExtractionStep] Unsupported data type for individual source plots: {type(sources)}")
                    
            except Exception as e:
                logging.error(f"[ICAExtractionStep] Error creating individual source plots: {e}")
            
            # In interactive mode, show the plots
            if params["interactive"]:
                for fig in all_figs:
                    plt.figure(fig.number)
                    plt.show()
                    
        except Exception as e:
            logging.error(f"[ICAExtractionStep] Error in _plot_source_timecourses: {e}")
            
        return all_figs
    
    def _plot_component_properties(self, ica, epochs, plot_dir, params):
        """Generate detailed plots of component properties."""
        logging.info("[ICAExtractionStep] Generating component property plots")
        
        all_figs = []
        
        try:
            # Create a directory for component properties
            props_dir = os.path.join(plot_dir, "properties")
            os.makedirs(props_dir, exist_ok=True)
            
            for comp_idx in range(ica.n_components_):
                try:
                    logging.info(f"[ICAExtractionStep] Plotting properties for component {comp_idx}")
                    
                    # Generate the property plots
                    figs = ica.plot_properties(epochs, picks=comp_idx, show=False)
                    
                    # Save the plots to files and collect them
                    for i, fig in enumerate(figs):
                        fig.savefig(os.path.join(props_dir, f"comp_{comp_idx:03d}_property_{i}.png"), dpi=300)
                        all_figs.append(fig)
                    
                    # In interactive mode, show the plots
                    if params["interactive"]:
                        for fig in figs:
                            plt.figure(fig.number)
                            plt.show()
                
                except Exception as e:
                    logging.error(f"[ICAExtractionStep] Error processing properties for component {comp_idx}: {e}")
                    
        except Exception as e:
            logging.error(f"[ICAExtractionStep] Error in _plot_component_properties: {e}")
            
        return all_figs
    
    def _plot_component_psd(self, ica, data, plot_dir, params):
        """Generate PSD plots for each component."""
        logging.info("[ICAExtractionStep] Generating component PSD plots")
        
        all_figs = []
        
        try:
            # Create a directory for PSD plots
            psd_dir = os.path.join(plot_dir, "psd")
            os.makedirs(psd_dir, exist_ok=True)
            
            # Get the sources
            sources = ica.get_sources(data)
            sfreq = data.info['sfreq']
            
            # Extract source data based on data type
            if isinstance(sources, mne.Epochs) or str(type(sources).__name__) == 'EpochsFIF':
                # For epochs, take average across epochs
                source_data_array = sources.get_data().mean(axis=0)  # Average across epochs
            elif isinstance(sources, mne.io.Raw):
                # For Raw data
                source_data_array = sources.get_data()
            else:
                logging.warning(f"[ICAExtractionStep] Unsupported data type for PSD plots: {type(sources)}")
                return all_figs
                
            # Create plots comparing PSDs of all components
            try:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                for comp_idx in range(ica.n_components_):
                    source_data = source_data_array[comp_idx]
                    f, Pxx = signal.welch(source_data, fs=sfreq, nperseg=min(4096, len(source_data)))
                    ax.semilogy(f, Pxx, alpha=0.7, label=f'Comp {comp_idx}' if comp_idx < 10 else None)
                
                ax.set_title('Power Spectral Density of ICA Components')
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('PSD (μV²/Hz)')
                ax.set_xlim([0, min(100, sfreq/2)])
                ax.grid(True)
                
                if ica.n_components_ < 20:  # Only show legend if not too many components
                    ax.legend(loc='upper right')
                
                fig.tight_layout()
                fig.savefig(os.path.join(psd_dir, "all_components_psd.png"), dpi=300)
                all_figs.append(fig)
            except Exception as e:
                logging.error(f"[ICAExtractionStep] Error creating combined PSD plot: {e}")
            
            # Create individual PSD plots
            for comp_idx in range(ica.n_components_):
                try:
                    source_data = source_data_array[comp_idx]
                    f, Pxx = signal.welch(source_data, fs=sfreq, nperseg=min(4096, len(source_data)))
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.semilogy(f, Pxx)
                    ax.set_title(f'Component {comp_idx} - Power Spectral Density')
                    ax.set_xlabel('Frequency (Hz)')
                    ax.set_ylabel('PSD (μV²/Hz)')
                    ax.set_xlim([0, min(100, sfreq/2)])
                    ax.grid(True)
                    
                    # Add some statistics to the plot
                    max_freq_idx = np.argmax(Pxx)
                    peak_freq = f[max_freq_idx]
                    peak_power = Pxx[max_freq_idx]
                    
                    # Calculate power in frequency bands
                    delta_mask = (f >= 1) & (f <= 4)
                    theta_mask = (f >= 4) & (f <= 8)
                    alpha_mask = (f >= 8) & (f <= 13)
                    beta_mask = (f >= 13) & (f <= 30)
                    gamma_mask = (f >= 30) & (f <= 100)
                    
                    delta_power = np.mean(Pxx[delta_mask]) if np.any(delta_mask) else 0
                    theta_power = np.mean(Pxx[theta_mask]) if np.any(theta_mask) else 0
                    alpha_power = np.mean(Pxx[alpha_mask]) if np.any(alpha_mask) else 0
                    beta_power = np.mean(Pxx[beta_mask]) if np.any(beta_mask) else 0
                    gamma_power = np.mean(Pxx[gamma_mask]) if np.any(gamma_mask) else 0
                    
                    stats_text = (
                        f"Peak Frequency: {peak_freq:.1f} Hz\n"
                        f"Delta Power: {delta_power:.2e}\n"
                        f"Theta Power: {theta_power:.2e}\n"
                        f"Alpha Power: {alpha_power:.2e}\n"
                        f"Beta Power: {beta_power:.2e}\n"
                        f"Gamma Power: {gamma_power:.2e}"
                    )
                    
                    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
                           verticalalignment='top', bbox=dict(boxstyle='round',
                                                            facecolor='white', alpha=0.8))
                    
                    fig.tight_layout()
                    fig.savefig(os.path.join(psd_dir, f"comp_{comp_idx:03d}_psd.png"), dpi=300)
                    all_figs.append(fig)
                except Exception as e:
                    logging.error(f"[ICAExtractionStep] Error creating PSD plot for component {comp_idx}: {e}")
            
            # In interactive mode, show the plots
            if params["interactive"]:
                for fig in all_figs:
                    plt.figure(fig.number)
                    plt.show()
                    
        except Exception as e:
            logging.error(f"[ICAExtractionStep] Error in _plot_component_psd: {e}")
            
        return all_figs
    
    def _plot_data_overlay(self, ica, data, plot_dir, params):
        """Generate overlay plots to compare original and component-projected data."""
        logging.info("[ICAExtractionStep] Generating data overlay plots")
        
        try:
            # Create a directory for overlay plots
            overlay_dir = os.path.join(plot_dir, "overlay")
            os.makedirs(overlay_dir, exist_ok=True)
            
            # Create plots for each component
            for comp_idx in range(ica.n_components_):
                try:
                    # Get the pattern for this component
                    pattern = ica.get_components()[:, comp_idx]
                    
                    # Create a copy of ICA
                    ica_comp = ica.copy()
                    ica_comp.exclude = list(range(ica.n_components_))
                    ica_comp.exclude.remove(comp_idx)  # Keep only this component
                    
                    # Apply this component to the data
                    comp_data = ica_comp.apply(data.copy())
                    
                    # Plot the original data and component-projected data
                    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
                    
                    # Find channels where this component has the strongest contribution
                    ch_idx = np.abs(pattern).argsort()[-5:]  # Top 5 channels
                    ch_names = [data.ch_names[i] for i in ch_idx]
                    
                    # Original data
                    data_array = data.get_data(picks=ch_names)
                    times = np.arange(data_array.shape[1]) / data.info['sfreq']
                    for i, ch in enumerate(ch_names):
                        axs[0].plot(times, data_array[i], label=ch)
                    axs[0].set_title(f'Original Data (Top 5 channels for Component {comp_idx})')
                    axs[0].set_ylabel('Amplitude (μV)')
                    axs[0].legend()
                    
                    # Component-projected data
                    comp_array = comp_data.get_data(picks=ch_names)
                    for i, ch in enumerate(ch_names):
                        axs[1].plot(times, comp_array[i], label=ch)
                    axs[1].set_title(f'Component {comp_idx} Contribution')
                    axs[1].set_xlabel('Time (s)')
                    axs[1].set_ylabel('Amplitude (μV)')
                    
                    fig.tight_layout()
                    fig.savefig(os.path.join(overlay_dir, f"comp_{comp_idx:03d}_overlay.png"), dpi=300)
                    plt.close(fig)
                    
                except Exception as e:
                    logging.error(f"[ICAExtractionStep] Error creating overlay plot for component {comp_idx}: {e}")
                
        except Exception as e:
            logging.error(f"[ICAExtractionStep] Error in _plot_data_overlay: {e}")
    
    def _generate_component_details(self, ica, data, epochs, plot_dir, params):
        """Generate a comprehensive report with details for each component."""
        logging.info("[ICAExtractionStep] Generating component details")
        
        try:
            # Create a directory for component details
            details_dir = os.path.join(plot_dir, "details")
            os.makedirs(details_dir, exist_ok=True)
            
            # For each component, create a comprehensive figure with multiple subplots
            for comp_idx in range(ica.n_components_):
                try:
                    fig = plt.figure(figsize=(15, 12))
                    gs = plt.GridSpec(3, 3, figure=fig)
                    
                    # Title
                    fig.suptitle(f"Component {comp_idx} Detailed Analysis", fontsize=16, fontweight='bold')
                    
                    # 1. Topography (top-left)
                    ax_topo = fig.add_subplot(gs[0, 0])
                    ica.plot_components(picks=comp_idx, axes=ax_topo, show=False, colorbar=False)
                    
                    # 2. PSD (top-center)
                    ax_psd = fig.add_subplot(gs[0, 1])
                    sources = ica.get_sources(data)
                    source_data = sources.get_data()[comp_idx]
                    f, Pxx = signal.welch(source_data, fs=data.info['sfreq'], nperseg=min(4096, len(source_data)))
                    ax_psd.semilogy(f, Pxx)
                    ax_psd.set_title('Power Spectral Density')
                    ax_psd.set_xlabel('Frequency (Hz)')
                    ax_psd.set_ylabel('PSD')
                    ax_psd.set_xlim([0, min(100, data.info['sfreq']/2)])
                    ax_psd.grid(True)
                    
                    # 3. Time course (top-right)
                    ax_time = fig.add_subplot(gs[0, 2])
                    times = np.arange(min(10000, sources.n_times)) / sources.info['sfreq']
                    ax_time.plot(times, source_data[:len(times)])
                    ax_time.set_title('Time Course')
                    ax_time.set_xlabel('Time (s)')
                    ax_time.set_ylabel('Amplitude')
                    ax_time.grid(True)
                    
                    # 4. Time-frequency representation (middle row, span all columns)
                    ax_tf = fig.add_subplot(gs[1, :])
                    try:
                        from mne.time_frequency import tfr_array_morlet
                        
                        # Reshape data for tfr_array_morlet
                        data_arr = source_data.reshape(1, 1, -1)
                        sfreq = data.info['sfreq']
                        
                        # Compute time-frequency representation
                        freqs = np.logspace(np.log10(1), np.log10(min(100, sfreq/2.5)), 20)
                        n_cycles = freqs / 2.
                        power = tfr_array_morlet(data_arr, sfreq=sfreq, freqs=freqs, 
                                              n_cycles=n_cycles, output='power')
                        
                        # Plot the result
                        extent = [0, data_arr.shape[2]/sfreq, freqs[0], freqs[-1]]
                        im = ax_tf.imshow(np.squeeze(power), aspect='auto', origin='lower', 
                                       extent=extent, cmap='viridis')
                        ax_tf.set_title('Time-Frequency Representation')
                        ax_tf.set_xlabel('Time (s)')
                        ax_tf.set_ylabel('Frequency (Hz)')
                        ax_tf.set_yscale('log')
                        plt.colorbar(im, ax=ax_tf, label='Power')
                    except Exception as tf_e:
                        logging.warning(f"[ICAExtractionStep] Error computing time-frequency plot: {tf_e}")
                        ax_tf.text(0.5, 0.5, "Time-frequency plot could not be generated", 
                                ha='center', va='center', fontsize=12)
                    
                    # 5. Component pattern (bottom-left)
                    ax_pattern = fig.add_subplot(gs[2, 0])
                    pattern = ica.get_components()[:, comp_idx]
                    ch_names = data.ch_names
                    
                    # Sort channels by absolute pattern value
                    sorted_idx = np.abs(pattern).argsort()[::-1]
                    sorted_ch_names = [ch_names[i] for i in sorted_idx[:10]]  # Top 10 channels
                    sorted_pattern = pattern[sorted_idx[:10]]
                    
                    colors = ['r' if p < 0 else 'b' for p in sorted_pattern]
                    ax_pattern.barh(range(len(sorted_ch_names)), np.abs(sorted_pattern), color=colors)
                    ax_pattern.set_yticks(range(len(sorted_ch_names)))
                    ax_pattern.set_yticklabels(sorted_ch_names)
                    ax_pattern.set_title('Component Pattern (Top 10 Channels)')
                    ax_pattern.set_xlabel('Contribution (Abs Value)')
                    
                    # 6. Histogram (bottom-center)
                    ax_hist = fig.add_subplot(gs[2, 1])
                    ax_hist.hist(source_data, bins=50, density=True, alpha=0.8, color='steelblue')
                    ax_hist.set_title('Amplitude Distribution')
                    ax_hist.set_xlabel('Amplitude')
                    ax_hist.set_ylabel('Density')
                    ax_hist.grid(True, linestyle='--', alpha=0.7)
                    
                    # Add some summary statistics
                    mean = np.mean(source_data)
                    median = np.median(source_data)
                    std_dev = np.std(source_data)
                    skew = np.mean(((source_data - mean) / std_dev) ** 3) if std_dev > 0 else 0
                    kurtosis = np.mean(((source_data - mean) / std_dev) ** 4) - 3 if std_dev > 0 else 0
                    
                    stats_text = (
                        f"Mean: {mean:.2f}\n"
                        f"Median: {median:.2f}\n"
                        f"Std Dev: {std_dev:.2f}\n"
                        f"Skewness: {skew:.2f}\n"
                        f"Kurtosis: {kurtosis:.2f}"
                    )
                    
                    ax_hist.text(0.02, 0.95, stats_text, transform=ax_hist.transAxes,
                               verticalalignment='top', bbox=dict(boxstyle='round',
                                                                facecolor='white', alpha=0.8))
                    
                    # 7. Autocorrelation (bottom-right)
                    ax_acf = fig.add_subplot(gs[2, 2])
                    try:
                        from statsmodels.graphics.tsaplots import plot_acf
                        plot_acf(source_data[:10000], lags=100, ax=ax_acf)
                        ax_acf.set_title('Autocorrelation')
                    except Exception as acf_e:
                        logging.warning(f"[ICAExtractionStep] Error computing autocorrelation: {acf_e}")
                        ax_acf.text(0.5, 0.5, "Autocorrelation plot could not be generated", 
                                  ha='center', va='center', fontsize=12)
                    
                    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
                    fig.savefig(os.path.join(details_dir, f"component_{comp_idx:03d}_details.png"), dpi=300)
                    plt.close(fig)
                    
                except Exception as e:
                    logging.error(f"[ICAExtractionStep] Error generating details for component {comp_idx}: {e}")
                    
        except Exception as e:
            logging.error(f"[ICAExtractionStep] Error in _generate_component_details: {e}") 
