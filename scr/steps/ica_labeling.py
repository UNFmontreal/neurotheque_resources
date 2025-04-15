# File: scr/steps/ica_labeling.py

import logging
import mne
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from .base import BaseStep
from ..utils.mne_utils import clean_mne_object
import sys
import subprocess

# Helper function to check and install dependencies
def _check_install_dependencies():
    """Check for dependencies needed for advanced ICA visualization and try to install if missing."""
    missing_deps = []
    
    # Check for mne-icalabel
    try:
        import mne_icalabel
    except ImportError:
        missing_deps.append("mne-icalabel")
        
    # Check for statsmodels
    try:
        import statsmodels
    except ImportError:
        missing_deps.append("statsmodels")
    
    # Try to install missing dependencies
    if missing_deps:
        try:
            logging.info(f"[ICALabelingStep] Installing missing dependencies: {', '.join(missing_deps)}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--user"] + missing_deps)
            logging.info("[ICALabelingStep] Successfully installed dependencies")
            
            # Re-import to ensure they're now available
            if "mne-icalabel" in missing_deps:
                try:
                    import mne_icalabel
                    logging.info("[ICALabelingStep] mne-icalabel is now available")
                except ImportError:
                    logging.warning("[ICALabelingStep] Failed to import mne-icalabel even after installation")
            
            if "statsmodels" in missing_deps:
                try:
                    import statsmodels
                    logging.info("[ICALabelingStep] statsmodels is now available")
                except ImportError:
                    logging.warning("[ICALabelingStep] Failed to import statsmodels even after installation")
                    
        except Exception as e:
            logging.warning(f"[ICALabelingStep] Failed to install missing dependencies: {e}")
            logging.warning("[ICALabelingStep] Some advanced ICA visualizations may not be available")
            logging.warning(f"[ICALabelingStep] To manually install, run: pip install {' '.join(missing_deps)}")

def _open_folder(path):
    """Open the folder in the file explorer."""
    try:
        if sys.platform == 'win32':
            os.startfile(path)
        elif sys.platform == 'darwin':  # macOS
            subprocess.call(['open', path])
        else:  # Linux
            subprocess.call(['xdg-open', path])
        print(f"Opening folder: {path}")
    except Exception as e:
        print(f"Failed to open folder: {e}")

class ICALabelingStep(BaseStep):
    """
    ICA labeling step that identifies artifact components using various methods,
    and optionally removes them from the data.
    
    This step focuses on labeling components and removing artifacts, assuming
    that ICA decomposition has already been computed.
    
    Parameters
    ----------
    methods : list of str
        Methods to use for labeling components ('iclabel', 'correlation', 'eog', 'ecg')
    threshold : dict
        Thresholds for each method to identify components as artifacts
    plot_labeled : bool
        Whether to generate plots of labeled components
    plot_before_after : bool
        Whether to generate plots of data before and after artifact removal
    interactive : bool
        Whether to allow interactive plots
    plot_dir : str
        Directory to save plots
    reconstruct : bool
        Whether to reconstruct cleaned data by removing artifact components
    auto_exclude : bool
        Whether to automatically exclude components based on thresholds
    """
    
    def run(self, data):
        """Run ICA labeling on the input data with pre-computed ICA."""
        # Check for dependencies needed for advanced visualizations
        _check_install_dependencies()
        
        if data is None:
            raise ValueError("[ICALabelingStep] No data provided.")

        # --------------------------
        # 1) Merge Default Params
        # --------------------------
        default_params = {
            "methods": ["iclabel"],    # Methods for artifact detection: 'iclabel', 'correlation', 'eog', 'ecg'
            "thresholds": {            # Thresholds for each method
                "iclabel": {
                    "eye": 0.8,
                    "heart": 0.8,
                    "muscle": 0.8,
                    "line_noise": 0.8,
                    "channel_noise": 0.8,
                    "other": 0.8
                },
                "correlation": {
                    "eog": 0.8,
                    "ecg": 0.8
                }
            },
            "eog_ch_names": ["Fp1", "Fp2"],
            "ecg_channel": None,
            "manual_selection": True,
            "plot_labeled": True,
            "plot_before_after": True,
            "interactive": True,
            "plot_dir": None,
            "reconstruct": True,
            "auto_exclude": False,
            "verbose": True
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
        import matplotlib
        original_backend = matplotlib.get_backend()
        
        # For Jupyter notebooks, it's better to use inline backend
        if not params["interactive"] and 'ipykernel' in sys.modules:
            try:
                logging.info("[ICALabelingStep] Jupyter environment detected, using inline backend")
                matplotlib.use('inline')
            except Exception as e:
                logging.warning(f"[ICALabelingStep] Could not switch to inline backend: {e}")
        # Check if we need to switch backends for interactive plotting
        elif params["interactive"]:
            try:
                # Try to switch to TkAgg for interactive plots or other suitable backends
                if original_backend != 'TkAgg' and original_backend != 'Qt5Agg' and original_backend != 'WXAgg':
                    # Try multiple backends in order of preference
                    for backend in ['TkAgg', 'Qt5Agg', 'Qt4Agg', 'WXAgg', 'MacOSX', 'Agg']:
                        try:
                            matplotlib.use(backend, force=True)
                            print(f"[ICALabelingStep] Switched matplotlib backend to {backend} for interactive plotting")
                            break
                        except Exception:
                            continue
                    
                    # If interactive plotting is not working properly, we should still save to disk
                    if matplotlib.get_backend() == 'Agg':
                        print("[ICALabelingStep] WARNING: Using non-interactive Agg backend")
                        print("[ICALabelingStep] Plots will be saved to disk but not displayed")
                        print(f"[ICALabelingStep] Plots will be saved to: {os.path.abspath(plot_dir)}")
                    else:
                        plt.ion()  # Turn on interactive mode
                    
                    print(f"[ICALabelingStep] Using matplotlib backend: {matplotlib.get_backend()}")
            except Exception as e:
                logging.warning(f"[ICALabelingStep] Could not switch to interactive backend: {e}")
                logging.warning("[ICALabelingStep] Interactive plots may not work properly")
        
        # --------------------------
        # 2) Retrieve ICA from data
        # --------------------------
        if 'temp' in data.info and "ica" in data.info["temp"]:
            ica = data.info["temp"]["ica"]
            logging.info("[ICALabelingStep] Retrieved ICA from data.info['temp']")
        else:
            # Try to load ICA from file if not found in data
            if paths is not None:
                ica_file = paths.get_derivative_path(sub_id, ses_id) / f'sub-{sub_id}_ses-{ses_id}_desc-ica_decomposition.fif'
                
                if os.path.isfile(ica_file):
                    try:
                        ica = mne.preprocessing.read_ica(ica_file)
                        logging.info(f"[ICALabelingStep] Loaded ICA from file: {ica_file}")
                    except Exception as e:
                        raise ValueError(f"[ICALabelingStep] Failed to load ICA from file: {e}")
                else:
                    raise ValueError(f"[ICALabelingStep] ICA file not found: {ica_file}")
            else:
                raise ValueError("[ICALabelingStep] No ICA found in data and no paths provided to load from file.")
        
        # --------------------------
        # 3) Create plots directory
        # --------------------------
        if paths is not None:
            plot_dir = paths.get_ica_report_dir(sub_id, ses_id, task_id, run_id)
        else:
            plot_dir = params.get("plot_dir", f"./ica_labeling_plots/sub-{sub_id}_ses-{ses_id}")
        
        os.makedirs(plot_dir, exist_ok=True)
        logging.info(f"[ICALabelingStep] Saving plots to {plot_dir}")
        print(f"[ICALabelingStep] Plots will be saved to: {os.path.abspath(plot_dir)}")
        
        # --------------------------
        # 4) Label components using specified methods
        # --------------------------
        labeled_components = {}  # Will store components labeled by each method
        all_artifacts = set()    # Union of all artifact components
        
        # Method 1: ICLabel
        if "iclabel" in params["methods"]:
            ic_labels = self._label_with_iclabel(ica, data, params)
            if ic_labels:
                labeled_components["iclabel"] = ic_labels
                
                # Add components that exceed threshold to artifacts list
                thresholds = params["thresholds"]["iclabel"]
                for idx, (label, prob) in enumerate(zip(ic_labels["labels"], ic_labels["y_pred_proba"])):
                    # Skip if label is 'brain' (not an artifact)
                    if label == 'brain':
                        continue
                        
                    # Check if probability exceeds threshold
                    label_idx = ic_labels["labels_set"].index(label)
                    label_prob = prob[label_idx]
                    
                    if label in thresholds and label_prob >= thresholds[label]:
                        all_artifacts.add(idx)
                        logging.info(f"[ICALabelingStep] Component {idx} classified as '{label}' (prob={label_prob:.2f})")
        
        # Method 2: Correlation with EOG/ECG
        if "correlation" in params["methods"]:
            corr_artifacts = self._label_with_correlation(ica, data, params)
            if corr_artifacts:
                labeled_components["correlation"] = corr_artifacts
                
                # Add EOG components to artifacts list
                if "eog" in corr_artifacts and corr_artifacts["eog"] and "eog_scores" in corr_artifacts:
                    for idx, score in zip(corr_artifacts["eog"], corr_artifacts["eog_scores"]):
                        # Only include components with correlation score of 0.8 or higher
                        if float(score) >= 0.8:
                            all_artifacts.add(idx)
                            logging.info(f"[ICALabelingStep] Component {idx} identified as EOG with score {float(score):.2f}")
                
                # Add ECG components to artifacts list
                if "ecg" in corr_artifacts and corr_artifacts["ecg"] and "ecg_scores" in corr_artifacts:
                    for idx, score in zip(corr_artifacts["ecg"], corr_artifacts["ecg_scores"]):
                        # Only include components with correlation score of 0.8 or higher
                        if score >= 0.8:
                            all_artifacts.add(idx)
                            logging.info(f"[ICALabelingStep] Component {idx} identified as ECG with score {score:.2f}")
        
        # --------------------------
        # 5) Plot labeled components
        # --------------------------
        if params["plot_labeled"]:
            labeled_figs = self._plot_labeled_components(ica, data, labeled_components, all_artifacts, plot_dir, params)
            if labeled_figs:
                self.figures.extend(labeled_figs)
        
        # --------------------------
        # 6) Manual selection of components to exclude
        # --------------------------
        final_exclude = set()
        
        if params["manual_selection"] and params["interactive"]:
            # Print identified artifact components
            print("\n[ICALabelingStep] Components identified as artifacts:")
            for method, labels in labeled_components.items():
                if method == "iclabel":
                    print(f"  ICLabel:")
                    for comp_idx, (label, prob) in enumerate(zip(labels["labels"], labels["y_pred_proba"])):
                        if label != "brain":  # Skip brain components
                            label_idx = labels["labels_set"].index(label)
                            print(f"    Component {comp_idx}: {label.capitalize()} (prob={prob[label_idx]:.2f})")
                elif method == "correlation":
                    if "eog" in labels and labels["eog"]:
                        print(f"  EOG correlation: {labels['eog']}")
                    if "ecg" in labels and labels["ecg"]:
                        print(f"  ECG correlation: {labels['ecg']}")
            
            # Ask for user input
            print(f"\nSuggested artifacts: {sorted(list(all_artifacts))}")
            user_input = input("\nEnter components to exclude (comma-separated), or press Enter to use suggestions: ")
            
            if user_input.strip():
                try:
                    final_exclude = set(int(x) for x in user_input.split(","))
                    logging.info(f"[ICALabelingStep] User selected components to exclude: {sorted(list(final_exclude))}")
                except ValueError:
                    logging.error("[ICALabelingStep] Invalid input format. Using suggested components instead.")
                    final_exclude = all_artifacts
            else:
                logging.info("[ICALabelingStep] Using automatically identified artifact components")
                final_exclude = all_artifacts
        elif params["auto_exclude"]:
            # Automatically exclude identified artifacts
            final_exclude = all_artifacts
            logging.info(f"[ICALabelingStep] Automatically excluding components: {sorted(list(final_exclude))}")
        else:
            # No components excluded by default if not manual_selection and not auto_exclude
            logging.info("[ICALabelingStep] No components will be excluded (manual_selection=False, auto_exclude=False)")
        
        # --------------------------
        # 7) Reconstruct cleaned data
        # --------------------------
        if params["reconstruct"] and final_exclude:
            # Set the exclude property on the ICA object
            ica.exclude = sorted(list(final_exclude))
            logging.info(f"[ICALabelingStep] Excluding components: {ica.exclude}")
            
            # Apply ICA to get cleaned data
            data_clean = ica.apply(data.copy())
            
            # Plot before and after
            if params["plot_before_after"]:
                ba_figs = self._plot_before_after(data, data_clean, ica, plot_dir, params)
                if ba_figs:
                    self.figures.extend(ba_figs)
            
            # Try to open the plot directory in file explorer
            if params.get("open_plot_dir", True):
                _open_folder(plot_dir)
            
            # Save the cleaned data to disk
            if params.get("save_data", False):
                try:
                    # Create standardized file paths
                    if "output_file" in params:
                        clean_file = params["output_file"]
                    elif paths is not None:
                        clean_file = os.path.join(
                            os.path.normpath(os.path.dirname(str(paths.get_derivative_path(sub_id, ses_id)))),
                            f"sub-{sub_id}_ses-{ses_id}_task-{task_id}_run-{run_id}_desc-ica_labeled_epo.fif"
                        )
                    else:
                        logging.error("[ICALabelingStep] No output_file or paths provided, cannot save data")
                        return data_clean
                    
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(str(clean_file)), exist_ok=True)
                    
                    # Make a clean copy without the complex ICA object
                    data_to_save = clean_mne_object(data_clean)
                    
                    # Remove all complex objects from info dict
                    if 'temp' in data_to_save.info:
                        del data_to_save.info['temp']
                    
                    # CRITICAL: Create a completely new subject_info dictionary with only string values
                    # This is needed to avoid the "data type '>a' not understood" error
                    data_to_save.info['subject_info'] = {'his_id': f"sub-{sub_id}"}
                    
                    # Save the data
                    logging.info(f"[ICALabelingStep] Saving cleaned data to {clean_file}")
                    data_to_save.save(str(clean_file), overwrite=True)
                    logging.info(f"[ICALabelingStep] Successfully saved cleaned data")
                except Exception as e:
                    logging.error(f"[ICALabelingStep] Error saving cleaned data: {e}")
                    logging.error("[ICALabelingStep] Continuing without saving cleaned data")
            
            # Update data.info
            if not hasattr(data_clean.info, "temp"):
                data_clean.info["temp"] = {}
            
            data_clean.info["temp"]["ica_labeled"] = {
                "excluded": ica.exclude,
                "labeled_components": labeled_components
            }
            
            # Display figures in notebook if running in Jupyter
            if 'ipykernel' in sys.modules and self.figures:
                for fig in self.figures:
                    if fig is not None:
                        plt.figure(fig.number)
                        plt.show()
            
            return data_clean
        else:
            # Just store labeling information in data.info
            if not hasattr(data.info, "temp"):
                data.info["temp"] = {}
            
            data.info["temp"]["ica_labeled"] = {
                "labeled_components": labeled_components,
                "suggested_exclude": sorted(list(all_artifacts))
            }
            
            # Save data if requested with custom file path
            save_data = params.get("save_data", False)
            output_file = params.get("output_file", None)
            
            if save_data and output_file:
                try:
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    # Clean the data before saving
                    data_to_save = clean_mne_object(data)
                    # Save to specified path
                    data_to_save.save(output_file, overwrite=True)
                    logging.info(f"[ICALabelingStep] Saved data with ICA labels to {output_file}")
                except Exception as e:
                    logging.error(f"[ICALabelingStep] Error saving data to {output_file}: {e}")
            
            # Display figures in notebook if running in Jupyter
            if 'ipykernel' in sys.modules and self.figures:
                for fig in self.figures:
                    if fig is not None:
                        plt.figure(fig.number)
                        plt.show()
            
            return data
        
    def _label_with_iclabel(self, ica, data, params):
        """Label components using ICLabel."""
        logging.info("[ICALabelingStep] Attempting to label components with ICLabel")
        
        try:
            # Check if mne-icalabel is available
            import importlib.util
            if importlib.util.find_spec("mne_icalabel") is None:
                logging.warning("[ICALabelingStep] mne-icalabel package is not installed. Cannot use ICLabel.")
                return None
            
            import mne_icalabel
            from mne_icalabel.label_components import label_components
            
            logging.info("[ICALabelingStep] Running ICLabel classification")
            ic_labels = label_components(ica=ica, inst=data, method="iclabel")
            
            logging.info(f"[ICALabelingStep] ICLabel classification successful: {ic_labels['labels']}")
            return ic_labels
            
        except Exception as e:
            logging.error(f"[ICALabelingStep] ICLabel classification failed: {e}")
            return None
    
    def _label_with_correlation(self, ica, data, params):
        """Label components by correlation with EOG and ECG channels."""
        logging.info("[ICALabelingStep] Labeling components by correlation with EOG/ECG")
        
        result = {}
        thresholds = params["thresholds"]["correlation"]
        
        # EOG detection
        if params["eog_ch_names"]:
            try:
                eog_indices, eog_scores = ica.find_bads_eog(
                    inst=data,
                    ch_name=params["eog_ch_names"],
                    threshold=thresholds["eog"]
                )
                
                if eog_indices:
                    logging.info(f"[ICALabelingStep] Identified EOG components: {eog_indices}")
                    result["eog"] = eog_indices
                    result["eog_scores"] = eog_scores[0]
                else:
                    logging.info("[ICALabelingStep] No EOG components identified")
            except Exception as e:
                logging.error(f"[ICALabelingStep] Error identifying EOG components: {e}")
        
        # ECG detection
        if params["ecg_channel"] and (params["ecg_channel"] in data.ch_names):
            try:
                ecg_indices, ecg_scores = ica.find_bads_ecg(
                    data,
                    ch_name=params["ecg_channel"],
                    threshold=thresholds["ecg"]
                )
                
                if ecg_indices:
                    logging.info(f"[ICALabelingStep] Identified ECG components: {ecg_indices}")
                    result["ecg"] = ecg_indices
                    result["ecg_scores"] = ecg_scores
                else:
                    logging.info("[ICALabelingStep] No ECG components identified")
            except Exception as e:
                logging.error(f"[ICALabelingStep] Error identifying ECG components: {e}")
        
        return result if result else None
    
    def _plot_labeled_components(self, ica, data, labeled_components, all_artifacts, plot_dir, params):
        """
        Plot ICA components with classification and artifact details.
        This version uses built‐in MNE plotting functions for improved representation.
        """
        logging.info("[ICALabelingStep] Plotting labeled components")
        all_figs = []
        try:
            # Create a directory for labeled component plots
            labeled_dir = os.path.join(plot_dir, "labeled_components")
            os.makedirs(labeled_dir, exist_ok=True)
            print(f"[ICALabelingStep] Saving plots to: {os.path.abspath(labeled_dir)}")
            
            # ----- Overview Plot -----
            # Use a bar plot for a quick overview that color-codes components by type.
            n_components = ica.n_components_
            fig_overview, ax_overview = plt.subplots(figsize=(15, 4))
            ax_overview.set_title("ICA Components Classification Overview", fontsize=14, fontweight='bold')
            ax_overview.set_xlabel("Component Index")
            ax_overview.set_xticks(np.arange(n_components))
            
            # Define a simple color map.
            color_map = {
                'brain': 'green',
                'eye': 'blue',
                'heart': 'red',
                'muscle': 'purple',
                'line_noise': 'orange',
                'channel_noise': 'brown',
                'other': 'gray'
            }
            # Create a summary label per component based on results from ICLabel and correlation.
            comp_colors = []
            for comp in range(n_components):
                # Default label is brain.
                label = "brain"
                if comp in labeled_components.get("iclabel", {}).get("labels", []):
                    # Use the first non-brain label if available.
                    cand = labeled_components["iclabel"]["labels"][comp]
                    if cand != 'brain':
                        label = cand
                # If correlation detected an artifact, mark it.
                if comp in all_artifacts:
                    # Overwrite with black for artifacts.
                    comp_colors.append("black")
                else:
                    comp_colors.append(color_map.get(label, "gray"))
                    
            ax_overview.bar(np.arange(n_components), [1] * n_components, color=comp_colors)
            ax_overview.set_yticks([])  # Hide y-axis
            
            fig_file = os.path.join(labeled_dir, "overview.png")
            fig_overview.tight_layout()
            fig_overview.savefig(fig_file, dpi=300)
            all_figs.append(fig_overview)
            logging.info(f"[ICALabelingStep] Saved overview plot to {fig_file}")

            # ----- Detailed Per-Component Plots -----
            # For each artifact component, use plot_properties which shows topography, time-series, PSD, and sensor-level properties.
            for comp_idx in sorted(all_artifacts):
                try:
                    # Plot component properties; this function creates a multipanel figure.
                    # It is recommended in MNE examples for detailed component inspection.
                    fig_props = ica.plot_properties(data, picks=comp_idx, psd_args={'fmax': 50},
                                                 show=False)
                    # Save the figure.
                    comp_file = os.path.join(labeled_dir, f"component_{comp_idx:02d}_properties.png")
                    fig_props.savefig(comp_file, dpi=300)
                    all_figs.append(fig_props)
                    print(f"[ICALabelingStep] Saved detailed plot for component {comp_idx} to {comp_file}")
                except Exception as e:
                    logging.error(f"[ICALabelingStep] Error plotting properties for component {comp_idx}: {e}")
            
            # Return list of created figures.
            return all_figs

        except Exception as e:
            logging.error(f"[ICALabelingStep] Error in _plot_labeled_components: {e}")
            return None

        
    def _plot_before_after(self, data_orig, data_clean, ica, plot_dir, params):
        """
        Plot data before and after ICA cleaning using robust MNE functions.
        For raw data, plot overlay using plot; for Epochs, show a representative subset.
        Also plot PSD using psd_welch.
        """
        logging.info("[ICALabelingStep] Plotting data before and after ICA cleaning")
        all_figs = []
        try:
            ba_dir = os.path.join(plot_dir, "before_after")
            os.makedirs(ba_dir, exist_ok=True)
            print(f"[ICALabelingStep] Saving before/after plots to: {os.path.abspath(ba_dir)}")
        
            # ----- Raw Data or Evoked -----
            if not isinstance(data_orig, mne.BaseEpochs):
                fig_raw, ax_raw = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
                ax_raw[0].set_title("Before ICA Cleaning")
                data_orig.plot(duration=10, show=False, axes=ax_raw[0])
                ax_raw[1].set_title("After ICA Cleaning")
                data_clean.plot(duration=10, show=False, axes=ax_raw[1])
            else:
                # For Epochs, plot a representative average of a few epochs.
                fig_raw, ax_raw = plt.subplots(1, 2, figsize=(15, 5))
                avg_before = data_orig.average()
                avg_after = data_clean.average()
                avg_before.plot(axes=ax_raw[0], show=False, time_unit='s')
                ax_raw[0].set_title("Average Before ICA Cleaning")
                avg_after.plot(axes=ax_raw[1], show=False, time_unit='s')
                ax_raw[1].set_title("Average After ICA Cleaning")
        
            fig_raw.tight_layout()
            raw_file = os.path.join(ba_dir, "raw_before_after.png")
            fig_raw.savefig(raw_file, dpi=300)
            all_figs.append(fig_raw)
            logging.info(f"[ICALabelingStep] Saved raw before/after plot to {raw_file}")
        
            # ----- PSD Comparison -----
            # Use mne.time_frequency.psd_welch to compute and plot PSDs.
            fig_psd, ax_psd = plt.subplots(figsize=(12, 8))
            psds_orig, freqs_orig = mne.time_frequency.psd_welch(data_orig, n_per_seg=4096, average=True)
            psds_clean, freqs_clean = mne.time_frequency.psd_welch(data_clean, n_per_seg=4096, average=True)
            ax_psd.semilogy(freqs_orig, np.mean(psds_orig, axis=0), color='red', alpha=0.8, label='Before')
            ax_psd.semilogy(freqs_clean, np.mean(psds_clean, axis=0), color='blue', alpha=0.8, label='After')
            ax_psd.set_title("PSD: Before vs. After ICA Cleaning")
            ax_psd.set_xlabel("Frequency (Hz)")
            ax_psd.set_ylabel("Power Spectral Density")
            ax_psd.legend()
        
            fig_psd.tight_layout()
            psd_file = os.path.join(ba_dir, "psd_before_after.png")
            fig_psd.savefig(psd_file, dpi=300)
            all_figs.append(fig_psd)
            logging.info(f"[ICALabelingStep] Saved PSD before/after plot to {psd_file}")
        
            return all_figs
            
        except Exception as e:
            logging.error(f"[ICALabelingStep] Error in _plot_before_after: {e}")
            return None
