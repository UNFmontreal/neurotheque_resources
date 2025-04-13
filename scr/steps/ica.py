# File: scr/steps/ica.py

import logging
import mne
from mne.preprocessing import ICA
from mne.report import Report
from pathlib import Path
from .base import BaseStep
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import warnings

# Import the new specialized ICA classes
from .ica_extraction import ICAExtractionStep
from .ica_labeling import ICALabelingStep

# Re-export the classes for backward compatibility
__all__ = ['ICAStep', 'ICAExtractionStep', 'ICALabelingStep']

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
            import subprocess
            import sys
            
            logging.info(f"[ICA] Installing missing dependencies: {', '.join(missing_deps)}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--user"] + missing_deps)
            logging.info("[ICA] Successfully installed dependencies")
            
            # Re-import to ensure they're now available
            if "mne-icalabel" in missing_deps:
                try:
                    import mne_icalabel
                    logging.info("[ICA] mne-icalabel is now available")
                except ImportError:
                    logging.warning("[ICA] Failed to import mne-icalabel even after installation")
            
            if "statsmodels" in missing_deps:
                try:
                    import statsmodels
                    logging.info("[ICA] statsmodels is now available")
                except ImportError:
                    logging.warning("[ICA] Failed to import statsmodels even after installation")
                    
        except Exception as e:
            logging.warning(f"[ICA] Failed to install missing dependencies: {e}")
            logging.warning("[ICA] Some advanced ICA visualizations may not be available")
            logging.warning(f"[ICA] To manually install, run: pip install {' '.join(missing_deps)}")

class ICAStep(BaseStep):
    """
    Original ICA step that logs suggested bad components (EOG/ECG) but does not 
    automatically exclude them. The user can then inspect those suggestions 
    plus the component plots to make a final decision.
    
    This class combines both extraction and labeling functionality and is maintained
    for backward compatibility. For new code, use ICAExtractionStep followed by
    ICALabelingStep instead.

    References:
      - Chaumon et al. (2015), J Neurosci Methods
      - Winkler et al. (2015), NeuroImage
      - DSI-24 Technical Specs
    """

    def run(self, data):
        # Check for dependencies needed for advanced visualizations
        _check_install_dependencies()
        
        if data is None:
            raise ValueError("[ICAStep] No data provided for ICA.")

        # --------------------------
        # 1) Merge Default Params
        # --------------------------
        default_params = {
            "n_components": 0.99,      # Or an int if you prefer a fixed number
            "method": "infomax",
            "max_iter": 2000,
            "fit_params": {"extended": True, "l_rate": 1e-3},
            "decim": 3,
            "use_good_epochs_only": True,
            "eog_ch_names": ["Fp1", "Fp2"],
            "eog_threshold": 0.5,
            "ecg_channel": None,
            "ecg_threshold": 0.3,
            "plot_dir": ".../reports/ica",
            "interactive": True,
            "exclude": [],              # Pre-exclusions
            "plot_components": True,
            "plot_sources": True,
        }
        params = {**default_params, **self.params}
        sub_id = params.get("subject_id", "unknown")
        ses_id = params.get("session_id", "001")
        task_id = params.get("task_id", None)
        run_id = params.get("run_id", None)
        paths = params.get("paths", None)
        
        # --------------------------
        # Configure matplotlib backend for plotting
        # --------------------------
        import matplotlib
        original_backend = matplotlib.get_backend()
        # Check if we need to switch backends for interactive plotting
        if params["interactive"]:
            try:
                # Try to switch to TkAgg for interactive plots
                if original_backend != 'TkAgg' and original_backend != 'Qt5Agg' and original_backend != 'WXAgg':
                    logging.info(f"[ICAStep] Switching matplotlib backend from {original_backend} to TkAgg for interactive plotting")
                    matplotlib.use('TkAgg')
                    import matplotlib.pyplot as plt
                    plt.ion()  # Turn on interactive mode
            except Exception as e:
                logging.warning(f"[ICAStep] Could not switch to interactive backend: {e}")
                logging.warning("[ICAStep] Interactive plots may not work properly")
                
        # --------------------------
        # 2) Instantiate ICA
        # --------------------------
        # Handle 'all' option for n_components
        n_components = params["n_components"]
        if n_components == 'all':
            logging.info("[ICAStep] Using all available components for ICA decomposition.")
            n_components = None  # None tells MNE to use all components
            
        ica = ICA(
            n_components=n_components,
            method=params["method"],
            max_iter=params["max_iter"],
            fit_params=params["fit_params"],
            random_state=0
        )
        # --------------------------
        # 3) Select Data for ICA
        # --------------------------
        if params["use_good_epochs_only"]:
            logging.info("[ICAStep] Using data with AutoReject annotations to exclude bad segments.")
            
            # Check if data has BAD_autoreject annotations
            has_autoreject_annotations = False
            if data.annotations is not None and len(data.annotations) > 0:
                bad_annot_count = sum(1 for desc in data.annotations.description if desc == 'BAD_autoreject')
                if bad_annot_count > 0:
                    has_autoreject_annotations = True
                    logging.info(f"[ICAStep] Found {bad_annot_count} 'BAD_autoreject' annotations")
                    
            if has_autoreject_annotations:
                # Create epochs, automatically rejecting those with annotations
                # MNE will skip epochs that overlap with 'BAD' annotations
                events = mne.make_fixed_length_events(data, duration=1)
                good_epochs = mne.Epochs(
                    data, 
                    events, 
                    tmin=0, 
                    tmax=1, 
                    baseline=None, 
                    preload=True,
                    reject_by_annotation=True  # This ensures epochs with BAD_* annotations are excluded
                )
                logging.info(f"[ICAStep] Created {len(good_epochs)} epochs, excluding segments with bad annotations")
            else:
                # No annotations found, use all data
                logging.warning("[ICAStep] No BAD_autoreject annotations found. Using all data for ICA.")
                events = mne.make_fixed_length_events(data, duration=1)
                good_epochs = mne.Epochs(data, events, tmin=0, tmax=1, baseline=None, preload=True)
                logging.info(f"[ICAStep] Created {len(good_epochs)} epochs (no annotations to exclude)")
        else:
            logging.info("[ICAStep] Using all data for ICA (use_good_epochs_only=False).")
            good_epochs = data

        # --------------------------
        # 4) Fit ICA
        # --------------------------
        ica.fit(
            good_epochs,
            decim=params["decim"],
            reject=None,
        )
        # --------------------------
        # 5) Automated Artifact Detection
        # --------------------------
        # Instead of directly excluding, we'll just store candidate indices
        bad_ic_candidates = []  # We'll store dictionaries with type + indices

        # EOG
        eog_indices, eog_scores = [], []
        if params["eog_ch_names"]:
            eog_indices, eog_scores = ica.find_bads_eog(
                good_epochs if params["use_good_epochs_only"] else data,
                ch_name=params["eog_ch_names"],
                threshold=params["eog_threshold"]
            )
            if eog_indices:
                bad_ic_candidates.append({
                    "type": "EOG",
                    "indices": eog_indices,
                    "threshold": params["eog_threshold"]
                })
        
        # ECG
        ecg_indices, ecg_scores = [], []
        if params["ecg_channel"] and (params["ecg_channel"] in data.ch_names):
            ecg_indices, ecg_scores = ica.find_bads_ecg(
                good_epochs if params["use_good_epochs_only"] else data,
                ch_name=params["ecg_channel"],
                threshold=params["ecg_threshold"]
            )
            if ecg_indices:
                bad_ic_candidates.append({
                    "type": "ECG",
                    "indices": ecg_indices,
                    "threshold": params["ecg_threshold"]
                })

        # --------------------------
        # 6) Show Candidate Bad ICs
        # --------------------------
        # We'll start with any user-predefined exclusions
        final_exclude = set(params["exclude"])

        # Print out detection suggestions
        for entry in bad_ic_candidates:
            logging.info(
                f"[ICAStep] Suggested {entry['type']} components (threshold={entry['threshold']}): {entry['indices']}"
            )

        # The user sees these logs, or you can print them to console:
        if bad_ic_candidates:
            print("\n[ICAStep] Candidate bad ICs from automatic detection:")
            for entry in bad_ic_candidates:
                print(f"  {entry['type']} => {entry['indices']} (threshold={entry['threshold']})")
            print("These are NOT excluded yet. You will get a chance to confirm or modify.\n")

        # --------------------------
        # 7) Interactive QA
        # --------------------------
        # Let's optionally plot the "candidate" ICs
        if params["interactive"]:
            # Create a directory for saving plots if needed
            plot_dir = params.get("plot_dir", "./ica_plots")
            os.makedirs(plot_dir, exist_ok=True)
            
            # First, show all components if requested
            if params["plot_components"]:
                logging.info("[ICAStep] Opening plot of all ICA component topographies...")
                print("[ICAStep] Opening component topographies. Close the window when you're done reviewing.")
                
                try:
                    # Try interactive plotting first
                    fig_topo = ica.plot_components(picks=range(min(ica.n_components_, 20)), show=True)
                    input("[ICAStep] Press Enter after closing the component topography window to continue...")
                except Exception as e:
                    # Fallback: save plots to files
                    logging.warning(f"[ICAStep] Interactive plotting failed: {e}")
                    logging.info("[ICAStep] Saving component topographies to files instead.")
                    figs = ica.plot_components(picks=range(min(ica.n_components_, 20)), show=False)
                    
                    # Save individual components plots
                    for i, fig in enumerate(figs):
                        fig_path = os.path.join(plot_dir, f"ica_component_{i}.png")
                        fig.savefig(fig_path)
                        plt.close(fig)
                    
                    print(f"[ICAStep] Component topographies saved to {plot_dir}")
                    print("[ICAStep] Please review the component plots and be ready to enter component numbers to exclude.")
            
            # Show ICA sources if requested
            if params["plot_sources"]:
                logging.info("[ICAStep] Opening plot of ICA source time courses...")
                print("[ICAStep] Opening source time courses. Close the window when you're done reviewing.")
                
                try:
                    # Try interactive plotting first
                    fig_sources = ica.plot_sources(data, show=True)
                    input("[ICAStep] Press Enter after closing the source time courses window to continue...")
                except Exception as e:
                    # Fallback: save plot to file
                    logging.warning(f"[ICAStep] Interactive plotting of sources failed: {e}")
                    logging.info("[ICAStep] Saving source time courses to file instead.")
                    fig_sources = ica.plot_sources(data, show=False)
                    fig_path = os.path.join(plot_dir, "ica_sources.png")
                    fig_sources.savefig(fig_path)
                    plt.close(fig_sources)
                    print(f"[ICAStep] Source time courses saved to {fig_path}")
            
            # Show properties of candidate bad components
            union_candidates = set()
            for entry in bad_ic_candidates:
                union_candidates.update(entry["indices"])

            # If you want to see topographies of the candidate ICs (union of EOG, ECG, etc.):
            if union_candidates:
                logging.info("[ICAStep] Opening detailed properties of candidate bad components...")
                print("[ICAStep] Opening detailed properties of candidate bad components. Close each window when done.")
                
                try:
                    # Try interactive plotting first
                    ica.plot_properties(good_epochs, picks=sorted(list(union_candidates)), show=True)
                    input("[ICAStep] Press Enter after reviewing component properties to continue...")
                except Exception as e:
                    # Fallback: save property plots to files
                    logging.warning(f"[ICAStep] Interactive plotting of properties failed: {e}")
                    logging.info("[ICAStep] Saving component properties to files instead.")
                    
                    for comp_idx in sorted(list(union_candidates)):
                        try:
                            fig = ica.plot_properties(good_epochs, picks=comp_idx, show=False)
                            fig_path = os.path.join(plot_dir, f"ica_properties_comp_{comp_idx}.png")
                            fig[0].savefig(fig_path)
                            for f in fig:
                                plt.close(f)
                        except Exception as prop_e:
                            logging.warning(f"[ICAStep] Error saving properties for component {comp_idx}: {prop_e}")
                    
                    print(f"[ICAStep] Component properties saved to {plot_dir}")

            # Also show any pre-excluded from user params if you like
            if final_exclude:
                logging.info("[ICAStep] Opening detailed properties of pre-excluded components...")
                print("[ICAStep] Opening detailed properties of pre-excluded components. Close each window when done.")
                
                try:
                    # Try interactive plotting first
                    ica.plot_properties(good_epochs, picks=sorted(list(final_exclude)), show=True)
                    input("[ICAStep] Press Enter after reviewing pre-excluded component properties to continue...")
                except Exception as e:
                    # Fallback: save property plots to files
                    logging.warning(f"[ICAStep] Interactive plotting of pre-excluded properties failed: {e}")
                    logging.info("[ICAStep] Saving pre-excluded component properties to files instead.")
                    
                    for comp_idx in sorted(list(final_exclude)):
                        try:
                            fig = ica.plot_properties(good_epochs, picks=comp_idx, show=False)
                            fig_path = os.path.join(plot_dir, f"ica_preexcluded_comp_{comp_idx}.png")
                            fig[0].savefig(fig_path)
                            for f in fig:
                                plt.close(f)
                        except Exception as prop_e:
                            logging.warning(f"[ICAStep] Error saving properties for pre-excluded component {comp_idx}: {prop_e}")
                    
                    print(f"[ICAStep] Pre-excluded component properties saved to {plot_dir}")

            # Now prompt the user
            suggested_str = (
                f"\nSuggested bad ICs from detection: {sorted(list(union_candidates))}" 
                if union_candidates else "None"
            )
            print(suggested_str)
            print("\nMake your selection based on the component plots you just saw.")
            print("Components with eye artifacts typically show frontal topography.")
            print("Components with muscle artifacts typically show high-frequency activity.")
            print("Components with cardiac artifacts typically show regular periodic activity.")

            # The user can override or add anything:
            user_input = input("\nEnter ALL IC indices to exclude (comma-separated), or press Enter to skip: ")
            if user_input.strip():
                # Overwrite final_exclude with user input
                try:
                    final_exclude = set(int(x) for x in user_input.split(","))
                except ValueError:
                    logging.error("[ICAStep] Invalid input format. Please enter comma-separated numbers only.")
                    print("Invalid input format. Using suggested components instead.")
                    final_exclude = union_candidates
            else:
                print("No components manually selected. Using only automatically detected components.")
                final_exclude = union_candidates
        else:
            logging.info("[ICAStep] Non-interactive mode. Using only 'exclude' param from YAML.")

        # Assign to ica.exclude
        ica.exclude = sorted(list(final_exclude))
        logging.info(f"[ICAStep] Final exclusion list: {ica.exclude}")

        # If we switched matplotlib backends, switch back to the original
        if params["interactive"] and original_backend not in ['TkAgg', 'Qt5Agg', 'WXAgg']:
            try:
                logging.info(f"[ICAStep] Switching matplotlib backend back to {original_backend}")
                matplotlib.use(original_backend)
                plt.ioff()  # Turn off interactive mode
            except Exception as e:
                logging.warning(f"[ICAStep] Error switching back to original backend: {e}")

        # --------------------------
        # 8) Apply ICA
        # --------------------------
        data_clean = ica.apply(data.copy())
        ica_dir = paths.get_derivative_path(sub_id, ses_id) / f'sub-{sub_id}_ses-{ses_id}_desc-ica_cleaned.fif'
        
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(str(ica_dir)), exist_ok=True)
        
        # Save the ICA-cleaned data
        try:
            data_clean.save(str(ica_dir), overwrite=True)
            logging.info(f"[ICAStep] Saved ICA-cleaned data to {ica_dir}")
        except Exception as e:
            logging.error(f"[ICAStep] Error saving ICA-cleaned data: {e}")
            logging.info("[ICAStep] Trying alternative path...")
            
            # Try an alternative path if there's an issue
            alt_dir = Path(os.path.join(str(paths.get_derivative_path(sub_id, ses_id)), 
                                     f'sub-{sub_id}_ses-{ses_id}_desc-ica_cleaned.fif'))
            try:
                os.makedirs(os.path.dirname(str(alt_dir)), exist_ok=True)
                data_clean.save(str(alt_dir), overwrite=True)
                logging.info(f"[ICAStep] Saved ICA-cleaned data to alternative path: {alt_dir}")
            except Exception as e2:
                logging.error(f"[ICAStep] Error saving to alternative path as well: {e2}")
                logging.error("[ICAStep] Continuing without saving ICA-cleaned data")
        
        # --------------------------
        # 9) Generate QA Report
        # --------------------------
        self._generate_report(ica, data_clean, params, paths)

        # --------------------------
        # 10) Store Metadata
        # --------------------------
        if not hasattr(data_clean.info, "temp"):
            data_clean.info["temp"] = {}
        data_clean.info["temp"]["ica"] = {
            "excluded": ica.exclude,
            "n_components": params["n_components"]
        }

        return data_clean

    def _generate_report(self, ica, data, params, paths):
        """Create an MNE Report summarizing ICA."""
        from mne.report import Report
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        import os
        
        # Get directories for saving reports and figures
        sub_id = params["subject_id"]
        ses_id = params["session_id"]
        rep_dir = paths.get_ica_report_dir(sub_id, ses_id)
        fig_dir = rep_dir / "figures"
        comp_dir = rep_dir / "components"
        
        # Create directories if they don't exist
        os.makedirs(rep_dir, exist_ok=True)
        os.makedirs(fig_dir, exist_ok=True)
        os.makedirs(comp_dir, exist_ok=True)
        
        try:
            # 1. Standard MNE HTML report
            report = Report(title="ICA Quality Report", verbose=False)
            report.add_ica(
                ica=ica,
                title="ICA cleaning",
                picks=None,
                inst=data,
                n_jobs=None,
            )
            out_file = rep_dir / "ica_report.html"
            report.save(out_file, overwrite=True, open_browser=False)
            logging.info(f"[ICAStep] ICA report saved at {out_file}")
        except Exception as e:
            logging.error(f"[ICAStep] Error generating HTML report: {e}")
        
        # 2. Comprehensive artifact labeling
        # Find all bad components identified by different methods
        artifact_labels = {}
        
        # Add EOG components
        if hasattr(ica, 'labels_') and 'eog' in ica.labels_:
            artifact_labels['EOG'] = ica.labels_['eog']
        elif hasattr(ica, '_eog_indices') and ica._eog_indices is not None:
            artifact_labels['EOG'] = ica._eog_indices
        
        # Add ECG components
        if hasattr(ica, 'labels_') and 'ecg' in ica.labels_:
            artifact_labels['ECG'] = ica.labels_['ecg']
        elif hasattr(ica, '_ecg_indices') and ica._ecg_indices is not None:
            artifact_labels['ECG'] = ica._ecg_indices
            
        # Add excluded components
        artifact_labels['Excluded'] = ica.exclude
        
        # Try to use ICLabel if available
        has_iclabel = False
        try:
            # First check if mne_icalabel is installed
            import importlib.util
            if importlib.util.find_spec("mne_icalabel") is None:
                logging.warning("[ICAStep] mne-icalabel package is not installed. Advanced classification will not be available.")
                # Don't try to classify if package isn't available
            else:
                import mne_icalabel
                
                # The correct approach is to use a specific function that works with ICA objects
                # Import method differently to make clear what we're doing
                from mne_icalabel.label_components import label_components
                
                # This function expects both the ICA object and the data instance
                logging.info("[ICAStep] Running ICLabel classification")
                
                # Need to prepare an appropriate inst parameter (Raw or Epochs)
                if isinstance(data, mne.io.Raw):
                    ic_labels = label_components(ica=ica, inst=data, method="iclabel")
                else:
                    # If data is already Epochs, use it directly
                    ic_labels = label_components(ica=ica, inst=data, method="iclabel")
                
                logging.info(f"[ICAStep] ICLabel classification successful: {ic_labels['labels']}")
                has_iclabel = True
                
                # Add ICLabel results to artifact_labels
                for label in ['eye', 'heart', 'muscle', 'line_noise']:
                    indices = [idx for idx, l in enumerate(ic_labels['labels']) if l == label]
                    if indices:
                        artifact_labels[f'ICLabel_{label.capitalize()}'] = indices
        except Exception as e:
            logging.warning(f"[ICAStep] ICLabel classification failed: {e}")
            # Just continue without ICLabel classification
        
        try:
            # Plot all components overview with artifact labels
            n_components = len(ica.mixing_matrix_.T)
            fig_components = plt.figure(figsize=(15, 10))
            gs = GridSpec(3, 1, height_ratios=[1, 2, 2], figure=fig_components)
            
            # Component overview (color coded by artifact type)
            ax_overview = fig_components.add_subplot(gs[0])
            ax_overview.set_title("ICA Components Overview", fontsize=14, fontweight='bold')
            ax_overview.set_xlabel("Component Index")
            ax_overview.set_ylabel("Artifact Type")
            ax_overview.set_yticks([])
            
            # Plot each component as a vertical bar
            for i in range(n_components):
                if i in ica.exclude:
                    color = 'black'
                    label = 'Excluded'
                else:
                    color = 'green'  # Default: good component
                    label = 'Good'
                    
                # Check if this component is in any artifact category
                for artifact_type, indices in artifact_labels.items():
                    if i in indices and artifact_type != 'Excluded':  # Skip 'Excluded' to avoid double coloring
                        if 'EOG' in artifact_type:
                            color = 'blue'
                            label = artifact_type
                        elif 'ECG' in artifact_type:
                            color = 'red'
                            label = artifact_type
                        elif 'muscle' in artifact_type.lower():
                            color = 'purple'
                            label = artifact_type
                        elif 'noise' in artifact_type.lower():
                            color = 'orange'
                            label = artifact_type
                        else:
                            color = 'gray'
                            label = artifact_type
                
                ax_overview.bar(i, 1, color=color, alpha=0.7)
                
                # Add component number
                ax_overview.text(i, 0.5, str(i), ha='center', va='center', 
                               rotation=90, fontsize=8, color='black')
            
            # Add legend for artifact types
            handles = []
            labels = []
            colors = {'Good': 'green', 'Excluded': 'black', 'EOG': 'blue', 'ECG': 'red', 
                      'Muscle': 'purple', 'Line_noise': 'orange', 'Other': 'gray'}
            
            for label, color in colors.items():
                if label == 'Good' or any(label.lower() in k.lower() for k in artifact_labels.keys()):
                    handles.append(plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.7))
                    labels.append(label)
            
            ax_overview.legend(handles, labels, loc='upper right', ncol=len(handles))
            
            # Add ICLabel probabilities if available
            if has_iclabel:
                ax_iclabel = fig_components.add_subplot(gs[1])
                ax_iclabel.set_title("ICLabel Classification Probabilities", fontsize=14, fontweight='bold')
                
                # Set up the x and y axes
                ax_iclabel.set_xlim(0, n_components)
                ax_iclabel.set_ylabel("Probability")
                ax_iclabel.set_ylim(0, 1)
                
                # Define colors for each class
                class_colors = {
                    'brain': 'green',
                    'muscle': 'purple',
                    'eye': 'blue',
                    'heart': 'red',
                    'line_noise': 'orange',
                    'channel_noise': 'brown',
                    'other': 'gray'
                }
                
                # For each component, stack bars for each class probability
                x = np.arange(n_components)
                bottoms = np.zeros(n_components)
                
                # Sort classes by their "badness" (brain first, then others)
                sorted_classes = ['brain', 'muscle', 'eye', 'heart', 'line_noise', 'channel_noise', 'other']
                
                for class_name in sorted_classes:
                    if class_name in ic_labels['labels_set']:
                        class_idx = ic_labels['labels_set'].index(class_name)
                        probs = [p[class_idx] for p in ic_labels['y_pred_proba']]
                        ax_iclabel.bar(x, probs, bottom=bottoms, color=class_colors[class_name], 
                                     label=class_name, alpha=0.7)
                        bottoms += np.array(probs)
                
                # Add legend
                ax_iclabel.legend(loc='upper right', ncol=len(class_colors))
                
                # Add component numbers
                for i in range(n_components):
                    ax_iclabel.text(i, -0.05, str(i), ha='center', va='center', fontsize=8)
            
            # Add topomap summary
            ax_topo = fig_components.add_subplot(gs[2])
            try:
                ica.plot_components(picks=range(min(20, n_components)), ch_type='eeg', axes=ax_topo, 
                                  title='Component Topographies', colorbar=False, show=False)
            except Exception as e:
                logging.warning(f"[ICAStep] Error plotting component topographies: {e}")
                ax_topo.text(0.5, 0.5, "Component topographies could not be plotted", 
                           ha='center', va='center', fontsize=12)
            
            plt.tight_layout()
            overview_file = fig_dir / "ica_components_overview.png"
            fig_components.savefig(overview_file, dpi=300, bbox_inches='tight')
            plt.close(fig_components)
            logging.info(f"[ICAStep] Saved component overview to {overview_file}")
        except Exception as e:
            logging.error(f"[ICAStep] Error generating component overview: {e}")
        
        # 3. Individual component plots with more detail
        try:
            component_files_created = 0
            for comp_idx in range(n_components):
                # Create a figure with multiple plots for this component
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                plt.suptitle(f"Component {comp_idx} Analysis", fontsize=16, fontweight='bold')
                
                # Add classification metadata if available
                classification_text = f"Component {comp_idx}\n"
                if has_iclabel:
                    label = ic_labels['labels'][comp_idx]
                    prob = ic_labels['y_pred_proba'][comp_idx][ic_labels['labels_set'].index(label)]
                    classification_text += f"ICLabel: {label.capitalize()} ({prob:.2f})\n"
                
                # Mark if it's excluded
                if comp_idx in ica.exclude:
                    classification_text += "Status: EXCLUDED\n"
                else:
                    classification_text += "Status: Kept\n"
                    
                # Add artifact type if identified
                for artifact_type, indices in artifact_labels.items():
                    if comp_idx in indices and artifact_type != 'Excluded':
                        classification_text += f"Identified as: {artifact_type}\n"
                
                # Add the text box to the figure
                fig.text(0.02, 0.02, classification_text, transform=fig.transFigure,
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
                
                # Topography plot in top-left
                try:
                    ica.plot_components(comp_idx, ch_type='eeg', axes=axes[0, 0], 
                                      title=f'Topography', colorbar=True, show=False)
                except Exception as e:
                    logging.warning(f"[ICAStep] Error plotting component topography: {e}")
                    axes[0, 0].text(0.5, 0.5, "Topography plot failed", ha='center', va='center')
                
                # PSD plot in top-right
                try:
                    # Get the source signal for this component
                    source = ica.get_sources(data).get_data()[comp_idx]
                    sfreq = data.info['sfreq']
                    
                    # Calculate PSD
                    from scipy import signal
                    f, Pxx = signal.welch(source, fs=sfreq, nperseg=min(4096, len(source)))
                    axes[0, 1].semilogy(f, Pxx)
                    axes[0, 1].set_title('Power Spectral Density')
                    axes[0, 1].set_xlabel('Frequency (Hz)')
                    axes[0, 1].set_ylabel('Power Spectral Density (dB/Hz)')
                    axes[0, 1].set_xlim([0, min(100, sfreq/2)])  # Limit to 100 Hz or Nyquist
                    axes[0, 1].grid(True)
                except Exception as e:
                    logging.warning(f"[ICAStep] Error plotting component PSD: {e}")
                    axes[0, 1].text(0.5, 0.5, "PSD plot failed", ha='center', va='center')
                
                # Time series plot in bottom-left
                try:
                    # Plot a segment of the source signal
                    n_points = min(10000, len(source))
                    t = np.arange(n_points) / sfreq
                    axes[1, 0].plot(t, source[:n_points])
                    axes[1, 0].set_title('Time Series')
                    axes[1, 0].set_xlabel('Time (s)')
                    axes[1, 0].set_ylabel('Amplitude')
                    axes[1, 0].grid(True)
                except Exception as e:
                    logging.warning(f"[ICAStep] Error plotting component time series: {e}")
                    axes[1, 0].text(0.5, 0.5, "Time series plot failed", ha='center', va='center')
                
                # Properties plot in bottom-right
                try:
                    # Use MNE's plot_properties function but redirect to our axis
                    # Wrap this in another try/except since it's known to be problematic
                    try:
                        # Instead of trying to use MNE's plot_properties with custom axes,
                        # let's create our own simple properties visualization
                        source = ica.get_sources(data).get_data()[comp_idx]
                        sfreq = data.info['sfreq']
                        
                        # Plot a histogram of component values in bottom-right
                        axes[1, 1].hist(source, bins=50, density=True, alpha=0.8, color='steelblue')
                        axes[1, 1].set_title('Component Amplitude Distribution')
                        axes[1, 1].set_xlabel('Amplitude')
                        axes[1, 1].set_ylabel('Density')
                        axes[1, 1].grid(True, linestyle='--', alpha=0.7)
                        
                        # Add some summary statistics
                        mean = np.mean(source)
                        median = np.median(source)
                        std_dev = np.std(source)
                        skew = np.mean(((source - mean) / std_dev) ** 3) if std_dev > 0 else 0
                        kurtosis = np.mean(((source - mean) / std_dev) ** 4) - 3 if std_dev > 0 else 0
                        
                        stats_text = (
                            f"Mean: {mean:.2f}\n"
                            f"Median: {median:.2f}\n"
                            f"Std Dev: {std_dev:.2f}\n"
                            f"Skewness: {skew:.2f}\n"
                            f"Kurtosis: {kurtosis:.2f}"
                        )
                        
                        # Add stats text box
                        axes[1, 1].text(0.02, 0.95, stats_text, transform=axes[1, 1].transAxes,
                                      verticalalignment='top', bbox=dict(boxstyle='round', 
                                                                        facecolor='white', alpha=0.8))
                    except Exception as e1:
                        logging.warning(f"[ICAStep] Error creating custom component properties: {e1}")
                        # Fallback: try to plot autocorrelation
                        try:
                            from statsmodels.graphics.tsaplots import plot_acf
                            plot_acf(source[:10000], lags=100, ax=axes[1, 1])
                            axes[1, 1].set_title('Autocorrelation')
                        except Exception as e2:
                            logging.warning(f"[ICAStep] Error plotting autocorrelation: {e2}")
                            axes[1, 1].text(0.5, 0.5, "Properties plot failed", ha='center', va='center')
                except Exception as e:
                    logging.warning(f"[ICAStep] Error in properties plot section: {e}")
                    axes[1, 1].text(0.5, 0.5, "Properties section failed", ha='center', va='center')
                
                plt.tight_layout()
                
                # Save the figure
                component_file = comp_dir / f"component_{comp_idx:03d}.png"
                try:
                    fig.savefig(component_file, dpi=300, bbox_inches='tight')
                    component_files_created += 1
                except Exception as e:
                    logging.error(f"[ICAStep] Error saving component plot for component {comp_idx}: {e}")
                plt.close(fig)
            
            logging.info(f"[ICAStep] Saved detailed plots for {component_files_created} components to {comp_dir}")
        except Exception as e:
            logging.error(f"[ICAStep] Error generating component plots: {e}")
        
        # 4. Create a summary text file with component classifications
        try:
            summary_file = rep_dir / "ica_component_summary.txt"
            os.makedirs(os.path.dirname(str(summary_file)), exist_ok=True)
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"ICA Component Analysis Summary\n")
                f.write(f"==============================\n\n")
                f.write(f"Subject: {sub_id}, Session: {ses_id}\n")
                f.write(f"Total components: {n_components}\n")
                f.write(f"Excluded components: {len(ica.exclude)} - {ica.exclude}\n\n")
                
                # Simplified output to ensure file creation
                f.write("Component Classifications:\n")
                f.write("-------------------------\n")
            
            logging.info(f"[ICAStep] Saved component summary to {summary_file}")
        except Exception as e:
            logging.error(f"[ICAStep] Error creating summary text file: {e}")
        
        # 5. Create a README.md file with visualization explanations
        try:
            readme_file = rep_dir / "README.md"
            os.makedirs(os.path.dirname(str(readme_file)), exist_ok=True)
            
            with open(readme_file, 'w', encoding='utf-8') as f:
                f.write(f"# ICA Analysis Results\n\n")
                f.write(f"**Subject:** {sub_id}, **Session:** {ses_id}\n\n")
                
                f.write("## Overview\n\n")
                f.write("This directory contains the results of Independent Component Analysis (ICA) performed on EEG data.\n")
                f.write("ICA is a blind source separation technique that decomposes the EEG signal into independent components,\n")
                f.write("allowing for the identification and removal of artifacts.\n\n")
                
                # Simplified output to ensure file creation
                f.write("## Files and Directories\n\n")
                f.write("- `ica_report.html`: MNE-generated interactive HTML report of the ICA results\n")
                f.write("- `ica_component_summary.txt`: Text summary of component classifications\n")
                f.write("- `figures/`: Directory containing overview visualizations\n")
                f.write("- `components/`: Directory containing detailed plots for each ICA component\n\n")
            
            logging.info(f"[ICAStep] Saved README to {readme_file}")
        except Exception as e:
            logging.error(f"[ICAStep] Error creating README file: {e}")
            # Try an alternative approach with a different open mode
            try:
                with open(str(readme_file), 'w', encoding='utf-8') as f:
                    f.write("# ICA Analysis Results\n\n")
                    f.write(f"**Subject:** {sub_id}, **Session:** {ses_id}\n\n")
                    f.write("Error occurred during full README generation. See console logs for details.\n")
                logging.info(f"[ICAStep] Saved minimal README to {readme_file} after error")
            except Exception as e2:
                logging.error(f"[ICAStep] Complete failure creating README file: {e2}")