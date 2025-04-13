# File: scr/steps/ica_labeling.py

import logging
import mne
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from .base import BaseStep

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
                    "eog": 0.5,
                    "ecg": 0.3
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
                    logging.info(f"[ICALabelingStep] Switching matplotlib backend from {original_backend} to TkAgg for interactive plotting")
                    matplotlib.use('TkAgg')
                    plt.ion()  # Turn on interactive mode
            except Exception as e:
                logging.warning(f"[ICALabelingStep] Could not switch to interactive backend: {e}")
                logging.warning("[ICALabelingStep] Interactive plots may not work properly")
                
        # --------------------------
        # 2) Retrieve ICA from data
        # --------------------------
        if hasattr(data.info, "temp") and "ica" in data.info["temp"]:
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
            plot_dir = paths.get_ica_label_plots_dir(sub_id, ses_id)
        else:
            plot_dir = params.get("plot_dir", f"./ica_labeling_plots/sub-{sub_id}_ses-{ses_id}")
        
        os.makedirs(plot_dir, exist_ok=True)
        logging.info(f"[ICALabelingStep] Saving plots to {plot_dir}")
        
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
                if "eog" in corr_artifacts and corr_artifacts["eog"]:
                    for idx in corr_artifacts["eog"]:
                        all_artifacts.add(idx)
                        logging.info(f"[ICALabelingStep] Component {idx} identified as EOG")
                
                # Add ECG components to artifacts list
                if "ecg" in corr_artifacts and corr_artifacts["ecg"]:
                    for idx in corr_artifacts["ecg"]:
                        all_artifacts.add(idx)
                        logging.info(f"[ICALabelingStep] Component {idx} identified as ECG")
        
        # --------------------------
        # 5) Plot labeled components
        # --------------------------
        if params["plot_labeled"]:
            self._plot_labeled_components(ica, data, labeled_components, all_artifacts, plot_dir, params)
        
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
                self._plot_before_after(data, data_clean, ica, plot_dir, params)
            
            # Save cleaned data
            if paths is not None:
                clean_file = paths.get_derivative_path(sub_id, ses_id) / f'sub-{sub_id}_ses-{ses_id}_desc-ica_cleaned.fif'
                os.makedirs(os.path.dirname(str(clean_file)), exist_ok=True)
                
                try:
                    data_clean.save(str(clean_file), overwrite=True)
                    logging.info(f"[ICALabelingStep] Saved cleaned data to {clean_file}")
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
            
            return data_clean
        else:
            # Just store labeling information in data.info
            if not hasattr(data.info, "temp"):
                data.info["temp"] = {}
            
            data.info["temp"]["ica_labeled"] = {
                "labeled_components": labeled_components,
                "suggested_exclude": sorted(list(all_artifacts))
            }
            
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
                    data,
                    ch_name=params["eog_ch_names"],
                    threshold=thresholds["eog"]
                )
                
                if eog_indices:
                    logging.info(f"[ICALabelingStep] Identified EOG components: {eog_indices}")
                    result["eog"] = eog_indices
                    result["eog_scores"] = eog_scores
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
        """Plot components with their labels."""
        logging.info("[ICALabelingStep] Plotting labeled components")
        
        try:
            # Create a directory for labeled component plots
            labeled_dir = os.path.join(plot_dir, "labeled_components")
            os.makedirs(labeled_dir, exist_ok=True)
            
            # 1. Plot an overview of all components with their labels
            fig = plt.figure(figsize=(15, 10))
            n_components = ica.n_components_
            
            # Define colors for different artifact types
            color_map = {
                'brain': 'green',
                'eye': 'blue',
                'heart': 'red',
                'muscle': 'purple',
                'line_noise': 'orange',
                'channel_noise': 'brown',
                'other': 'gray',
                'eog': 'blue',
                'ecg': 'red',
                'artifact': 'black'
            }
            
            # Create a colorbar for component types
            gs = plt.GridSpec(3, 1, height_ratios=[1, 2, 2], figure=fig)
            
            # Component overview (color coded by artifact type)
            ax_overview = fig.add_subplot(gs[0])
            ax_overview.set_title("ICA Components Classification Overview", fontsize=14, fontweight='bold')
            ax_overview.set_xlabel("Component Index")
            ax_overview.set_ylabel("Artifact Type")
            ax_overview.set_yticks([])
            
            # Get component labels from different methods
            component_labels = {}
            
            # Add ICLabel results
            if "iclabel" in labeled_components:
                ic_labels = labeled_components["iclabel"]
                for i, label in enumerate(ic_labels["labels"]):
                    if i not in component_labels:
                        component_labels[i] = []
                    component_labels[i].append(label)
            
            # Add correlation results
            if "correlation" in labeled_components:
                corr_labels = labeled_components["correlation"]
                if "eog" in corr_labels:
                    for i in corr_labels["eog"]:
                        if i not in component_labels:
                            component_labels[i] = []
                        component_labels[i].append("eog")
                
                if "ecg" in corr_labels:
                    for i in corr_labels["ecg"]:
                        if i not in component_labels:
                            component_labels[i] = []
                        component_labels[i].append("ecg")
            
            # Plot bars for each component
            for i in range(n_components):
                # Default color (no label)
                color = 'green'  # Default: good component
                label = 'brain'
                
                # Check if this component has labels
                if i in component_labels:
                    # Use the first non-brain label
                    for comp_label in component_labels[i]:
                        if comp_label != 'brain':
                            label = comp_label
                            color = color_map.get(label, 'gray')
                            break
                
                # Mark artifacts
                if i in all_artifacts:
                    color = 'black'  # Black border for artifacts
                    edge_color = 'black'
                    linewidth = 2
                else:
                    edge_color = 'none'
                    linewidth = 0
                
                ax_overview.bar(i, 1, color=color, alpha=0.7, edgecolor=edge_color, linewidth=linewidth)
                
                # Add component number
                ax_overview.text(i, 0.5, str(i), ha='center', va='center', 
                              rotation=90, fontsize=8, color='black')
            
            # Add legend for artifact types
            handles = []
            labels = []
            
            # Only add labels that are actually used
            used_labels = set()
            for comp_labels in component_labels.values():
                used_labels.update(comp_labels)
            
            for label in ['brain', 'eye', 'heart', 'muscle', 'line_noise', 'channel_noise', 'other', 'eog', 'ecg']:
                if label in used_labels:
                    handles.append(plt.Rectangle((0, 0), 1, 1, color=color_map[label], alpha=0.7))
                    labels.append(label)
            
            # Add artifact indicator
            handles.append(plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', linewidth=2))
            labels.append('marked as artifact')
            
            ax_overview.legend(handles, labels, loc='upper right', ncol=len(handles)//2 + 1)
            
            # Add ICLabel probability bars if available
            if "iclabel" in labeled_components:
                ic_labels = labeled_components["iclabel"]
                ax_iclabel = fig.add_subplot(gs[1])
                ax_iclabel.set_title("ICLabel Classification Probabilities", fontsize=14, fontweight='bold')
                
                # Set up the x and y axes
                ax_iclabel.set_xlim(0, n_components)
                ax_iclabel.set_ylabel("Probability")
                ax_iclabel.set_ylim(0, 1)
                
                # For each component, stack bars for each class probability
                x = np.arange(n_components)
                bottoms = np.zeros(n_components)
                
                # Sort classes by their "badness" (brain first, then others)
                sorted_classes = ['brain', 'muscle', 'eye', 'heart', 'line_noise', 'channel_noise', 'other']
                
                for class_name in sorted_classes:
                    if class_name in ic_labels['labels_set']:
                        class_idx = ic_labels['labels_set'].index(class_name)
                        probs = [p[class_idx] for p in ic_labels['y_pred_proba']]
                        ax_iclabel.bar(x, probs, bottom=bottoms, color=color_map[class_name], 
                                     label=class_name, alpha=0.7)
                        bottoms += np.array(probs)
                
                # Add legend
                ax_iclabel.legend(loc='upper right', ncol=len(sorted_classes))
                
                # Add component numbers
                for i in range(n_components):
                    ax_iclabel.text(i, -0.05, str(i), ha='center', va='center', fontsize=8)
            
            # Add correlation scores if available
            if "correlation" in labeled_components and ("eog_scores" in labeled_components["correlation"] or 
                                                      "ecg_scores" in labeled_components["correlation"]):
                corr_labels = labeled_components["correlation"]
                ax_corr = fig.add_subplot(gs[2])
                ax_corr.set_title("Correlation with EOG/ECG", fontsize=14, fontweight='bold')
                
                # Set up the x and y axes
                ax_corr.set_xlim(0, n_components)
                ax_corr.set_xlabel("Component Index")
                ax_corr.set_ylabel("Correlation Score")
                ax_corr.set_ylim(0, 1)
                
                # Add EOG scores
                if "eog_scores" in corr_labels:
                    eog_scores = np.zeros(n_components)
                    for i, score in zip(corr_labels.get("eog", []), corr_labels["eog_scores"]):
                        eog_scores[i] = score
                    
                    ax_corr.bar(x, eog_scores, color=color_map["eog"], alpha=0.7, label="EOG Correlation")
                
                # Add ECG scores
                if "ecg_scores" in corr_labels:
                    ecg_scores = np.zeros(n_components)
                    for i, score in zip(corr_labels.get("ecg", []), corr_labels["ecg_scores"]):
                        ecg_scores[i] = score
                    
                    ax_corr.bar(x + 0.3, ecg_scores, color=color_map["ecg"], alpha=0.7, label="ECG Correlation")
                
                # Add legend
                ax_corr.legend(loc='upper right')
                
                # Add thresholds
                thresholds = params["thresholds"]["correlation"]
                if "eog" in thresholds:
                    ax_corr.axhline(y=thresholds["eog"], color=color_map["eog"], linestyle='--', alpha=0.7)
                    ax_corr.text(n_components-1, thresholds["eog"]+0.02, f"EOG threshold: {thresholds['eog']}", 
                               ha='right', va='bottom', color=color_map["eog"])
                
                if "ecg" in thresholds:
                    ax_corr.axhline(y=thresholds["ecg"], color=color_map["ecg"], linestyle='--', alpha=0.7)
                    ax_corr.text(n_components-1, thresholds["ecg"]+0.02, f"ECG threshold: {thresholds['ecg']}", 
                               ha='right', va='bottom', color=color_map["ecg"])
            
            # Save the figure
            plt.tight_layout()
            overview_file = os.path.join(labeled_dir, "labeled_components_overview.png")
            fig.savefig(overview_file, dpi=300)
            plt.close(fig)
            logging.info(f"[ICALabelingStep] Saved labeled components overview to {overview_file}")
            
            # 2. Generate detailed plots for each labeled artifact component
            for comp_idx in all_artifacts:
                try:
                    fig = plt.figure(figsize=(12, 8))
                    plt.suptitle(f"Component {comp_idx} - Artifact Details", fontsize=16, fontweight='bold')
                    
                    # Create a GridSpec layout
                    gs = plt.GridSpec(2, 3, figure=fig)
                    
                    # Topography
                    ax_topo = fig.add_subplot(gs[0, 0])
                    ica.plot_components(comp_idx, axes=ax_topo, show=False, colorbar=False)
                    
                    # Time course
                    ax_time = fig.add_subplot(gs[0, 1:])
                    sources = ica.get_sources(data)
                    source_data = sources.get_data()[comp_idx]
                    times = np.arange(min(10000, len(source_data))) / data.info['sfreq']
                    ax_time.plot(times, source_data[:len(times)])
                    ax_time.set_title("Time Course")
                    ax_time.set_xlabel("Time (s)")
                    ax_time.set_ylabel("Amplitude")
                    ax_time.grid(True)
                    
                    # PSD
                    ax_psd = fig.add_subplot(gs[1, 0])
                    f, Pxx = signal.welch(source_data, fs=data.info['sfreq'], nperseg=min(4096, len(source_data)))
                    ax_psd.semilogy(f, Pxx)
                    ax_psd.set_title("Power Spectral Density")
                    ax_psd.set_xlabel("Frequency (Hz)")
                    ax_psd.set_ylabel("PSD")
                    ax_psd.set_xlim([0, min(100, data.info['sfreq']/2)])
                    ax_psd.grid(True)
                    
                    # Label information
                    ax_info = fig.add_subplot(gs[1, 1:])
                    ax_info.axis('off')
                    
                    info_text = f"Component {comp_idx} - Artifact Details\n\n"
                    
                    # Add ICLabel information
                    if "iclabel" in labeled_components and comp_idx < len(labeled_components["iclabel"]["labels"]):
                        ic_labels = labeled_components["iclabel"]
                        label = ic_labels["labels"][comp_idx]
                        probs = ic_labels["y_pred_proba"][comp_idx]
                        
                        info_text += "ICLabel Classification:\n"
                        for i, class_name in enumerate(ic_labels["labels_set"]):
                            info_text += f"  {class_name.capitalize()}: {probs[i]:.3f}"
                            if class_name == label:
                                info_text += " (selected)"
                            info_text += "\n"
                        
                        info_text += "\n"
                    
                    # Add correlation information
                    if "correlation" in labeled_components:
                        corr_labels = labeled_components["correlation"]
                        
                        if "eog" in corr_labels and comp_idx in corr_labels["eog"]:
                            idx = corr_labels["eog"].index(comp_idx)
                            score = corr_labels["eog_scores"][idx]
                            info_text += f"EOG Correlation Score: {score:.3f}\n"
                        
                        if "ecg" in corr_labels and comp_idx in corr_labels["ecg"]:
                            idx = corr_labels["ecg"].index(comp_idx)
                            score = corr_labels["ecg_scores"][idx]
                            info_text += f"ECG Correlation Score: {score:.3f}\n"
                    
                    # Add component statistics
                    mean = np.mean(source_data)
                    std = np.std(source_data)
                    skew = np.mean(((source_data - mean) / std) ** 3) if std > 0 else 0
                    kurtosis = np.mean(((source_data - mean) / std) ** 4) - 3 if std > 0 else 0
                    
                    info_text += f"\nComponent Statistics:\n"
                    info_text += f"  Mean: {mean:.3f}\n"
                    info_text += f"  Std Dev: {std:.3f}\n"
                    info_text += f"  Skewness: {skew:.3f}\n"
                    info_text += f"  Kurtosis: {kurtosis:.3f}\n"
                    
                    # Display the text
                    ax_info.text(0, 1, info_text, va='top', fontfamily='monospace')
                    
                    # Save the figure
                    plt.tight_layout()
                    fig.savefig(os.path.join(labeled_dir, f"artifact_comp_{comp_idx:03d}.png"), dpi=300)
                    plt.close(fig)
                
                except Exception as e:
                    logging.error(f"[ICALabelingStep] Error creating detailed plot for component {comp_idx}: {e}")
            
            logging.info(f"[ICALabelingStep] Saved detailed plots for {len(all_artifacts)} artifact components")
            
        except Exception as e:
            logging.error(f"[ICALabelingStep] Error in _plot_labeled_components: {e}")
    
    def _plot_before_after(self, data_orig, data_clean, ica, plot_dir, params):
        """Plot data before and after ICA cleaning."""
        logging.info("[ICALabelingStep] Plotting data before and after ICA cleaning")
        
        try:
            # Create a directory for before/after plots
            ba_dir = os.path.join(plot_dir, "before_after")
            os.makedirs(ba_dir, exist_ok=True)
            
            # 1. Plot raw data overlay (before/after)
            fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
            plt.suptitle("Data Before and After ICA Cleaning", fontsize=16, fontweight='bold')
            
            # Before cleaning
            data_orig.plot(axes=axs[0], duration=10, show=False)
            axs[0].set_title("Before ICA Cleaning")
            
            # After cleaning
            data_clean.plot(axes=axs[1], duration=10, show=False)
            axs[1].set_title("After ICA Cleaning")
            
            plt.tight_layout()
            fig.savefig(os.path.join(ba_dir, "raw_before_after.png"), dpi=300)
            plt.close(fig)
            
            # 2. Plot PSD before and after
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Calculate and plot PSDs
            data_orig.compute_psd().plot(ax=ax, show=False, average=True, color='red', alpha=0.8, label='Before')
            data_clean.compute_psd().plot(ax=ax, show=False, average=True, color='blue', alpha=0.8, label='After')
            
            ax.set_title("Power Spectral Density: Before vs. After ICA Cleaning")
            ax.legend()
            
            plt.tight_layout()
            fig.savefig(os.path.join(ba_dir, "psd_before_after.png"), dpi=300)
            plt.close(fig)
            
            # 3. Plot time-frequency representation
            try:
                from mne.time_frequency import tfr_multitaper
                
                # Select a subset of channels that are most affected
                ch_picks = []
                
                # Find channels most affected by excluded components
                if ica.exclude:
                    patterns = ica.get_components()[:, ica.exclude]
                    if patterns.size > 0:
                        # Sum absolute values across components
                        pattern_sum = np.sum(np.abs(patterns), axis=1)
                        # Get top 5 channels
                        top_ch_idx = np.argsort(pattern_sum)[-5:]
                        ch_picks = [data_orig.ch_names[i] for i in top_ch_idx]
                
                # If no specific channels selected, use a default selection
                if not ch_picks:
                    ch_picks = data_orig.ch_names[:5]  # First 5 channels
                
                # Compute TFR for original and cleaned data
                freqs = np.logspace(np.log10(4), np.log10(40), 20)  # Frequencies from 4-40 Hz
                n_cycles = freqs / 2.  # Different number of cycles per frequency
                
                power_orig = tfr_multitaper(data_orig, freqs=freqs, n_cycles=n_cycles, 
                                         picks=ch_picks, return_itc=False, average=False)
                power_clean = tfr_multitaper(data_clean, freqs=freqs, n_cycles=n_cycles, 
                                          picks=ch_picks, return_itc=False, average=False)
                
                # Plot TFR for each selected channel
                for ch_idx, ch_name in enumerate(ch_picks):
                    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True, sharey=True)
                    
                    # Original data
                    power_orig.plot([ch_idx], baseline=None, mode='logratio', title=f"{ch_name}: Before ICA",
                                  axes=axs[0], show=False)
                    
                    # Cleaned data
                    power_clean.plot([ch_idx], baseline=None, mode='logratio', title=f"{ch_name}: After ICA",
                                   axes=axs[1], show=False)
                    
                    plt.suptitle(f"Time-Frequency Representation: {ch_name}", fontsize=16, fontweight='bold')
                    plt.tight_layout()
                    
                    fig.savefig(os.path.join(ba_dir, f"tfr_{ch_name}.png"), dpi=300)
                    plt.close(fig)
                
            except Exception as e:
                logging.warning(f"[ICALabelingStep] Error creating TFR plots: {e}")
            
            # 4. Plot specific artifact segments (if found)
            # For EOG artifacts
            try:
                if "ica_labeled" in data_clean.info.get("temp", {}) and "labeled_components" in data_clean.info["temp"]["ica_labeled"]:
                    labeled = data_clean.info["temp"]["ica_labeled"]["labeled_components"]
                    
                    # Plot EOG artifact segments if found
                    if "correlation" in labeled and "eog" in labeled["correlation"] and labeled["correlation"]["eog"]:
                        # Create a figure comparing before/after on a segment with eye blinks
                        eog_picks = params.get("eog_ch_names", ["Fp1", "Fp2"])
                        if not all(ch in data_orig.ch_names for ch in eog_picks):
                            # Find frontal channels if specified EOG not available
                            frontal_chs = ["Fp1", "Fp2", "F7", "F8", "F3", "F4", "Fz"]
                            eog_picks = [ch for ch in frontal_chs if ch in data_orig.ch_names][:2]
                        
                        if eog_picks:
                            # Find a segment with eye blinks
                            # This is a simplified approach - in a real implementation, 
                            # you might want to use peak detection on EOG channels
                            eog_data = data_orig.copy().pick_channels(eog_picks)
                            eog_data.filter(1, 10)  # Filter to enhance eye blinks
                            times = eog_data.times
                            eog_signal = eog_data.get_data()[0]
                            
                            # Get the time of maximum amplitude as a proxy for blink
                            max_idx = np.argmax(np.abs(eog_signal))
                            blink_time = times[max_idx]
                            
                            # Plot a window around this time
                            window_sec = 2  # seconds on each side
                            start_time = max(0, blink_time - window_sec)
                            
                            fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
                            
                            # Before cleaning
                            data_orig.plot(axes=axs[0], duration=window_sec*2, start=start_time, show=False)
                            axs[0].set_title("Eye Blink Artifact: Before Cleaning")
                            
                            # After cleaning
                            data_clean.plot(axes=axs[1], duration=window_sec*2, start=start_time, show=False)
                            axs[1].set_title("Eye Blink Artifact: After Cleaning")
                            
                            plt.tight_layout()
                            fig.savefig(os.path.join(ba_dir, "eog_artifact_removal.png"), dpi=300)
                            plt.close(fig)
            
            except Exception as e:
                logging.warning(f"[ICALabelingStep] Error creating artifact segment plots: {e}")
            
            logging.info(f"[ICALabelingStep] Saved before/after plots to {ba_dir}")
            
        except Exception as e:
            logging.error(f"[ICALabelingStep] Error in _plot_before_after: {e}") 