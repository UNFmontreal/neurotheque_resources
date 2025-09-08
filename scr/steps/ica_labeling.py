# File: scr/steps/ica_labeling.py

import logging
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import mne

from .base import BaseStep
from ..utils.mne_utils import clean_mne_object


def _optional_dep_available(modname: str) -> bool:
    try:
        __import__(modname)
        return True
    except Exception:
        return False


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
    thresholds : dict
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
        
        if data is None:
            raise ValueError("[ICALabelingStep] No data provided.")
        
        # Merge default parameters
        default_params = {
            "methods": ["iclabel"],
            "thresholds": {
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
            "mode": "auto",  # 'auto' | 'interactive'
            "plot_labeled": True,
            "plot_before_after": True,
            "interactive": False,
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
        
        # Initialize figure storage.
        self.figures = []
        
        # Configure matplotlib backend: default to non-interactive/headless
        import matplotlib
        if not params["interactive"]:
            try:
                matplotlib.use("Agg", force=True)
            except Exception:
                pass
        
        # Retrieve ICA.
        if 'temp' in data.info and "ica" in data.info["temp"]:
            ica = data.info["temp"]["ica"]
            logging.info("[ICALabelingStep] Retrieved ICA from data.info['temp']")
        else:
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
        
        # Create plot directory.
        if paths is not None:
            plot_dir = paths.get_ica_report_dir(sub_id, ses_id, task_id, run_id)
        else:
            plot_dir = params.get("plot_dir", f"./ica_labeling_plots/sub-{sub_id}_ses-{ses_id}")
        os.makedirs(plot_dir, exist_ok=True)
        logging.info(f"[ICALabelingStep] Saving plots to {plot_dir}")
        print(f"[ICALabelingStep] Plots will be saved to: {os.path.abspath(plot_dir)}")
        
        # Label components.
        labeled_components = {}
        all_artifacts = set()
        
        # Method 1: ICLabel (if available)
        if "iclabel" in params["methods"]:
            if not _optional_dep_available("mne_icalabel"):
                logging.warning("[ICALabelingStep] mne-icalabel not installed; skipping ICLabel. Install mne-icalabel to enable.")
                ic_labels = None
            else:
                ic_labels = self._label_with_iclabel(ica, data, params)
            if ic_labels:
                labeled_components["iclabel"] = ic_labels
                thresholds_ic = params["thresholds"]["iclabel"]
                # Iterate over components (using index checking rather than "in" on list)
                for idx in range(len(ic_labels["labels"])):
                    label = ic_labels["labels"][idx]
                    if label == 'brain':
                        continue
                    label_idx = ic_labels["labels_set"].index(label)
                    label_prob = ic_labels["y_pred_proba"][idx][label_idx]
                    if label in thresholds_ic and label_prob >= thresholds_ic[label]:
                        all_artifacts.add(idx)
                        logging.info(f"[ICALabelingStep] Component {idx} classified as '{label}' (prob={label_prob:.2f})")
        
        # Method 2: Correlation.
        if "correlation" in params["methods"]:
            corr_artifacts = self._label_with_correlation(ica, data, params)
            if corr_artifacts:
                labeled_components["correlation"] = corr_artifacts
                if "eog" in corr_artifacts and corr_artifacts["eog"] and "eog_scores" in corr_artifacts:
                    for idx, score in zip(corr_artifacts["eog"], corr_artifacts["eog_scores"]):
                        if float(score) >= 0.8:
                            all_artifacts.add(idx)
                            logging.info(f"[ICALabelingStep] Component {idx} identified as EOG with score {float(score):.2f}")
                if "ecg" in corr_artifacts and corr_artifacts["ecg"] and "ecg_scores" in corr_artifacts:
                    for idx, score in zip(corr_artifacts["ecg"], corr_artifacts["ecg_scores"]):
                        if score >= 0.8:
                            all_artifacts.add(idx)
                            logging.info(f"[ICALabelingStep] Component {idx} identified as ECG with score {score:.2f}")
        
        # Plot labeled components.
        if params["plot_labeled"]:
            labeled_figs = self._plot_labeled_components(ica, data, labeled_components, all_artifacts, plot_dir, params)
            if labeled_figs:
                self.figures.extend(labeled_figs)
        
        # Manual selection.
        final_exclude = set()
        mode = params.get("mode", "auto")
        if params["manual_selection"] and params["interactive"] and mode == "interactive":
            print("\n[ICALabelingStep] Components identified as artifacts:")
            if "iclabel" in labeled_components:
                print("  ICLabel:")
                for comp_idx, label in enumerate(labeled_components["iclabel"]["labels"]):
                    if label != "brain":
                        if comp_idx < len(labeled_components["iclabel"]["y_pred_proba"]):
                            label_idx = labeled_components["iclabel"]["labels_set"].index(label)
                            prob = labeled_components["iclabel"]["y_pred_proba"][comp_idx][label_idx]
                            print(f"    Component {comp_idx}: {label.capitalize()} (prob={prob:.2f})")
            if "correlation" in labeled_components:
                corr = labeled_components["correlation"]
                if "eog" in corr:
                    print(f"  EOG correlation: {corr['eog']}")
                if "ecg" in corr:
                    print(f"  ECG correlation: {corr['ecg']}")
            
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
        elif params["auto_exclude"] or mode == "auto":
            final_exclude = all_artifacts
            logging.info(f"[ICALabelingStep] Automatically excluding components: {sorted(list(final_exclude))}")
        else:
            logging.info("[ICALabelingStep] No components will be excluded (manual_selection=False, auto_exclude=False)")
        
        # Reconstruct cleaned data.
        if params["reconstruct"] and final_exclude:
            ica.exclude = sorted(list(final_exclude))
            logging.info(f"[ICALabelingStep] Excluding components: {ica.exclude}")
            data_clean = ica.apply(data.copy())
            if params["plot_before_after"]:
                ba_figs = self._plot_before_after(data, data_clean, ica, plot_dir, params)
                if ba_figs:
                    self.figures.extend(ba_figs)
            # Do not auto-open folders in headless/server mode
            if params.get("save_data", False):
                try:
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
                    os.makedirs(os.path.dirname(str(clean_file)), exist_ok=True)
                    data_to_save = clean_mne_object(data_clean)
                    if 'temp' in data_to_save.info:
                        del data_to_save.info['temp']
                    data_to_save.info['subject_info'] = {'his_id': f"sub-{sub_id}"}
                    logging.info(f"[ICALabelingStep] Saving cleaned data to {clean_file}")
                    data_to_save.save(str(clean_file), overwrite=True)
                    logging.info("[ICALabelingStep] Successfully saved cleaned data")
                except Exception as e:
                    logging.error(f"[ICALabelingStep] Error saving cleaned data: {e}")
                    logging.error("[ICALabelingStep] Continuing without saving cleaned data")
            if 'ipykernel' in sys.modules and self.figures:
                for fig in self.figures:
                    if fig is not None:
                        plt.figure(fig.number)
                        plt.show()
            return data_clean
        else:
            if not hasattr(data.info, "temp"):
                data.info["temp"] = {}
            data.info["temp"]["ica_labeled"] = {
                "labeled_components": labeled_components,
                "suggested_exclude": sorted(list(all_artifacts))
            }
            if params.get("save_data", False) and params.get("output_file", None):
                try:
                    os.makedirs(os.path.dirname(params["output_file"]), exist_ok=True)
                    data_to_save = clean_mne_object(data)
                    data_to_save.save(params["output_file"], overwrite=True)
                    logging.info(f"[ICALabelingStep] Saved data with ICA labels to {params['output_file']}")
                except Exception as e:
                    logging.error(f"[ICALabelingStep] Error saving data to {params['output_file']}: {e}")
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
            labeled_dir = os.path.join(plot_dir, "labeled_components")
            os.makedirs(labeled_dir, exist_ok=True)
            print(f"[ICALabelingStep] Saving plots to: {os.path.abspath(labeled_dir)}")
            
            # Overview plot: a bar graph showing component classification.
            n_components = ica.n_components_
            fig_overview, ax_overview = plt.subplots(figsize=(15, 4))
            ax_overview.set_title("ICA Components Classification Overview", fontsize=14, fontweight='bold')
            ax_overview.set_xlabel("Component Index")
            ax_overview.set_xticks(np.arange(n_components))
            
            color_map = {
                'brain': 'green',
                'eye': 'blue',
                'heart': 'red',
                'muscle': 'purple',
                'line_noise': 'orange',
                'channel_noise': 'brown',
                'other': 'gray'
            }
            comp_colors = []
            for comp in range(n_components):
                label = "brain"
                # Check if there is a label for this component from ICLabel.
                if "iclabel" in labeled_components and comp < len(labeled_components["iclabel"]["labels"]):
                    cand = labeled_components["iclabel"]["labels"][comp]
                    if cand != 'brain':
                        label = cand
                if comp in all_artifacts:
                    comp_colors.append("black")
                else:
                    comp_colors.append(color_map.get(label, "gray"))
                    
            ax_overview.bar(np.arange(n_components), [1] * n_components, color=comp_colors)
            ax_overview.set_yticks([])
            
            overview_file = os.path.join(labeled_dir, "overview.png")
            fig_overview.tight_layout()
            fig_overview.savefig(overview_file, dpi=300)
            all_figs.append(fig_overview)
            logging.info(f"[ICALabelingStep] Saved overview plot to {overview_file}")
            
            # Detailed per-component plots using plot_properties.
            for comp_idx in sorted(all_artifacts):
                try:
                    fig_props = ica.plot_properties(data, picks=comp_idx, psd_args={'fmax': 50}, show=False)
                    if not isinstance(fig_props, list):
                        fig_props = [fig_props]
                    for i, fig in enumerate(fig_props):
                        comp_file = os.path.join(labeled_dir, f"component_{comp_idx:02d}_properties_{i}.png")
                        fig.savefig(comp_file, dpi=300)
                        all_figs.append(fig)
                        print(f"[ICALabelingStep] Saved detailed plot for component {comp_idx} to {comp_file}")
                except Exception as e:
                    logging.error(f"[ICALabelingStep] Error plotting properties for component {comp_idx}: {e}")
            
            return all_figs
        except Exception as e:
            logging.error(f"[ICALabelingStep] Error in _plot_labeled_components: {e}")
            return None

    def _plot_before_after(self, data_orig, data_clean, ica, plot_dir, params):
        """
        Plot data before and after ICA cleaning using robust MNE functions.
        For raw data, use data.plot(); for Epochs, average and plot Evoked data.
        Also plot PSD comparison.
        """
        logging.info("[ICALabelingStep] Plotting data before and after ICA cleaning")
        all_figs = []
        try:
            ba_dir = os.path.join(plot_dir, "before_after")
            os.makedirs(ba_dir, exist_ok=True)
            print(f"[ICALabelingStep] Saving before/after plots to: {os.path.abspath(ba_dir)}")
        
            # Raw data or Evoked.
            if not isinstance(data_orig, mne.BaseEpochs):
                fig_raw, ax_raw = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
                ax_raw[0].set_title("Before ICA Cleaning")
                data_orig.plot(duration=10, show=False, axes=ax_raw[0])
                ax_raw[1].set_title("After ICA Cleaning")
                data_clean.plot(duration=10, show=False, axes=ax_raw[1])
            else:
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
            fig_psd, ax_psd = plt.subplots(figsize=(12, 8))

            # Determine n_fft such that it does not exceed the length of the time axis.
            n_fft_val = min(4096, len(data_orig.times))
            n_fft_val_clean = min(4096, len(data_clean.times))

            # Compute the PSD using average='mean' so that the PSDs are averaged internally.
            psds_orig, freqs_orig = data_orig.compute_psd(method='welch', n_fft=n_fft_val, average='mean')[:2]
            psds_clean, freqs_clean = data_clean.compute_psd(method='welch', n_fft=n_fft_val_clean, average='mean')[:2]

            # If the frequency array has more than one dimension (e.g., shape (n_epochs, n_freqs)), select the first row.
            if freqs_orig.ndim == 2:
                freqs_orig = freqs_orig[0, :]
            if freqs_clean.ndim == 2:
                freqs_clean = freqs_clean[0, :]

            # If PSD arrays are 2D, average across channels (they should now be 1D after averaging)
            if psds_orig.ndim == 2:
                psds_orig = psds_orig.mean(axis=0)
            if psds_clean.ndim == 2:
                psds_clean = psds_clean.mean(axis=0)

            ax_psd.semilogy(freqs_orig, psds_orig, color='red', alpha=0.8, label='Before')
            ax_psd.semilogy(freqs_clean, psds_clean, color='blue', alpha=0.8, label='After')
            ax_psd.set_title("PSD: Before vs. After ICA Cleaning")
            ax_psd.set_xlabel("Frequency (Hz)")
            ax_psd.set_ylabel("Power Spectral Density (V²/Hz)")
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
