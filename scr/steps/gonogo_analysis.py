# File: scr/steps/gonogo_analysis.py

import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import mne
from mne.report import Report
import yaml

from .base import BaseStep  # Adjust import as needed

class GoNoGoAnalysisStep(BaseStep):
    """
    A pipeline step to perform a comprehensive Go/No-Go analysis with multi-method plotting:
      1. Merging triggers for Go/No-Go with correct/incorrect
      2. (Optional) band-pass filtering
      3. Epoching correct trials
      4. Computing ERPs
      5. Combining channels into ROIs
      6. Plotting:
         - Raw trigger channel
         - Average ERPs for single conditions
         - Side-by-side ROI plots
         - Compare evokeds for each ROI
         - Combined ROI plot with custom line styling
      7. Generating an MNE Report
    """

    def __init__(self, config_file=None, repo_root=None):
        self.config_dict = config_file
        config = self._load_config()
        self.params = config.get("pipeline", {}).get("steps", {})[0].get("params", {})
        if not self.params:
            raise ValueError("No parameters defined under pipeline.steps.gonogo_analysis.params in config.")
        self.repo_root = repo_root

    def _load_config(self):
        """Load YAML or dictionary config."""
        if isinstance(self.config_dict, dict):
            return self.config_dict
        elif isinstance(self.config_dict, (str, Path)):
            path = Path(self.config_dict)
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {path}")
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        else:
            raise ValueError("config_file must be a dict or a path to a YAML file.")

    def run(self, data):
        """
        Main entry to run the entire Go/No-Go analysis pipeline step.
        """
        # 1) Gather params
        fif_path = self.repo_root / self.params.get("fif_path", {})
        stim_channel = self.params.get("stim_channel", "Trigger")
        filter_range = self.params.get("filter_range", (1, 30))
        output_dir = self.repo_root / self.params.get("output_dir", "reports/gonogo")
        event_id = self.params.get("event_id", {'Go': 1, 'NoGo': 2, 'Correct': 3, 'Incorrect': 4})
        rois = self.params.get("rois", {
            "Middle_ROI": ["Fz", "Cz", "F3", "F4", "C3", "C4"],
            "Back_ROI": ["P3", "P4", "O1", "O2"]
        })

        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"[GoNoGoAnalysisStep] Loading data from {fif_path}")

        # 2) Load raw data
        raw = mne.io.read_raw_fif(fif_path, preload=True)

        # (Optional) Re-filter
        if filter_range is not None:
            l_freq, h_freq = filter_range
            logging.info(f"[GoNoGoAnalysisStep] Filtering {l_freq} - {h_freq} Hz")
            raw.filter(l_freq, h_freq, phase='zero')

        # --- EXAMPLE PLOT: RAW TRIGGER CHANNEL ---
        self._plot_raw_trigger_channel(raw, output_dir, duration=5)

        # 3) Find events
        logging.info(f"[GoNoGoAnalysisStep] Finding events using stim_channel='{stim_channel}'...")
        events = mne.find_events(raw, stim_channel=stim_channel, min_duration=0.01)

        # 4) Merge triggers into new event codes (Go_Correct, etc.)
        merged_events = self._merge_go_nogo_events(events, event_id)

        # 5) Create epochs for correct trials only
        combined_event_id = {
            'Go_Correct': 101,
            'NoGo_Correct': 201
        }
        tmin, tmax = -0.2, 0.8
        baseline = (None, 0)
        logging.info("[GoNoGoAnalysisStep] Creating epochs for correct trials only...")
        epochs_correct = mne.Epochs(raw, merged_events, event_id=combined_event_id,
                                    tmin=tmin, tmax=tmax, baseline=baseline,
                                    preload=True)

        # --- EXAMPLE PLOTS: AVERAGE ERPs FOR GO & NO-GO ---
        self._plot_average_erp_condition(epochs_correct, 'Go_Correct', output_dir)
        self._plot_average_erp_condition(epochs_correct, 'NoGo_Correct', output_dir)

        # 6) Compute average ERPs
        evoked_go = epochs_correct['Go_Correct'].average()
        evoked_nogo = epochs_correct['NoGo_Correct'].average()

        # 7) Combine channels into ROIs
        logging.info("[GoNoGoAnalysisStep] Combining channels into ROIs...")
        roi_dict = {}
        for roi_name, ch_names in rois.items():
            picks = mne.pick_channels(raw.info['ch_names'], include=ch_names)
            roi_dict[roi_name] = picks

        evoked_go_roi = mne.channels.combine_channels(evoked_go, roi_dict, method='mean')
        evoked_nogo_roi = mne.channels.combine_channels(evoked_nogo, roi_dict, method='mean')

        # --- EXAMPLE PLOT: SIDE-BY-SIDE ROI PLOTS ---
        self._plot_roi_erps_side_by_side(evoked_go_roi, evoked_nogo_roi, output_dir)

        # --- EXAMPLE PLOT: COMPARE EVOKEDS FOR EACH ROI ---
        self._plot_compare_evokeds_rois(evoked_go_roi, evoked_nogo_roi, rois, output_dir)

        # --- ADDITIONAL PLOT: COMBINED ROI (with line styling) ---
        self._plot_combined_rois(evoked_go_roi, evoked_nogo_roi, output_dir)

        # 8) Generate final MNE Report
        self._generate_report(output_dir, evoked_go_roi, evoked_nogo_roi, evoked_go, evoked_nogo)

        logging.info("[GoNoGoAnalysisStep] Done analyzing Go/No-Go.")
        return data

    # ----------------------------------------------------------------
    # EVENT MERGING
    # ----------------------------------------------------------------
    def _merge_go_nogo_events(self, events, event_id):
        """
        Merge pairs: (Go=1 or NoGo=2) followed by (Correct=3 or Incorrect=4) into:
            101 (Go_Correct), 102 (Go_Incorrect),
            201 (NoGo_Correct), 202 (NoGo_Incorrect).
        """
        logging.info("[GoNoGoAnalysisStep] Merging onset/response triggers...")
        new_events = []
        new_event_id = {
            'Go_Correct': 101,
            'Go_Incorrect': 102,
            'NoGo_Correct': 201,
            'NoGo_Incorrect': 202
        }
        i = 0
        while i < len(events) - 1:
            onset_evt = events[i]
            resp_evt = events[i + 1]
            onset = onset_evt[2]
            resp = resp_evt[2]
            # Check if onset in [Go=1, NoGo=2], resp in [Correct=3, Incorrect=4]
            if (onset in [event_id['Go'], event_id['NoGo']] and
                resp in [event_id['Correct'], event_id['Incorrect']]):
                if onset == event_id['Go'] and resp == event_id['Correct']:
                    new_events.append([onset_evt[0], 0, new_event_id['Go_Correct']])
                elif onset == event_id['Go'] and resp == event_id['Incorrect']:
                    new_events.append([onset_evt[0], 0, new_event_id['Go_Incorrect']])
                elif onset == event_id['NoGo'] and resp == event_id['Correct']:
                    new_events.append([onset_evt[0], 0, new_event_id['NoGo_Correct']])
                elif onset == event_id['NoGo'] and resp == event_id['Incorrect']:
                    new_events.append([onset_evt[0], 0, new_event_id['NoGo_Incorrect']])
                i += 2
            else:
                i += 1

        new_events = np.array(new_events)
        logging.info(f"[GoNoGoAnalysisStep] Final merged events shape: {new_events.shape}")
        return new_events

    # ----------------------------------------------------------------
    # PLOTTING METHODS
    # ----------------------------------------------------------------
    def _plot_raw_trigger_channel(self, raw, output_dir, duration=5):
        """
        Plot the Trigger channel in the raw data for a quick visual check.
        Saves a PNG figure.
        """
        logging.info("[GoNoGoAnalysisStep] Plotting raw trigger channel...")
        fig = raw.plot(duration=duration, picks='Trigger', show=False)
        if isinstance(fig, list):
            for idx, f_ in enumerate(fig):
                out_path = Path(output_dir) / f"raw_trigger_channel_{idx}.png"
                f_.savefig(out_path)
                plt.close(f_)
        else:
            out_path = Path(output_dir) / "raw_trigger_channel.png"
            fig.savefig(out_path)
            plt.close(fig)

    def _plot_average_erp_condition(self, epochs, condition_label, output_dir):
        """
        Plot average ERP for a single condition (e.g., 'Go_Correct') and save the figure.
        """
        logging.info(f"[GoNoGoAnalysisStep] Plotting average ERP for condition: {condition_label}")
        evoked = epochs[condition_label].average()
        fig = evoked.plot(spatial_colors=True, show=False,
                          titles=f"{condition_label} Average ERP")

        if isinstance(fig, list):
            for idx, f_ in enumerate(fig):
                out_path = Path(output_dir) / f"{condition_label}_ERP_{idx}.png"
                f_.savefig(out_path)
                plt.close(f_)
        else:
            out_path = Path(output_dir) / f"{condition_label}_ERP.png"
            fig.savefig(out_path)
            plt.close(fig)

    def _plot_roi_erps_side_by_side(self, evoked_go_roi, evoked_nogo_roi, output_dir):
        """
        Plot ROI-averaged Go vs No-Go in a single figure with two subplots.
        """
        logging.info("[GoNoGoAnalysisStep] Plotting side-by-side ROI ERPs for Go vs. No-Go...")
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Plot Go ROI average
        figs_go = evoked_go_roi.plot(axes=axes[0], spatial_colors=False, show=False)
        axes[0].set_title("Go Correct - ROI average")

        # Plot No-Go ROI average
        figs_nogo = evoked_nogo_roi.plot(axes=axes[1], spatial_colors=False, show=False)
        axes[1].set_title("No-Go Correct - ROI average")

        # Save single combined figure:
        fig_path = Path(output_dir) / "roi_erps_go_vs_nogo_side_by_side.png"
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close(fig)

    def _plot_compare_evokeds_rois(self, evoked_go_roi, evoked_nogo_roi, rois, output_dir):
        """
        Compare Go vs. No-Go for each ROI using mne.viz.plot_compare_evokeds.
        Saves one file per ROI.
        """
        logging.info("[GoNoGoAnalysisStep] Plotting compare_evokeds for each ROI...")
        evokeds_compare = {
            'Go Correct': evoked_go_roi,
            'No-Go Correct': evoked_nogo_roi
        }
        colors_compare = {
            'Go Correct': 'green',
            'No-Go Correct': 'red'
        }

        # For each ROI, we use the newly created
        # ROI channel name (e.g., 'Middle_ROI', 'Back_ROI') as 'picks'
        for roi_name in rois.keys():
            fig = mne.viz.plot_compare_evokeds(
                evokeds_compare,
                picks=roi_name,
                combine='mean',
                colors=colors_compare,
                title=f"Go vs No-Go Correct ERP Comparison - {roi_name}",
                show=False
            )
            # plot_compare_evokeds might return either a Figure or a dict
            if isinstance(fig, dict):
                fig = fig['fig']
            out_path = Path(output_dir) / f"compare_evokeds_{roi_name}.png"
            fig.tight_layout()
            fig.savefig(out_path)
            plt.close(fig)

    def _plot_combined_rois(self, evoked_go_roi, evoked_nogo_roi, output_dir):
        """
        Plot Go vs. No-Go ROI Evoked with two subplots (like side-by-side),
        but applying custom line styling using a colormap.
        """
        logging.info("[GoNoGoAnalysisStep] Plotting combined ROI analysis with custom styling...")
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Plot Go responses in the top subplot
        evoked_go_roi.plot(axes=axes[0], spatial_colors=False, show=False)
        axes[0].set_title("Go Correct - ROI average (styled)")

        # Style the lines with our helper
        self._style_lines(axes[0], cmap_name='Greens', legend_text='Go Correct')

        # Plot No-Go responses in the bottom subplot
        evoked_nogo_roi.plot(axes=axes[1], spatial_colors=False, show=False)
        axes[1].set_title("No-Go Correct - ROI average (styled)")

        # Style the lines with our helper
        self._style_lines(axes[1], cmap_name='Reds', legend_text='No-Go Correct')

        out_path = Path(output_dir) / "combined_roi_analysis.png"
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)

    def _style_lines(self, ax, cmap_name, legend_text):
        """
        Apply consistent styling to ERP lines in a given subplot.
        - Uses the specified colormap (e.g. 'Reds', 'Greens') 
        - Appends the legend with the provided text
        """
        cmap = plt.get_cmap(cmap_name)
        lines = ax.get_lines()
        n_lines = len(lines)
        for idx, line in enumerate(lines):
            # vary color across [0.3, 0.8], for instance
            fraction = 0.3 + (0.5 * idx / max(1, n_lines - 1))
            line.set_color(cmap(fraction))

        # Build a customized legend
        ax.legend(
            lines,
            [f"{legend_text} - {line.get_label()}" for line in lines],
            loc='upper right', frameon=True
        )

    # ----------------------------------------------------------------
    # REPORT GENERATION
    # ----------------------------------------------------------------
    def _generate_report(self, output_dir, evoked_go_roi, evoked_nogo_roi, evoked_go, evoked_nogo):
        """
        Create an MNE report that includes final summary plots.
        We'll add a known list of figures that we generated above.
        """
        logging.info("[GoNoGoAnalysisStep] Generating final MNE report for Go/No-Go...")
        report = Report(title="Go/No-Go Analysis Report")

        # Here’s a structured approach to collecting figure files:
        figure_files = [
            "raw_trigger_channel.png",
            "Go_Correct_ERP.png",
            "NoGo_Correct_ERP.png",
            "roi_erps_go_vs_nogo_side_by_side.png",
            "combined_roi_analysis.png",
        ]

        # Add any 'compare_evokeds_ROI' plots for each ROI
        for roi_name in evoked_go_roi.info["ch_names"]:
            # These are the ROI channels named in combine_channels
            # E.g. 'Middle_ROI', 'Back_ROI'
            fname = f"compare_evokeds_{roi_name}.png"
            figure_files.append(fname)

        for fname in figure_files:
            path = Path(output_dir) / fname
            if path.exists():
                # Convert filename to a user-friendly title
                title_str = fname.replace(".png", "").replace("_", " ").title()
                report.add_image(
                    image=path,
                    title=title_str,
                    caption=f"{fname}"
                )

        # Save the final HTML report
        report_fname = Path(output_dir) / "gonogo_analysis_report.html"
        report.save(report_fname, overwrite=True, open_browser=False)
        logging.info(f"[GoNoGoAnalysisStep] Report saved => {report_fname}")
