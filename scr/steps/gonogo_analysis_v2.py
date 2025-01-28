# File: scr/steps/gonogo_analysis.py

import logging
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import mne
from mne.report import Report
from mne.time_frequency import tfr_morlet
from mne.stats import ttest_rel  # or from scipy.stats import ttest_rel

from .base import BaseStep


class GoNoGoAnalysisStep(BaseStep):
    """
    Enhanced Go/No-Go analysis supporting:
    - Multiple subjects/sessions with flexible file patterns
    - Per-subject loading, event merging, ROI-based ERP analysis
    - Optional time-frequency (Morlet wavelet) analysis
    - N2/P3 amplitude extraction for Go vs. NoGo
    - Individual subject and group-level HTML reports
    """

    def __init__(self, config_file=None, repo_root=None):
        self.config_dict = config_file
        config = self._load_config()
        self.params = config.get("pipeline", {}).get("steps", {})[0].get("params", {})
        # Root directory for the pipeline (or current working dir)
        self.repo_root = Path(repo_root) if repo_root else Path.cwd()

        # Store results for each subject in a dictionary
        # e.g., self.subject_data[(sub_id, ses_id)] = { ...analysis outputs... }
        self.subject_data = {}

        # Store any group-level results (e.g., grand averages, stats)
        self.group_data = {}

    def _load_config(self):
        """Load YAML or dictionary config."""
        if isinstance(self.config_dict, dict):
            return self.config_dict
        else:
            path = Path(self.config_dict)
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {path}")
            with open(path, 'r') as f:
                return yaml.safe_load(f)

    # -------------------------------------------------------------------------
    # Main pipeline entry
    # -------------------------------------------------------------------------
    def run(self, data):
        """
        Main method to:
         1) Find all subject FIF files
         2) Process each subject
         3) Generate a group-level analysis & report
        """
        files = self._get_subject_files()

        if not files:
            logging.error("No files found matching the pattern!")
            return data

        # Process each file => subject/session
        for fif_path in files:
            subj_info = self._parse_filename(fif_path)
            if self._should_process(subj_info):
                self._process_single_file(fif_path, subj_info)

        # If more than one subject, produce group-level stats
        if len(self.subject_data) > 1:
            self._perform_group_analysis()
            self._generate_group_report()

        return data

    # -------------------------------------------------------------------------
    # Locating & Selecting Subjects
    # -------------------------------------------------------------------------
    def _get_subject_files(self):
        """
        Returns a list of FIF files that match the user-defined pattern in config.
        E.g. "data/pilot_data/sub-*_ses-*_raw_preprocessed_GonoGo_noEpoched.fif"
        """
        # The user can store a pattern in self.params["fif_path"], e.g.:
        #   "data/pilot_data/sub-*_ses-*_raw_preprocessed_GonoGo_noEpoched.fif"
        search_pattern = str(self.repo_root / self.params["fif_path"])
        # or store it separately in "fif_path_pattern" – adapt as you prefer.
        return sorted(Path(self.repo_root).glob(self.params["fif_path"]))

    def _parse_filename(self, path):
        """Extract subject/session info from filename via regex or custom logic."""
        fname = path.name
        subj_match = re.search(r'sub-(\d+)', fname)
        sess_match = re.search(r'ses-(\d+)', fname)
        return {
            'subject': subj_match.group(1) if subj_match else 'unknown',
            'session': sess_match.group(1) if sess_match else '001',
            'full_path': path
        }

    def _should_process(self, subj_info):
        """
        Decide if we should process this subject/session.
        For example, you can exclude certain sessions or check a subject whitelist.
        """
        return True

    # -------------------------------------------------------------------------
    # Subject-Level Analysis
    # -------------------------------------------------------------------------
    def _process_single_file(self, fif_path, subj_info):
        """Process a single subject's data (load, filter, event merge, epoch, analyze)."""
        subj_id = f"sub-{subj_info['subject']}"
        sess_id = f"ses-{subj_info['session']}"

        # Create an output directory for this subject/session
        output_dir = (self.repo_root / self.params["output_dir"] / subj_id / sess_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize a place to store results & a subject-level MNE Report
        self.subject_data[(subj_id, sess_id)] = {
            'output_dir': output_dir,
            'report': Report(title=f"{subj_id} {sess_id} Report"),
        }

        try:
            # --- 1) Load & Filter ---
            raw = self._load_and_filter(fif_path)

            # (Optional) Quick plot of raw trigger channel
            self._plot_raw_trigger_channel(raw, output_dir)

            # --- 2) Find & Merge Events ---
            events = self._find_and_merge_events(raw)

            # --- 3) Epoch Data (Correct Trials) ---
            epochs = self._create_epochs(raw, events)

            # --- 4) Subject-level ERP & ROI Analysis ---
            evoked_go = epochs['Go_Correct'].average()
            evoked_nogo = epochs['NoGo_Correct'].average()

            # ROI
            evoked_go_roi, evoked_nogo_roi = self._roi_analysis(evoked_go, evoked_nogo, raw, output_dir)

            # (Optional) Time-Frequency
            tfr_data = None
            if self.params.get("time_frequency", False):
                tfr_data = self._time_frequency_analysis(epochs, output_dir)

            # (Optional) Basic N2/P3 amplitude extraction
            # e.g., measure amplitude in 200-300 ms (N2), 300-500 ms (P3)
            comp_amps = self._detect_n2_p3_amplitudes(evoked_go_roi, evoked_nogo_roi)

            # Store analysis results in self.subject_data
            self.subject_data[(subj_id, sess_id)].update({
                'epochs': epochs,
                'evoked_go': evoked_go,
                'evoked_nogo': evoked_nogo,
                'evoked_go_roi': evoked_go_roi,
                'evoked_nogo_roi': evoked_nogo_roi,
                'component_amps': comp_amps,
                'tfr': tfr_data
            })

            # Create a subject-level HTML report
            self._generate_subject_report(subj_id, sess_id)

        except Exception as e:
            logging.error(f"Failed processing {subj_id}/{sess_id}: {str(e)}")

    def _load_and_filter(self, fif_path):
        """Load raw data from .fif and optionally apply band-pass filter."""
        raw = mne.io.read_raw_fif(fif_path, preload=True)
        if self.params.get("filter_range"):
            l_freq, h_freq = self.params["filter_range"]
            raw.filter(l_freq, h_freq, phase='zero')
        return raw

    def _find_and_merge_events(self, raw):
        """Find raw events and merge them into Go_Correct, NoGo_Correct, etc."""
        stim_channel = self.params.get("stim_channel", "Trigger")
        event_id_map = self.params.get("event_id", {'Go':1, 'NoGo':2, 'Correct':3, 'Incorrect':4})

        events = mne.find_events(raw, stim_channel=stim_channel, min_duration=0.01)
        merged = self._merge_events(events, event_id_map)
        return merged

    def _merge_events(self, events, event_id):
        """
        Merge pairs: (Go=1 or NoGo=2) followed by (Correct=3 or Incorrect=4) 
        => new codes 101, 102, 201, 202.
        """
        new_events = []
        new_id = {'Go_Correct': 101, 'Go_Incorrect': 102, 'NoGo_Correct': 201, 'NoGo_Incorrect': 202}
        i = 0
        while i < len(events) - 1:
            onset_evt = events[i]
            resp_evt = events[i + 1]
            onset = onset_evt[2]
            resp = resp_evt[2]
            if onset in [event_id['Go'], event_id['NoGo']] and resp in [event_id['Correct'], event_id['Incorrect']]:
                if onset == event_id['Go'] and resp == event_id['Correct']:
                    new_events.append([onset_evt[0], 0, new_id['Go_Correct']])
                elif onset == event_id['Go'] and resp == event_id['Incorrect']:
                    new_events.append([onset_evt[0], 0, new_id['Go_Incorrect']])
                elif onset == event_id['NoGo'] and resp == event_id['Correct']:
                    new_events.append([onset_evt[0], 0, new_id['NoGo_Correct']])
                elif onset == event_id['NoGo'] and resp == event_id['Incorrect']:
                    new_events.append([onset_evt[0], 0, new_id['NoGo_Incorrect']])
                i += 2
            else:
                i += 1
        return np.array(new_events)

    def _create_epochs(self, raw, events):
        """
        Create epochs for correct trials only:
         - event_id = {Go_Correct=101, NoGo_Correct=201}
         - tmin=-0.2, tmax=0.8 by default
        """
        return mne.Epochs(
            raw, events,
            event_id={'Go_Correct': 101, 'NoGo_Correct': 201},
            tmin=-0.2, tmax=0.8,
            baseline=(None, 0), preload=True
        )

    def _roi_analysis(self, evoked_go, evoked_nogo, raw, output_dir):
        """Combine channels into ROIs & produce ROI plots (side-by-side, compare_evokeds, etc.)."""
        rois = self.params.get("rois", {
            "Middle_ROI": ["Fz", "Cz", "F3", "F4", "C3", "C4"],
            "Back_ROI": ["P3", "P4", "O1", "O2"]
        })
        # Convert channel names -> picks
        roi_dict = {roi_name: mne.pick_channels(raw.info['ch_names'], include=chs)
                    for roi_name, chs in rois.items()}

        # Combine channels
        evoked_go_roi = mne.channels.combine_channels(evoked_go, roi_dict, method='mean')
        evoked_nogo_roi = mne.channels.combine_channels(evoked_nogo, roi_dict, method='mean')

        # Plot side-by-side
        self._plot_roi_erps_side_by_side(evoked_go_roi, evoked_nogo_roi, output_dir)
        # Compare evokeds for each ROI
        self._plot_compare_evokeds_rois(evoked_go_roi, evoked_nogo_roi, rois, output_dir)
        # Combined custom style
        self._plot_combined_rois(evoked_go_roi, evoked_nogo_roi, output_dir)

        return evoked_go_roi, evoked_nogo_roi

    def _time_frequency_analysis(self, epochs, output_dir):
        """Example Morlet wavelet time-frequency transform. Adjust freq range / baselining as needed."""
        freqs = np.logspace(*np.log10([3, 30]), num=15)
        n_cycles = freqs / 2  # or a custom approach

        power = tfr_morlet(epochs, picks='eeg', freqs=freqs, n_cycles=n_cycles,
                           return_itc=False, average=True, decim=3)
        # Example: topographic plot of average power
        tfr_fig = power.average().plot_topo(baseline=(None, 0), mode='logratio', show=False)
        out_path = output_dir / "time_frequency_topo.png"
        tfr_fig.savefig(out_path)
        plt.close(tfr_fig)

        return power

    def _detect_n2_p3_amplitudes(self, evoked_go_roi, evoked_nogo_roi):
        """
        Measure average amplitude in N2 (200-300 ms) and P3 (300-500 ms) windows
        for each ROI channel. Return a structure like:
          {
            'ROI_NAME': {
               'Go_Correct': {'N2': val, 'P3': val},
               'NoGo_Correct': {'N2': val, 'P3': val}
            },
            ...
          }
        """
        time_windows = {
            'N2': (0.2, 0.3),
            'P3': (0.3, 0.5)
        }

        rois = evoked_go_roi.ch_names  # e.g. ["Middle_ROI", "Back_ROI"]
        results = {}

        # We'll handle each ROI channel
        for roi_name in rois:
            idx_go = evoked_go_roi.ch_names.index(roi_name)
            idx_nogo = evoked_nogo_roi.ch_names.index(roi_name)
            data_go = evoked_go_roi.data[idx_go, :]  # shape = (n_times,)
            data_nogo = evoked_nogo_roi.data[idx_nogo, :]

            times = evoked_go_roi.times

            results[roi_name] = {'Go_Correct': {}, 'NoGo_Correct': {}}
            for comp_name, (start_t, end_t) in time_windows.items():
                mask = (times >= start_t) & (times <= end_t)
                mean_go = np.mean(data_go[mask]) * 1e6  # convert to microvolts
                mean_nogo = np.mean(data_nogo[mask]) * 1e6
                results[roi_name]['Go_Correct'][comp_name] = mean_go
                results[roi_name]['NoGo_Correct'][comp_name] = mean_nogo

        return results

    # -------------------------------------------------------------------------
    # Subject-Level Reporting
    # -------------------------------------------------------------------------
    def _generate_subject_report(self, subj_id, sess_id):
        """
        Gather subject-level figures from the output directory, add to MNE Report,
        and save an HTML file. 
        """
        out_dir = self.subject_data[(subj_id, sess_id)]['output_dir']
        report = self.subject_data[(subj_id, sess_id)]['report']

        # Add all PNG images we generated
        for fig_file in out_dir.glob("*.png"):
            # Turn filename into a title-ish label
            title_str = fig_file.stem.replace("_", " ").title()
            report.add_image(fig_file, title=title_str, caption=str(fig_file.name))

        html_name = out_dir / f"{subj_id}_{sess_id}_gonogo_report.html"
        report.save(html_name, overwrite=True, open_browser=False)
        logging.info(f"[GoNoGoAnalysisStep] Subject report saved => {html_name}")

    # -------------------------------------------------------------------------
    # Group-Level Analysis
    # -------------------------------------------------------------------------
    def _perform_group_analysis(self):
        """
        1) Gather subject-level Evoked or amplitude data
        2) Compute grand averages or do T-tests
        3) Store results in self.group_data
        """
        # Example: gather each subject's evoked_go and evoked_nogo
        evoked_go_list = []
        evoked_nogo_list = []
        comp_amps = []  # store e.g. N2/P3 from each subject

        for (subj_id, sess_id), data_dict in self.subject_data.items():
            if 'evoked_go' in data_dict and 'evoked_nogo' in data_dict:
                evoked_go_list.append(data_dict['evoked_go'])
                evoked_nogo_list.append(data_dict['evoked_nogo'])
            if 'component_amps' in data_dict:
                comp_amps.append(data_dict['component_amps'])

        # If we have more than one subject, make a grand average
        if len(evoked_go_list) > 1:
            grand_go = mne.grand_average(evoked_go_list)
            grand_nogo = mne.grand_average(evoked_nogo_list)
            grand_diff = grand_nogo - grand_go
            self.group_data['grand_go'] = grand_go
            self.group_data['grand_nogo'] = grand_nogo
            self.group_data['grand_diff'] = grand_diff

        # Example group stats on N2 amplitude for ROI=Back_ROI, etc.
        # You can iterate over each ROI, do a paired t-test on Go vs. NoGo
        # This is a simplistic illustration
        if comp_amps:
            # comp_amps is a list of dicts => each dict: ROI -> {Go_Correct:{N2:val,P3:val},NoGo_Correct:{N2:val,P3:val}}
            all_rois = list(comp_amps[0].keys())  # e.g. ['Middle_ROI','Back_ROI']
            stats_results = {}
            for roi in all_rois:
                go_n2_vals = []
                nogo_n2_vals = []
                for s_amps in comp_amps:
                    go_n2_vals.append(s_amps[roi]['Go_Correct']['N2'])
                    nogo_n2_vals.append(s_amps[roi]['NoGo_Correct']['N2'])

                if len(go_n2_vals) > 1:
                    t_stat, p_val = ttest_rel(nogo_n2_vals, go_n2_vals)
                    stats_results[roi] = (t_stat, p_val)

            self.group_data['stats_n2'] = stats_results

    def _generate_group_report(self):
        """
        Produce a group-level MNE report summarizing:
        - Grand averages for Go, NoGo, and difference
        - Possibly TFR group results, plus textual stats
        """
        group_dir = self.repo_root / self.params["output_dir"] / "group"
        group_dir.mkdir(parents=True, exist_ok=True)
        group_report = Report(title="Go/NoGo Group-Level Analysis")

        # Add grand-average figures
        if 'grand_go' in self.group_data:
            fig_go = self.group_data['grand_go'].plot(spatial_colors=True, show=False)
            group_report.add_figure(fig=fig_go, title="Grand Average - Go", caption="All Subjects")
            plt.close(fig_go)

        if 'grand_nogo' in self.group_data:
            fig_nogo = self.group_data['grand_nogo'].plot(spatial_colors=True, show=False)
            group_report.add_figure(fig=fig_nogo, title="Grand Average - NoGo", caption="All Subjects")
            plt.close(fig_nogo)

        if 'grand_diff' in self.group_data:
            fig_diff = self.group_data['grand_diff'].plot(spatial_colors=True, show=False)
            group_report.add_figure(fig=fig_diff, title="Grand Average - Difference (NoGo - Go)",
                                    caption="All Subjects")
            plt.close(fig_diff)

        # If we have stats results, add them
        stats_n2 = self.group_data.get('stats_n2', {})
        if stats_n2:
            stats_html = "<h2>N2 Stats (Paired t-test: NoGo vs. Go)</h2><ul>"
            for roi, (t_val, p_val) in stats_n2.items():
                stats_html += f"<li>{roi}: t={t_val:.3f}, p={p_val:.3g}</li>"
            stats_html += "</ul>"
            group_report.add_html(stats_html, title="N2 Group Stats")

        # Save final group report
        report_fname = group_dir / "gonogo_group_report.html"
        group_report.save(report_fname, overwrite=True, open_browser=False)
        logging.info(f"[GoNoGoAnalysisStep] Group report saved => {report_fname}")

    # -------------------------------------------------------------------------
    # Plotting Helpers
    # -------------------------------------------------------------------------
    def _plot_raw_trigger_channel(self, raw, out_dir, duration=5):
        """Optional: Quick look at the Trigger channel in the raw data."""
        fig = raw.plot(duration=duration, picks='Trigger', show=False)
        outpath = out_dir / "raw_trigger_channel.png"
        if isinstance(fig, list):
            fig[0].savefig(outpath)
            plt.close(fig[0])
        else:
            fig.savefig(outpath)
            plt.close(fig)

    def _plot_roi_erps_side_by_side(self, evoked_go_roi, evoked_nogo_roi, out_dir):
        """Plot ROI-averaged Go vs. No-Go in a single figure with two subplots."""
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        evoked_go_roi.plot(axes=axes[0], spatial_colors=False, show=False)
        axes[0].set_title("Go Correct - ROI Average")
        evoked_nogo_roi.plot(axes=axes[1], spatial_colors=False, show=False)
        axes[1].set_title("No-Go Correct - ROI Average")

        out_path = out_dir / "roi_erps_go_vs_nogo_side_by_side.png"
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)

    def _plot_compare_evokeds_rois(self, evoked_go_roi, evoked_nogo_roi, rois, out_dir):
        """Compare Go vs No-Go for each ROI via mne.viz.plot_compare_evokeds."""
        evokeds_compare = {'Go Correct': evoked_go_roi, 'No-Go Correct': evoked_nogo_roi}
        colors_compare = {'Go Correct': 'green', 'No-Go Correct': 'red'}

        for roi_name in rois.keys():
            fig = mne.viz.plot_compare_evokeds(
                evokeds_compare,
                picks=roi_name,
                combine='mean',
                colors=colors_compare,
                title=f"Go vs No-Go Comparison - {roi_name}",
                show=False
            )
            # If dict returned, extract the figure
            if isinstance(fig, dict):
                fig = fig['fig']
            out_path = out_dir / f"compare_evokeds_{roi_name}.png"
            fig.savefig(out_path)
            plt.close(fig)

    def _plot_combined_rois(self, evoked_go_roi, evoked_nogo_roi, out_dir):
        """Plot ROI waveforms for Go and No-Go with custom color styling."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        evoked_go_roi.plot(axes=axes[0], spatial_colors=False, show=False)
        axes[0].set_title("Go Correct - ROI (styled)")
        self._style_lines(axes[0], cmap_name='Greens', legend_text='Go Correct')

        evoked_nogo_roi.plot(axes=axes[1], spatial_colors=False, show=False)
        axes[1].set_title("No-Go Correct - ROI (styled)")
        self._style_lines(axes[1], cmap_name='Reds', legend_text='No-Go Correct')

        out_path = out_dir / "combined_roi_analysis.png"
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)

    def _style_lines(self, ax, cmap_name, legend_text):
        """Apply consistent styling to ERP lines in a subplot via a colormap."""
        cmap = plt.get_cmap(cmap_name)
        lines = ax.get_lines()
        n_lines = len(lines)
        for idx, line in enumerate(lines):
            fraction = 0.3 + (0.5 * idx / max(1, n_lines - 1))
            line.set_color(cmap(fraction))
        ax.legend(lines,
                  [f"{legend_text} - {line.get_label()}" for line in lines],
                  loc='upper right', frameon=True)
