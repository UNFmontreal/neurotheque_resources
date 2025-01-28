# File: scr/steps/gonogo_analysis.py

import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import mne
from mne.report import Report
from .base import BaseStep
import yaml

class GoNoGoAnalysisStep(BaseStep):
    """
    A pipeline step to perform a comprehensive Go/No-Go analysis:
      1. Merge triggers: (Go=1, NoGo=2, Correct=3, Incorrect=4) → 
         (Go_Correct=101, Go_Incorrect=102, NoGo_Correct=201, NoGo_Incorrect=202)
      2. Band-pass filter (optional) for 1-30 Hz (if desired)
      3. Epoch data around correct trials only
      4. Compute ERPs, combine channels into ROIs
      5. Compare Go vs. No-Go in specific ROIs
      6. Generate a final MNE report with figures

    Literature Refs:
    - Falkenstein et al. (1999): Differences between Go and NoGo in the N2 and P3
    - Kok et al. (1986): The amplitude of the P3 component as an index of processing capacity

    Expected pipeline params:
    --------------------------------
    fif_path : str
        Path to the preprocessed .fif file containing Go/No-Go data
    stim_channel : str
        Name of the stimulus/trigger channel (default='Trigger')
    filter_range : tuple or None
        e.g., (1, 30) if you want to re-filter, or None to skip
    output_dir : str
        Where to save figures and final report
    event_id : dict
        Mapping for basic triggers, e.g., {'Go':1, 'NoGo':2, 'Correct':3, 'Incorrect':4}
    rois : dict of list
        Dictionary of ROI name -> list of channel names. E.g.,
            {
                "Middle_ROI": ["Fz","Cz","F3","F4","C3","C4"],
                "Back_ROI": ["P3","P4","O1","O2"]
            }
    """
    def __init__(self, config_file=None,repo_root=None):
        self.config_dict = config_file
        config=self._load_config()
        self.params = config.get("pipeline", {}).get("steps", {})[0].get("params", {})
        if not self.params:
            raise ValueError("No parameters defined under pipeline.steps.gonogo_analysis.params in config.")
        self.repo_root=repo_root
    def _load_config(self):
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
        # 1) Gather params
        fif_path=self.repo_root / self.params.get("fif_path", {})
        stim_channel = self.params.get("stim_channel", "Trigger")
        filter_range = self.params.get("filter_range", (1, 30))
        output_dir = self.repo_root / self.params.get("output_dir", "reports/gonogo")
        event_id = self.params.get("event_id", {'Go':1, 'NoGo':2, 'Correct':3, 'Incorrect':4})
        rois = self.params.get("rois", {
            "Middle_ROI": ["Fz","Cz","F3","F4","C3","C4"],
            "Back_ROI": ["P3","P4","O1","O2"]
        })

        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"[GoNoGoAnalysisStep] Loading data from {fif_path}")
        
        # 2) Load raw data
        raw = mne.io.read_raw_fif(fif_path, preload=True)
        
        # Optional: re-filter if desired
        if filter_range is not None:
            l_freq, h_freq = filter_range
            logging.info(f"[GoNoGoAnalysisStep] Filtering {l_freq} - {h_freq} Hz")
            raw.filter(l_freq, h_freq, phase='zero')
        
        # 3) Find events (if not already in raw.info)
        logging.info(f"[GoNoGoAnalysisStep] Finding events using stim_channel='{stim_channel}'...")
        events = mne.find_events(raw, stim_channel=stim_channel, min_duration=0.01)
        
        # 4) Merge triggers into new event codes
        merged_events = self._merge_go_nogo_events(events, event_id)

        # 5) Create epochs for correct trials only
        combined_event_id = {
            'Go_Correct': 101,
            'NoGo_Correct': 201
            # If you want to analyze Incorrect too, add them
        }
        tmin, tmax = -0.2, 0.8
        baseline = (None, 0)
        logging.info("[GoNoGoAnalysisStep] Creating epochs for correct trials only...")
        epochs_correct = mne.Epochs(
            raw, merged_events, event_id=combined_event_id,
            tmin=tmin, tmax=tmax,
            baseline=baseline, preload=True
        )

        # 6) Compute average ERPs
        evoked_go = epochs_correct['Go_Correct'].average()
        evoked_nogo = epochs_correct['NoGo_Correct'].average()

        # 7) Combine channels into ROIs
        # We'll create new Evoked objects for each condition
        # using e.g. mne.channels.combine_channels
        logging.info("[GoNoGoAnalysisStep] Combining channels into ROIs...")
        roi_dict = {}
        for roi_name, ch_names in rois.items():
            picks = mne.pick_channels(raw.info['ch_names'], include=ch_names)
            roi_dict[roi_name] = picks  # so combine_channels() can find them

        evoked_go_roi   = mne.channels.combine_channels(evoked_go,   roi_dict, method='mean')
        evoked_nogo_roi = mne.channels.combine_channels(evoked_nogo, roi_dict, method='mean')

        # 8) Compare Go vs No-Go in each ROI
        # We'll produce some figures and save them
        self._plot_roi_erps(evoked_go_roi, evoked_nogo_roi, output_dir)

        # 9) Generate final MNE Report
        self._generate_report(output_dir, evoked_go_roi, evoked_nogo_roi)

        logging.info("[GoNoGoAnalysisStep] Done analyzing Go/No-Go.")
        return data
    
    def _merge_go_nogo_events(self, events, event_id):
        """
        Merge pairs: (1 or 2) followed by (3 or 4) into:
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
            resp_evt  = events[i+1]
            onset = onset_evt[2]
            resp  = resp_evt[2]

            # Check if onset in [Go=1, NoGo=2], resp in [Correct=3, Incorrect=4]
            if onset in [event_id['Go'], event_id['NoGo']] and resp in [event_id['Correct'], event_id['Incorrect']]:
                # Go + Correct => 101
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

    def _plot_roi_erps(self, evoked_go_roi, evoked_nogo_roi, output_dir):
        """
        Compare the ROI waveforms for Go vs No-Go.
        Save figures in output_dir.
        """
        logging.info("[GoNoGoAnalysisStep] Plotting ROI ERPs for Go vs. No-Go...")

        # We'll just do a couple of example plots:
        # 1) Combined side-by-side figure
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot Go top subplot
        evoked_go_roi.plot(axes=axes[0], spatial_colors=False, show=False,
            titles=dict(eeg='Go Correct (ROI)'))
        axes[0].set_title("Go Correct - ROI average")

        # Plot No-Go bottom subplot
        evoked_nogo_roi.plot(axes=axes[1], spatial_colors=False, show=False,
            titles=dict(eeg='No-Go Correct (ROI)'))
        axes[1].set_title("No-Go Correct - ROI average")

        fig_path = Path(output_dir) / "roi_erps_go_vs_nogo.png"
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close(fig)
        logging.info(f"[GoNoGoAnalysisStep] ROI ERP figure saved: {fig_path}")

    def _generate_report(self, output_dir, evoked_go_roi, evoked_nogo_roi):
        """
        Create an MNE report that includes final summary plots.
        """
        logging.info("[GoNoGoAnalysisStep] Generating final MNE report for Go/No-Go...")

        report = Report(title="Go/No-Go Analysis Report")

        # Add the ROI ERP figure
        roi_erp_path = Path(output_dir) / "roi_erps_go_vs_nogo.png"
        if roi_erp_path.exists():
            report.add_image(
                image=roi_erp_path,
                title="Go vs NoGo ROI ERPs",
                caption="Comparison of ROI waveforms for Go_Correct and NoGo_Correct"
            )

        # If you want to add a topomap or difference wave
        #   diff = evoked_nogo_roi - evoked_go_roi
        #   create a figure, etc.

        report_fname = Path(output_dir) / "gonogo_analysis_report.html"
        report.save(report_fname, overwrite=True, open_browser=False)
        logging.info(f"[GoNoGoAnalysisStep] Report saved => {report_fname}")
