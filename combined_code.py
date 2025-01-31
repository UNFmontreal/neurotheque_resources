

# ==================================================
# FILE: .\combine_py_files.py
# ==================================================

import os
import argparse

def combine_py_files(root_dir, output_file):
    """
    Recursively combines all .py files from a root directory into a single output file.
    Includes the file path as a header before each file's content.
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for root, dirs, files in os.walk(root_dir):
                for file in files:
                    if file.endswith('.py') and file != os.path.basename(output_file):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as infile:
                                content = infile.read()
                                outfile.write(f"\n\n# {'=' * 50}\n")
                                outfile.write(f"# FILE: {file_path}\n")
                                outfile.write(f"# {'=' * 50}\n\n")
                                outfile.write(content)
                        except Exception as e:
                            print(f"Error reading {file_path}: {str(e)}")
        print(f"Successfully combined files to {output_file}")
    except Exception as e:
        print(f"Error creating output file: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine Python files from a directory into a single file')
    parser.add_argument('--root', default='.', help='Root directory to search from')
    parser.add_argument('--output', default='combined_code.py', help='Output filename')
    args = parser.parse_args()
    
    combine_py_files(args.root, args.output)

# ==================================================
# FILE: .\export_repo_structure.py
# ==================================================

import os
import ast

def get_function_names(file_path):
    """Extract top-level function names from a Python file."""
    function_names = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=file_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_names.append(node.name)
    except (SyntaxError, UnicodeDecodeError, Exception):
        pass  # Skip unreadable or non-Python files
    return function_names

def generate_repo_structure(start_path, output_file="repo_structure.txt", exclude_dirs=None):
    """Generate a directory tree with function names in Python files."""
    if exclude_dirs is None:
        exclude_dirs = {".git", "__pycache__", "venv", "env", "node_modules"}

    structure = []
    for root, dirs, files in os.walk(start_path, topdown=True):
        # Exclude unwanted directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        level = root.replace(start_path, "").count(os.sep)
        indent = " " * 4 * level
        structure.append(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 4 * (level + 1)

        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                functions = get_function_names(file_path)
                func_list = f" ({', '.join(functions)})" if functions else ""
                structure.append(f"{subindent}{file}{func_list}")
            else:
                structure.append(f"{subindent}{file}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(structure))

if __name__ == "__main__":
    repo_root = os.getcwd()  # Run from the repository root
    generate_repo_structure(repo_root)

# ==================================================
# FILE: .\scr\pipeline.py
# ==================================================

# File: src/pipeline.py

import yaml
import mne
import pickle
import re
from glob import glob
from pathlib import Path
from scr.steps.project_paths import ProjectPaths
# A global STEP_REGISTRY that maps "step name" -> "step class"
from scr.registery import STEP_REGISTRY
import logging
logging.basicConfig(level=logging.INFO)
class Pipeline:
    """
    A pipeline that executes a list of steps in order.
    Steps can be specified via a YAML file or a Python dict.
    Supports single-subject (original) or multi-subject (new) mode.
    """

    def __init__(self, config_file="config/pipeline_config.yaml", config_dict=None):  
        self.config = self._load_config(config_file, config_dict)
        self.paths = ProjectPaths(self.config)
        
        # Set default subject/session from config
        self.default_subject = self.config["default_subject"]
        self.default_session = self.config["default_session"]
        

    def _load_config(self, config_file, config_dict):
        if config_dict:
            return config_dict
            
        with open(config_file) as f:
            config = yaml.safe_load(f)
            
        # Set absolute paths
        config["project"]["root"] = str(Path(config["project"]["root"]).resolve())
        return config

    def _resolve_checkpoints(self, steps_def):
        """
        Attempt to find a valid checkpoint from a 'SaveCheckpoint' step
        in the pipeline steps (searching from last to first).
        If found, load that .fif file into self.data, then skip steps
        up to that point. 
        (In multi-subject mode, this logic may load only one global checkpoint.)
        """
        for i, step in reversed(list(enumerate(steps_def))):
            if step["name"] == "SaveCheckpoint":
                output_path = Path(step["params"]["output_path"])
                full_path = (self.project_root / output_path).resolve()
                
                if full_path.exists():
                    try:
                        # Load the checkpoint file
                        self.data = mne.io.read_raw_fif(full_path, preload=True)
                        # If there's an autoreject log, load it
                        log_path = full_path.with_name(full_path.stem + "_rejectlog.pkl")
                        if log_path.exists():
                            with open(log_path, "rb") as f:
                                if not hasattr(self.data.info, 'temp') or self.data.info['temp'] is None:
                                    self.data.info['temp'] = {}
                                self.data.info["temp"]["autoreject_log"] = pickle.load(f)
                        print(f"Resuming from checkpoint: {full_path}")
                        return steps_def[i+1:]  # Return steps after the checkpoint
                    except Exception as e:
                        print(f"Error loading checkpoint: {e}")
        return steps_def  # No checkpoint found or load failed

    def run(self):
        """
        Main pipeline entry:
         1) Load config
         2) If multi_subject is False => single-subject approach
         3) If multi_subject is True => discover all subject files, parse sub/ses, 
        """
        config=self.config
        steps_def = config.get("pipeline", {}).get("steps", [])
        if not steps_def:
            raise ValueError("No steps defined under pipeline.steps in config.")

        # Check if multi-subject mode is enabled
        multi_subj = config.get("pipeline", {}).get("multi_subject", False)

        if not multi_subj:
            # === SINGLE-SUBJECT (original logic) ===
            remaining_steps = self._resolve_checkpoints(steps_def.copy())
            for step_info in remaining_steps:
                self._run_step(step_info)
            print("[SUCCESS] Pipeline completed (single subject).")
            return self.data

        else:
            # === MULTI-SUBJECT MODE ===
            # 1) Find the pattern from "LoadData" step
            file_pattern = self._get_file_pattern(steps_def)
            # 2) Glob for matching files
            all_files = sorted(glob(str(self.project_root / file_pattern)))
            print(f"[INFO] Found {len(all_files)} file(s): {all_files}")

            # 3) Process each subject file
            for file_path in all_files:
                print(f"\n=== Processing file: {file_path} ===")
                sub_id, ses_id = self._parse_sub_ses(file_path)
                
                #Look for an "after_autoreject" checkpoint for this subject
                ckpt_path = self.paths.get_checkpoint_file(sub_id, ses_id,checkpoint_key="after_autoreject")
                skip_index = None  # We'll find the index of "AutoRejectStep"

                if ckpt_path.exists():
                    print(f"[INFO] Found existing checkpoint => {ckpt_path}")
                    # Load it
                    try:
                        self.data = mne.io.read_raw_fif(ckpt_path, preload=True)
                        # If there's an autoreject log
                        log_path = ckpt_path.with_name(ckpt_path.stem + "_rejectlog.pkl")
                        if log_path.exists():
                            with open(log_path, "rb") as f:
                                if not hasattr(self.data.info, 'temp') or self.data.info['temp'] is None:
                                    self.data.info['temp'] = {}
                                self.data.info["temp"]["autoreject_log"] = pickle.load(f)
                        print("[INFO] Checkpoint loaded successfully.")

                        # 2) Skip all steps up to and including "AutoRejectStep"
                        skip_index = self._find_step_index(steps_def, "AutoRejectStep")
                        if skip_index is not None:
                            print(f"[INFO] Will skip steps [0..{skip_index}] because checkpoint found.")
                    except Exception as e:
                        print(f"[ERROR] Could not load checkpoint: {e}")
                        self.data = None
                else:
                    print("[INFO] No checkpoint found; starting from scratch.")
                    self.data = None

                # 3) Actually run the steps for this subject, skipping if needed
                self._run_steps(steps_def, skip_index, sub_id, ses_id, file_path)

            print("[SUCCESS] Pipeline completed (multi-subject).")
            return None                


    
    def _run_steps(self, steps_def, skip_index=None, subject_id=None, session_id=None, file_path=None):
        """
        Runs the pipeline steps, optionally skipping steps up to skip_index inclusive.
        If subject_id/session_id are provided, pass them to each step along with 'paths'.
        """
        for i, step_info in enumerate(steps_def):
            if skip_index is not None and i <= skip_index:
                # skip the step
                continue

            step_info.setdefault("params", {})
            # If we have subject context, pass it in
            if subject_id and session_id:
                step_info["params"]["subject_id"] = subject_id
                step_info["params"]["session_id"] = session_id
                step_info["params"]["paths"] = self.paths
            # If we have a file_path for "LoadData"
            if file_path:
                step_info["params"]["input_file"] = file_path

            self._run_step(step_info)

    def _find_step_index(self, steps_def, step_name):
        """
        Returns the index of the named step if found, else None.
        For example, to skip all steps <= index of "AutoRejectStep".
        """
        for i, st in enumerate(steps_def):
            if st["name"] == step_name:
                return i
        return None
    def _get_file_pattern(self, steps_def):
        for st in steps_def:
            if st["name"] == "LoadData":
                return st["params"].get("file_path_pattern", None)
        raise ValueError("No file_path_pattern found in LoadData step for multi-subject.")
    def _parse_sub_ses(self, file_path):
        """
        Extract sub-xx, ses-yy from a filename, e.g. 'sub-01_ses-001_raw.edf' -> ('01','001')
        """
        fname = Path(file_path).name
        sub_match = re.search(r'sub-(\d+)', fname)
        ses_match = re.search(r'ses-(\d+)', fname)
        sub_id = sub_match.group(1) if sub_match else "unknown"
        ses_id = ses_match.group(1) if ses_match else "001"
        return sub_id, ses_id

    def _run_step(self, step_info):
        """Instantiate and execute one pipeline step."""
        step_name = step_info["name"]
        params = step_info.get("params", {})

        if step_name not in STEP_REGISTRY:
            raise ValueError(f"Step '{step_name}' not registered.")
        step_class = STEP_REGISTRY[step_name]
        step = step_class(params)
        self.data = step.run(self.data)




# ==================================================
# FILE: .\scr\registery.py
# ==================================================

# File: scr/registry.py
STEP_REGISTRY = {}

# ==================================================
# FILE: .\scr\__init__.py
# ==================================================



# ==================================================
# FILE: .\scr\steps\autoreject.py
# ==================================================

# File: src/steps/autoreject.py

import logging
import mne
from autoreject import AutoReject
from .base import BaseStep

class AutoRejectStep(BaseStep):
    """
    A professional AutoReject step for EEG pipelines.

    This step can:
      - Create short (1s) epochs from the incoming data (Raw or continuous Epochs).
      - Fit AutoReject on those short epochs to estimate artifact thresholds.
      - Optionally transform (clean) those short epochs and store the reject log.

    Expected params:
    --------------------------------------------------------------------------
    ar_params (dict):
        Dictionary of parameters passed directly to AutoReject's constructor,
        e.g. 'consensus', 'n_interpolate', 'thresh_func', 'n_jobs', etc.
        Example:
          ar_params:
            consensus: 0.5
            n_interpolate: 4
            random_state: 42

    mode (str): "fit" or "fit_transform"
        "fit" (default):
            Fits AutoReject on short epochs but does NOT transform them,
            i.e., no interpolation or epoch dropping is applied.
            Only logs the reject thresholds.
        "fit_transform":
            Additionally calls 'ar.transform(...)' on the ephemeral epochs,
            so the short segments are “cleaned.”
            Potentially you could store the result or the reject_log for future usage.

    store_log (bool):
        If True, stores the reject log in data.info['autoreject_log'] for reference.
        Default is False.

    Usage in YAML:
    --------------------------------------------------------------------------
    pipeline:
      steps:
        - name: AutoRejectStep
          params:
            ar_params:
              consensus: 0.5
              n_interpolate: 5
              random_state: 42
            mode: "transform"
            store_log: true
    """

    def run(self, data):
        if data is None:
            raise ValueError("[AutoRejectStep] No data provided.")

        ar_params = self.params.get("ar_params", {})
        store_log = self.params.get("store_log", False)

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

        logging.info(f"[AutoRejectStep] Initializing AutoReject with params: {ar_params}")
        ar = AutoReject(**ar_params)
        # ar = AutoReject()
        logging.info("[AutoRejectStep] Fitting AutoReject on short epochs.")
        # ar.fit(epochs_tmp)
        # reject_log = ar.get_reject_log(epochs_tmp)
        reject_log = [1,2,3] #for debugging
        logging.info("[AutoRejectStep] AutoReject thresholds:")
        
        sub_id = self.params.get("subject_id", "unknown")
        ses_id = self.params.get("session_id", "001")
        paths = self.params.get("paths", None)
        # fig=reject_log.plot("horizontal",show=False)
        fig_dir = paths.get_autoreject_report_dir(sub_id, ses_id)
        # fig.savefig(fig_path / "autoreject_thresholds.png")
        # 4)  store the reject log in data.info
        if store_log:
            if not hasattr(data.info, 'temp') or data.info['temp'] is None:
                data.info['temp'] = {}
            data.info["temp"]["autoreject_log"] = reject_log  # ← Key fix

        logging.info("[AutoRejectStep] AutoReject finished. "
                     "Thresholds stored in data.info['temp']['autoreject_log']")
        return data


# ==================================================
# FILE: .\scr\steps\base.py
# ==================================================

# src/steps/base.py

from abc import ABC, abstractmethod

class BaseStep(ABC):
    """
    Abstract base class for a pipeline step. Each step must implement run().
    """
    def __init__(self, params=None):
        """
        Initialize the step with parameters.
        """
        self.params = params if params is not None else {}

    @abstractmethod
    def run(self, data):
        """
        Execute this step's logic on the incoming data.

        Parameters
        ----------
        data : object (e.g., mne.io.Raw, mne.Epochs, or None)
            The data object to process.

        Returns
        -------
        object
            The updated data object after processing.
        """
        pass


# ==================================================
# FILE: .\scr\steps\epoching.py
# ==================================================

# src/steps/epoching.py

import mne
from .base import BaseStep

class EpochingStep(BaseStep):
    """
    Step that converts Raw to Epochs based on parsed events.
    """

    def run(self, data):
        """
        Expected params:
        - event_id (dict): e.g. {'Go_Correct': 101, 'NoGo_Correct': 201}
        - tmin (float)
        - tmax (float)
        - baseline (tuple or None)
        """
        if data is None:
            raise ValueError("No data available for epoching.")

        if not hasattr(data.info, "parsed_events") and "parsed_events" not in data.info:
            # fallback to find_events or raise an error
            raise ValueError("No parsed_events found in data.info. Did you run TriggerParsingStep?")

        events = data.info["parsed_events"]
        event_id = self.params.get("event_id", {})
        tmin = self.params.get("tmin", -0.2)
        tmax = self.params.get("tmax", 0.8)
        baseline = self.params.get("baseline", (None, 0))

        epochs = mne.Epochs(data, events, event_id=event_id,
                            tmin=tmin, tmax=tmax, baseline=baseline,
                            preload=True)
        return epochs


# ==================================================
# FILE: .\scr\steps\epoching_gonogo.py
# ==================================================

# File: eeg_pipeline/src/steps/epoching_gonogo.py

import mne
from .base import BaseStep

class GoNoGoEpochingStep(BaseStep):
    """
    Epochs only correct responses (101, 201),
    tmin=-0.2, tmax=0.8 by default.
    """

    def run(self, data):
        if data is None:
            raise ValueError("No data in GoNoGoEpochingStep.")

        # We rely on data.info['parsed_events'] from GoNoGoTriggerStep
        if 'parsed_events' not in data.info:
            raise ValueError("No 'parsed_events' found. Run triggers step first?")

        events = data.info['parsed_events']
        event_id = {
            'Go_Correct': 101,
            'NoGo_Correct': 201
        }
        tmin = self.params.get("tmin", -0.2)
        tmax = self.params.get("tmax", 0.8)
        baseline = self.params.get("baseline", (None, 0))

        epochs = mne.Epochs(
            data, events, event_id=event_id,
            tmin=tmin, tmax=tmax,
            baseline=baseline, preload=True
        )
        return epochs


# ==================================================
# FILE: .\scr\steps\filter.py
# ==================================================

# src/steps/filter.py

from .base import BaseStep

class FilterStep(BaseStep):
    """
    Step to apply bandpass and/or notch filters on an mne.Raw or mne.Epochs object.
    """

    def run(self, data):
        """
        Expected params:
        - l_freq (float): low cutoff frequency
        - h_freq (float): high cutoff frequency
        - notch_freqs (list): list of freqs to notch out
        """
        if data is None:
            raise ValueError("No data available to filter.")

        l_freq = self.params.get("l_freq", 1.0)
        h_freq = self.params.get("h_freq", 40.0)
        notch_freqs = self.params.get("notch_freqs", [])

        data.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin")

        if notch_freqs:
            data.notch_filter(freqs=notch_freqs, fir_design="firwin")

        return data


# ==================================================
# FILE: .\scr\steps\gonogo_analysis.py
# ==================================================

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


# ==================================================
# FILE: .\scr\steps\gonogo_analysis_v2.py
# ==================================================

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


# ==================================================
# FILE: .\scr\steps\ica.py
# ==================================================

# File: scr/steps/ica.py

import logging
import mne
from mne.preprocessing import ICA
from mne.report import Report
from pathlib import Path
from .base import BaseStep

class ICAStep(BaseStep):
    """
    ICA step that logs suggested bad components (EOG/ECG) but does not 
    automatically exclude them. The user can then inspect those suggestions 
    plus the component plots to make a final decision.

    References:
      - Chaumon et al. (2015), J Neurosci Methods
      - Winkler et al. (2015), NeuroImage
      - DSI-24 Technical Specs
      - ADJUST, MARA, etc. for advanced automated IC classification
    """

    def run(self, data):
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

        # --------------------------
        # 2) Instantiate ICA
        # --------------------------
        ica = ICA(
            n_components=params["n_components"],
            method=params["method"],
            max_iter=params["max_iter"],
            fit_params=params["fit_params"],
            random_state=0
        )
        # ica = ICA(
        #     n_components=params["n_components"],
        #     method=params["method"],
        #     random_state=0
        # )
        # --------------------------
        # 3) Select Data for ICA
        # --------------------------
        if params["use_good_epochs_only"] and "autoreject_log" in data.info.get("temp", {}):
            logging.info("[ICAStep] Using only good epochs from AutoReject.")
            reject_log = data.info["temp"]["autoreject_log"]
            events = mne.make_fixed_length_events(data, duration=1)
            epochs = mne.Epochs(data, events, tmin=0, tmax=1, baseline=None,preload=True)
            good_mask = ~reject_log.bad_epochs
            good_epochs = epochs[good_mask] if len(good_mask) == len(epochs) else epochs
        else:
            logging.info("[ICAStep] No (or unused) AutoReject log; using all data for ICA.")
            good_epochs = data

        # --------------------------
        # 4) Fit ICA
        # --------------------------
        # ica.fit(
        #     good_epochs,
        #     decim=params["decim"],
        #     reject=None,
        #     tstep=4.0
        # )
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

        # (Optional) Additional metrics go here; for example:
        # - ADJUST
        # - MARA
        # - FASTER
        # Just store them in bad_ic_candidates in a similar fashion.

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
            # For example, you might unify all candidate indices to plot them in one go:
            union_candidates = set()
            for entry in bad_ic_candidates:
                union_candidates.update(entry["indices"])

            # If you want to see topographies of the candidate ICs (union of EOG, ECG, etc.):
            if union_candidates:
                ica.plot_properties(good_epochs, picks=sorted(list(union_candidates)))

            # Also show any pre-excluded from user params if you like
            if final_exclude:
                ica.plot_properties(good_epochs, picks=sorted(list(final_exclude)))

            # Now prompt the user
            suggested_str = (
                f"\nSuggested bad ICs from detection: {union_candidates}" 
                if union_candidates else "None"
            )
            print(suggested_str)

            # The user can override or add anything:
            user_input = input("Enter ALL IC indices to exclude (comma-separated), or press Enter to skip: ")
            if user_input.strip():
                # Overwrite final_exclude with user input
                final_exclude = set(int(x) for x in user_input.split(","))
        else:
            logging.info("[ICAStep] Non-interactive mode. Using only 'exclude' param from YAML.")

        # Assign to ica.exclude
        ica.exclude = sorted(list(final_exclude))
        logging.info(f"[ICAStep] Final exclusion list: {ica.exclude}")

        # --------------------------
        # 8) Apply ICA
        # --------------------------
        data_clean = ica.apply(data.copy())

        # --------------------------
        # 9) Generate QA Report
        # --------------------------
        self._generate_report(ica, data_clean, params)

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

    def _generate_report(self, ica, data, params):
        """Create an MNE Report summarizing ICA."""
        from mne.report import Report
        import matplotlib.pyplot as plt
        
        out_path = Path(params["plot_dir"])
        out_path.mkdir(parents=True, exist_ok=True)

        report = Report(title="ICA Quality Report", verbose=False)

        report.add_ica(
            ica=ica,
            title="ICA cleaning",
            picks=None,  # plot the excluded EOG components
            inst=data,
            n_jobs=None,  # could be increased!
        )
        out_file = out_path / "ica_report.html"
        report.save(out_file, overwrite=True, open_browser=False)
        logging.info(f"[ICAStep] ICA report saved at {out_file}")

# ==================================================
# FILE: .\scr\steps\load.py
# ==================================================

import mne
from pathlib import Path
from .base import BaseStep

class LoadData(BaseStep):
    """
        Step to load EEG data from a file. 
        If multi-subject, we rely on subject_id, session_id, 
        but we might also just read input_file directly if it was set.
    """

    def run(self, data):

        sub_id = self.params.get("subject_id", None)
        ses_id = self.params.get("session_id", None)
        paths = self.params.get("paths", None)

        file_path = self.params.get("input_file", None)

        if not file_path:
            # If we wanted to auto-generate the path from ProjectPaths:
            if sub_id and ses_id and paths:
                file_path = paths.get_raw_input_file(sub_id, ses_id)
            else:
                raise ValueError("[LoadData] No input_file or subject/session set.")

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        stim_channel = self.params.get("stim_channel", "Trigger")

        if file_path.suffix == ".edf":
            stim_channel = self.params.get("stim_channel", "Trigger")
            raw = mne.io.read_raw_edf(file_path, preload=True, stim_channel=stim_channel)
        elif file_path.suffix == ".fif":
            raw = mne.io.read_raw_fif(file_path, preload=True)
        else:
            raise ValueError(f"Unsupported file extension: {file_path.suffix}")

        return raw


# ==================================================
# FILE: .\scr\steps\prepchannels.py
# ==================================================

# src/steps/prepchannels.py

import logging
from mne.channels import make_standard_montage
from .base import BaseStep

class PrepChannelsStep(BaseStep):
    """
    Drops non-EEG channels, renames them, sets channel types & montage.
    Mirrors the notebook's channel preparation logic.
    """
    def run(self, data):
        if data is None:
            raise ValueError("[PrepChannelsStep] No data to process.")

        # 1) Drop non-EEG channels
        non_eeg_channels = [
            'EEG X1:ECG-Pz', 'EEG X2:-Pz', 'EEG X3:-Pz',
            'CM', 'EEG A1-Pz', 'EEG A2-Pz'
        ]
        existing_non_eeg = [ch for ch in non_eeg_channels if ch in data.info['ch_names']]
        data.drop_channels(existing_non_eeg)
        logging.info(f"Dropped non-EEG: {existing_non_eeg}")

        # 2) Rename EEG channels
        eeg_channels = [ch for ch in data.info['ch_names'] if 'EEG ' in ch]
        rename_mapping = {ch: ch.replace('EEG ', '').replace('-Pz', '') for ch in eeg_channels}
        data.rename_channels(rename_mapping)

        # 3) Set channel types
        for ch in data.info['ch_names']:
            if ch in rename_mapping.values():
                data.set_channel_types({ch: 'eeg'})
            elif 'Trigger' in ch:
                data.set_channel_types({ch: 'stim'})
            else:
                data.set_channel_types({ch: 'misc'})

        # 4) Montage
        montage = make_standard_montage('standard_1020')
        data.set_montage(montage)

        logging.info("[PrepChannelsStep] Channels prepared successfully.")
        return data


# ==================================================
# FILE: .\scr\steps\project_paths.py
# ==================================================

# File: scr/project_paths.py

from pathlib import Path

class ProjectPaths:
    """
    Central class for all file and directory paths.
    Each method here returns the *final* path for reading/writing,
    based on subject/session and your folder conventions.
    """

    def __init__(self, base_dir,raw_data_dir='data/raw',reports_dir='reports'):
        self.base_dir = Path(base_dir).resolve()
        self.raw_data_dir = self.base_dir / raw_data_dir
        self.reports_dir = self.base_dir / reports_dir
        
    def validate_subject_session(self, subject_id, session_id):
        """Ensures valid BIDS-style identifiers"""
        if not isinstance(subject_id, str) or not subject_id.startswith("sub-"):
            raise ValueError(f"Invalid subject_id format: {subject_id}")
        if not isinstance(session_id, str) or not session_id.startswith("ses-"):
            raise ValueError(f"Invalid session_id format: {session_id}")

    def get_raw_input_path(self, subject_id, session_id):
        """BIDS-compliant raw data path"""
        self.validate_subject_session(subject_id, session_id)
        return (
            self.raw_data_dir
            / subject_id
            / session_id
            / "eeg"
            / f"{subject_id}_{session_id}_task-rest_eeg.edf"
        )

    def get_checkpoint_path(self, subject_id, session_id, checkpoint_name):
        """Standardized checkpoint pathing"""
        self.validate_subject_session(subject_id, session_id)
        path = (
            self.raw_data_dir
            / subject_id
            / session_id
            / "derivatives"
            / f"{subject_id}_{session_id}_desc-{checkpoint_name}_eeg.fif"
        )
        self.ensure_parent(path)
        return path

    def get_autoreject_report_dir(self, subject_id, session_id):
        """Standardized report directory with auto-creation"""
        self.validate_subject_session(subject_id, session_id)
        path = (
            self.reports_dir
            / "autoreject"
            / subject_id
            / session_id
        )
        self.ensure_parent(path)
        return path
        
    def get_ica_report_dir(self, subject_id, session_id):
        path = (
            self.reports_dir
            / "ica"
            / subject_id
            / session_id
        )
        self.ensure_parent(path)
        return path

    def ensure_parent(self, path: Path):
        """Convenience method to create the parent folder if missing."""
        path.parent.mkdir(parents=True, exist_ok=True)


# ==================================================
# FILE: .\scr\steps\reference.py
# ==================================================

# File: src/steps/reference.py

import logging
from .base import BaseStep
import mne

class ReferenceStep(BaseStep):
    """
    Re-references EEG data according to user-specified method.

    Expected params (all optional, with defaults):
    --------------------------------------------------------------------------
    method : str
        Either "average" for average reference (default) or "channels" for
        custom reference channels.

    channels : list of str
        Which channel(s) to use if method="channels". If None or empty, we error.

    projection : bool
        If True, add a projection to do the re-reference rather than directly
        modifying data. (default: False)

    Example usage in YAML:
    --------------------------------------------------------------------------
    pipeline:
      steps:
        - name: ReferenceStep
          params:
            method: "channels"
            channels: ["TP9", "TP10"]
            projection: false
    """

    def run(self, data):
        if data is None:
            raise ValueError("[ReferenceStep] No data to re-reference.")

        # Get params
        method = self.params.get("method", "average")
        channels = self.params.get("channels", [])
        projection = self.params.get("projection", False)

        logging.info(f"[ReferenceStep] Re-referencing method={method}, projection={projection}")

        if method == "average":
            # set_eeg_reference(ref_channels="average") => average re-ref
            logging.info("[ReferenceStep] Using average reference for EEG channels.")
            data.set_eeg_reference(ref_channels="average", projection=projection)

        elif method == "channels":
            if not channels:
                raise ValueError("[ReferenceStep] method='channels' requires 'channels' param.")
            logging.info(f"[ReferenceStep] Using custom channels {channels} for reference.")
            data.set_eeg_reference(ref_channels=channels, projection=projection)

        else:
            raise ValueError(f"[ReferenceStep] Unknown re-reference method '{method}'.")

        # MNE might add new reference channels to the data if channels were used.
        # If projection=True, we have an EEG ref projection added but not applied
        # until you do e.g. data.apply_proj().

        logging.info("[ReferenceStep] Re-reference complete.")
        return data


# ==================================================
# FILE: .\scr\steps\save.py
# ==================================================

# File: src/steps/save.py

import logging
from pathlib import Path
from .base import BaseStep
import pickle
class SaveData(BaseStep):
    """
    Step to save the data as an 'after_autoreject' checkpoint. 
    That means next run can skip 'AutoRejectStep' if the file is found.
    
    Example YAML usage:
    ------------------
    - name: SaveCheckpoint
      params:
        checkpoint_key: "after_autoreject"  # or "post_ica", etc.
    """

    def run(self, data):

        if data is None:
            raise ValueError("[SaveData] No data to save.")

        
        sub_id = self.params["subject_id"]
        ses_id = self.params["session_id"]
        paths = self.params["paths"]
        
        # We'll default to "after_autoreject" if not specified

    # some fallback or raise an error  
        
        # If "after_autoreject", we typically do "paths.get_autoreject_checkpoint(...)"
        # but you might have multiple keys => define more in ProjectPaths if needed
        # if ckpt_key == "after_autoreject":
        #     ckpt_path = paths.get_autoreject_checkpoint(sub_id, ses_id)
        # elif ckpt_key == "after_ica":
        #     ckpt_path = paths.get_ica_checkpoint(sub_id, ses_id)
        # else:
        
        ckpt_path = paths.get_autoreject_checkpoint(sub_id, ses_id)
        paths.ensure_parent(ckpt_path) 
        data.save(str(ckpt_path), overwrite=True)   
        
        
        # If there's an autoreject log, store it in a separate .pkl
        if "temp" in data.info and "autoreject_log" in data.info["temp"]:
            log_path = ckpt_path.with_name(ckpt_path.stem + "_rejectlog.pkl")
            with open(log_path, "wb") as f:
                pickle.dump(data.info["temp"]["autoreject_log"], f)

        print(f"[SaveCheckpoint] Saved => {ckpt_path}")
        return data


# ==================================================
# FILE: .\scr\steps\save_checkpoint.py
# ==================================================

# src/steps/save_checkpoint.py
import mne
from pathlib import Path
from .base import BaseStep
import pickle

class SaveCheckpoint(BaseStep):
    def run(self, data):
        output_path = Path(self.params["output_path"])
        project_root = Path(__file__).resolve().parent.parent.parent
        
        # Save Raw data
        raw_path = project_root / output_path
        data.save(raw_path, overwrite=True)
        
        # Save reject_log separately if it exists
        if "autoreject_log" in data.info.get("temp", {}):
            log_path = raw_path.with_name(raw_path.stem + "_rejectlog.pkl")
            with open(log_path, "wb") as f:
                pickle.dump(data.info["temp"]["autoreject_log"], f)
        
        return data

# ==================================================
# FILE: .\scr\steps\splittasks.py
# ==================================================

# src/steps/splittasks.py

import os
import logging
import mne
from .base import BaseStep

class SplitTasksStep(BaseStep):
    """
    Finds triggers for tasks: Rest_GoNoGo, GoNoGo, LandoitC, MentalImagery.
    Then crops each segment and saves as a separate .fif file (e.g. 'GonoGo.fif').
    """

    def run(self, data):
        if data is None:
            raise ValueError("[SplitTasksStep] No data to split.")

        output_folder = self.params.get("output_folder")
        if not output_folder:
            raise ValueError("[SplitTasksStep] 'output_folder' param is required.")

        os.makedirs(output_folder, exist_ok=True)

        events = mne.find_events(data, stim_channel='Trigger',
                                 min_duration=0.001, consecutive=False)
        logging.info(f"[SplitTasksStep] Found {len(events)} events.")

        # Build a dict of trigger -> [samples]
        trigger_dict = {}
        for samp, _, trig in events:
            trigger_dict.setdefault(trig, []).append(samp)

        task_periods = {
            'Rest_GoNoGo': {'start': None, 'end': None},
            'GoNoGo': {'start': None, 'end': None},
            'LandoitC': {'start': None, 'end': None},
            'MentalImagery': {'start': None, 'end': None}
        }

        _define_task_periods(task_periods, trigger_dict, data.info['sfreq'])

        # Crop each segment, save
        for task_name, period in task_periods.items():
            s = period['start']
            e = period['end']
            if s is None or e is None:
                logging.warning(f"[SplitTasksStep] {task_name} period is None; skipping.")
                continue
            tmin = s / data.info['sfreq']
            tmax = e / data.info['sfreq']
            sub_raw = data.copy().crop(tmin=tmin, tmax=tmax)

            save_path = os.path.join(output_folder, f"{task_name}.fif")
            sub_raw.save(save_path, overwrite=True)
            logging.info(f"[SplitTasksStep] Saved {task_name} => {save_path}")

        return data


def _define_task_periods(task_periods, trigger_dict, sfreq):
    """
    The same logic as in your notebook for:
      6->7 => Rest_GoNoGo
      7->(8,9) => GoNoGo
      second 8,9 => MentalImagery
      14 min after gonogo end => LandoitC
    etc.
    """

    import math

    def minutes_to_samples(minutes):
        return int(minutes * 60 * sfreq)

    # 1) Rest_GoNoGo
    if 6 in trigger_dict and 7 in trigger_dict:
        rest_start = trigger_dict[6][0]
        rest_end   = trigger_dict[7][0]
        task_periods['Rest_GoNoGo']['start'] = rest_start
        task_periods['Rest_GoNoGo']['end']   = rest_end
    else:
        logging.error("Missing triggers 6,7 for Rest_GoNoGo")

    # 2) GoNoGo: from 7->(8,9)
    if 7 in trigger_dict and 8 in trigger_dict and 9 in trigger_dict:
        gonogo_start = trigger_dict[7][0]
        gonogo_end   = trigger_dict[9][0]
        task_periods['GoNoGo']['start'] = gonogo_start
        task_periods['GoNoGo']['end']   = gonogo_end
    else:
        logging.error("Missing triggers (7,8,9) for GoNoGo")

    # 3) Mental Imagery: second 8, second 9
    if len(trigger_dict.get(8, [])) >= 2 and len(trigger_dict.get(9, [])) >= 2:
        mi_start = trigger_dict[8][-1]
        mi_end   = trigger_dict[9][-1]
        task_periods['MentalImagery']['start'] = mi_start
        task_periods['MentalImagery']['end']   = mi_end
    else:
        logging.error("Missing second triggers 8,9 for Mental Imagery")

    # 4) LandoitC
    go_no_go_end = task_periods['GoNoGo']['end']
    if go_no_go_end is not None:
        start = go_no_go_end + minutes_to_samples(14)
        rest_starts = trigger_dict.get(6, [])
        if len(rest_starts) >= 2:
            # second rest start
            mental_imagery_rest_start = rest_starts[-1]
            end = mental_imagery_rest_start - minutes_to_samples(1)
            task_periods['LandoitC']['start'] = start
            task_periods['LandoitC']['end']   = end
        else:
            logging.error("Second rest (trigger 6) for LandoitC not found.")
    else:
        logging.error("Cannot define LandoitC: no GoNoGo end found.")


# ==================================================
# FILE: .\scr\steps\splittasks_dynamic.py
# ==================================================

# File: src/steps/splittasks.py

import logging
import os
import mne
from pathlib import Path
from .base import BaseStep

class SplitTasksStep(BaseStep):
    """
    A flexible step to split Raw data into multiple tasks based on triggers,
    offsets, or references to previously-defined tasks.

    Example YAML snippet:
      - name: SplitTasksStep
        params:
          output_folder: "data/pilot_data/tasks"
          tasks:
            - name: "GoNoGo"
              start_trigger: 7
              end_trigger: 9
            - name: "Rest"
              start_trigger: 6
              end_trigger: 7

    The pipeline will rewrite `output_folder` if multi-subject mode is used,
    so each subject ends up with:
      data/pilot_data/tasks/sub-01/ses-001/GoNoGo.fif
      data/pilot_data/tasks/sub-01/ses-001/Rest.fif
    etc.
    """

    def run(self, data):
        if data is None:
            raise ValueError("[SplitTasksStep] No data to split.")

        # 1) Ensure we have an output folder (subject-specific if pipeline rewrote it)
        output_folder = self.params.get("output_folder")
        if not output_folder:
            raise ValueError("[SplitTasksStep] 'output_folder' param is required.")
        out_dir = Path(output_folder)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 2) Get tasks
        tasks = self.params.get("tasks", [])
        if not tasks:
            logging.warning("[SplitTasksStep] No tasks defined. Doing nothing.")
            return data

        # 3) Find triggers
        events = mne.find_events(data, stim_channel='Trigger', min_duration=0.001, consecutive=False)
        logging.info(f"[SplitTasksStep] Found {len(events)} events in the data.")

        # Build a dictionary of {trigger_value -> list of sample indices}
        trigger_dict = {}
        for samp, _, trig_val in events:
            trigger_dict.setdefault(trig_val, []).append(samp)

        sfreq = data.info['sfreq']

        # We'll store start/end sample for each completed task
        task_segments = {}

        # 4) Iterate over tasks in order
        for task_def in tasks:
            task_name = task_def["name"]
            start_sample, end_sample = self._find_task_segment(
                task_def, trigger_dict, sfreq, task_segments
            )

            if start_sample is None or end_sample is None:
                logging.warning(f"[SplitTasksStep] {task_name}: Could not define segment. Skipping.")
                continue

            if start_sample >= end_sample:
                logging.warning(f"[SplitTasksStep] {task_name}: start >= end => Skipping.")
                continue

            # Crop the raw data for this task
            tmin = start_sample / sfreq
            tmax = end_sample / sfreq
            sub_raw = data.copy().crop(tmin=tmin, tmax=tmax)

            # 5) Save to disk
            save_path = out_dir / f"{task_name}.fif"
            sub_raw.save(str(save_path), overwrite=True)
            logging.info(f"[SplitTasksStep] Saved {task_name} => {save_path}")

            # Store for reference by subsequent tasks
            task_segments[task_name] = {"start": start_sample, "end": end_sample}

        return data

    def _find_task_segment(self, task_def, trigger_dict, sfreq, task_segments):
        """
        Compute (start_sample, end_sample) for a given task definition.
        The definition can contain:
          - start_trigger (int)
          - end_trigger (int)
          - occurrence_index (int) [default=1 => first occurrence]
          - start_after_task (str) + start_offset_minutes (float)
          - end_before_trigger (int) + end_offset_minutes (float)
          etc.

        Return (None, None) if we cannot find a valid segment.
        """
        task_name = task_def["name"]
        occurrence_index = task_def.get("occurrence_index", 1) - 1

        start_sample = None
        end_sample = None

        # start_trigger logic
        if "start_trigger" in task_def:
            st_trig = task_def["start_trigger"]
            if st_trig in trigger_dict and len(trigger_dict[st_trig]) > occurrence_index:
                start_sample = trigger_dict[st_trig][occurrence_index]
            else:
                logging.error(f"{task_name}: Missing or insufficient occurrences of start_trigger={st_trig}.")
                return (None, None)

        # start_after_task logic
        if "start_after_task" in task_def:
            ref_task = task_def["start_after_task"]
            if ref_task not in task_segments:
                logging.error(f"{task_name}: The referenced task '{ref_task}' isn't defined yet.")
                return (None, None)
            ref_end = task_segments[ref_task]["end"]
            offset_mins = task_def.get("start_offset_minutes", 0)
            offset_samps = int(offset_mins * 60 * sfreq)
            candidate_start = ref_end + offset_samps
            if start_sample is None:
                start_sample = candidate_start
            else:
                start_sample = max(start_sample, candidate_start)

        # end_trigger logic
        if "end_trigger" in task_def:
            end_trig = task_def["end_trigger"]
            if end_trig in trigger_dict and len(trigger_dict[end_trig]) > occurrence_index:
                end_sample = trigger_dict[end_trig][occurrence_index]
            else:
                logging.error(f"{task_name}: Missing or insufficient occurrences of end_trigger={end_trig}.")
                return (None, None)

        # end_before_trigger logic
        if "end_before_trigger" in task_def:
            ebt = task_def["end_before_trigger"]
            ebt_occ_idx = task_def.get("end_before_occurrence_index", 1) - 1
            offset_mins = task_def.get("end_offset_minutes", 0)
            offset_samps = int(offset_mins * 60 * sfreq)

            if ebt in trigger_dict and len(trigger_dict[ebt]) > ebt_occ_idx:
                ebt_samp = trigger_dict[ebt][ebt_occ_idx]
                ebt_samp -= offset_samps
                if end_sample is None:
                    end_sample = ebt_samp
                else:
                    end_sample = min(end_sample, ebt_samp)
            else:
                logging.error(f"{task_name}: Missing end_before_trigger {ebt} or not enough occurrences.")
                return (None, None)

        # Return the final start/end
        return (start_sample, end_sample)


# ==================================================
# FILE: .\scr\steps\triggers.py
# ==================================================

# src/steps/triggers.py

import mne
import numpy as np
from .base import BaseStep

class TriggerParsingStep(BaseStep):
    """
    Step that finds events in raw data and applies custom parsing logic
    depending on the 'task' parameter (gonogo, finger_tapping, etc.).
    """

    def run(self, data):
        """
        Expected params:
        - stim_channel (str): e.g. 'Trigger'
        - task (str): e.g. 'gonogo', 'finger_tapping', etc.

        The parsed events are stored in data.info['parsed_events'] (or we can just store them in step).
        """
        if data is None:
            raise ValueError("No data available for trigger parsing.")

        stim_channel = self.params.get("stim_channel", "Trigger")
        task = self.params.get("task", "gonogo")

        events = mne.find_events(data, stim_channel=stim_channel, consecutive=True, min_duration=0.001)

        if task == "gonogo":
            parsed_events = self._parse_gonogo(events)
        elif task == "finger_tapping":
            parsed_events = self._parse_finger_tapping(events)
        elif task == "mental_imagery":
            parsed_events = self._parse_mental_imagery(events)
        else:
            # Fallback or user logic
            parsed_events = events

        # Store parsed events in the info dict or return them
        data.info["parsed_events"] = parsed_events
        return data

    def _parse_gonogo(self, events):
        """
        Example of combining triggers for go/nogo correctness.
        """
        # skeleton logic
        new_events = []
        i = 0
        while i < len(events)-1:
            onset = events[i][2]
            resp = events[i+1][2]
            # ... your go/nogo logic ...
            i += 1
        return np.array(new_events)

    def _parse_finger_tapping(self, events):
        """
        Example logic for beep vs mario triggers, key presses, etc.
        """
        # ...
        return events

    def _parse_mental_imagery(self, events):
        """
        Example logic for 30s blocks, etc.
        """
        # ...
        return events


# ==================================================
# FILE: .\scr\steps\triggers_gonogo.py
# ==================================================

# File: eeg_pipeline/src/steps/triggers_gonogo.py

import mne
import numpy as np
from .base import BaseStep

class GoNoGoTriggerStep(BaseStep):
    """
    Finds events (onset=1,2 / response=3,4),
    merges them into new events for Go_Correct, Go_Incorrect, NoGo_Correct, NoGo_Incorrect.
    Stores them in data.info['parsed_events'].
    """

    def run(self, data):
        if data is None:
            raise ValueError("No data in GoNoGoTriggerStep.")

        stim_channel = self.params.get("stim_channel", "Trigger")
        events = mne.find_events(data, stim_channel=stim_channel, min_duration=0.01)

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
            if onset in [1, 2] and resp in [3, 4]:
                if onset == 1 and resp == 3:
                    new_events.append([onset_evt[0], 0, new_event_id['Go_Correct']])
                elif onset == 1 and resp == 4:
                    new_events.append([onset_evt[0], 0, new_event_id['Go_Incorrect']])
                elif onset == 2 and resp == 3:
                    new_events.append([onset_evt[0], 0, new_event_id['NoGo_Correct']])
                elif onset == 2 and resp == 4:
                    new_events.append([onset_evt[0], 0, new_event_id['NoGo_Incorrect']])
                i += 2
            else:
                i += 1

        new_events = np.array(new_events)
        data.info["parsed_events"] = new_events

        return data


# ==================================================
# FILE: .\scr\steps\__init__.py
# ==================================================

# File: eeg_pipeline/src/steps/__init__.py
"""
Initialization file for the steps package.

This file imports and registers all step classes in the STEP_REGISTRY so
that the pipeline can reference them by name without extra imports.
"""

import logging

from scr.registery import STEP_REGISTRY

# Import each step class:
from .base import BaseStep
from .load import LoadData
from .filter import FilterStep
from .ica import ICAStep
from .autoreject import AutoRejectStep
from .save import SaveData
from .reference import ReferenceStep
from .save_checkpoint import SaveCheckpoint

# If you have additional steps:
from .prepchannels import PrepChannelsStep
# from .splittasks import SplitTasksStep
# etc...    
from .splittasks_dynamic import SplitTasksStep
# If you have specialized steps for analysis:
try:
    from .triggers_gonogo import GoNoGoTriggerStep
    from .epoching_gonogo import GoNoGoEpochingStep
except ImportError:
    logging.warning("GoNoGo specialized steps not found. Skipping...")


# Register them in the global STEP_REGISTRY
STEP_REGISTRY.update({
    "LoadData": LoadData,
    "ReferenceStep": ReferenceStep,
    "FilterStep": FilterStep,
    "ICAStep": ICAStep,
    "AutoRejectStep": AutoRejectStep,
    "SaveCheckpoint": SaveCheckpoint,
    "SaveData": SaveData,
    "PrepChannelsStep": PrepChannelsStep,
    "SplitTasksStep": SplitTasksStep,
    # If you have them:
    "GoNoGoTriggerStep": GoNoGoTriggerStep,
    "GoNoGoEpochingStep": GoNoGoEpochingStep,
})

logging.info("[__init__.py] All step classes have been registered in STEP_REGISTRY.")


# ==================================================
# FILE: .\scr\strategies\finger_tapping_strategy.py
# ==================================================

# src/strategies/finger_tapping_strategy.py

from src.pipeline import Pipeline

def run_finger_tapping_pipeline(input_path, output_path):
    """
    Example pipeline for finger tapping experiment.
    """
    config_dict = {
        "pipeline": {
            "steps": [
                {
                    "name": "LoadData",
                    "params": {"file_path": input_path}
                },
                {
                    "name": "Filter",
                    "params": {"l_freq": 1, "h_freq": 40}
                },
                {
                    "name": "TriggerParsing",
                    "params": {"task": "finger_tapping"}
                },
                # Possibly do TFR in a specialized step if you like,
                # or epoching for key presses
                {
                    "name": "SaveData",
                    "params": {"output_path": output_path}
                }
            ]
        }
    }
    pipeline = Pipeline(config_dict=config_dict)
    pipeline.run()


# ==================================================
# FILE: .\scr\strategies\mental_imagery_strategy.py
# ==================================================

# src/strategies/mental_imagery_strategy.py

from src.pipeline import Pipeline

def run_mental_imagery_pipeline(input_path, output_path):
    """
    Example pipeline for mental imagery tasks.
    """
    config_dict = {
        "pipeline": {
            "steps": [
                {
                    "name": "LoadData",
                    "params": {"file_path": input_path}
                },
                {
                    "name": "Filter",
                    "params": {"l_freq": 1, "h_freq": 100, "notch_freqs": [60, 120]}
                },
                {
                    "name": "TriggerParsing",
                    "params": {"task": "mental_imagery"}
                },
                {
                    "name": "Epoching",
                    "params": {
                        "event_id": {
                            "Iright_stimulus": 1,
                            "Ileft_stimulus": 2,
                            "right_stimulus": 3,
                            "left_stimulus": 4
                        },
                        "tmin": -5,
                        "tmax": 30,
                        "baseline": [None, 0]
                    }
                },
                {
                    "name": "SaveData",
                    "params": {"output_path": output_path}
                }
            ]
        }
    }
    pipeline = Pipeline(config_dict=config_dict)
    pipeline.run()


# ==================================================
# FILE: .\scr\strategies\__init__.py
# ==================================================

