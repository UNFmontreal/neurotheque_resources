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
                output_path = self.paths.get_derivative_path(stage="after_autoreject")
                full_path = output_path.resolve()
                
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
            all_files = sorted(glob(str(self.paths.raw_data_dir/ file_pattern)))
            print(f"[INFO] Found {len(all_files)} file(s): {all_files}")

            # 3) Process each subject file
            for file_path in all_files:
                print(f"\n=== Processing file: {file_path} ===")
                sub_id, ses_id = self._parse_sub_ses(file_path)
                
                #Look for an "after_autoreject" checkpoint for this subject
                ckpt_path = self.paths.get_checkpoint_path(sub_id, ses_id,checkpoint_name="after_autoreject")
                skip_index = None  # We'll find the index of "AutoRejectStep"

                if ckpt_path.exists():
                    print(f"[INFO] Found existing checkpoint => {ckpt_path}")
                    # Load it
                    try:
                        ckpt_path = ckpt_path.resolve() / f'sub-{sub_id}_ses-{ses_id}_desc-after_autoreject_eeg.fif'
                        self.data = mne.io.read_raw_fif(ckpt_path, preload=True)
                        # If there's an autoreject log
                        
                        log_path = self.paths.get_auto_reject_log_path(sub_id, ses_id) / 'autoreject_log.pickle'   #TODO: Debug f'sub-{sub_id}_ses-{ses_id}_desc-autoreject_log.pickle'
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
            if skip_index is not None and i <= skip_index: #TODO:  you need to skip the save checkpoint step too 
                # skip the step
                continue
            if skip_index is not None and i == skip_index+1:
                # Skip the SaveCheckpoint step
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


