# File: src/pipeline.py

import yaml
import mne
import pickle
import re
from glob import glob
from pathlib import Path

# A global STEP_REGISTRY that maps "step name" -> "step class"
STEP_REGISTRY = {}

class Pipeline:
    """
    A pipeline that executes a list of steps in order.
    Steps can be specified via a YAML file or a Python dict.
    Supports single-subject (original) or multi-subject (new) mode.
    """

    def __init__(self, config_file=None, config_dict=None):
        """
        You can either pass a YAML file path or a dict directly.
        """
        self.config_file = config_file
        self.config_dict = config_dict
        self.data = None  # Holds mne.Raw or mne.Epochs throughout pipeline
        self.project_root = self._get_project_root()

    def _get_project_root(self):
        """Calculate project root based on this file's location."""
        current_file = Path(__file__).resolve()
        # e.g. pipeline.py is in src/, so project root might be parent of src
        return current_file.parent.parent

    def _load_config(self):
        """Load YAML or dictionary config."""
        if self.config_dict is not None:
            return self.config_dict
        elif self.config_file:
            path = Path(self.config_file)
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {path}")
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        else:
            raise ValueError("No config file or dict provided.")

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
         3) If multi_subject is True => discover all subject files, parse sub/ses, rewrite paths, run steps.
        """
        config = self._load_config()
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
            file_pattern = None
            for st in steps_def:
                if st["name"] == "LoadData":
                    file_pattern = st["params"].get("file_path_pattern", None)
                    break
            if not file_pattern:
                raise ValueError("No 'file_path_pattern' found under LoadData step in multi_subject mode.")

            # 2) Glob for matching files
            all_files = sorted(glob(str(self.project_root / file_pattern)))
            if not all_files:
                print(f"[WARNING] No files found for pattern: {file_pattern}")
                return None

            print(f"[INFO] Found {len(all_files)} file(s): {all_files}")

            # 3) Process each subject file
            for file_path in all_files:
                print(f"\n=== Processing file: {file_path} ===")
                sub_id, ses_id = self._parse_sub_ses(file_path)

                # Reload steps each iteration
                steps_def_copy = steps_def.copy()
                remaining_steps = self._resolve_checkpoints(steps_def_copy)

                # Reset data = None for each subject run
                self.data = None

                for step_info in remaining_steps:
                    step_info.setdefault("params", {})

                    # Provide subject info
                    step_info["params"]["subject_id"] = sub_id
                    step_info["params"]["session_id"] = ses_id
                    step_info["params"]["input_file"] = file_path

                    # If a step has output_path or plot_dir, rewrite it
                    if "output_path" in step_info["params"]:
                        step_info["params"]["output_path"] = self._adjust_output_path(
                            step_info["params"]["output_path"], sub_id, ses_id
                        )
                    if "plot_dir" in step_info["params"]:
                        step_info["params"]["plot_dir"] = self._adjust_output_path(
                            step_info["params"]["plot_dir"], sub_id, ses_id
                        )
                    if "output_folder" in step_info["params"]:
                        step_info["params"]["output_folder"] = self._adjust_output_path(
                            step_info["params"]["output_folder"], sub_id, ses_id
                        )

                    # Run the step
                    self._run_step(step_info)

            print("[SUCCESS] Pipeline completed (multi-subject).")
            return None

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

    def _adjust_output_path(self, original_path, sub_id, ses_id):
        """
        Insert sub-id / ses-id subdirectories to store subject-specific outputs.
        e.g. "data/pilot_data/raw_preprocessed.fif" => 
             "data/pilot_data/sub-01/ses-001/raw_preprocessed.fif"
        """
        p = Path(original_path)
        new_path = p.parent / f"sub-{sub_id}" / f"ses-{ses_id}" / p.name
        return str((self.project_root / new_path).resolve())
    # def _adjust_output_path(self, path, sub_id, ses_id):
    #     """
    #     Adjust the output path to include subject and session information.
    #     Ensure the directory exists before returning the path.
    #     """
    #     adjusted_path = path.format(subject_id=sub_id, session_id=ses_id)
    #     directory = os.path.dirname(adjusted_path)
        
    #     if not os.path.exists(directory):
    #         os.makedirs(directory)
        
    #     return adjusted_path