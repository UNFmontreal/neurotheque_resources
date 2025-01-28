import yaml
from pathlib import Path
import mne
import pickle
import re  # <--- We'll use this for parsing sub/ses
from glob import glob  # <--- For enumerating files
from .pipeline import STEP_REGISTRY  # If needed, or from your existing code

class Pipeline:
    """
    A pipeline that executes a list of steps in order.
    Steps can be specified via a YAML config or a Python dict.
    """

    def __init__(self, config_file=None, config_dict=None):
        self.config_file = config_file
        self.config_dict = config_dict
        self.data = None
        self.project_root = self._get_project_root()

    # ... your existing _get_project_root(), _load_config(), etc. ...

    def run(self):
        """
        Load config. If multi_subject is False, do single-subject flow (original).
        If multi_subject is True, find all subject files & iterate.
        """
        config = self._load_config()
        steps_def = config.get("pipeline", {}).get("steps", [])
        if not steps_def:
            raise ValueError("No steps defined under pipeline.steps in config.")

        # 1) Check multi-subject
        multi_subj = config.get("pipeline", {}).get("multi_subject", False)
        if not multi_subj:
            # === SINGLE-SUBJECT CASE (ORIGINAL LOGIC) ===
            remaining_steps = self._resolve_checkpoints(steps_def.copy())
            for step_info in remaining_steps:
                self._run_step(step_info)
            print("[SUCCESS] Pipeline completed (single subject).")
            return self.data
        else:
            # === MULTI-SUBJECT CASE ===
            # Find the pattern from the "LoadData" step
            file_pattern = None
            for st in steps_def:
                if st["name"] == "LoadData":
                    # The param might be "file_path_pattern"
                    file_pattern = st["params"].get("file_path_pattern", None)
                    break
            if not file_pattern:
                raise ValueError("No 'file_path_pattern' found under LoadData step in multi_subject mode.")

            # 2) Glob for all files
            all_files = sorted(glob(str(self.project_root / file_pattern)))
            if not all_files:
                print(f"[WARNING] No files found for pattern: {file_pattern}")
                return None

            print(f"[INFO] Found {len(all_files)} file(s): {all_files}")

            for file_path in all_files:
                print(f"\n=== Processing file: {file_path} ===")
                # Parse sub/ses from the filename
                sub_id, ses_id = self._parse_sub_ses(file_path)

                # Re-load steps each iteration; handle checkpoints if desired
                steps_def_copy = steps_def.copy()
                remaining_steps = self._resolve_checkpoints(steps_def_copy)

                # Reset self.data = None before each subject
                self.data = None

                # 3) Run steps for this subject
                for step_info in remaining_steps:
                    # Insert subject/session info so steps can adapt
                    step_info.setdefault("params", {})
                    step_info["params"]["subject_id"] = sub_id
                    step_info["params"]["session_id"] = ses_id
                    step_info["params"]["input_file"] = file_path
                    # Potentially rewrite any output_path in step_info
                    if "output_path" in step_info["params"]:
                        step_info["params"]["output_path"] = self._adjust_output_path(
                            step_info["params"]["output_path"], sub_id, ses_id
                        )
                    if "plot_dir" in step_info["params"]:
                        step_info["params"]["plot_dir"] = self._adjust_output_path(
                            step_info["params"]["plot_dir"], sub_id, ses_id
                        )
                    # Actually run the step
                    self._run_step(step_info)

            print("[SUCCESS] Pipeline completed (multi-subject).")
            return None

    def _parse_sub_ses(self, file_path):
        """
        Extract sub-xx, ses-yy from a filename, e.g.:
         'sub-01_ses-001_raw.edf' -> ('01','001')
        """
        fname = Path(file_path).name
        sub_match = re.search(r'sub-(\d+)', fname)
        ses_match = re.search(r'ses-(\d+)', fname)
        sub_id = sub_match.group(1) if sub_match else "unknown"
        ses_id = ses_match.group(1) if ses_match else "001"
        return sub_id, ses_id

    def _run_step(self, step_info):
        """Helper method to instantiate and execute a single pipeline step."""
        step_name = step_info["name"]
        params = step_info.get("params", {})

        # The step class
        if step_name not in STEP_REGISTRY:
            raise ValueError(f"Step '{step_name}' not registered.")
        step_class = STEP_REGISTRY[step_name]

        # Create instance and run
        step = step_class(params)
        self.data = step.run(self.data)

    def _adjust_output_path(self, original_path, sub_id, ses_id):
        """
        Insert sub-id / ses-id subdirectories to store subject-specific outputs.
        E.g. "data/pilot_data/raw_preprocessed.fif" => "data/pilot_data/sub-01/ses-001/raw_preprocessed.fif"
        """
        p = Path(original_path)
        # Build a new path: p.parent / sub-XX / ses-YY / p.name
        new_path = p.parent / f"sub-{sub_id}" / f"ses-{ses_id}" / p.name
        return str((self.project_root / new_path).resolve())
