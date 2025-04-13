# File: src/pipeline.py

import yaml
import logging
import re
from pathlib import Path
import mne
import pickle
import glob  # Add this import for the glob module
import os
from importlib import import_module
# A global STEP_REGISTRY that maps "step name" -> "step class"
from scr.registry import STEP_REGISTRY
from scr.steps.project_paths import ProjectPaths

# Import all steps to register them
from .steps import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        self.default_run = self.config["default_run"]
        

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
                output_path = self.paths.get_derivative_path(
                    subject_id=self.default_subject,
                    session_id=self.default_session,
                    stage="after_autoreject"
                )
                full_path = output_path.resolve()
                
                if full_path.exists():
                    try:
                        # Load the checkpoint file
                        self.data = mne.io.read_raw_fif(full_path, preload=True)
                        logging.info(f"Resuming from checkpoint: {full_path}")
                        return steps_def[i+1:]  # Return steps after the checkpoint
                    except Exception as e:
                        logging.error(f"Error loading checkpoint: {e}")
        return steps_def  # No checkpoint found or load failed

    def run(self):
        """Runs the pipeline for a single subject or multi-subject, depending on config."""
        
        # Get the requested mode from config
        mode = self.config.get("pipeline_mode", "standard")
        start_from_step = self.config.get("start_from_step", None)
        
        logging.info(f"Running pipeline in mode: {mode}")
        if start_from_step:
            logging.info(f"Will attempt to start from step: {start_from_step}")
        
        # 1) Multi-subject mode - iterating through files matching a pattern
        if "file_path_pattern" in self.config:
            steps_def = self.config["pipeline"]["steps"]  
            
            # Get the list of files to process based on the pattern
            pattern = self.config["file_path_pattern"]
            
            # Make the pattern absolute if it's not already
            if not os.path.isabs(pattern):
                pattern = os.path.join(self.paths.raw_data_dir, pattern)
            
            logging.info(f"Looking for files matching pattern: {pattern}")
            files = glob.glob(pattern)
            
            if not files:
                logging.warning(f"No files found matching pattern: {pattern}")
                return None
                
            logging.info(f"Found {len(files)} files to process")
            
            # Process each file
            for file_path in files:
                logging.info(f"Processing file: {file_path}")
                
                # Extract context from filename
                sub_id, ses_id, task_id, run_id = self._parse_sub_ses(file_path)
                
                # Look for the latest checkpoint for this subject
                latest_ckpt = self._find_latest_checkpoint(sub_id, ses_id, task_id, run_id)
                skip_index = None
                
                # If we're starting from a specific step, find its index
                if start_from_step:
                    skip_index = self._find_step_index(steps_def, start_from_step)
                    if skip_index is not None:
                        # Skip to right before the requested step
                        skip_index -= 1  # We want to start AT the requested step
                        logging.info(f"Will start from step {start_from_step} (skipping steps [0..{skip_index}])")
                    else:
                        logging.warning(f"Requested step '{start_from_step}' not found in pipeline. Starting from beginning.")
                
                # Otherwise, use the latest checkpoint if available
                elif mode != "restart" and latest_ckpt and latest_ckpt.exists():
                    logging.info(f"Found existing checkpoint => {latest_ckpt}")
                    # Extract the checkpoint name to determine which steps to skip
                    parts = latest_ckpt.stem.split('_')
                    if len(parts) >= 2 and parts[-2] == "after":
                        step_name = parts[-1]
                        logging.info(f"Last completed step: {step_name}")
                        
                        # Find which step to skip to based on checkpoint name
                        for i, st in enumerate(steps_def):
                            if st["name"].lower() == step_name.lower() or \
                               st["name"].lower() == f"{step_name}step".lower():
                                skip_index = i
                                logging.info(f"Will resume after step {st['name']} (index {skip_index})")
                                break
                    
                    # Load the checkpoint if needed
                    try:
                        # Only load data if we're skipping steps
                        if skip_index is not None:
                            self.data = mne.io.read_raw_fif(latest_ckpt, preload=True)
                            logging.info("Checkpoint loaded successfully.")
                    except Exception as e:
                        logging.error(f"Could not load checkpoint: {e}")
                        self.data = None
                        skip_index = None
                else:
                    # Starting from scratch
                    if mode == "restart":
                        logging.info("Restart mode enabled. Starting from scratch.")
                    else:
                        logging.info("No checkpoint found or continuing from start. Starting from scratch.")
                    self.data = None
                
                # Run the steps for this subject, skipping if needed
                self._run_steps(steps_def, skip_index, sub_id, ses_id, run_id, file_path, task_id)
            
            logging.info("[SUCCESS] Pipeline completed (multi-subject).")
            return None

        # 2) Single-subject mode
        else:
            # TODO: Implement single subject mode if needed
            logging.warning("Single subject mode not fully implemented.")
            return None

    def _run_steps(self, steps_def, skip_index=None, subject_id=None, session_id=None, run_id=None, file_path=None, task_id=None):
        """
        Runs the pipeline steps, optionally skipping steps up to skip_index inclusive.
        If subject_id/session_id/run_id/task_id are provided, pass them to each step along with 'paths'.
        
        Automatically saves after each step if auto_save is enabled in the config.
        """
        auto_save = self.config.get("auto_save", True)
        
        for i, step_info in enumerate(steps_def):
            # Skip steps up to the specified index
            if skip_index is not None and i <= skip_index:
                logging.info(f"Skipping step {i}: {step_info['name']}")
                continue
            
            step_name = step_info["name"]
            logging.info(f"Running step {i}: {step_name}")
            
            # Prepare step parameters
            step_info.setdefault("params", {})
            # If we have subject context, pass it in
            if subject_id and session_id:
                step_info["params"]["subject_id"] = subject_id
                step_info["params"]["session_id"] = session_id
                step_info["params"]["run_id"] = run_id
                step_info["params"]["task_id"] = task_id
                step_info["params"]["paths"] = self.paths
            # If we have a file_path for "LoadData"
            if file_path and step_name == "LoadData":
                step_info["params"]["input_file"] = file_path

            # Run the step
            try:
                self._run_step(step_info)
                logging.info(f"Step {step_name} completed successfully")
                
                # Auto-save after this step if enabled (and it's not already a saving step)
                if auto_save and "Save" not in step_name and self.data is not None:
                    # Extract the base step name without "Step" suffix
                    base_name = step_name.replace("Step", "").lower()
                    
                    # Create an AutoSave step
                    save_params = {
                        "subject_id": subject_id,
                        "session_id": session_id,
                        "task_id": task_id,
                        "run_id": run_id,
                        "step_name": base_name,
                        "paths": self.paths
                    }
                    
                    # Import the AutoSave class
                    from .steps.auto_save import AutoSave
                    
                    # Run the auto-save
                    auto_save_step = AutoSave(save_params)
                    self.data = auto_save_step.run(self.data)
                    logging.info(f"Auto-saved after step {step_name}")
            except Exception as e:
                logging.error(f"Error executing step {step_name}: {e}")
                # Stop processing further steps on error
                raise

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
        Extract sub-xx, ses-yy, task-zz, run-ww from a filename
        e.g. 'sub-01_ses-001_task-stroop_run-01_raw.edf' -> ('01','001','stroop','01')
        """
        fname = Path(file_path).name
        
        # Initialize with default values
        sub_id = self.default_subject
        ses_id = self.default_session
        task_id = "unknown"
        run_id = self.default_run
        
        # Improved regex patterns with proper boundaries 
        # Look for patterns like 'sub-01_' or 'sub-01.' (at end of filename)
        sub_match = re.search(r'sub-([^_\.]+)', fname)
        ses_match = re.search(r'ses-([^_\.]+)', fname)
        task_match = re.search(r'task-([^_\.]+)', fname)
        run_match = re.search(r'run-([^_\.]+)', fname)
        
        if sub_match:
            sub_id = sub_match.group(1)
        if ses_match:
            ses_id = ses_match.group(1)  
        if task_match:
            task_id = task_match.group(1)
        if run_match:
            run_id = run_match.group(1)
        
        return sub_id, ses_id, task_id, run_id

    def _run_step(self, step_info):
        """Instantiate and execute one pipeline step."""
        step_name = step_info["name"]
        params = step_info.get("params", {})

        if step_name not in STEP_REGISTRY:
            raise ValueError(f"Step '{step_name}' not registered.")
        step_class = STEP_REGISTRY[step_name]
        step = step_class(params)
        self.data = step.run(self.data)

    def _find_latest_checkpoint(self, sub_id, ses_id, task_id=None, run_id=None):
        """Find the most recent checkpoint file for a subject/session."""
        # Create the base path for the subject/session
        sub_path = self.paths.processed_dir / f'sub-{sub_id}' / f'ses-{ses_id}'
        
        # Create the base filename pattern to match
        base_name = f"sub-{sub_id}_ses-{ses_id}"
        if task_id:
            base_name += f"_task-{task_id}"
        if run_id:
            base_name += f"_run-{run_id}"
        
        # Pattern to match any checkpoint for this subject
        pattern = f"{base_name}_*.fif"
        
        # Find all matching checkpoint files
        checkpoint_files = list(sub_path.glob(pattern))
        
        if not checkpoint_files:
            logging.info(f"No checkpoints found for {base_name}")
            return None
        
        # Sort by modification time (most recent first)
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        latest = checkpoint_files[0]
        logging.info(f"Found latest checkpoint: {latest}")
        return latest


