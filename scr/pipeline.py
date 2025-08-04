import yaml
import logging
import re
from pathlib import Path
import mne
import pickle
import glob
import os
from importlib import import_module
import json
from jsonschema import validate

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
        self.config_file_path = Path(config_file).resolve()
        self.project_root = self.config_file_path.parent.parent # Assuming config file is in a 'config' directory
        self.config = self._load_and_validate_config(config_file, config_dict)
        self.paths = ProjectPaths(self.config, project_root=self.project_root)
        
        # Set default subject/session from config
        self.default_subject = self.config["default_subject"]
        self.default_session = self.config["default_session"]
        self.default_run = self.config["default_run"]
        

    def _load_and_validate_config(self, config_file, config_dict):
        if config_dict:
            config = config_dict
        else:
            with open(config_file) as f:
                config = yaml.safe_load(f)

        # Load the schema
        schema_path = Path(__file__).parent / 'config_schema.json'
        with open(schema_path) as f:
            schema = json.load(f)

        # Validate the config against the schema
        try:
            validate(instance=config, schema=schema)
            logging.info("Configuration is valid.")
        except Exception as e:
            logging.error(f"Configuration validation error: {e}")
            raise

        return config

    def run(self):
        """Runs the pipeline for a single subject or multi-subject, depending on config."""
        
        mode = self.config.get("pipeline_mode", "standard")
        start_from_step = self.config.get("start_from_step", None)
        
        logging.info(f"Running pipeline in mode: {mode}")
        if start_from_step:
            logging.info(f"Will attempt to start from step: {start_from_step}")
        
        if "file_path_pattern" in self.config:
            self._run_multi_subject_mode(mode, start_from_step)
        else:
            self._run_single_subject_mode(mode, start_from_step)

    def _run_multi_subject_mode(self, mode, start_from_step):
        steps_def = self.config["pipeline"]["steps"]
        pattern = self.config["file_path_pattern"]

        if not os.path.isabs(pattern):
            pattern = os.path.join(self.paths.raw_data_dir, pattern)
        
        files = glob.glob(pattern)
        if not files:
            logging.warning(f"No files found matching pattern: {pattern}")
            return

        logging.info(f"Found {len(files)} files to process")
        for file_path in files:
            self.process_single_file(file_path, steps_def, mode, start_from_step)

    def process_single_file(self, file_path, steps_def, mode, start_from_step):
        logging.info(f"Processing file: {file_path}")
        sub_id, ses_id, task_id, run_id = self._parse_sub_ses(file_path)
        
        context = {
            "subject_id": sub_id,
            "session_id": ses_id,
            "task_id": task_id,
            "run_id": run_id,
            "metrics": {}
        }
        
        self.data = None
        skip_index = self._determine_start_index(steps_def, mode, start_from_step, sub_id, ses_id, task_id, run_id)
        
        self._run_steps(steps_def, context, skip_index, file_path)
        logging.info(f"[SUCCESS] Pipeline completed for {file_path}.")

    def _determine_start_index(self, steps_def, mode, start_from_step, sub_id, ses_id, task_id, run_id):
        if start_from_step:
            skip_index = self._find_step_index(steps_def, start_from_step)
            if skip_index is not None:
                logging.info(f"Starting from step {start_from_step} (index {skip_index})")
                return skip_index - 1
            else:
                logging.warning(f"Step '{start_from_step}' not found. Starting from beginning.")
                return None

        if mode == "restart":
            logging.info("Restart mode enabled. Starting from scratch.")
            return None
            
        latest_ckpt = self._find_latest_checkpoint(sub_id, ses_id, task_id, run_id)
        if latest_ckpt and latest_ckpt.exists():
            logging.info(f"Found existing checkpoint => {latest_ckpt}")
            step_name = latest_ckpt.stem.split('_')[-1]
            skip_index = self._find_step_index_by_name(steps_def, step_name)
            if skip_index is not None:
                try:
                    self.data = mne.io.read_raw_fif(latest_ckpt, preload=True)
                    logging.info("Checkpoint loaded successfully.")
                    return skip_index
                except Exception as e:
                    logging.error(f"Could not load checkpoint: {e}")
                    self.data = None
        
        logging.info("No checkpoint found or continuing from start. Starting from scratch.")
        return None

    def _run_steps(self, steps_def, context, skip_index=None, file_path=None):
        auto_save = self.config.get("auto_save", True)
        
        for i, step_info in enumerate(steps_def):
            if skip_index is not None and i <= skip_index:
                logging.info(f"Skipping step {i}: {step_info['name']}")
                continue
            
            step_name = step_info["name"]
            logging.info(f"Running step {i}: {step_name}")
            
            params = step_info.get("params", {})
            params.update(context)
            params["paths"] = self.paths
            
            if file_path and step_name == "LoadData":
                params["input_file"] = file_path

            try:
                self._run_step(step_name, params, context)
                logging.info(f"Step {step_name} completed successfully")
                
                if auto_save and "Save" not in step_name and self.data is not None:
                    self._auto_save_data(context)
            except Exception as e:
                logging.error(f"Error executing step {step_name}: {e}")
                raise

    def _run_step(self, step_name, params, context):
        if step_name not in STEP_REGISTRY:
            raise ValueError(f"Step '{step_name}' not registered.")
        
        step_class = STEP_REGISTRY[step_name]
        step = step_class(params)
        self.data = step.run(self.data)
        
        # Update context with metrics from the step
        if hasattr(step, 'get_metrics'):
            context['metrics'].update(step.get_metrics())

    def _auto_save_data(self, context):
        from .steps.auto_save import AutoSave
        save_params = {
            **context,
            "step_name": "autosave",
            "paths": self.paths
        }
        auto_save_step = AutoSave(save_params)
        self.data = auto_save_step.run(self.data)
        logging.info(f"Auto-saved data.")

    def _find_step_index(self, steps_def, step_name):
        for i, st in enumerate(steps_def):
            if st["name"] == step_name:
                return i
        return None
        
    def _find_step_index_by_name(self, steps_def, step_name_part):
        for i, st in enumerate(steps_def):
            if step_name_part.lower() in st["name"].lower():
                return i
        return None

    def _parse_sub_ses(self, file_path):
        fname = Path(file_path).name
        sub_match = re.search(r'sub-([^_\.]+)', fname)
        ses_match = re.search(r'ses-([^_\.]+)', fname)
        task_match = re.search(r'task-([^_\.]+)', fname)
        run_match = re.search(r'run-([^_\.]+)', fname)
        
        return (
            sub_match.group(1) if sub_match else self.default_subject,
            ses_match.group(1) if ses_match else self.default_session,
            task_match.group(1) if task_match else "unknown",
            run_match.group(1) if run_match else self.default_run
        )

    def _find_latest_checkpoint(self, sub_id, ses_id, task_id=None, run_id=None):
        sub_path = self.paths.processed_dir / f'sub-{sub_id}' / f'ses-{ses_id}'
        base_name = f"sub-{sub_id}_ses-{ses_id}"
        if task_id: base_name += f"_task-{task_id}"
        if run_id: base_name += f"_run-{run_id}"
        
        pattern = f"{base_name}_*.fif"
        checkpoint_files = sorted(sub_path.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
        
        if checkpoint_files:
            logging.info(f"Found latest checkpoint: {checkpoint_files[0]}")
            return checkpoint_files[0]
        
        logging.info(f"No checkpoints found for {base_name}")
        return None
        
    def _run_single_subject_mode(self, mode, start_from_step):
        logging.warning("Single subject mode not fully implemented.")
        return None
