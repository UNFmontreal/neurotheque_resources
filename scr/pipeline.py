"""
Pipeline entrypoint and runner.

Supports JSON (preferred) and YAML configuration files, with validation
against `scr/config_schema.json`.
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
import difflib
import glob

import mne
import yaml
from jsonschema import Draft7Validator, ValidationError

# A global STEP_REGISTRY that maps "step name" -> "step class"
from scr.registry import STEP_REGISTRY
from scr.steps.project_paths import ProjectPaths

# Import all steps to register them
from .steps import *  # noqa: F401,F403

logger = logging.getLogger(__name__)

class Pipeline:
    """
    A pipeline that executes a list of steps in order.
    Steps can be specified via a YAML file or a Python dict.
    Supports single-subject (original) or multi-subject (new) mode.
    """

    def __init__(self, config_file="configs/pipeline_config.yaml", config_dict=None, validate_config=True):
        self.config = self._load_config(config_file, config_dict, validate=validate_config)
        self.paths = ProjectPaths(self.config)
        
        # Set default subject/session from config
        self.default_subject = self.config["default_subject"]
        self.default_session = self.config["default_session"]
        self.default_run = self.config["default_run"]
        

    def _load_config(self, config_file, config_dict, validate=True):
        """Load config from dict or file (JSON or YAML) and optionally validate."""
        if config_dict is not None:
            config = config_dict
        else:
            if config_file is None:
                raise ValueError("No configuration provided. Use --config to specify a file.")
            cfg_path = Path(os.path.expandvars(os.path.expanduser(config_file)))
            if not cfg_path.exists():
                raise FileNotFoundError(f"Config file not found: {cfg_path}")

            ext = cfg_path.suffix.lower()
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    if ext in (".json", ".jsonc"):
                        # Basic JSONC support: strip // and /* */ comments
                        text = f.read()
                        text = _strip_json_comments(text)
                        config = json.loads(text)
                    elif ext in (".yml", ".yaml"):
                        config = yaml.safe_load(f)
                    else:
                        raise ValueError(f"Unsupported config extension '{ext}'. Use .json, .yaml, or .yml")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in config {cfg_path}: {e}") from e
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML in config {cfg_path}: {e}") from e

        if validate:
            _validate_config_schema(config)
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
        
        logger.info(f"Running pipeline in mode: {mode}")
        if start_from_step:
            logger.info(f"Will attempt to start from step: {start_from_step}")
        
        # 1) Multi-subject mode - iterating through files matching a pattern
        if "file_path_pattern" in self.config:
            steps_def = self.config["pipeline"]["steps"]  
            
            # Get the list of files to process based on the pattern
            pattern = self.config["file_path_pattern"]
            
            # Make the pattern absolute if it's not already
            if not os.path.isabs(pattern):
                pattern = os.path.join(self.paths.raw_data_dir, pattern)
            
            logger.info(f"Looking for files matching pattern: {pattern}")
            files = glob.glob(pattern)

            if not files:
                logger.warning(f"No files found matching pattern: {pattern}")
                return None
                
            logger.info(f"Found {len(files)} files to process")
            
            # Process each file
            for file_path in files:
                logger.info(f"Processing file: {file_path}")
                
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
                    logger.info(f"Will start from step {start_from_step} (skipping steps [0..{skip_index}])")
                else:
                    logger.warning(f"Requested step '{start_from_step}' not found in pipeline. Starting from beginning.")
                
                # Otherwise, use the latest checkpoint if available
                elif mode != "restart" and latest_ckpt and latest_ckpt.exists():
                    logger.info(f"Found existing checkpoint => {latest_ckpt}")
                    # Extract the checkpoint name to determine which steps to skip
                    parts = latest_ckpt.stem.split('_')
                    if len(parts) >= 2 and parts[-2] == "after":
                        step_name = parts[-1]
                        logger.info(f"Last completed step: {step_name}")
                        
                        # Find which step to skip to based on checkpoint name
                        for i, st in enumerate(steps_def):
                            if st["name"].lower() == step_name.lower() or \
                               st["name"].lower() == f"{step_name}step".lower():
                                skip_index = i
                                logger.info(f"Will resume after step {st['name']} (index {skip_index})")
                        break
                    
                    # Load the checkpoint if needed
                    try:
                        # Only load data if we're skipping steps
                        if skip_index is not None:
                            self.data = mne.io.read_raw_fif(latest_ckpt, preload=True)
                            logger.info("Checkpoint loaded successfully.")
                    except Exception as e:
                        logger.error(f"Could not load checkpoint: {e}")
                        self.data = None
                        skip_index = None
                else:
                    # Starting from scratch
                    if mode == "restart":
                        logger.info("Restart mode enabled. Starting from scratch.")
                    else:
                        logger.info("No checkpoint found or continuing from start. Starting from scratch.")
                    self.data = None
                
                # Run the steps for this subject, skipping if needed
                self._run_steps(steps_def, skip_index, sub_id, ses_id, run_id, file_path, task_id)
            
            logger.info("[SUCCESS] Pipeline completed (multi-subject).")
            return None

        # 2) Single-subject mode
        else:
            steps_def = self.config["pipeline"]["steps"]

            subject_id = self.default_subject
            session_id = self.default_session
            run_id = self.default_run
            task_id = self.config.get("default_task")

            # If LoadData specifies an explicit input_file, parse task/run from it when possible
            input_file = None
            for st in steps_def:
                if st.get("name") == "LoadData":
                    input_file = st.get("params", {}).get("input_file")
                    break

            file_path = None
            if input_file:
                file_path = str(Path(os.path.expandvars(os.path.expanduser(input_file))).resolve())
                # Try to parse IDs from filename for consistency
                p_sub, p_ses, p_task, p_run = self._parse_sub_ses(file_path)
                subject_id = p_sub or subject_id
                session_id = p_ses or session_id
                run_id = p_run or run_id
                task_id = p_task or task_id

            # Determine skip point: explicit step or latest checkpoint
            skip_index = None
            if start_from_step:
                skip_index = self._find_step_index(steps_def, start_from_step)
                if skip_index is not None:
                    skip_index -= 1  # start at requested step
                    logger.info(f"Will start from step {start_from_step} (skipping [0..{skip_index}])")
            else:
                # Try resuming from latest checkpoint unless in restart mode
                if mode != "restart":
                    latest_ckpt = self._find_latest_checkpoint(subject_id, session_id, task_id, run_id)
                    if latest_ckpt and latest_ckpt.exists():
                        logger.info(f"Found existing checkpoint => {latest_ckpt}")
                        parts = latest_ckpt.stem.split('_')
                        step_name = None
                        if len(parts) >= 2 and parts[-2] == "after":
                            step_name = parts[-1]
                        if step_name:
                            for i, st in enumerate(steps_def):
                                nm = st.get("name", "")
                                if nm.lower() == step_name.lower() or nm.lower() == f"{step_name}step".lower():
                                    skip_index = i
                                    logger.info(f"Will resume after step {nm} (index {skip_index})")
                                    break
                        try:
                            if skip_index is not None:
                                self.data = mne.io.read_raw_fif(latest_ckpt, preload=True)
                                logger.info("Checkpoint loaded successfully.")
                        except Exception as e:
                            logger.error(f"Could not load checkpoint: {e}")
                            self.data = None
                            skip_index = None

            # Run the configured steps
            self._run_steps(
                steps_def,
                skip_index=skip_index,
                subject_id=subject_id,
                session_id=session_id,
                run_id=run_id,
                file_path=file_path,
                task_id=task_id,
            )
            logger.info("[SUCCESS] Pipeline completed (single-subject).")
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
                logger.info(f"Skipping step {i}: {step_info['name']}")
                continue
            
            step_name = step_info["name"]
            logger.info(f"Running step {i}: {step_name}")
            
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
                logger.info(f"Step {step_name} completed successfully")
                
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
                    logger.info(f"Auto-saved after step {step_name}")
            except Exception as e:
                logger.error(f"Error executing step {step_name}: {e}")
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
            # Try friendly error with suggestions
            registered = list(STEP_REGISTRY.keys())
            # Common mistake: missing 'Step' suffix
            alt = f"{step_name}Step"
            hints = []
            if alt in STEP_REGISTRY:
                hints.append(alt)
            hints += difflib.get_close_matches(step_name, registered, n=3, cutoff=0.6)
            hint_txt = f" Did you mean: {', '.join(sorted(set(hints)))}?" if hints else ""
            raise ValueError(f"Step '{step_name}' not registered.{hint_txt}")
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
            logger.info(f"No checkpoints found for {base_name}")
            return None
        
        # Sort by modification time (most recent first)
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        latest = checkpoint_files[0]
        logger.info(f"Found latest checkpoint: {latest}")
        return latest


def _strip_json_comments(text: str) -> str:
    """Remove // and /* */ comments from JSON-like text for basic JSONC support."""
    import re as _re
    # Remove // comments
    text = _re.sub(r"//.*", "", text)
    # Remove /* */ comments
    text = _re.sub(r"/\*.*?\*/", "", text, flags=_re.DOTALL)
    return text


def _validate_config_schema(config: dict) -> None:
    """Validate config against scr/config_schema.json and raise helpful errors."""
    schema_path = Path(__file__).with_name("config_schema.json")
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load config schema at {schema_path}: {e}") from e

    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(config), key=lambda e: e.path)
    if errors:
        msgs = []
        for err in errors:
            loc = "/".join([str(x) for x in err.path]) or "<root>"
            msgs.append(f"- {loc}: {err.message}")
        hint = (
            "Common fixes: check 'directory' paths, ensure 'pipeline.steps' is set, "
            "and verify option names."
        )
        raise ValueError("Configuration validation failed:\n" + "\n".join(msgs) + f"\n{hint}")


def _setup_logging(verbosity: int = 0, log_file: Path | None = None) -> None:
    """Configure console and optional file logging with timestamps."""
    level = logging.INFO if verbosity <= 0 else logging.DEBUG
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=handlers,
    )


def main(argv: list[str] | None = None) -> int:
    """Simple CLI to run the pipeline from a config file."""
    parser = argparse.ArgumentParser(description="Run Neurotheque pipeline")
    parser.add_argument("--config", required=True, help="Path to JSON/YAML config file")
    parser.add_argument("--no-validate", action="store_true", help="Disable schema validation")
    parser.add_argument("--verbose", "-v", action="count", default=0, help="Increase verbosity")
    parser.add_argument("--log-file", default="pipeline.log", help="Log file path (under repo root by default)")
    args = parser.parse_args(argv)

    # Setup logging early
    log_path = Path(args.log_file)
    _setup_logging(args.verbose, log_file=log_path)

    try:
        pipe = Pipeline(config_file=args.config, validate_config=not args.no_validate)
        pipe.run()
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error("Hint: check path spelling and that the file exists.")
        return 2
    except ValueError as e:
        logger.error(str(e))
        return 3
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


