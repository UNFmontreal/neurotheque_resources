# File: eeg_pipeline/src/pipeline.py

import yaml
from pathlib import Path
import mne
import pickle

# A global STEP_REGISTRY that maps "step name" -> "step class"
STEP_REGISTRY = {}

class Pipeline:
    """
    A pipeline that executes a list of steps in order.
    Steps can be specified via a YAML config or a Python dict.
    """

    def __init__(self, config_file=None, config_dict=None):
        """
        You can either pass a YAML file path or a dict.
        """
        self.config_file = config_file
        self.config_dict = config_dict
        self.data = None  # Will hold mne.Raw or mne.Epochs throughout the pipeline
        self.project_root = self._get_project_root()
        
    def _get_project_root(self):
        """Calculate project root based on this file's location"""
        current_file = Path(__file__).resolve()
        return current_file.parent.parent  # Adjust based on actual structure
    
    def _load_config(self):
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
        """Find latest valid checkpoint and load data"""
        for i, step in reversed(list(enumerate(steps_def))):
            if step["name"] == "SaveCheckpoint":
                output_path = Path(step["params"]["output_path"])
                full_path = (self.project_root / output_path).resolve()
                
                if full_path.exists():
                    try:
                        self.data = mne.io.read_raw_fif(full_path, preload=True)
                        log_path=full_path.with_name(full_path.stem + "_rejectlog.pkl")
                        if log_path.exists():
                            with open(log_path, "rb") as f:
                                if not hasattr(self.data.info, 'temp') or self.data.info['temp'] is None:
                                    self.data.info['temp'] = {}
                                self.data.info["temp"]["autoreject_log"] = pickle.load(f)
                                
                        print(f"Resuming from checkpoint: {full_path}")
                        return steps_def[i+1:]  # Return remaining steps
                    except Exception as e:
                        print(f"Error loading checkpoint: {e}")
        return steps_def  # No checkpoint found
    
    def run(self):
        """
        Load the pipeline config, iterate through each step,
        instantiate the step class from STEP_REGISTRY, run it.
        """
        config = self._load_config()
        steps_def = config.get("pipeline", {}).get("steps", [])
        if not steps_def:
            raise ValueError("No steps defined under pipeline.steps in config.")

        # Try to resume from last checkpoint
        remaining_steps = self._resolve_checkpoints(steps_def.copy())

        for step_info in remaining_steps:
            step_name = step_info["name"]
            params = step_info.get("params", {})
            
            # Resolve relative paths in parameters
            if "output_path" in params:
                params["output_path"] = str(
                    (self.project_root / params["output_path"]).resolve()
                )

            if step_name not in STEP_REGISTRY:
                raise ValueError(f"Step '{step_name}' not registered.")
            
            step = STEP_REGISTRY[step_name](params)
            self.data = step.run(self.data)

        print("[SUCCESS] Pipeline completed.")
        return self.data
