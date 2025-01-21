# File: eeg_pipeline/src/pipeline.py

import yaml
from pathlib import Path

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

    def run(self):
        """
        Load the pipeline config, iterate through each step,
        instantiate the step class from STEP_REGISTRY, run it.
        """
        config = self._load_config()
        steps_def = config.get("pipeline", {}).get("steps", [])
        if not steps_def:
            raise ValueError("No steps defined under pipeline.steps in config.")

        for step_info in steps_def:
            step_name = step_info["name"]
            params = step_info.get("params", {})

            # Retrieve the step class from registry
            if step_name not in STEP_REGISTRY:
                raise ValueError(f"Step '{step_name}' not found in STEP_REGISTRY.")
            step_cls = STEP_REGISTRY[step_name]

            # Instantiate and run
            step_instance = step_cls(params)
            self.data = step_instance.run(self.data)

        print("[INFO] Pipeline finished successfully.")
        return self.data
