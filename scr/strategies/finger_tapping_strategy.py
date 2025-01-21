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
