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
