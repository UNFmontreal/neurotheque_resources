# src/strategies/gonogo_strategy.py

from src.pipeline import Pipeline

def run_gonogo_pipeline(raw_file_path, output_file):
    """
    A convenience function for a typical Go/No-Go pipeline:
    1. Load
    2. Filter (1-30 Hz)
    3. Trigger parsing (gonogo logic)
    4. Epoching
    5. Save
    """
    config_dict = {
        "pipeline": {
            "steps": [
                {
                    "name": "LoadData",
                    "params": {"file_path": raw_file_path}
                },
                {
                    "name": "Filter",
                    "params": {"l_freq": 1, "h_freq": 30}
                },
                {
                    "name": "TriggerParsing",
                    "params": {"task": "gonogo"}
                },
                {
                    "name": "Epoching",
                    "params": {
                        "event_id": {"Go_Correct": 101, "NoGo_Correct": 201},
                        "tmin": -0.2,
                        "tmax": 0.8,
                        "baseline": [None, 0]
                    }
                },
                {
                    "name": "SaveData",
                    "params": {"output_path": output_file}
                }
            ]
        }
    }
    pipeline = Pipeline(config_dict=config_dict)
    pipeline.run()
