# src/strategies/basic_cleaning.py

from src.pipeline import Pipeline

def run_basic_cleaning(input_path, output_path, ica_components=0.95):
    """
    A convenience function that runs a 'basic cleaning' pipeline:
    1. Load data
    2. Filter
    3. ICA
    4. Save
    """
    config_dict = {
        "pipeline": {
            "steps": [
                {
                    "name": "LoadData",
                    "params": {
                        "file_path": input_path
                    }
                },
                {
                    "name": "Filter",
                    "params": {
                        "l_freq": 1,
                        "h_freq": 40,
                        "notch_freqs": [50]
                    }
                },
                {
                    "name": "ICA",
                    "params": {
                        "n_components": ica_components,
                        "random_state": 42
                    }
                },
                {
                    "name": "SaveData",
                    "params": {
                        "output_path": output_path,
                        "overwrite": True
                    }
                }
            ]
        }
    }
    pipeline = Pipeline(config_dict=config_dict)
    pipeline.run()
