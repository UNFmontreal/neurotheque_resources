# src/strategies/finger_tapping_strategy.py

from scr.pipeline import Pipeline

def run_finger_tapping_pipeline(input_path, output_path, run_id=None):
    """
    Example pipeline for finger tapping experiment.
    
    Parameters:
    -----------
    input_path : str
        Path to the input file
    output_path : str
        Path to save the output file
    run_id : str, optional
        Run identifier if multiple runs are available
    """
    config_dict = {
        "default_subject": "01",
        "default_session": "001",
        "default_run": run_id if run_id else "01",
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
