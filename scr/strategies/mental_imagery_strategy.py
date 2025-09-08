# scr/strategies/mental_imagery_strategy.py

from scr.pipeline import Pipeline


def run_mental_imagery_pipeline(input_path, output_path, run_id=None):
    """
    Example pipeline for mental imagery tasks.
    
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
                    "params": {"input_file": input_path}
                },
                {
                    "name": "FilterStep",
                    "params": {"l_freq": 1, "h_freq": 100, "notch_freqs": [60, 120]}
                },
                {
                    "name": "EpochingStep",
                    "params": {
                        "task_type": "custom",
                        "trigger_ids": {
                            "Iright_stimulus": 1,
                            "Ileft_stimulus": 2,
                            "right_stimulus": 3,
                            "left_stimulus": 4
                        },
                        "epoch_params": {"tmin": -5, "tmax": 30, "baseline": [None, 0]},
                        "stim_channel": "Trigger",
                        "returns_epochs": True
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
