# src/preprocessing/global_and_task_preprocessing.py

import os
import os.path as op
import logging
import numpy as np

import mne
from mne.channels import make_standard_montage

# Import your Pipeline class and step classes/registry
# (Adjust these import paths as needed to match your actual folder structure.)
from neuropipe.src.pipeline import Pipeline
from neuropipe.src.steps.base import BaseStep
# We assume you already have these steps in your steps/ folder:
#   LoadData, FilterStep, AutoRejectStep, ICAStep, SaveData
# If not, create or import them accordingly.
# Example:
from neuropipe.src.steps.filter import FilterStep
from neuropipe.src.steps.autoreject import AutoRejectStep
from neuropipe.src.steps.ica import ICAStep
from neuropipe.src.steps.load import LoadData
from neuropipe.src.steps.save import SaveData





class SplitTasksStep(BaseStep):
    """
    A custom step that:
    1. Reads the triggers from the final preprocessed raw.
    2. Crops out each task's segment (GoNoGo, Rest, etc.).
    3. Saves each segment as a new .fif file.
    """

    def run(self, data):
        """
        data : mne.io.Raw (the globally preprocessed entire session)
        We'll parse triggers, define tasks, and save them.
        Expected params in self.params:
            - output_folder (str): where to save sub-task .fif files
        """
        if data is None:
            raise ValueError("No raw data to split.")

        output_folder = self.params.get("output_folder", ".")
        os.makedirs(output_folder, exist_ok=True)

        # 1. Find events
        events = mne.find_events(data, stim_channel='Trigger', min_duration=0.001, consecutive=False)
        logging.info(f"Found {len(events)} events in the global preprocessed data.")

        # Build a dict: trigger => [samples]
        trigger_dict = {}
        for sample, _, trig in events:
            trigger_dict.setdefault(trig, []).append(sample)

        # 2. Define your tasks (Rest_GoNoGo, GoNoGo, LandoitC, MentalImagery)
        task_periods = {
            'Rest_GoNoGo': {'start': None, 'end': None},
            'GoNoGo': {'start': None, 'end': None},
            'LandoitC': {'start': None, 'end': None},
            'MentalImagery': {'start': None, 'end': None}
        }

        self._define_task_periods(task_periods, trigger_dict, data.info['sfreq'])

        # 3. Crop and save
        for task_name, period in task_periods.items():
            start = period['start']
            end = period['end']
            if start is None or end is None:
                logging.warning(f"[SplitTasksStep] Could not define task {task_name}; skipping.")
                continue
            tmin = start / data.info['sfreq']
            tmax = end / data.info['sfreq']
            sub_raw = data.copy().crop(tmin=tmin, tmax=tmax)

            # Save
            save_path = op.join(output_folder, f"{task_name}.fif")
            sub_raw.save(save_path, overwrite=True)
            logging.info(f"[SplitTasksStep] Saved {task_name} => {save_path}")

        return data  # We return the entire raw unmodified (besides saving subtasks)

    def _define_task_periods(self, task_periods, trigger_dict, sfreq):
        """
        Helper function that sets the start/end samples for each task,
        just like in your notebook code.
        """
        def minutes_to_samples(minutes, sf):
            return int(minutes * 60 * sf)

        # Example logic from your code:
        # Rest_GoNoGo: triggers 6->7
        if 6 in trigger_dict and 7 in trigger_dict:
            rest_start = trigger_dict[6][0]
            rest_end = trigger_dict[7][0]
            task_periods['Rest_GoNoGo']['start'] = rest_start
            task_periods['Rest_GoNoGo']['end'] = rest_end
        else:
            logging.error("Missing triggers 6,7 for Rest_GoNoGo.")
            # We won't raise an error, just skip.

        # GoNoGo: triggers 7-> (8,9)
        if 7 in trigger_dict and 8 in trigger_dict and 9 in trigger_dict:
            gonogo_start = trigger_dict[7][0]
            gonogo_end = trigger_dict[9][0]
            task_periods['GoNoGo']['start'] = gonogo_start
            task_periods['GoNoGo']['end'] = gonogo_end
        else:
            logging.error("Missing triggers 7,8,9 for GoNoGo.")

        # MentalImagery: second occurrence of 8/9
        if len(trigger_dict.get(8, [])) >= 2 and len(trigger_dict.get(9, [])) >= 2:
            mental_imagery_start = trigger_dict[8][-1]
            mental_imagery_end = trigger_dict[9][-1]
            task_periods['MentalImagery']['start'] = mental_imagery_start
            task_periods['MentalImagery']['end'] = mental_imagery_end
        else:
            logging.error("Missing second triggers 8,9 for MentalImagery")

        # LandoitC: from gonogo_end + 14 min => second rest start - 1 min
        # (like your logic)
        if task_periods['GoNoGo']['end'] is not None:
            start = task_periods['GoNoGo']['end'] + minutes_to_samples(14, sfreq)
            rest_starts = trigger_dict.get(6, [])
            if len(rest_starts) >= 2:
                mental_imagery_rest_start = rest_starts[-1]
                end = mental_imagery_rest_start - minutes_to_samples(1, sfreq)
                task_periods['LandoitC']['start'] = start
                task_periods['LandoitC']['end'] = end
            else:
                logging.error("Missing second rest trigger for LandoitC definition.")
        else:
            logging.error("Cannot define LandoitC without a valid GoNoGo end.")


def run_global_and_task_preprocessing(
    input_file,
    output_preprocessed,
    output_folder_for_tasks,
    ica_bad_components=(1,2,3)
):
    """
    A single "strategy" function that builds a pipeline config to:
      1. Load data (LoadData)
      2. Prep channels (PrepChannelsStep)
      3. Filter (FilterStep)
      4. AutoReject (AutoRejectStep)
      5. ICA (ICAStep)
      6. Save global (SaveData)
      7. Split tasks (SplitTasksStep)

    Parameters
    ----------
    input_file : str
        Path to the raw data file (e.g., 'sub-01_ses-001_raw.edf').
    output_preprocessed : str
        Path to save the globally preprocessed data (e.g., '..._preprocessed.fif').
    output_folder_for_tasks : str
        Directory to store the splitted tasks (e.g. '.../pilot_data/')
    ica_bad_components : tuple or list
        Components to exclude after ICA.
    """

    # Build a pipeline config dictionary
    pipeline_config = {
        "pipeline": {
            "steps": [
                {
                    "name": "LoadData",
                    "params": {
                        "file_path": input_file
                    }
                },
                {
                    "name": "PrepChannelsStep",
                    "params": {}  # no special params needed
                },
                {
                    "name": "FilterStep",
                    "params": {
                        "l_freq": 1.0,
                        "h_freq": 100.0,
                        "notch_freqs": [60, 120]
                    }
                },
                {
                    "name": "AutoRejectStep",
                    "params": {
                        # can pass additional auto-reject params if needed
                    }
                },
                {
                    "name": "ICAStep",
                    "params": {
                        "n_components": 0.95,
                        "random_state": 0,
                        "exclude": list(ica_bad_components)
                    }
                },
                {
                    "name": "SaveData",
                    "params": {
                        "output_path": output_preprocessed,
                        "overwrite": True
                    }
                },
                {
                    "name": "SplitTasksStep",
                    "params": {
                        "output_folder": output_folder_for_tasks
                    }
                }
            ]
        }
    }

    # Build the pipeline object (your Pipeline class)
    pipe = Pipeline(config_dict=pipeline_config)

    # Register our custom steps
    # (If you have a STEP_REGISTRY, ensure these classes are in it,
    # or you can add them dynamically as below.)
    from neuropipe.src.pipeline import STEP_REGISTRY
    STEP_REGISTRY["PrepChannelsStep"] = PrepChannelsStep
    STEP_REGISTRY["SplitTasksStep"]   = SplitTasksStep

    # If other steps aren't already in the registry, also add them:
    STEP_REGISTRY["LoadData"]      = LoadData
    STEP_REGISTRY["FilterStep"]    = FilterStep
    STEP_REGISTRY["AutoRejectStep"] = AutoRejectStep
    STEP_REGISTRY["ICAStep"]       = ICAStep
    STEP_REGISTRY["SaveData"]      = SaveData

    # Run the pipeline
    pipe.run()
    print("[INFO] Global + Task Preprocessing pipeline completed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    folderpath = "../data/pilot_data/"
    filename = "sub-01_ses-001_raw.edf"
    input_path = op.join(folderpath, filename)
    output_global = op.join(folderpath, filename.replace('.edf', '_preprocessed.fif'))
    output_folder = folderpath  # same folder for splitted tasks

    run_global_and_task_preprocessing(
        input_file=input_path,
        output_preprocessed=output_global,
        output_folder_for_tasks=output_folder,
        ica_bad_components=(1,2,3)
    )
