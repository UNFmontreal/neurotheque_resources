# File: src/steps/splittasks.py

import os
import logging
import mne
from .base import BaseStep

class SplitTasksStep(BaseStep):
    """
    A flexible step to split Raw data into multiple tasks based on triggers,
    offsets, or references to previously-defined tasks.

    Expected params in YAML (example):
    ---------------------------------
    output_folder: "data/pilot_data/tasks"
    tasks:
      - name: "Rest_GoNoGo"
        start_trigger: 6
        end_trigger: 7
        occurrence_index: 1   # default 1 => first occurrence
      - name: "GoNoGo"
        start_trigger: 7
        end_trigger: 9
      - name: "MentalImagery"
        start_trigger: 8
        end_trigger: 9
        occurrence_index: 2   # use second occurrence
      - name: "LandoitC"
        start_after_task: "GoNoGo"
        start_offset_minutes: 14
        end_before_trigger: 6
        end_offset_minutes: 1
    """

    def run(self, data):
        if data is None:
            raise ValueError("[SplitTasksStep] No data to split.")

        # 1) Ensure we have an output folder
        output_folder = self.params.get("output_folder")
        if not output_folder:
            raise ValueError("[SplitTasksStep] 'output_folder' param is required.")
        os.makedirs(output_folder, exist_ok=True)

        # 2) Tasks to define
        tasks = self.params.get("tasks", [])
        if not tasks:
            logging.warning("[SplitTasksStep] No tasks defined. Doing nothing.")
            return data

        # 3) Find all triggers in the raw
        events = mne.find_events(data, stim_channel='Trigger',
                                 min_duration=0.001, consecutive=False)
        logging.info(f"[SplitTasksStep] Found {len(events)} events.")

        # Build a dictionary of {trigger_value -> list of sample indices}
        trigger_dict = {}
        for samp, _, trig in events:
            trigger_dict.setdefault(trig, []).append(samp)

        sfreq = data.info['sfreq']

        # 4) We'll store the start/end sample for each completed task, so later tasks
        #    can reference "start_after_task": "GoNoGo", etc.
        task_segments = {}

        # 5) Iterate over the tasks in order
        for task_def in tasks:
            task_name = task_def["name"]
            start_sample, end_sample = self._find_task_segment(
                task_def, trigger_dict, sfreq, task_segments
            )

            if start_sample is None or end_sample is None:
                logging.warning(f"[SplitTasksStep] {task_name}: Could not define segment. Skipping.")
                continue

            # Enforce order: if start > end, skip
            if start_sample >= end_sample:
                logging.warning(f"[SplitTasksStep] {task_name}: start >= end. Skipping.")
                continue

            # 6) Crop the raw data for this task
            tmin = start_sample / sfreq
            tmax = end_sample / sfreq
            sub_raw = data.copy().crop(tmin=tmin, tmax=tmax)

            # 7) Save to disk
            save_path = os.path.join(output_folder, f"{task_name}.fif")
            sub_raw.save(save_path, overwrite=True)
            logging.info(f"[SplitTasksStep] Saved {task_name} => {save_path}")

            # 8) Store for reference by subsequent tasks
            task_segments[task_name] = {
                "start": start_sample,
                "end": end_sample
            }

        return data

    def _find_task_segment(self, task_def, trigger_dict, sfreq, task_segments):
        """
        Compute (start_sample, end_sample) for a given task definition.
        The definition can contain:
          - start_trigger (int)
          - end_trigger (int)
          - occurrence_index (int) [default=1 => first occurrence]
          - start_after_task (str) + start_offset_minutes (float)
          - end_before_trigger (int) + end_offset_minutes (float)
          etc.

        Return (None, None) if we cannot find a valid segment.
        """
        task_name = task_def["name"]
        occurrence_index = task_def.get("occurrence_index", 1) - 1

        start_sample = None
        end_sample = None

        # 1) start_trigger logic
        if "start_trigger" in task_def:
            start_trig = task_def["start_trigger"]
            # check existence
            if start_trig in trigger_dict and len(trigger_dict[start_trig]) > occurrence_index:
                start_sample = trigger_dict[start_trig][occurrence_index]
            else:
                logging.error(f"{task_name}: Missing or insufficient occurrences of start_trigger={start_trig}.")
                return (None, None)

        # 2) If there's "start_after_task"
        if "start_after_task" in task_def:
            ref_task_name = task_def["start_after_task"]
            if ref_task_name not in task_segments:
                logging.error(f"{task_name}: The referenced task '{ref_task_name}' isn't defined yet.")
                return (None, None)
            # get the end of that ref task
            ref_end = task_segments[ref_task_name]["end"]
            offset_minutes = task_def.get("start_offset_minutes", 0)
            offset_samples = int(offset_minutes * 60 * sfreq)
            candidate_start = ref_end + offset_samples

            # If we didn't have a start_sample from a trigger, use this.
            # Or if we had both, we pick whichever is LATER
            if start_sample is None:
                start_sample = candidate_start
            else:
                # take the max => ensures we start after both conditions
                start_sample = max(start_sample, candidate_start)

        # 3) end_trigger logic
        if "end_trigger" in task_def:
            end_trig = task_def["end_trigger"]
            if end_trig in trigger_dict and len(trigger_dict[end_trig]) > occurrence_index:
                end_sample = trigger_dict[end_trig][occurrence_index]
            else:
                logging.error(f"{task_name}: Missing or insufficient occurrences of end_trigger={end_trig}.")
                return (None, None)

        # 4) end_before_trigger logic
        if "end_before_trigger" in task_def:
            ebt = task_def["end_before_trigger"]
            # We want 1 minute before that trigger, for instance
            ebt_occ_idx = task_def.get("end_before_occurrence_index", 1) - 1
            offset_minutes = task_def.get("end_offset_minutes", 0)
            offset_samples = int(offset_minutes * 60 * sfreq)

            if ebt in trigger_dict and len(trigger_dict[ebt]) > ebt_occ_idx:
                ebt_sample = trigger_dict[ebt][ebt_occ_idx]
                ebt_sample = ebt_sample - offset_samples
                # If we had no end_sample, use this
                if end_sample is None:
                    end_sample = ebt_sample
                else:
                    # pick the min => we stop whichever comes first
                    end_sample = min(end_sample, ebt_sample)
            else:
                logging.error(f"{task_name}: Missing end_before_trigger {ebt} or not enough occurrences.")
                return (None, None)

        # 5) Possibly more logic if you want other conditions
        # e.g., "end_after_task" or "start_before_trigger", etc.

        return (start_sample, end_sample)
