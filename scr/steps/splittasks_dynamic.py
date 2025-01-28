# File: src/steps/splittasks.py

import logging
import os
import mne
from pathlib import Path
from .base import BaseStep

class SplitTasksStep(BaseStep):
    """
    A flexible step to split Raw data into multiple tasks based on triggers,
    offsets, or references to previously-defined tasks.

    Example YAML snippet:
      - name: SplitTasksStep
        params:
          output_folder: "data/pilot_data/tasks"
          tasks:
            - name: "GoNoGo"
              start_trigger: 7
              end_trigger: 9
            - name: "Rest"
              start_trigger: 6
              end_trigger: 7

    The pipeline will rewrite `output_folder` if multi-subject mode is used,
    so each subject ends up with:
      data/pilot_data/tasks/sub-01/ses-001/GoNoGo.fif
      data/pilot_data/tasks/sub-01/ses-001/Rest.fif
    etc.
    """

    def run(self, data):
        if data is None:
            raise ValueError("[SplitTasksStep] No data to split.")

        # 1) Ensure we have an output folder (subject-specific if pipeline rewrote it)
        output_folder = self.params.get("output_folder")
        if not output_folder:
            raise ValueError("[SplitTasksStep] 'output_folder' param is required.")
        out_dir = Path(output_folder)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 2) Get tasks
        tasks = self.params.get("tasks", [])
        if not tasks:
            logging.warning("[SplitTasksStep] No tasks defined. Doing nothing.")
            return data

        # 3) Find triggers
        events = mne.find_events(data, stim_channel='Trigger', min_duration=0.001, consecutive=False)
        logging.info(f"[SplitTasksStep] Found {len(events)} events in the data.")

        # Build a dictionary of {trigger_value -> list of sample indices}
        trigger_dict = {}
        for samp, _, trig_val in events:
            trigger_dict.setdefault(trig_val, []).append(samp)

        sfreq = data.info['sfreq']

        # We'll store start/end sample for each completed task
        task_segments = {}

        # 4) Iterate over tasks in order
        for task_def in tasks:
            task_name = task_def["name"]
            start_sample, end_sample = self._find_task_segment(
                task_def, trigger_dict, sfreq, task_segments
            )

            if start_sample is None or end_sample is None:
                logging.warning(f"[SplitTasksStep] {task_name}: Could not define segment. Skipping.")
                continue

            if start_sample >= end_sample:
                logging.warning(f"[SplitTasksStep] {task_name}: start >= end => Skipping.")
                continue

            # Crop the raw data for this task
            tmin = start_sample / sfreq
            tmax = end_sample / sfreq
            sub_raw = data.copy().crop(tmin=tmin, tmax=tmax)

            # 5) Save to disk
            save_path = out_dir / f"{task_name}.fif"
            sub_raw.save(str(save_path), overwrite=True)
            logging.info(f"[SplitTasksStep] Saved {task_name} => {save_path}")

            # Store for reference by subsequent tasks
            task_segments[task_name] = {"start": start_sample, "end": end_sample}

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

        # start_trigger logic
        if "start_trigger" in task_def:
            st_trig = task_def["start_trigger"]
            if st_trig in trigger_dict and len(trigger_dict[st_trig]) > occurrence_index:
                start_sample = trigger_dict[st_trig][occurrence_index]
            else:
                logging.error(f"{task_name}: Missing or insufficient occurrences of start_trigger={st_trig}.")
                return (None, None)

        # start_after_task logic
        if "start_after_task" in task_def:
            ref_task = task_def["start_after_task"]
            if ref_task not in task_segments:
                logging.error(f"{task_name}: The referenced task '{ref_task}' isn't defined yet.")
                return (None, None)
            ref_end = task_segments[ref_task]["end"]
            offset_mins = task_def.get("start_offset_minutes", 0)
            offset_samps = int(offset_mins * 60 * sfreq)
            candidate_start = ref_end + offset_samps
            if start_sample is None:
                start_sample = candidate_start
            else:
                start_sample = max(start_sample, candidate_start)

        # end_trigger logic
        if "end_trigger" in task_def:
            end_trig = task_def["end_trigger"]
            if end_trig in trigger_dict and len(trigger_dict[end_trig]) > occurrence_index:
                end_sample = trigger_dict[end_trig][occurrence_index]
            else:
                logging.error(f"{task_name}: Missing or insufficient occurrences of end_trigger={end_trig}.")
                return (None, None)

        # end_before_trigger logic
        if "end_before_trigger" in task_def:
            ebt = task_def["end_before_trigger"]
            ebt_occ_idx = task_def.get("end_before_occurrence_index", 1) - 1
            offset_mins = task_def.get("end_offset_minutes", 0)
            offset_samps = int(offset_mins * 60 * sfreq)

            if ebt in trigger_dict and len(trigger_dict[ebt]) > ebt_occ_idx:
                ebt_samp = trigger_dict[ebt][ebt_occ_idx]
                ebt_samp -= offset_samps
                if end_sample is None:
                    end_sample = ebt_samp
                else:
                    end_sample = min(end_sample, ebt_samp)
            else:
                logging.error(f"{task_name}: Missing end_before_trigger {ebt} or not enough occurrences.")
                return (None, None)

        # Return the final start/end
        return (start_sample, end_sample)
