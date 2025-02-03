# src/steps/splittasks.py

import os
import logging
import mne
from .base import BaseStep

class SplitTasksStep(BaseStep):
    """
    Finds triggers for tasks: Rest_GoNoGo, GoNoGo, LandoitC, MentalImagery.
    Then crops each segment and saves as a separate .fif file (e.g. 'GonoGo.fif').
    """

    def run(self, data):
        if data is None:
            raise ValueError("[SplitTasksStep] No data to split.")

        output_folder = self.params.paths.get_split_task_path(self.params.subject_id, self.params.session_id)
        if not output_folder:
            raise ValueError("[SplitTasksStep] 'output_folder' param is required.")

        events = mne.find_events(data, stim_channel='Trigger',
                                 min_duration=0.001, consecutive=False)
        logging.info(f"[SplitTasksStep] Found {len(events)} events.")

        # Build a dict of trigger -> [samples]
        trigger_dict = {}
        for samp, _, trig in events:
            trigger_dict.setdefault(trig, []).append(samp)

        task_periods = {
            'rest': {'start': None, 'end': None},
            'gng_image': {'start': None, 'end': None},
            'landoitc': {'start': None, 'end': None},
            'mental_imagery': {'start': None, 'end': None}
        }

        _define_task_periods(task_periods, trigger_dict, data.info['sfreq'])

        # Crop each segment, save
        for task_name, period in task_periods.items():
            s = period['start']
            e = period['end']
            if s is None or e is None:
                logging.warning(f"[SplitTasksStep] {task_name} period is None; skipping.")
                continue
            tmin = s / data.info['sfreq']
            tmax = e / data.info['sfreq']
            sub_raw = data.copy().crop(tmin=tmin, tmax=tmax)

            save_path = os.path.join(output_folder, f"{task_name}.fif")
            sub_raw.save(save_path, overwrite=True)
            logging.info(f"[SplitTasksStep] Saved {task_name} => {save_path}")

        return data


def _define_task_periods(task_periods, trigger_dict, sfreq):
    """
    The same logic as in your notebook for:
      6->7 => Rest_GoNoGo
      7->(8,9) => GoNoGo
      second 8,9 => MentalImagery
      14 min after gonogo end => LandoitC
    etc.
    """

    import math

    def minutes_to_samples(minutes):
        return int(minutes * 60 * sfreq)

    # 1) Rest_GoNoGo
    if 6 in trigger_dict and 7 in trigger_dict:
        rest_start = trigger_dict[6][0]
        rest_end   = trigger_dict[7][0]
        task_periods['Rest_GoNoGo']['start'] = rest_start
        task_periods['Rest_GoNoGo']['end']   = rest_end
    else:
        logging.error("Missing triggers 6,7 for Rest_GoNoGo")

    # 2) GoNoGo: from 7->(8,9)
    if 7 in trigger_dict and 8 in trigger_dict and 9 in trigger_dict:
        gonogo_start = trigger_dict[7][0]
        gonogo_end   = trigger_dict[9][0]
        task_periods['GoNoGo']['start'] = gonogo_start
        task_periods['GoNoGo']['end']   = gonogo_end
    else:
        logging.error("Missing triggers (7,8,9) for GoNoGo")

    # 3) Mental Imagery: second 8, second 9
    if len(trigger_dict.get(8, [])) >= 2 and len(trigger_dict.get(9, [])) >= 2:
        mi_start = trigger_dict[8][-1]
        mi_end   = trigger_dict[9][-1]
        task_periods['MentalImagery']['start'] = mi_start
        task_periods['MentalImagery']['end']   = mi_end
    else:
        logging.error("Missing second triggers 8,9 for Mental Imagery")

    # 4) LandoitC
    go_no_go_end = task_periods['GoNoGo']['end']
    if go_no_go_end is not None:
        start = go_no_go_end + minutes_to_samples(14)
        rest_starts = trigger_dict.get(6, [])
        if len(rest_starts) >= 2:
            # second rest start
            mental_imagery_rest_start = rest_starts[-1]
            end = mental_imagery_rest_start - minutes_to_samples(1)
            task_periods['LandoitC']['start'] = start
            task_periods['LandoitC']['end']   = end
        else:
            logging.error("Second rest (trigger 6) for LandoitC not found.")
    else:
        logging.error("Cannot define LandoitC: no GoNoGo end found.")
