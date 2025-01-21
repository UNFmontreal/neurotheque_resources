# File: eeg_pipeline/src/steps/triggers_gonogo.py

import mne
import numpy as np
from .base import BaseStep

class GoNoGoTriggerStep(BaseStep):
    """
    Finds events (onset=1,2 / response=3,4),
    merges them into new events for Go_Correct, Go_Incorrect, NoGo_Correct, NoGo_Incorrect.
    Stores them in data.info['parsed_events'].
    """

    def run(self, data):
        if data is None:
            raise ValueError("No data in GoNoGoTriggerStep.")

        stim_channel = self.params.get("stim_channel", "Trigger")
        events = mne.find_events(data, stim_channel=stim_channel, min_duration=0.01)

        new_events = []
        new_event_id = {
            'Go_Correct': 101,
            'Go_Incorrect': 102,
            'NoGo_Correct': 201,
            'NoGo_Incorrect': 202
        }

        i = 0
        while i < len(events) - 1:
            onset_evt = events[i]
            resp_evt  = events[i+1]
            onset = onset_evt[2]
            resp  = resp_evt[2]
            if onset in [1, 2] and resp in [3, 4]:
                if onset == 1 and resp == 3:
                    new_events.append([onset_evt[0], 0, new_event_id['Go_Correct']])
                elif onset == 1 and resp == 4:
                    new_events.append([onset_evt[0], 0, new_event_id['Go_Incorrect']])
                elif onset == 2 and resp == 3:
                    new_events.append([onset_evt[0], 0, new_event_id['NoGo_Correct']])
                elif onset == 2 and resp == 4:
                    new_events.append([onset_evt[0], 0, new_event_id['NoGo_Incorrect']])
                i += 2
            else:
                i += 1

        new_events = np.array(new_events)
        data.info["parsed_events"] = new_events

        return data
