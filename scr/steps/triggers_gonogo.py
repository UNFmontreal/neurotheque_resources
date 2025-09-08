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
        code_map = self.params.get(
            "events",
            {"go_onset": 1, "nogo_onset": 2, "correct": 3, "incorrect": 4},
        )
        try:
            events = mne.find_events(data, stim_channel=stim_channel, min_duration=0.01)
        except Exception as e:
            raise ValueError(
                f"Failed to find events on stim_channel '{stim_channel}': {e}. "
                "Hint: set 'stim_channel' (e.g., 'Trigger') or check channel list via raw.ch_names."
            ) from e

        if events is None or len(events) == 0:
            raise ValueError(
                f"No events found with stim_channel='{stim_channel}'. "
                "Hint: verify the correct stim channel name or ensure triggers were recorded."
            )

        new_events = []
        new_event_id = {
            'Go_Correct': 101,
            'Go_Incorrect': 102,
            'NoGo_Correct': 201,
            'NoGo_Incorrect': 202
        }

        go = code_map.get("go_onset", 1)
        nogo = code_map.get("nogo_onset", 2)
        correct = code_map.get("correct", 3)
        incorrect = code_map.get("incorrect", 4)

        i = 0
        while i < len(events) - 1:
            onset_evt = events[i]
            resp_evt  = events[i+1]
            onset = onset_evt[2]
            resp  = resp_evt[2]
            if onset in [go, nogo] and resp in [correct, incorrect]:
                if onset == go and resp == correct:
                    new_events.append([onset_evt[0], 0, new_event_id['Go_Correct']])
                elif onset == go and resp == incorrect:
                    new_events.append([onset_evt[0], 0, new_event_id['Go_Incorrect']])
                elif onset == nogo and resp == correct:
                    new_events.append([onset_evt[0], 0, new_event_id['NoGo_Correct']])
                elif onset == nogo and resp == incorrect:
                    new_events.append([onset_evt[0], 0, new_event_id['NoGo_Incorrect']])
                i += 2
            else:
                i += 1

        new_events = np.array(new_events)
        data.info["parsed_events"] = new_events
        return data
