# src/steps/triggers.py

import mne
import numpy as np
from .base import BaseStep

class TriggerParsingStep(BaseStep):
    """
    Step that finds events in raw data and applies custom parsing logic
    depending on the 'task' parameter (gonogo, finger_tapping, etc.).
    """

    def run(self, data):
        """
        Expected params:
        - stim_channel (str): e.g. 'Trigger'
        - task (str): e.g. 'gonogo', 'finger_tapping', etc.

        The parsed events are stored in data.info['parsed_events'] (or we can just store them in step).
        """
        if data is None:
            raise ValueError("No data available for trigger parsing.")

        stim_channel = self.params.get("stim_channel", "Trigger")
        task = self.params.get("task", "gonogo")

        events = mne.find_events(data, stim_channel=stim_channel, consecutive=True, min_duration=0.001)

        if task == "gonogo":
            parsed_events = self._parse_gonogo(events)
        elif task == "finger_tapping":
            parsed_events = self._parse_finger_tapping(events)
        elif task == "mental_imagery":
            parsed_events = self._parse_mental_imagery(events)
        else:
            # Fallback or user logic
            parsed_events = events

        # Store parsed events in the info dict or return them
        data.info["parsed_events"] = parsed_events
        return data

    def _parse_gonogo(self, events):
        """
        Example of combining triggers for go/nogo correctness.
        """
        # skeleton logic
        new_events = []
        i = 0
        while i < len(events)-1:
            onset = events[i][2]
            resp = events[i+1][2]
            # ... your go/nogo logic ...
            i += 1
        return np.array(new_events)

    def _parse_finger_tapping(self, events):
        """
        Example logic for beep vs mario triggers, key presses, etc.
        """
        # ...
        return events

    def _parse_mental_imagery(self, events):
        """
        Example logic for 30s blocks, etc.
        """
        # ...
        return events
