# src/steps/epoching.py

import mne
from .base import BaseStep

class EpochingStep(BaseStep):
    """
    Step that converts Raw to Epochs based on parsed events.
    """

    def run(self, data):
        """
        Expected params:
        - event_id (dict): e.g. {'Go_Correct': 101, 'NoGo_Correct': 201}
        - tmin (float)
        - tmax (float)
        - baseline (tuple or None)
        """
        if data is None:
            raise ValueError("No data available for epoching.")

        if not hasattr(data.info, "parsed_events") and "parsed_events" not in data.info:
            # fallback to find_events or raise an error
            raise ValueError("No parsed_events found in data.info. Did you run TriggerParsingStep?")

        events = data.info["parsed_events"]
        event_id = self.params.get("event_id", {})
        tmin = self.params.get("tmin", -0.2)
        tmax = self.params.get("tmax", 0.8)
        baseline = self.params.get("baseline", (None, 0))

        epochs = mne.Epochs(data, events, event_id=event_id,
                            tmin=tmin, tmax=tmax, baseline=baseline,
                            preload=True)
        return epochs
