import mne
from .base import BaseStep

class GoNoGoEpochingStep(BaseStep):
    """
    Epochs only correct responses (101, 201),
    tmin=-0.2, tmax=0.8 by default.
    """

    def run(self, data):
        if data is None:
            raise ValueError("No data in GoNoGoEpochingStep.")

        # We rely on data.info['parsed_events'] from GoNoGoTriggerStep
        if 'parsed_events' not in data.info:
            raise ValueError("No 'parsed_events' found. Run triggers step first?")

        events = data.info['parsed_events']
        event_id = self.params.get("event_id", {
            'Go_Correct': 101,
            'NoGo_Correct': 201
        })
        tmin = self.params.get("tmin", -0.2)
        tmax = self.params.get("tmax", 0.8)
        baseline = self.params.get("baseline", (None, 0))

        epochs = mne.Epochs(
            data, events, event_id=event_id,
            tmin=tmin, tmax=tmax,
            baseline=baseline, preload=True
        )
        return epochs
