# src/steps/base.py

from abc import ABC, abstractmethod

class BaseStep(ABC):
    """
    Abstract base class for a pipeline step. Each step must implement run().
    """
    def __init__(self, params=None):
        """
        Initialize the step with parameters.
        """
        self.params = params if params is not None else {}

    @abstractmethod
    def run(self, data):
        """
        Execute this step's logic on the incoming data.

        Parameters
        ----------
        data : object (e.g., mne.io.Raw, mne.Epochs, or None)
            The data object to process.

        Returns
        -------
        object
            The updated data object after processing.
        """
        pass
