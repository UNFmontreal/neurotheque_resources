# File: eeg_pipeline/src/steps/__init__.py
"""
Initialization file for the steps package.

This file imports and registers all step classes in the STEP_REGISTRY so
that the pipeline can reference them by name without extra imports.
"""

import logging

from ..pipeline import STEP_REGISTRY

# Import each step class:
from .base import BaseStep
from .load import LoadData
from .filter import FilterStep
from .ica import ICAStep
from .autoreject import AutoRejectStep
from .save import SaveData
from .reference import ReferenceStep
from .save_checkpoint import SaveCheckpoint

# If you have additional steps:
from .prepchannels import PrepChannelsStep
# from .splittasks import SplitTasksStep
# etc...    
from .splittasks_dynamic import SplitTasksStep
# If you have specialized steps for analysis:
try:
    from .triggers_gonogo import GoNoGoTriggerStep
    from .epoching_gonogo import GoNoGoEpochingStep
except ImportError:
    logging.warning("GoNoGo specialized steps not found. Skipping...")


# Register them in the global STEP_REGISTRY
STEP_REGISTRY.update({
    "LoadData": LoadData,
    "ReferenceStep": ReferenceStep,
    "FilterStep": FilterStep,
    "ICAStep": ICAStep,
    "AutoRejectStep": AutoRejectStep,
    "SaveCheckpoint": SaveCheckpoint,
    "SaveData": SaveData,
    "PrepChannelsStep": PrepChannelsStep,
    "SplitTasksStep": SplitTasksStep,
    # If you have them:
    "GoNoGoTriggerStep": GoNoGoTriggerStep,
    "GoNoGoEpochingStep": GoNoGoEpochingStep,
})

logging.info("[__init__.py] All step classes have been registered in STEP_REGISTRY.")
