# File: src/steps/autoreject.py

import logging
import mne
from autoreject import AutoReject
from .base import BaseStep

class AutoRejectStep(BaseStep):
    """
    A professional AutoReject step for EEG pipelines.

    This step can:
      - Create short (1s) epochs from the incoming data (Raw or continuous Epochs).
      - Fit AutoReject on those short epochs to estimate artifact thresholds.
      - Optionally transform (clean) those short epochs and store the reject log.

    Expected params:
    --------------------------------------------------------------------------
    ar_params (dict):
        Dictionary of parameters passed directly to AutoReject's constructor,
        e.g. 'consensus', 'n_interpolate', 'thresh_func', 'n_jobs', etc.
        Example:
          ar_params:
            consensus: 0.5
            n_interpolate: 4
            random_state: 42

    mode (str): "fit" or "fit_transform"
        "fit" (default):
            Fits AutoReject on short epochs but does NOT transform them,
            i.e., no interpolation or epoch dropping is applied.
            Only logs the reject thresholds.
        "fit_transform":
            Additionally calls 'ar.transform(...)' on the ephemeral epochs,
            so the short segments are “cleaned.”
            Potentially you could store the result or the reject_log for future usage.

    store_log (bool):
        If True, stores the reject log in data.info['autoreject_log'] for reference.
        Default is False.

    Usage in YAML:
    --------------------------------------------------------------------------
    pipeline:
      steps:
        - name: AutoRejectStep
          params:
            ar_params:
              consensus: 0.5
              n_interpolate: 5
              random_state: 42
            mode: "fit_transform"
            store_log: true
    """

    def run(self, data):
        if data is None:
            raise ValueError("[AutoRejectStep] No data available for AutoReject.")

        # 1) Unpack parameters
        ar_params = self.params.get("ar_params", {})
        # mode = self.params.get("mode", "fit")  # either "fit" or "fit_transform"
        store_log = self.params.get("store_log", False)

        # 2) Create short ephemeral epochs for AutoReject
        #    Typically 1s windows are used for continuous artifact detection
        logging.info("[AutoRejectStep] Creating 1-second epochs for AR fitting.")
        events_tmp = mne.make_fixed_length_events(data, duration=1)
        epochs_tmp = mne.Epochs(
            data,
            events_tmp,
            tmin=0,
            tmax=1,
            baseline=None,
            detrend=0,
            preload=True
        )

        # 3) Initialize and fit
        logging.info(f"[AutoRejectStep] Initializing AutoReject with params: {ar_params}")
        # ar = AutoReject(**ar_params)
        ar = AutoReject()
        logging.info("[AutoRejectStep] Fitting AutoReject on short epochs.")
        ar.fit(epochs_tmp)

        # # 4) Optional transform
        # if mode == "fit_transform":
        #     logging.info("[AutoRejectStep] Applying transform on ephemeral epochs.")
        #     epochs_tmp_clean = ar.transform(epochs_tmp)
        #     # In principle, you could store these cleaned short epochs somewhere or
        #     # merge them back into data, but typically this is just demonstration or QA.

        # 5) Optionally store the reject log in data.info
        if store_log:
            reject_log = ar.get_reject_log(epochs_tmp)
            data.info["autoreject_log"] = reject_log
            logging.info("[AutoRejectStep] Stored reject log in data.info['autoreject_log'].")

        logging.info("[AutoRejectStep] AutoReject finished. "
                     "Thresholds can be used in condition-specific epochs or QA steps.")
        return data
