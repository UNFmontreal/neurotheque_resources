# File: src/steps/autoreject.py

import logging
import mne
from autoreject import AutoReject
from .base import BaseStep
import json
import os
import pickle
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
            mode: "transform"
            store_log: true
    """

    def run(self, data):
        if data is None:
            raise ValueError("[AutoRejectStep] No data provided.")

        ar_params = self.params.get("ar_params", {})
        store_log = self.params.get("store_log", False)

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

        logging.info(f"[AutoRejectStep] Initializing AutoReject with params: {ar_params}")
        ar = AutoReject(**ar_params)
        # ar = AutoReject()
        logging.info("[AutoRejectStep] Fitting AutoReject on short epochs.")
        ar.fit(epochs_tmp)
        reject_log = ar.get_reject_log(epochs_tmp)
        # reject_log = [1,2,3] #for debugging
        logging.info("[AutoRejectStep] AutoReject thresholds:")
        
        sub_id = self.params.get("subject_id", "unknown")
        ses_id = self.params.get("session_id", "001")
        paths = self.params.get("paths", None)
        fig=reject_log.plot("horizontal",show=False)
        fig_dir = paths.get_autoreject_report_dir (sub_id, ses_id)
        fig.savefig(fig_dir / "autoreject_thresholds.png")
        # 4)  store the reject log in data.info
        if store_log:
            log_dir = paths.get_auto_reject_log_path(sub_id, ses_id)
            log_file=os.path.join(log_dir,"autoreject_log.pickle")
            with open(log_file, 'wb') as f:
                pickle.dump(reject_log, f)
                logging.info(f"[AutoRejectStep] AutoReject log saved to {log_dir}")
        logging.info("[AutoRejectStep] AutoReject finished.")
        return data
