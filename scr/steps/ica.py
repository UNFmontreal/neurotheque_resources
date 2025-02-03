# File: scr/steps/ica.py

import logging
import mne
from mne.preprocessing import ICA
from mne.report import Report
from pathlib import Path
from .base import BaseStep
import pickle
import os
class ICAStep(BaseStep):
    """
    ICA step that logs suggested bad components (EOG/ECG) but does not 
    automatically exclude them. The user can then inspect those suggestions 
    plus the component plots to make a final decision.

    References:
      - Chaumon et al. (2015), J Neurosci Methods
      - Winkler et al. (2015), NeuroImage
      - DSI-24 Technical Specs
    """

    def run(self, data):
        if data is None:
            raise ValueError("[ICAStep] No data provided for ICA.")

        # --------------------------
        # 1) Merge Default Params
        # --------------------------
        default_params = {
            "n_components": 0.99,      # Or an int if you prefer a fixed number
            "method": "infomax",
            "max_iter": 2000,
            "fit_params": {"extended": True, "l_rate": 1e-3},
            "decim": 3,
            "use_good_epochs_only": True,
            "eog_ch_names": ["Fp1", "Fp2"],
            "eog_threshold": 0.5,
            "ecg_channel": None,
            "ecg_threshold": 0.3,
            "plot_dir": ".../reports/ica",
            "interactive": True,
            "exclude": [],              # Pre-exclusions
            "plot_components": True,
            "plot_sources": True,
        }
        params = {**default_params, **self.params}
        sub_id = params.get("subject_id", "unknown")
        ses_id = params.get("session_id", "001")
        paths = params.get("paths", None)
        # --------------------------
        # 2) Instantiate ICA
        # --------------------------
        ica = ICA(
            n_components=params["n_components"],
            method=params["method"],
            max_iter=params["max_iter"],
            fit_params=params["fit_params"],
            random_state=0
        )
        # ica = ICA(
        #     n_components=params["n_components"],
        #     method=params["method"],
        #     random_state=0
        # )
        # --------------------------
        # 3) Select Data for ICA
        # --------------------------
        if params["use_good_epochs_only"]:
            logging.info("[ICAStep] Using only good epochs from AutoReject.")
            autoreject_log = None

            log_dir = paths.get_auto_reject_log_path(sub_id, ses_id)
            log_file = os.path.join(log_dir, "autoreject_log.pickle")

            if os.path.exists(log_file):
                with open(log_file,'rb') as f:
                    autoreject_log = pickle.load(f)
                    logging.info("[ICAStep] AutoReject log loaded.")
                    
            events = mne.make_fixed_length_events(data, duration=1)
            epochs = mne.Epochs(data, events, tmin=0, tmax=1, baseline=None,preload=True)
            good_mask = ~autoreject_log.bad_epochs
            good_epochs = epochs[good_mask] if len(good_mask) == len(epochs) else epochs
        else:
            logging.info("[ICAStep] No (or unused) AutoReject log; using all data for ICA.")
            good_epochs = data

        # --------------------------
        # 4) Fit ICA
        # --------------------------
        # ica.fit(
        #     good_epochs,
        #     decim=params["decim"],
        #     reject=None,
        #     tstep=4.0
        # )
        ica.fit(
            good_epochs,
            decim=params["decim"],
            reject=None,
        )
        # --------------------------
        # 5) Automated Artifact Detection
        # --------------------------
        # Instead of directly excluding, we'll just store candidate indices
        bad_ic_candidates = []  # We'll store dictionaries with type + indices

        # EOG
        eog_indices, eog_scores = [], []
        if params["eog_ch_names"]:
            eog_indices, eog_scores = ica.find_bads_eog(
                good_epochs if params["use_good_epochs_only"] else data,
                ch_name=params["eog_ch_names"],
                threshold=params["eog_threshold"]
            )
            if eog_indices:
                bad_ic_candidates.append({
                    "type": "EOG",
                    "indices": eog_indices,
                    "threshold": params["eog_threshold"]
                })
        
        # ECG
        ecg_indices, ecg_scores = [], []
        if params["ecg_channel"] and (params["ecg_channel"] in data.ch_names):
            ecg_indices, ecg_scores = ica.find_bads_ecg(
                good_epochs if params["use_good_epochs_only"] else data,
                ch_name=params["ecg_channel"],
                threshold=params["ecg_threshold"]
            )
            if ecg_indices:
                bad_ic_candidates.append({
                    "type": "ECG",
                    "indices": ecg_indices,
                    "threshold": params["ecg_threshold"]
                })

        # (Optional) Additional metrics go here; for example:
        # - ADJUST
        # - MARA
        # - FASTER
        # Just store them in bad_ic_candidates in a similar fashion.

        # --------------------------
        # 6) Show Candidate Bad ICs
        # --------------------------
        # We'll start with any user-predefined exclusions
        final_exclude = set(params["exclude"])

        # Print out detection suggestions
        for entry in bad_ic_candidates:
            logging.info(
                f"[ICAStep] Suggested {entry['type']} components (threshold={entry['threshold']}): {entry['indices']}"
            )

        # The user sees these logs, or you can print them to console:
        if bad_ic_candidates:
            print("\n[ICAStep] Candidate bad ICs from automatic detection:")
            for entry in bad_ic_candidates:
                print(f"  {entry['type']} => {entry['indices']} (threshold={entry['threshold']})")
            print("These are NOT excluded yet. You will get a chance to confirm or modify.\n")

        # --------------------------
        # 7) Interactive QA
        # --------------------------
        # Let's optionally plot the "candidate" ICs
        if params["interactive"]:
            # For example, you might unify all candidate indices to plot them in one go:
            union_candidates = set()
            for entry in bad_ic_candidates:
                union_candidates.update(entry["indices"])

            # If you want to see topographies of the candidate ICs (union of EOG, ECG, etc.):
            if union_candidates:
                ica.plot_properties(good_epochs, picks=sorted(list(union_candidates)))

            # Also show any pre-excluded from user params if you like
            if final_exclude:
                ica.plot_properties(good_epochs, picks=sorted(list(final_exclude)))

            # Now prompt the user
            suggested_str = (
                f"\nSuggested bad ICs from detection: {union_candidates}" 
                if union_candidates else "None"
            )
            print(suggested_str)

            # The user can override or add anything:
            user_input = input("Enter ALL IC indices to exclude (comma-separated), or press Enter to skip: ")
            if user_input.strip():
                # Overwrite final_exclude with user input
                final_exclude = set(int(x) for x in user_input.split(","))
        else:
            logging.info("[ICAStep] Non-interactive mode. Using only 'exclude' param from YAML.")

        # Assign to ica.exclude
        ica.exclude = sorted(list(final_exclude))
        logging.info(f"[ICAStep] Final exclusion list: {ica.exclude}")

        # --------------------------
        # 8) Apply ICA
        # --------------------------
        data_clean = ica.apply(data.copy())
        ica_dir=paths.get_derivative_path(sub_id, ses_id)/f'sub-{sub_id}_ses-{ses_id}_desc-ica_cleaned.fif'
        data_clean.save(ica_dir, overwrite=True)
        # --------------------------
        # 9) Generate QA Report
        # --------------------------
        self._generate_report(ica, data_clean, params,paths)

        # --------------------------
        # 10) Store Metadata
        # --------------------------
        if not hasattr(data_clean.info, "temp"):
            data_clean.info["temp"] = {}
        data_clean.info["temp"]["ica"] = {
            "excluded": ica.exclude,
            "n_components": params["n_components"]
        }

        return data_clean

    def _generate_report(self, ica, data, params,paths):
        """Create an MNE Report summarizing ICA."""
        from mne.report import Report
        import matplotlib.pyplot as plt
        
        rep_dir = paths.get_ica_report_dir(params["subject_id"], params["session_id"])
        
        report = Report(title="ICA Quality Report", verbose=False)

        report.add_ica(
            ica=ica,
            title="ICA cleaning",
            picks=None,  # plot the excluded EOG components
            inst=data,
            n_jobs=None,  # could be increased!
        )
        out_file = rep_dir / "ica_report.html"
        report.save(out_file, overwrite=True, open_browser=False)
        logging.info(f"[ICAStep] ICA report saved at {out_file}")