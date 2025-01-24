# File: src/steps/ica.py

import logging
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
from .base import BaseStep

class ICAStep(BaseStep):
    """
    Step to perform ICA on raw EEG data, optionally detect EOG components,
    plot components/sources for user inspection, and exclude certain comps.

    Expected params (all are optional, with defaults):
    --------------------------------------------------------------------------
    n_components (float/int): 
        Number of ICA components, e.g. 0.95 or an integer (default: 0.95).

    random_state (int): 
        Random seed for ICA reproducibility (default: 42).

    decim (int): 
        Decimation factor to speed up ICA fitting (default: 3).

    use_eog_detection (bool): 
        If True, attempts to auto-detect EOG components with find_bads_eog() 
        (default: False).

    eog_ch_name (str or list): 
        Channel(s) to use for EOG detection if use_eog_detection=True 
        (default: None, e.g. 'Fp1' or an actual EOG channel).

    plot_components (bool): 
        If True, calls ica.plot_components() for user inspection (default: False).

    plot_sources (bool): 
        If True, calls ica.plot_sources(data) for user inspection (default: False).

    interactive (bool): 
        If True, prompts user to type which components to exclude 
        after any auto-detection/plotting (default: False).

    exclude (list): 
        A list of component indices to exclude automatically (default: []).

    Example usage in YAML:
    --------------------------------------------------------------------------
    pipeline:
      steps:
        - name: ICAStep
          params:
            n_components: 0.95
            random_state: 42
            decim: 3
            use_eog_detection: True
            eog_ch_name: "Fp1"
            plot_components: True
            plot_sources: True
            interactive: True
            exclude: [0, 2]  # Additional known bad comps
    """

    def run(self, data):
        if data is None:
            raise ValueError("[ICAStep] No data provided for ICA.")

        # Extract parameters (with defaults)
        n_components        = self.params.get("n_components", 0.95)
        random_state        = self.params.get("random_state", 42)
        decim               = self.params.get("decim", 3)
        use_eog_detection   = self.params.get("use_eog_detection", False)
        eog_ch_name         = self.params.get("eog_ch_name", None)
        plot_comps          = self.params.get("plot_components", False)
        plot_srcs           = self.params.get("plot_sources", False)
        interactive         = self.params.get("interactive", False)
        exclude_list        = self.params.get("exclude", [])
        
        if "autoreject_log" not in data.info.get("temp", {}):
            logging.info(f"[ICAStep] Fitting ICA (n_components={n_components}, decim={decim})...")
            ica = ICA(n_components=n_components, random_state=random_state)
            ica.fit(data, decim=decim)
            logging.info("[ICAStep] ICA fit complete.")
        else:
            reject_log = data.info["temp"]["autoreject_log"]
            events = mne.make_fixed_length_events(data, duration=1)
            epochs = mne.Epochs(data, events, tmin=0, tmax=1,
                           baseline=None, detrend=0, preload=True)
            good_epochs = epochs[~reject_log.bad_epochs]

            logging.info(f"[ICAStep] Fitting ICA (n_components={n_components}, decim={decim})...")
            ica = ICA(n_components=n_components, random_state=random_state)
            ica.fit(good_epochs, decim=decim)
            logging.info("[ICAStep] ICA fit complete.")

        # 2) Optional EOG-based detection
        auto_suggested_comps = []
        if use_eog_detection and eog_ch_name:
            logging.info(f"[ICAStep] Attempting EOG detection with eog_ch_name={eog_ch_name}...")
            eog_indices, eog_scores = ica.find_bads_eog(data, ch_name=eog_ch_name)
            if eog_indices:
                auto_suggested_comps = eog_indices
                logging.info(f"[ICAStep] EOG-based detection suggests comps {auto_suggested_comps}")
            else:
                logging.info("[ICAStep] EOG detection found no obvious comps.")
        else:
            logging.info("[ICAStep] Skipping EOG detection.")

        # 3) Plot components or sources if requested
        if plot_comps:
            logging.info("[ICAStep] Plotting ICA components.")
            ica.plot_components(inst=data)  # pass inst to get sensor layout
            plt.show(block=True)

        if plot_srcs:
            logging.info("[ICAStep] Plotting ICA sources.")
            ica.plot_sources(data)
            plt.show(block=True)

        # 4) Merge auto-suggested with user config 'exclude' param
        combined_exclude = list(set(auto_suggested_comps + exclude_list))

        # 5) If interactive => ask user to confirm/adjust final exclude
        final_exclude = combined_exclude
        if interactive:
            logging.info(f"[ICAStep] Interactive mode: current exclude list {combined_exclude}.")
            user_input = input("Enter additional component indices to exclude (comma-separated), or press Enter to skip: ")
            if user_input.strip():
                user_comps = [int(x.strip()) for x in user_input.split(',')]
                final_exclude = list(set(final_exclude + user_comps))

            # Optionally let user remove comps if they decided they're not artifact
            logging.info(f"[ICAStep] Current final exclude list: {final_exclude}.")
            user_input_remove = input("Enter any component indices to *remove* from exclude list (comma-separated), or Enter to skip: ")
            if user_input_remove.strip():
                remove_comps = [int(x.strip()) for x in user_input_remove.split(',')]
                final_exclude = [c for c in final_exclude if c not in remove_comps]

        # 6) Exclude final set
        logging.info(f"[ICAStep] Excluding components: {final_exclude}")
        ica.exclude = final_exclude

        # 7) Apply
        ica.apply(data)
        logging.info("[ICAStep] ICA application complete.")

        return data
