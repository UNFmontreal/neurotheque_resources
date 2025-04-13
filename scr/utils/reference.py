# File: scr/utils/reference.py

import logging
import mne

def apply_reference(data, params=None):
    """
    Re-references EEG data according to user-specified method.
    
    Parameters
    ----------
    data : mne.io.Raw or mne.Epochs
        The MNE Raw or Epochs object to be re-referenced
    params : dict
        Dictionary with referencing parameters:
        - method : str
            Either "average" for average reference (default) or "channels" for
            custom reference channels.
        - channels : list of str
            Which channel(s) to use if method="channels". Required when method="channels".
        - projection : bool
            If True, add a projection to do the re-reference rather than directly
            modifying data. (default: False)
    
    Returns
    -------
    data : mne.io.Raw or mne.Epochs
        The re-referenced data
    
    Notes
    -----
    For systems that already have a hardware reference (like Pz), you might want 
    to change to a different reference scheme based on your analysis needs.
    """
    if data is None:
        raise ValueError("[apply_reference] No data to re-reference.")
    
    # Default parameters
    if params is None:
        params = {}
    
    method = params.get("method", "average")
    channels = params.get("channels", [])
    projection = params.get("projection", False)
    
    logging.info(f"[apply_reference] Re-referencing method={method}, projection={projection}")
    
    if method == "average":
        # set_eeg_reference(ref_channels="average") => average re-ref
        logging.info("[apply_reference] Using average reference for EEG channels.")
        try:
            data.set_eeg_reference(ref_channels="average", projection=projection)
        except Exception as e:
            logging.error(f"[apply_reference] Error applying average reference: {str(e)}")
            raise
    
    elif method == "channels":
        if not channels:
            raise ValueError("[apply_reference] method='channels' requires 'channels' param.")
            
        # Check if all reference channels exist in the data
        missing_channels = [ch for ch in channels if ch not in data.ch_names]
        if missing_channels:
            error_msg = f"[apply_reference] Reference channel(s) {missing_channels} not found in data. Available channels: {data.ch_names}"
            logging.error(error_msg)
            raise ValueError(error_msg)
            
        logging.info(f"[apply_reference] Using custom channels {channels} for reference.")
        try:
            data.set_eeg_reference(ref_channels=channels, projection=projection)
        except Exception as e:
            logging.error(f"[apply_reference] Error applying custom channel reference: {str(e)}")
            raise
    
    else:
        raise ValueError(f"[apply_reference] Unknown re-reference method '{method}'.")
    
    # MNE might add new reference channels to the data if channels were used.
    # If projection=True, we have an EEG ref projection added but not applied
    # until you do e.g. data.apply_proj().
    
    logging.info("[apply_reference] Re-reference complete.")
    return data 