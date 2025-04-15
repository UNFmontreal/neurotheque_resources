#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for MNE-Python data handling.
"""

import logging
import mne
import numpy as np

def clean_info(info):
    """
    Convert any fixed-length string arrays in the info to lists of Python strings.
    
    This function addresses the 'data type >a not understood' error by converting
    problematic NumPy string arrays to standard Python strings.
    
    Parameters
    ----------
    info : mne.Info or dict
        The MNE info object or dictionary to clean
    
    Returns
    -------
    None
        The function modifies the info object in place
    """
    for key in list(info.keys()):
        value = info[key]
        if isinstance(value, np.ndarray):
            # Check if the dtype is of type fixed-length bytes ('S') or Unicode ('U')
            if value.dtype.kind in ('S', 'U'):
                # Convert the array to a list of standard Python strings
                info[key] = value.astype(str).tolist()
        elif isinstance(value, dict):
            # Recursively clean dictionary values
            clean_info(value)
        elif isinstance(value, list):
            # Iterate over list items, if an item is a numpy array then clean it
            new_list = []
            for item in value:
                if isinstance(item, np.ndarray) and item.dtype.kind in ('S', 'U'):
                    new_list.append(item.astype(str).tolist())
                elif isinstance(item, dict):
                    clean_info(item)
                    new_list.append(item)
                else:
                    new_list.append(item)
            info[key] = new_list

def clean_mne_object(obj):
    """
    Clean an MNE object to ensure it can be serialized properly.
    
    Parameters
    ----------
    obj : mne object
        The MNE object to clean (Raw, Epochs, Evoked, ICA, etc.)
    
    Returns
    -------
    mne object
        The cleaned MNE object
    """
    try:
        # Create a copy of the object
        clean_obj = obj.copy()
        
        # Clean the info object if it exists
        if hasattr(clean_obj, 'info'):
            clean_info(clean_obj.info)
        
        return clean_obj
    except Exception as e:
        logging.error(f"Error cleaning MNE object: {e}")
        return obj

def clean_info_for_saving(info):
    """
    Clean mne.Info object to ensure it can be serialized properly in fif format.
    
    This function preserves as much information as possible while fixing only
    problematic fields that cause serialization errors.
    
    Parameters
    ----------
    info : mne.Info
        The original Info object to be fixed
    
    Returns
    -------
    mne.Info
        A fixed version of the info object that can be safely serialized
    """
    try:
        # Create a copy of the original info to preserve as much as possible
        fixed_info = info.copy()
        
        # Fix only the problematic string fields
        if 'subject_info' in fixed_info:
            # Convert all string values to ASCII
            subject_info = fixed_info['subject_info']
            if subject_info is not None:
                for key, value in subject_info.items():
                    if isinstance(value, str):
                        subject_info[key] = value.encode('ascii', 'replace').decode('ascii')
        
        # Ensure description field is ASCII
        if 'description' in fixed_info and fixed_info['description'] is not None:
            if isinstance(fixed_info['description'], str):
                fixed_info['description'] = fixed_info['description'].encode('ascii', 'replace').decode('ascii')
        
        # Ensure experimenter field is ASCII
        if 'experimenter' in fixed_info and fixed_info['experimenter'] is not None:
            if isinstance(fixed_info['experimenter'], str):
                fixed_info['experimenter'] = fixed_info['experimenter'].encode('ascii', 'replace').decode('ascii')
        
        return fixed_info
    except Exception as e:
        logging.error(f"Error fixing info for saving: {e}")
        logging.info("Falling back to creating minimal info")
        
        # Get essential info fields
        ch_names = info['ch_names']
        sfreq = info['sfreq']
        ch_types = ['eeg'] * len(ch_names)  # Assume EEG for all channels
        
        # Create new minimal info only as a last resort
        minimal_info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        
        # Copy a few essential fields if they exist
        if 'bads' in info:
            minimal_info['bads'] = info['bads']
        
        return minimal_info

def clean_ica_for_saving(ica):
    """
    Clean ICA object to ensure it can be serialized properly.
    
    Parameters
    ----------
    ica : mne.preprocessing.ICA
        The ICA object to clean
    
    Returns
    -------
    mne.preprocessing.ICA
        A cleaned version of the ICA object
    """
    try:
        # Create a copy of the ICA object
        clean_ica = ica.copy()
        
        # Clean the info object if it exists
        if hasattr(clean_ica, 'info'):
            clean_ica.info = clean_info_for_saving(clean_ica.info)
        
        return clean_ica
    except Exception as e:
        logging.error(f"Error cleaning ICA object: {e}")
        return ica

def clean_epochs_for_saving(epochs):
    """
    Clean epochs object to ensure it can be serialized properly.
    
    Parameters
    ----------
    epochs : mne.Epochs
        The epochs object to clean
    
    Returns
    -------
    mne.Epochs
        A cleaned version of the epochs object
    """
    try:
        # Create a copy of the epochs object
        clean_epochs = epochs.copy()
        
        # Clean the info object
        clean_epochs.info = clean_info_for_saving(clean_epochs.info)
        
        return clean_epochs
    except Exception as e:
        logging.error(f"Error cleaning epochs object: {e}")
        return epochs 