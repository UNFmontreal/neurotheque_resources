#!/usr/bin/env python3
"""
Generate a YAML configuration file for the Neurotheque pipeline.
This script reads debug_log.txt to extract successful parameters
and creates a corresponding YAML configuration file.
"""

import os
import re
import yaml
import logging
from pathlib import Path

def parse_log_file(log_file="debug_log.txt"):
    """Parse the debug log file to extract parameters."""
    if not os.path.exists(log_file):
        print(f"Error: Log file {log_file} not found.")
        return None
    
    # Initialize parameters dictionary
    parameters = {
        "load": {},
        "prepchannels": {},
        "reference": {},
        "filter": {},
        "ica": {},
        "autoreject": {},
        "epoching": {},
    }
    
    # Extract parameters from log file
    with open(log_file, 'r') as f:
        log_content = f.read()
        
        # Extract input file
        input_file_match = re.search(r'Loading data from: ([^\n]+)', log_content)
        if input_file_match:
            parameters['load']['input_file'] = input_file_match.group(1)
        
        # Extract filtering parameters
        filter_match = re.search(r'l_freq": ([0-9.]+).+h_freq": ([0-9.]+).+notch_freqs": \[([^\]]+)\]', log_content, re.DOTALL)
        if filter_match:
            parameters['filter']['l_freq'] = float(filter_match.group(1))
            parameters['filter']['h_freq'] = float(filter_match.group(2))
            notch_freqs_str = filter_match.group(3).strip()
            if notch_freqs_str:
                parameters['filter']['notch_freqs'] = [float(f) for f in notch_freqs_str.split(',')]
        
        # Extract ICA parameters
        ica_match = re.search(r'n_components": ([0-9.]+).+method": "([^"]+)"', log_content, re.DOTALL)
        if ica_match:
            parameters['ica']['n_components'] = int(float(ica_match.group(1)))
            parameters['ica']['method'] = ica_match.group(2)
        
        # Extract EOG channels
        eog_match = re.search(r'Using EOG channels: \[([^\]]+)\]', log_content)
        if eog_match:
            eog_channels_str = eog_match.group(1).replace("'", "").strip()
            if eog_channels_str:
                parameters['ica']['eog_ch_names'] = [c.strip() for c in eog_channels_str.split(',')]
        
        # Extract AutoReject parameters
        ar_match = re.search(r'n_interpolate": \[([^\]]+)\].+consensus": \[([^\]]+)\]', log_content, re.DOTALL)
        if ar_match:
            parameters['autoreject']['n_interpolate'] = [int(n) for n in ar_match.group(1).split(',')]
            parameters['autoreject']['consensus'] = [float(c) for c in ar_match.group(2).split(',')]
        
        # Extract epoching parameters
        tmin_match = re.search(r'"tmin": (-?[0-9.]+)', log_content)
        tmax_match = re.search(r'"tmax": ([0-9.]+)', log_content)
        if tmin_match and tmax_match:
            parameters['epoching']['tmin'] = float(tmin_match.group(1))
            parameters['epoching']['tmax'] = float(tmax_match.group(1))
        
        # Extract events information
        events_match = re.search(r'Unique event IDs: \[([^\]]+)\]', log_content)
        if events_match:
            event_ids_str = events_match.group(1).strip()
            if event_ids_str:
                event_ids = [int(e) for e in event_ids_str.split()]
                event_id_dict = {f"event_{e}": e for e in event_ids}
                parameters['epoching']['event_id'] = event_id_dict
    
    return parameters

def generate_pipeline_config(parameters, output_file="pipeline_config.yaml"):
    """Generate a pipeline configuration file from parameters."""
    if parameters is None:
        print("Error: No parameters provided.")
        return False
    
    # Create pipeline configuration
    pipeline_config = {
        "directory": {
            "root": "./data",
            "raw_data_dir": "raw",
            "processed_dir": "processed",
            "reports_dir": "reports",
            "derivatives_dir": "derivatives"
        },
        "pipeline_mode": "restart",
        "default_subject": "01",
        "default_session": "001",
        "default_run": "01",
        "auto_save": True,
        "file_path_pattern": parameters['load'].get('input_file', "data/pilot_data/*.edf"),
        "pipeline": {
            "steps": [
                {
                    "name": "LoadData",
                    "params": {
                        "stim_channel": "auto"
                    }
                },
                {
                    "name": "PrepChannelsStep",
                    "params": {
                        "rename_dict": None,
                        "drop_channels": ["PHOTO"] if "drop_channels" not in parameters['prepchannels'] else parameters['prepchannels']['drop_channels']
                    }
                },
                {
                    "name": "ReferenceStep",
                    "params": {
                        "reference": parameters['reference'].get('reference', "average")
                    }
                },
                {
                    "name": "FilterStep",
                    "params": {
                        "l_freq": parameters['filter'].get('l_freq', 1.0),
                        "h_freq": parameters['filter'].get('h_freq', 40.0),
                        "notch_freqs": parameters['filter'].get('notch_freqs', [50])
                    }
                },
                {
                    "name": "ICAStep",
                    "params": {
                        "n_components": parameters['ica'].get('n_components', 15),
                        "method": parameters['ica'].get('method', "fastica"),
                        "max_iter": 500,
                        "fit_params": {"extended": True},
                        "eog_ch_names": parameters['ica'].get('eog_ch_names', []),
                        "interactive": False,
                        "plot_components": True,
                        "plot_sources": True
                    }
                },
                {
                    "name": "AutoRejectStep",
                    "params": {
                        "ar_params": {
                            "n_interpolate": parameters['autoreject'].get('n_interpolate', [1, 4]), 
                            "consensus": parameters['autoreject'].get('consensus', [0.2, 0.5])
                        },
                        "mode": "fit",
                        "plot_results": True,
                        "store_reject_log": True,
                        "save_model": True
                    }
                },
                {
                    "name": "EpochingStep",
                    "params": {
                        "tmin": parameters['epoching'].get('tmin', -0.2),
                        "tmax": parameters['epoching'].get('tmax', 0.5),
                        "baseline": [None, 0],
                        "preload": True,
                        "event_id": parameters['epoching'].get('event_id', None),
                        "reject_by_annotation": True
                    }
                },
                {
                    "name": "SaveData",
                    "params": {
                        "output_file": "processed_data.fif",
                        "overwrite": True
                    }
                }
            ]
        }
    }
    
    # Write configuration to file
    with open(output_file, 'w') as f:
        yaml.dump(pipeline_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Pipeline configuration saved to {output_file}")
    return True

def main():
    # Parse log file
    print("Parsing debug log...")
    parameters = parse_log_file()
    
    if parameters:
        # Generate pipeline configuration
        output_file = "pipeline_config.yaml"
        if generate_pipeline_config(parameters, output_file):
            print(f"Success! Pipeline configuration saved to {output_file}")
            print("You can use this configuration with the pipeline by running:")
            print(f"python scr/pipeline.py --config {output_file}")
    else:
        print("Error: Failed to extract parameters from log file.")
        print("Make sure you've run the test script first to generate debug_log.txt")

if __name__ == "__main__":
    main() 