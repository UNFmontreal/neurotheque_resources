# src/steps/epoching.py

import mne
import numpy as np
import logging
import pandas as pd
from .base import BaseStep
from pathlib import Path
import os

class EpochingStep(BaseStep):
    """
    Flexible epoching step that supports multiple task-specific epoching strategies.
    
    This step handles:
    1. Continuous data extraction between start/end triggers
    2. Event-related epoching around specific triggers
    3. Task-specific epoching methods (5-point test, go/no-go, etc.)
    
    Parameters:
    -----------
    task_type : str
        Type of task to epoch ('5pt', 'gng', 'continuous', 'custom')
    
    trigger_ids : dict
        Dictionary mapping trigger names to their numeric IDs
        Example: {'start': 1, 'end': 2, 'onset': 3, 'response': 4}
    
    epoch_params : dict
        Dictionary containing epoching parameters:
        - tmin: start time in seconds relative to trigger (default: -0.2)
        - tmax: end time in seconds relative to trigger (default: 1.0)
        - baseline: baseline correction period (default: (None, 0))
        - preload: whether to preload data (default: True)
        - reject_by_annotation: whether to reject epochs with annotations (default: True)
    
    extract_continuous : bool
        If True, extracts continuous data between start/end triggers (default: False)
    
    returns_epochs : bool
        If True, returns Epochs object; if False, returns Raw object with selections (default: True)
    
    event_id : dict or None
        Custom event_id dictionary for MNE Epochs (default: None)
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        logging.info(f"[EpochingStep.__init__] Initialized with params: {self.params}")
    
    def run(self, data):
        if data is None:
            raise ValueError("[EpochingStep] No data provided.")
        
        logging.info(f"[EpochingStep.run] Running with params: {self.params}")
        
        # Get parameters
        task_type = self.params.get("task_type", "custom")
        trigger_ids = self.params.get("trigger_ids", {})
        epoch_params = self.params.get("epoch_params", {})
        extract_continuous = self.params.get("extract_continuous", False)
        returns_epochs = self.params.get("returns_epochs", True)
        
        # Set up default epoching parameters
        tmin = epoch_params.get("tmin", -0.2)
        tmax = epoch_params.get("tmax", 1.0)
        baseline = epoch_params.get("baseline", (None, 0))
        preload = epoch_params.get("preload", True)
        reject_by_annotation = epoch_params.get("reject_by_annotation", True)
        
        # Get plotting parameters
        auto_plot = self.params.get("auto_plot", False)
        plot_params = self.params.get("plot_params", {})
        
        logging.info(f"[EpochingStep.run] Task type: {task_type}, Trigger IDs: {trigger_ids}")
        logging.info(f"[EpochingStep.run] Epoch params: tmin={tmin}, tmax={tmax}, baseline={baseline}")
        
        # Route to task-specific epoching method
        if task_type.lower() == "5pt":
            result = self._epoch_five_point_test(data, trigger_ids, tmin, tmax, baseline, preload, 
                                              reject_by_annotation, extract_continuous, returns_epochs)
        elif task_type.lower() == "gng":
            result = self._epoch_go_nogo(data, trigger_ids, tmin, tmax, baseline, preload, 
                                      reject_by_annotation, extract_continuous, returns_epochs)
        elif task_type.lower() == "continuous":
            result = self._extract_continuous_segment(data, trigger_ids, preload)
        elif task_type.lower() == "fixed":
            result = self._epoch_fixed_length(data, trigger_ids, epoch_params, reject_by_annotation, 
                                           extract_continuous, returns_epochs)
        else:
            # Generic/custom epoching
            result = self._epoch_custom(data, trigger_ids, tmin, tmax, baseline, preload, 
                                     reject_by_annotation, returns_epochs)
        
        # Optionally plot the results if auto_plot is True and we have epochs
        if auto_plot and hasattr(result, 'event_id') and returns_epochs:
            logging.info("[EpochingStep.run] Auto-plotting enabled, generating plots")
            
            # Get plotting parameters with defaults
            plot_type = plot_params.get("plot_type", "average")
            event_names = plot_params.get("event_names", None)
            channels = plot_params.get("channels", None)
            time_window = plot_params.get("time_window", None)
            combine = plot_params.get("combine", False)
            title = plot_params.get("title", None)
            
            # Generate plots
            figures = self.plot_epochs(
                result, 
                plot_type=plot_type,
                event_names=event_names,
                channels=channels,
                time_window=time_window,
                combine=combine,
                title=title
            )
            
            # Store the figures in the result's metadata if available
            if hasattr(result, 'metadata') and result.metadata is not None:
                result.metadata['figures'] = figures
            
            # If save_plots is specified, save the figures
            save_plots = plot_params.get("save_plots", False)
            if save_plots:
                self._save_figures(figures, plot_params.get("save_dir", None))
        
        # Visualize the detected events to verify correct detection
        if self.params.get("visualize_events", False):
            # Get visualization parameters
            plot_params = self.params.get("plot_params", {})
            duration = plot_params.get("duration", 10.0)  # Default: 10s window
            tstart = plot_params.get("tstart", 0.0)       # Default: start at 0s
            interactive = plot_params.get("interactive", False)  # Default: non-interactive
            save_fig = {
                'save': plot_params.get("save_plots", False),
                'dir': plot_params.get("save_dir", "figures/events")
            }
            
            # Use the simple plot function with new parameters
            fig = self.plot_events_simple(
                data, 
                events, 
                stim_channel,
                duration=duration,
                tstart=tstart,
                save_fig=save_fig,
                interactive=interactive
            )
            
            if fig:
                logging.info(f"[EpochingStep] Simple event visualization created: {duration}s window from {tstart}s")
        
        return result

    def _epoch_five_point_test(self, data, trigger_ids, tmin, tmax, baseline, preload, 
                              reject_by_annotation, extract_continuous, returns_epochs):
        """
        Epoch data for the 5-point test task.
        
        For 5PT task:
        - All triggers have the same ID (8)
        - Simple pattern-based approach:
          - First event = Start trigger
          - Last event = End trigger
          - Events in between alternate: Onset -> Response -> Onset -> Response, etc.
          - Ignore any orphaned onset without corresponding response
        """
        # Log the epoching strategy
        logging.info(f"[EpochingStep] Epoching 5-point test data using pattern-based approach")
        
        # Get trigger ID (same for all events)
        trigger_id = trigger_ids.get("trigger_id", 8)  # Default to 8 if not specified
        
        # Find events in the data
        stim_channel = self.params.get("stim_channel", "Trigger")
        events = mne.find_events(data, stim_channel=stim_channel, shortest_event=1, verbose=True)
        
        if len(events) == 0:
            logging.error(f"[EpochingStep] No events found with stim_channel={stim_channel}. Available channels: {data.ch_names}")
            # Try with STI as a fallback
            if 'STI' in data.ch_names and stim_channel != 'STI':
                logging.info("[EpochingStep] Trying with STI channel as fallback")
                events = mne.find_events(data, stim_channel='STI', shortest_event=1, verbose=True)
        
        logging.info(f"[EpochingStep] Found {len(events)} total events")
        
        if len(events) == 0:
            logging.warning("[EpochingStep] No events found in the data, cannot epoch")
            return data
        
        if len(events) < 3:  # Need at least start, one pair (onset+response), and end
            logging.warning(f"[EpochingStep] Not enough events found ({len(events)}), need at least 4")
            return data
            
        # Get sampling frequency
        sfreq = data.info['sfreq']
        
        # Using pattern-based approach to identify events:
        # - First event = Start
        # - Last event = End
        # - Events in between alternate: Onset -> Response -> Onset -> Response...
        
        # Identify start and end events
        start_event = events[0].copy()
        end_event = events[-1].copy()
        
        # Log start/end timing
        start_time = start_event[0] / sfreq
        end_time = end_event[0] / sfreq
        logging.info(f"[EpochingStep] Start event at {start_time:.2f}s, End event at {end_time:.2f}s")
        
        # STEP 1: Always extract the continuous segment from start to end
        # Apply optional buffer before/after (can be configured in parameters)
        buffer_pre = self.params.get("buffer_pre", 0.0)  # seconds before start
        buffer_post = self.params.get("buffer_post", 0.0)  # seconds after end
        
        # Calculate times with buffer
        start_time_with_buffer = max(0, start_time - buffer_pre)  # Ensure we don't go below 0
        end_time_with_buffer = min(end_time + buffer_post, data.times[-1])  # Don't exceed data length
        
        # Crop the data
        data_cropped = data.copy().crop(tmin=start_time_with_buffer, tmax=end_time_with_buffer)
        
        # Recalculate events after cropping to ensure correct timing
        if self.params.get("stim_channel", "Trigger") in data_cropped.ch_names:
            logging.info("[EpochingStep] Recalculating events after cropping")
            events = mne.find_events(
                data_cropped, 
                stim_channel=self.params.get("stim_channel", "Trigger"),
                shortest_event=1,
                verbose=False
            )
            
            # Identify start and end events in the cropped data
            if len(events) >= 2:
                # Use the first and last events as start and end
                start_event = events[0]
                end_event = events[-1]
                # Update times
                start_time = start_event[0] / sfreq
                end_time = end_event[0] / sfreq
                logging.info(f"[EpochingStep] Updated start_time={start_time:.2f}s and end_time={end_time:.2f}s")
            else:
                logging.warning("[EpochingStep] Could not find enough events in cropped data")
        else:
            logging.warning(f"[EpochingStep] Stim channel {self.params.get('stim_channel', 'Trigger')} not found in cropped data")
        
        # Extract middle events
        if len(events) >= 3:
            middle_events = events[1:-1]
        else:
            middle_events = []
            
        onset_events = []
        response_events = []
        
        # Only process middle events if there are any
        if len(middle_events) > 0:
            # Events should alternate: onset, response, onset, response...
            for i in range(len(middle_events)):
                if i % 2 == 0:  # Even indices (0, 2, 4...) are onsets
                    onset_events.append(middle_events[i].copy())
                else:  # Odd indices (1, 3, 5...) are responses
                    response_events.append(middle_events[i].copy())
            
            # If there's an odd number of middle events, the last onset has no corresponding response
            # In this case, ignore the last onset as specified by the user
            if len(middle_events) % 2 != 0:
                # Remove the last onset
                if onset_events:
                    onset_events.pop()
                    logging.info("[EpochingStep] Removed last onset event because it has no response (time ran out)")
        else:
            logging.warning("[EpochingStep] No onset/response events found between start and end triggers")
        
        # Add annotations marking the task, onset and response events
        if self.params.get("add_annotations", True):
            # Create annotations for onset and response events (relative to the cropped data)
            onset_adjusted = 0.0  # In cropped data, task starts at 0
            task_duration = end_time - start_time
            
            # Create annotation for the main task segment
            task_annotation = mne.Annotations(
                onset=[onset_adjusted],
                duration=[task_duration],
                description=["5PT_task_segment"],
                orig_time=data_cropped.annotations.orig_time if data_cropped.annotations is not None else None
            )
            
            # Add existing annotations from the original data
            if data_cropped.annotations is not None and len(data_cropped.annotations) > 0:
                data_cropped.set_annotations(task_annotation + data_cropped.annotations)
            else:
                data_cropped.set_annotations(task_annotation)
            
            # Add onset and response annotations if we have any
            if onset_events and response_events:
                # Create arrays of onsets and responses
                onset_times_rel = [evt[0]/sfreq for evt in onset_events]
                response_times_rel = [evt[0]/sfreq for evt in response_events]
                
                # Create a single annotation object with all onsets and responses
                all_onsets = onset_times_rel
                all_durations = [0.1] * len(onset_times_rel)
                all_descriptions = ["5PT_onset"] * len(onset_times_rel)
                
                # Add responses
                all_onsets.extend(response_times_rel)
                all_durations.extend([0.1] * len(response_times_rel))
                all_descriptions.extend(["5PT_response"] * len(response_times_rel))
                
                # Create combined annotation 
                event_annotations = mne.Annotations(
                    onset=all_onsets,
                    duration=all_durations,
                    description=all_descriptions,
                    orig_time=data_cropped.annotations.orig_time
                )
                
                # Add to the data's annotations
                data_cropped.set_annotations(data_cropped.annotations + event_annotations)
        
        # Add task metadata to the data
        if not hasattr(data_cropped.info, 'temp'):
            data_cropped.info['temp'] = {}
        
        # Store task metadata
        data_cropped.info['temp']['5pt_task'] = {
            'total_duration': end_time - start_time,
            'n_trials': len(onset_events),
            'buffer_pre': buffer_pre,
            'buffer_post': buffer_post,
            'onset_times': [evt[0]/sfreq for evt in onset_events],
            'response_times': [evt[0]/sfreq for evt in response_events]
        }
        
        # Calculate and store response times
        if onset_events and response_events:
            response_latencies = []
            for onset_evt, response_evt in zip(onset_events, response_events):
                onset_time = onset_evt[0]
                response_time = response_evt[0]
                latency = (response_time - onset_time) / sfreq
                response_latencies.append(latency)
            
            if response_latencies:
                data_cropped.info['temp']['5pt_task']['avg_response_time'] = np.mean(response_latencies)
                data_cropped.info['temp']['5pt_task']['min_response_time'] = np.min(response_latencies)
                data_cropped.info['temp']['5pt_task']['max_response_time'] = np.max(response_latencies)
        
        logging.info(f"[EpochingStep] Extracted continuous segment from {start_time_with_buffer:.2f}s to {end_time_with_buffer:.2f}s")
        logging.info(f"[EpochingStep] Task duration: {end_time - start_time:.2f}s with {len(onset_events)} trials")
        
        # STEP 2: If extract_continuous is True, return the cropped data without creating epochs
        if extract_continuous:
            logging.info("[EpochingStep] Returning continuous data segment without creating epochs (extract_continuous=True)")
            return data_cropped
        
        # STEP 3: If extract_continuous is False, create epochs from the cropped data
        logging.info("[EpochingStep] Creating epochs from cropped data (extract_continuous=False)")
        
        # If there are no middle events, just return the cropped data
        if len(middle_events) == 0:
            logging.warning("[EpochingStep] No events found between start and end")
            return data_cropped
            
        # Log the classified events
        logging.info(f"[EpochingStep] Classified {len(onset_events)} onset events and {len(response_events)} response events")
        
        # Create new events for epoching - only using onset events
        new_events = []
        event_id = {}
        
        ONSET_ID = 801
        # We'll keep RESPONSE_ID for annotations and metadata, even though we won't epoch around them
        RESPONSE_ID = 802
        
        # Add onset events with a unique ID
        if onset_events:
            for evt in onset_events:
                new_evt = evt.copy()
                new_evt[2] = ONSET_ID  # Set a unique ID for onset
                new_events.append(new_evt)
            event_id['onset'] = ONSET_ID
        
        # Keep response events for annotations and metadata, but don't create epochs around them
        
        # Convert to numpy array and sort by time
        if new_events:
            new_events = np.array(new_events)
            new_events = new_events[np.argsort(new_events[:, 0])]
        else:
            logging.warning("[EpochingStep] No onset events to epoch")
            return data_cropped
        
        # Create epochs only around onset events
        epochs = mne.Epochs(data_cropped, new_events, event_id=event_id, tmin=tmin, tmax=tmax,
                           baseline=baseline, preload=preload, 
                           reject_by_annotation=reject_by_annotation)
        
        logging.info(f"[EpochingStep] Created {len(epochs)} epochs around onset events for 5-point test")
        
        # Calculate and log response times
        if onset_events and response_events:
            try:
                # Calculate response times directly from matched pairs
                response_latencies = []
                
                # Since we ensure equal numbers of onsets and responses, we can pair them directly
                for onset_evt, response_evt in zip(onset_events, response_events):
                    onset_time = onset_evt[0] 
                    response_time = response_evt[0]
                    latency = (response_time - onset_time) / sfreq
                    response_latencies.append(latency)
                
                if response_latencies:
                    avg_latency = np.mean(response_latencies)
                    min_latency = np.min(response_latencies)
                    max_latency = np.max(response_latencies)
                    
                    logging.info(f"[EpochingStep] Response time metrics:")
                    logging.info(f"  - Average: {avg_latency:.3f}s")
                    logging.info(f"  - Minimum: {min_latency:.3f}s")
                    logging.info(f"  - Maximum: {max_latency:.3f}s")
                    
                    # Store in metadata
                    if not hasattr(epochs, 'metadata') or epochs.metadata is None:
                        epochs.metadata = pd.DataFrame(index=range(len(epochs)))
                    
                    epochs.metadata['avg_response_time'] = avg_latency
                    epochs.metadata['min_response_time'] = min_latency
                    epochs.metadata['max_response_time'] = max_latency
            except Exception as e:
                logging.warning(f"[EpochingStep] Could not calculate response times: {e}")
        
        return epochs if returns_epochs else data_cropped

    def _epoch_go_nogo(self, data, trigger_ids, tmin, tmax, baseline, preload, 
                      reject_by_annotation, extract_continuous, returns_epochs):
        """
        Epoch data for Go/No-Go task.
        
        Typically has:
        - Go stimuli triggers
        - NoGo stimuli triggers
        - Response triggers (for Go trials)
        """
        # Log the epoching strategy
        logging.info(f"[EpochingStep] Epoching Go/No-Go data")
        
        # Get trigger IDs (with defaults)
        go_id = trigger_ids.get("go", None)
        nogo_id = trigger_ids.get("nogo", None)
        response_id = trigger_ids.get("response", None)
        
        # Get stim channel parameter
        stim_channel = self.params.get("stim_channel", "Trigger")
        
        # Find events in the data
        events = mne.find_events(data, stim_channel=stim_channel, shortest_event=5,min_duration=0.01)

        if len(events) == 0:
            logging.error(f"[EpochingStep] No events found with stim_channel={stim_channel}. Available channels: {data.ch_names}")
            # Try with STI as a fallback
            if 'STI' in data.ch_names and stim_channel != 'STI':
                logging.info("[EpochingStep] Trying with STI channel as fallback")
                events = mne.find_events(data, stim_channel='STI', shortest_event=5, min_duration=0.001,verbose=True)
        
        logging.info(f"[EpochingStep] Found {len(events)} total events")
        
        # Create event dictionary
        event_id = {}
        if go_id is not None:
            event_id['go'] = go_id
        if nogo_id is not None:
            event_id['nogo'] = nogo_id
        if response_id is not None:
            event_id['response'] = response_id
            
        if not event_id:
            logging.warning("[EpochingStep] No Go/NoGo triggers defined for epoching")
            return data
        
        # Visualize the detected events if requested
        if self.params.get("visualize_events", False):
            fig = self.plot_events_simple(data, events, stim_channel)
            if fig:
                logging.info("[EpochingStep] Simple event visualization created")
                # Save the figure if requested
                if self.params.get("plot_params", {}).get("save_plots", False):
                    save_dir = self.params.get("plot_params", {}).get("save_dir", "figures/events")
                    self._save_figures(fig, save_dir)
        
        # Create epochs
        epochs = mne.Epochs(data, events, event_id=event_id, tmin=tmin, tmax=tmax,
                           baseline=baseline, preload=preload, 
                           reject_by_annotation=reject_by_annotation)
        
        logging.info(f"[EpochingStep] Created {len(epochs)} epochs for Go/No-Go task")
        
        # Calculate performance metrics if possible
        if 'go' in event_id and 'nogo' in event_id and response_id is not None:
            try:
                go_events = events[events[:, 2] == go_id]
                nogo_events = events[events[:, 2] == nogo_id]
                response_events = events[events[:, 2] == response_id]
                
                # Calculate hit rate (responses to Go stimuli)
                go_count = len(go_events)
                hit_count = 0
                
                for go_evt in go_events[:, 0]:
                    # Look for responses within 1 second after Go stimulus
                    response_window = 1.0  # seconds
                    response_samples = int(response_window * data.info['sfreq'])
                    responses_after_go = response_events[
                        (response_events[:, 0] > go_evt) & 
                        (response_events[:, 0] < go_evt + response_samples)
                    ]
                    if len(responses_after_go) > 0:
                        hit_count += 1
                
                # Calculate false alarm rate (responses to NoGo stimuli)
                nogo_count = len(nogo_events)
                fa_count = 0
                
                for nogo_evt in nogo_events[:, 0]:
                    # Look for responses within 1 second after NoGo stimulus
                    response_window = 1.0  # seconds
                    response_samples = int(response_window * data.info['sfreq'])
                    responses_after_nogo = response_events[
                        (response_events[:, 0] > nogo_evt) & 
                        (response_events[:, 0] < nogo_evt + response_samples)
                    ]
                    if len(responses_after_nogo) > 0:
                        fa_count += 1
                
                # Calculate metrics
                hit_rate = hit_count / go_count if go_count > 0 else 0
                fa_rate = fa_count / nogo_count if nogo_count > 0 else 0
                
                logging.info(f"[EpochingStep] Go/NoGo Performance - Hit rate: {hit_rate:.2f}, False alarm rate: {fa_rate:.2f}")
                
                # Store in metadata if epochs object supports it
                if hasattr(epochs, 'metadata') and epochs.metadata is not None:
                    epochs.metadata['hit_rate'] = hit_rate
                    epochs.metadata['false_alarm_rate'] = fa_rate
            except Exception as e:
                logging.warning(f"[EpochingStep] Could not calculate performance metrics: {e}")
        
        return epochs if returns_epochs else data

    def _extract_continuous_segment(self, data, trigger_ids, preload):
        """
        Extract a continuous segment of data between start and end triggers.
        """
        # Get start and end trigger IDs
        start_id = trigger_ids.get("start", None)
        end_id = trigger_ids.get("end", None)
        
        if start_id is None or end_id is None:
            logging.warning("[EpochingStep] Missing start or end trigger IDs for continuous extraction")
            return data
        
        # Get stim channel parameter
        stim_channel = self.params.get("stim_channel", "Trigger")
        
        # Find events
        events = mne.find_events(data, stim_channel=stim_channel, shortest_event=1)
        
        if len(events) == 0:
            logging.error(f"[EpochingStep] No events found with stim_channel={stim_channel}")
            # Try with STI as a fallback
            if 'STI' in data.ch_names and stim_channel != 'STI':
                logging.info("[EpochingStep] Trying with STI channel as fallback")
                events = mne.find_events(data, stim_channel='STI', shortest_event=1, verbose=True)
        
        # Find start and end events
        start_events = events[events[:, 2] == start_id]
        end_events = events[events[:, 2] == end_id]
        
        if len(start_events) == 0 or len(end_events) == 0:
            logging.warning("[EpochingStep] No start or end events found for continuous extraction")
            return data
        
        # Get the first start and last end event
        start_time = start_events[0, 0] / data.info['sfreq']
        end_time = end_events[-1, 0] / data.info['sfreq']
        
        # Crop the data
        data_cropped = data.copy().crop(tmin=start_time, tmax=end_time)
        logging.info(f"[EpochingStep] Extracted continuous segment from {start_time:.2f}s to {end_time:.2f}s")
        
        return data_cropped

    def _epoch_custom(self, data, trigger_ids, tmin, tmax, baseline, preload, 
                     reject_by_annotation, returns_epochs):
        """
        Generic epoching for custom trigger configurations.
        """
        # Get stim channel parameter
        stim_channel = self.params.get("stim_channel", "Trigger")
        
        # Find events
        events = mne.find_events(data, stim_channel=stim_channel, shortest_event=1)
        
        if len(events) == 0:
            logging.error(f"[EpochingStep] No events found with stim_channel={stim_channel}")
            # Try with STI as a fallback
            if 'STI' in data.ch_names and stim_channel != 'STI':
                logging.info("[EpochingStep] Trying with STI channel as fallback")
                events = mne.find_events(data, stim_channel='STI', shortest_event=1, verbose=True)
        
        # Convert trigger_ids to event_id format expected by MNE
        event_id = {}
        for trigger_name, trigger_id in trigger_ids.items():
            event_id[trigger_name] = trigger_id
        
        if not event_id:
            logging.warning("[EpochingStep] No triggers defined for custom epoching")
            return data
        
        # Create epochs
        epochs = mne.Epochs(data, events, event_id=event_id, tmin=tmin, tmax=tmax,
                           baseline=baseline, preload=preload, 
                           reject_by_annotation=reject_by_annotation)
        
        logging.info(f"[EpochingStep] Created {len(epochs)} epochs with custom trigger configuration")
        
        return epochs if returns_epochs else data

    def _epoch_fixed_length(self, data, trigger_ids, epoch_params, reject_by_annotation, 
                           extract_continuous, returns_epochs):
        """
        Create fixed-length epochs with configurable duration.
        
        This method creates evenly spaced epochs across the data or within a task segment.
        
        Parameters
        ----------
        data : mne.io.Raw
            The raw data to epoch
        trigger_ids : dict
            Dict of trigger IDs (used only for task segment detection if extract_continuous=True)
        epoch_params : dict
            Epoching parameters including:
            - duration: Length of each epoch in seconds (default: 1.0)
            - overlap: Overlap between epochs in seconds (default: 0.0)
            - baseline: Baseline correction period (default: None)
        reject_by_annotation : bool
            Whether to reject epochs that overlap with annotations
        extract_continuous : bool
            If True, extract fixed-length epochs only within the task segment
            (defined by first and last trigger)
        returns_epochs : bool
            If True, return Epochs object; if False, return Raw object
            
        Returns
        -------
        epochs : mne.Epochs or mne.io.Raw
            The fixed-length epochs or the original data
        """
        # Get parameters
        duration = epoch_params.get("duration", 1.0)  # Default 1s epochs
        overlap = epoch_params.get("overlap", 0.0)    # Default no overlap
        baseline = epoch_params.get("baseline", None) # Default no baseline correction
        preload = epoch_params.get("preload", True)   # Default preload data
        
        # Log the epoching strategy
        logging.info(f"[EpochingStep] Creating fixed-length epochs: duration={duration}s, overlap={overlap}s")
        
        # If we want to epoch only within a task segment
        if extract_continuous:
            # Get the task segment based on first and last trigger
            # First, find all events
            stim_channel = self.params.get("stim_channel", "Trigger")
            events = mne.find_events(data, stim_channel=stim_channel, shortest_event=1)
            
            if len(events) < 2:  # Need at least start and end events
                logging.warning(f"[EpochingStep] Not enough events found ({len(events)}) for task segment detection")
                return data
                
            # Get sampling frequency
            sfreq = data.info['sfreq']
            
            # Use first and last event as task boundaries
            start_event = events[0]
            end_event = events[-1]
            
            # Calculate times
            start_time = start_event[0] / sfreq
            end_time = end_event[0] / sfreq
            
            # Apply optional buffer
            buffer_pre = self.params.get("buffer_pre", 0.0)
            buffer_post = self.params.get("buffer_post", 0.0)
            
            start_time_with_buffer = max(0, start_time - buffer_pre)
            end_time_with_buffer = min(end_time + buffer_post, data.times[-1])
            
            # First crop the data to the task segment with buffer
            data_cropped = data.copy().crop(tmin=start_time_with_buffer, tmax=end_time_with_buffer)
            
            # Recalculate events after cropping to ensure correct timing
            if self.params.get("stim_channel", "Trigger") in data_cropped.ch_names:
                logging.info("[EpochingStep] Recalculating events after cropping")
                events = mne.find_events(
                    data_cropped, 
                    stim_channel=self.params.get("stim_channel", "Trigger"),
                    shortest_event=1,
                    verbose=False
                )
                
                # Identify start and end events in the cropped data
                if len(events) >= 2:
                    # Use the first and last events as start and end
                    start_event = events[0]
                    end_event = events[-1]
                    # Update times
                    start_time = start_event[0] / sfreq
                    end_time = end_event[0] / sfreq
                    logging.info(f"[EpochingStep] Updated start_time={start_time:.2f}s and end_time={end_time:.2f}s")
                else:
                    logging.warning("[EpochingStep] Could not find enough events in cropped data")
            else:
                logging.warning(f"[EpochingStep] Stim channel {self.params.get('stim_channel', 'Trigger')} not found in cropped data")
            
            # Create fixed-length events within the cropped data
            # Use this data for further processing
            data_for_epochs = data_cropped
            logging.info(f"[EpochingStep] Using task segment from {start_time_with_buffer:.2f}s to {end_time_with_buffer:.2f}s")
        else:
            # Use the entire data
            data_for_epochs = data
        
        # Create fixed-length events
        # Step size is (duration - overlap)
        step = duration - overlap
        
        # Create events with MNE function
        events = mne.make_fixed_length_events(
            data_for_epochs,
            id=1,  # All events have the same ID
            duration=step,
            start=0,
            stop=None,  # Use entire file
            first_samp=True
        )
        
        # Create an event_id dictionary for MNE Epochs
        event_id = {'fixed': 1}
        
        # Create epochs
        epochs = mne.Epochs(
            data_for_epochs,
            events,
            event_id=event_id,
            tmin=0,
            tmax=duration,
            baseline=baseline,
            preload=preload,
            reject_by_annotation=reject_by_annotation
        )
        
        # Log creation
        logging.info(f"[EpochingStep] Created {len(epochs)} fixed-length epochs")
        
        # Add metadata about the epoching
        if not hasattr(epochs, 'metadata') or epochs.metadata is None:
            epochs.metadata = pd.DataFrame(index=range(len(epochs)))
        
        # Add duration and overlap info
        epochs.metadata['duration'] = duration
        epochs.metadata['overlap'] = overlap
        
        # Add task segment info if we used extract_continuous
        if extract_continuous:
            epochs.metadata['task_start'] = start_time
            epochs.metadata['task_end'] = end_time
            epochs.metadata['total_task_duration'] = end_time - start_time
        
        # Return the appropriate object
        return epochs if returns_epochs else data_for_epochs

    def save_epochs(self, epochs, output_dir=None):
        """
        Save epochs to disk.
        
        Parameters:
        -----------
        epochs : mne.Epochs
            Epochs object to save
        output_dir : str or Path, optional
            Directory to save epochs in (default: use paths from params)
        """
        if not isinstance(epochs, mne.Epochs):
            logging.warning("[EpochingStep] Cannot save non-Epochs object")
            return
        
        sub_id = self.params.get("subject_id", "unknown")
        ses_id = self.params.get("session_id", "001")
        task_id = self.params.get("task_id", None)
        run_id = self.params.get("run_id", None)
        
        # Get paths object if available
        paths = self.params.get("paths", None)
        
        if output_dir is None and paths is not None:
            # Use ProjectPaths to get the save path
            save_path = paths.get_derivative_path(
                subject_id=sub_id,
                session_id=ses_id,
                task_id=task_id,
                run_id=run_id,
                stage="epochs"
            )
        else:
            # Use provided output directory or current directory
            output_dir = output_dir or "."
            
            # Build filename
            filename = f"sub-{sub_id}_ses-{ses_id}"
            if task_id:
                filename += f"_task-{task_id}"
            if run_id:
                filename += f"_run-{run_id}"
            filename += "_epo.fif"
            
            save_path = Path(output_dir) / filename
        
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(str(save_path)), exist_ok=True)
        
        # Save epochs
        epochs.save(save_path, overwrite=True)
        logging.info(f"[EpochingStep] Saved epochs to {save_path}")

    def plot_epochs(self, epochs, plot_type='average', event_names=None, channels=None, 
                   time_window=None, combine=False, title=None):
        """
        Plot epochs in different ways
        
        Parameters:
        -----------
        epochs : mne.Epochs
            The epochs object to plot
        plot_type : str
            Type of plot ('average', 'butterfly', 'image', 'psd', 'topo', 'compare')
        event_names : list or None
            Names of events to include (if None, uses all)
        channels : list or None
            Channels to include (if None, uses all EEG channels)
        time_window : tuple or None
            (start, end) in seconds for time window to plot
        combine : bool
            Whether to combine channels (for image plot)
        title : str or None
            Plot title
            
        Returns:
        --------
        fig : matplotlib.figure.Figure or list of figures
            The created figure(s)
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Default event names if not specified
        if event_names is None:
            event_names = list(epochs.event_id.keys())
        
        # Default channels selection
        if channels is None:
            channels = 'eeg'  # Use all EEG channels
        
        # Set time window if specified
        if time_window is not None:
            epochs_plot = epochs.copy().crop(tmin=time_window[0], tmax=time_window[1])
        else:
            epochs_plot = epochs
        
        figures = []
        
        # Choose plot type
        if plot_type == 'average':
            # Plot average ERPs for each event type
            for event_name in event_names:
                try:
                    # Create the evoked object first
                    evoked = epochs_plot[event_name].average()
                    # Set the comment field which will be used as title
                    if title is None:
                        evoked.comment = f'Average ERP for {event_name} events'
                    else:
                        evoked.comment = title
                    # Plot with comment as title
                    fig = evoked.plot(picks=channels, spatial_colors=True)
                    plt.tight_layout()
                    figures.append(fig)
                except Exception as e:
                    logging.error(f"Error in average plot: {str(e)}")
        
        elif plot_type == 'butterfly':
            # Butterfly plot (all channels overlaid)
            for event_name in event_names:
                try:
                    # Create the evoked object first
                    evoked = epochs_plot[event_name].average()
                    # Set the comment field which will be used as title
                    if title is None:
                        evoked.comment = f'Butterfly plot for {event_name} events'
                    else:
                        evoked.comment = title
                    # Plot with comment as title
                    fig = evoked.plot(picks=channels, spatial_colors=False)
                    plt.tight_layout()
                    figures.append(fig)
                except Exception as e:
                    logging.error(f"Error in butterfly plot: {str(e)}")
        
        elif plot_type == 'image':
            # Image plot (epochs x time)
            for event_name in event_names:
                try:
                    fig = epochs_plot[event_name].plot_image(
                        picks=channels,
                        combine=combine,
                        title=f'Epochs image for {event_name}' if title is None else title
                    )
                    figures.append(fig)
                except Exception as e:
                    logging.error(f"Error in image plot: {str(e)}")
        
        elif plot_type == 'psd':
            # Power spectral density
            for event_name in event_names:
                try:
                    fig = epochs_plot[event_name].plot_psd(
                        picks=channels,
                        title=f'PSD for {event_name} epochs' if title is None else title
                    )
                    figures.append(fig)
                except Exception as e:
                    logging.error(f"Error in PSD plot: {str(e)}")
        
        elif plot_type == 'topo':
            # Topographic plots at specific times
            times = np.linspace(epochs_plot.tmin, epochs_plot.tmax, 5)
            for event_name in event_names:
                try:
                    fig = epochs_plot[event_name].average().plot_topomap(
                        times=times,
                        title=f'Topography for {event_name}' if title is None else title
                    )
                    figures.append(fig)
                except Exception as e:
                    logging.error(f"Error in topo plot: {str(e)}")
        
        elif plot_type == 'compare':
            # Compare average ERPs between conditions
            if len(event_names) < 2:
                logging.warning("[EpochingStep] Need at least 2 event types to compare")
                return None
            
            try:
                # Create averages
                evokeds = [epochs_plot[event].average() for event in event_names]
                
                # Set different colors for each condition
                colors = ['blue', 'red', 'green', 'orange', 'purple']
                
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot each condition
                for i, (evoked, event_name) in enumerate(zip(evokeds, event_names)):
                    times = evoked.times * 1000  # Convert to ms
                    if channels == 'eeg' or isinstance(channels, list) and len(channels) > 1:
                        # Average across selected channels if multiple
                        data = evoked.data.mean(axis=0)
                    else:
                        # Get data for single channel
                        ch_idx = evoked.ch_names.index(channels[0]) if isinstance(channels, list) else 0
                        data = evoked.data[ch_idx]
                    
                    color = colors[i % len(colors)]
                    ax.plot(times, data, label=event_name, color=color, linewidth=2)
                
                # Add vertical line at stimulus onset
                ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
                
                # Add horizontal line at 0 μV
                ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
                
                # Customize plot
                ax.set_xlabel('Time (ms)')
                ax.set_ylabel('Amplitude (μV)')
                ax.set_title('Comparison of conditions' if title is None else title)
                ax.legend()
                plt.tight_layout()
                figures.append(fig)
            except Exception as e:
                logging.error(f"Error in compare plot: {str(e)}")
            
        elif plot_type == 'difference':
            # Plot difference between two conditions
            if len(event_names) != 2:
                logging.warning("[EpochingStep] Need exactly 2 event types for difference plot")
                return None
                
            try:
                # Get averages for the two conditions
                evoked1 = epochs_plot[event_names[0]].average()
                evoked2 = epochs_plot[event_names[1]].average()
                
                # Create difference wave
                diff_wave = mne.combine_evoked([evoked1, evoked2], weights=[1, -1])
                
                # Set the comment field which will be used as title
                if title is None:
                    diff_wave.comment = f'Difference: {event_names[0]} - {event_names[1]}'
                else:
                    diff_wave.comment = title
                
                # Plot the difference - don't pass title parameter
                fig = diff_wave.plot(picks=channels, spatial_colors=True)
                
                # Alternative approach: create a custom figure
                # fig, ax = plt.subplots(figsize=(10, 6))
                # times = diff_wave.times * 1000  # Convert to ms
                # if channels == 'eeg' or isinstance(channels, list) and len(channels) > 1:
                #     # Average across selected channels if multiple channels
                #     data = diff_wave.data.mean(axis=0)
                #     ax.plot(times, data, linewidth=2)
                # else:
                #     # Plot the selected channel
                #     ch_idx = diff_wave.ch_names.index(channels[0]) if isinstance(channels, list) else 0
                #     data = diff_wave.data[ch_idx]
                #     ax.plot(times, data, linewidth=2)
                # ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
                # ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
                # ax.set_xlabel('Time (ms)')
                # ax.set_ylabel('Amplitude (μV)')
                # ax.set_title('Difference: {event_names[0]} - {event_names[1]}' if title is None else title)
                
                plt.tight_layout()
                figures.append(fig)
            except Exception as e:
                logging.error(f"Error in difference plot: {str(e)}")
            
        elif plot_type == 'gfp':
            # Global Field Power plot
            for event_name in event_names:
                try:
                    evoked = epochs_plot[event_name].average()
                    fig, ax = plt.subplots(figsize=(10, 6))
                    times = evoked.times * 1000  # Convert to ms
                    gfp = np.sqrt(np.mean(evoked.data ** 2, axis=0))
                    ax.plot(times, gfp, linewidth=2)
                    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
                    ax.set_xlabel('Time (ms)')
                    ax.set_ylabel('GFP (μV)')
                    ax.set_title(f'Global Field Power: {event_name}' if title is None else title)
                    plt.tight_layout()
                    figures.append(fig)
                except Exception as e:
                    logging.error(f"Error in GFP plot: {str(e)}")
        
        else:
            logging.warning(f"[EpochingStep] Unknown plot type: {plot_type}")
            return None
        
        # Return either a single figure or a list of figures
        if len(figures) == 1:
            return figures[0]
        else:
            return figures

    def plot_events_simple(self, raw, events, stim_channel=None, duration=10.0, 
                            tstart=0, save_fig=None, interactive=False):
        """
        Simple visualization of trigger channel and detected events.
        
        Parameters:
        -----------
        raw : mne.io.Raw
            Raw data containing the trigger channel
        events : ndarray
            Events array from mne.find_events
        stim_channel : str or None
            Name of the stim channel (if None, use the one from params)
        duration : float
            Duration in seconds to display (default: 10s)
        tstart : float
            Start time in seconds for the visualization window (default: 0)
        save_fig : dict or None
            Dictionary with parameters for saving the figure:
            - 'save': bool, whether to save the figure
            - 'dir': str, directory to save the figure in
            - 'fname': str, filename to use (without extension)
        interactive : bool
            If True, uses interactive plotting for notebooks (default: False)
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The created figure
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from pathlib import Path
        
        # Get stim channel if not provided
        if stim_channel is None:
            stim_channel = self.params.get("stim_channel", "Trigger")
        
        # Check if stim channel exists in the data
        if stim_channel not in raw.ch_names:
            logging.warning(f"[EpochingStep] Stim channel {stim_channel} not found in raw data")
            return None
        
        # Calculate sample range based on duration
        sfreq = raw.info['sfreq']
        tmax = tstart + duration
        start_sample = int(tstart * sfreq)
        end_sample = int(tmax * sfreq)
        
        # Ensure we don't exceed data boundaries
        start_sample = max(0, start_sample)
        end_sample = min(len(raw.times), end_sample)
        
        # Get the stim channel data for the specified duration
        pick_idx = raw.ch_names.index(stim_channel)
        stim_data, times = raw[pick_idx, start_sample:end_sample]
        stim_data = stim_data.flatten()
        
        # Configure interactive mode if requested
        if interactive:
            plt.ion()  # Turn on interactive mode
        else:
            plt.ioff()  # Make sure interactive mode is off
        
        # Create a simple figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot trigger channel
        ax.plot(times, stim_data, 'k-', linewidth=1, label='Trigger Channel')
        
        # Adjust for raw.first_samp to align events with times
        event_times = (events[:, 0] - raw.first_samp) / raw.info['sfreq']
        event_values = events[:, 2]
        
        # Filter events to visible time range
        mask = (event_times >= tstart) & (event_times <= tmax)
        visible_events = events[mask]
        visible_times = event_times[mask]
        visible_values = event_values[mask]
        
        # Log visible events for debugging
        logging.info(f"[EpochingStep] Showing {len(visible_events)} events in time window [{tstart:.1f}, {tmax:.1f}]s")
        for i in range(min(5, len(visible_events))):
            logging.info(f"  Event {i}: ID={visible_events[i, 2]}, time={visible_times[i]:.3f}s, sample={visible_events[i, 0]}")
        
        # Get unique event values
        unique_values = np.unique(visible_values) if len(visible_values) > 0 else np.unique(event_values)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_values)))
        
        # Plot each event type
        for i, val in enumerate(unique_values):
            # Find events with this value
            mask = visible_values == val if len(visible_values) > 0 else event_values == val
            if np.sum(mask) > 0:
                ev_times = visible_times[mask] if len(visible_values) > 0 else event_times[mask]
                # Plot vertical lines for these events
                for t in ev_times:
                    if tstart <= t <= tmax:  # Double-check time is in range
                        ax.axvline(t, color=colors[i], linestyle='--', alpha=0.7)
                # Add one point for the legend
                ax.plot([], [], color=colors[i], linestyle='--', 
                       label=f'Event ID: {val} (n={np.sum(mask)})')
        
        # Add labels and legend
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Trigger Channel ({stim_channel}) - {duration}s window from {tstart}s to {tmax}s')
        ax.legend(loc='upper right')
        
        # Set x-axis limits to match the requested time window
        ax.set_xlim(times[0], times[-1])
        
        # Add text showing raw.first_samp for clarity
        plt.figtext(0.02, 0.02, f"raw.first_samp = {raw.first_samp}", fontsize=8,
                   bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_fig is not None and save_fig.get('save', False):
            save_dir = save_fig.get('dir', 'figures/events')
            
            # Use paths from params if available
            paths = self.params.get("paths", None)
            if paths is not None and save_dir == 'figures/events':
                sub_id = self.params.get("subject_id", "unknown")
                ses_id = self.params.get("session_id", "001")
                task_id = self.params.get("task_id", None)
                run_id = self.params.get("run_id", None)
                
                save_dir = paths.get_derivative_path(
                    subject_id=sub_id,
                    session_id=ses_id,
                    task_id=task_id,
                    run_id=run_id,
                    stage="figures/events"
                )
            
            # Create filename
            if 'fname' in save_fig:
                filename = save_fig['fname']
            else:
                # Create a default filename
                sub_id = self.params.get("subject_id", "unknown")
                ses_id = self.params.get("session_id", "001")
                task_id = self.params.get("task_id", None)
                run_id = self.params.get("run_id", None)
                
                filename = f"sub-{sub_id}_ses-{ses_id}"
                if task_id:
                    filename += f"_task-{task_id}"
                if run_id:
                    filename += f"_run-{run_id}"
                filename += f"_events_{tstart}to{tmax}s"
            
            # Ensure directory exists
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save figure
            filepath = save_path / f"{filename}.png"
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logging.info(f"[EpochingStep] Saved event visualization to {filepath}")
        
        # Show the figure if interactive mode is on
        if interactive:
            plt.show()
            
        return fig

    def _save_figures(self, figures, save_dir=None):
        """
        Save generated figures to disk.
        
        Parameters:
        -----------
        figures : matplotlib.figure.Figure or list
            Figure or list of figures to save
        save_dir : str or Path or None
            Directory to save figures in (default: current directory)
        """
        import matplotlib.pyplot as plt
        from pathlib import Path
        import os
        
        # Convert single figure to list for uniform handling
        if not isinstance(figures, list):
            figures = [figures]
            
        # Set the save directory
        if save_dir is None:
            # Try to get from params
            paths = self.params.get("paths", None)
            if paths is not None:
                sub_id = self.params.get("subject_id", "unknown")
                ses_id = self.params.get("session_id", "001")
                task_id = self.params.get("task_id", None)
                run_id = self.params.get("run_id", None)
                
                save_dir = paths.get_derivative_path(
                    subject_id=sub_id,
                    session_id=ses_id,
                    task_id=task_id,
                    run_id=run_id,
                    stage="figures/epochs"
                )
            else:
                save_dir = "figures/epochs"
        
        # Ensure directory exists
        os.makedirs(str(save_dir), exist_ok=True)
        
        # Get the base filename
        sub_id = self.params.get("subject_id", "unknown")
        ses_id = self.params.get("session_id", "001")
        task_id = self.params.get("task_id", None)
        run_id = self.params.get("run_id", None)
        
        base_filename = f"sub-{sub_id}_ses-{ses_id}"
        if task_id:
            base_filename += f"_task-{task_id}"
        if run_id:
            base_filename += f"_run-{run_id}"
        
        # Save each figure
        for i, fig in enumerate(figures):
            if fig is not None:
                # Create filename
                filename = f"{base_filename}_plot-{i+1}.png"
                filepath = Path(save_dir) / filename
                
                # Save the figure
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                logging.info(f"[EpochingStep] Saved figure to {filepath}")

    def generate_plots(self, epochs, plot_params=None):
        """
        Generate plots from existing epochs without rerunning the epoching step.
        This provides a convenient way to create different visualizations after epoching.
        
        Parameters:
        -----------
        epochs : mne.Epochs
            The epochs to plot
        plot_params : dict, optional
            Dictionary containing plotting parameters:
            - plot_type: Type of plot ('average', 'butterfly', 'image', 'psd', etc.)
            - event_names: Names of events to include (if None, uses all)
            - channels: Channels to include (if None, uses all EEG channels)
            - time_window: (start, end) in seconds for time window to plot
            - combine: Whether to combine channels (for image plot)
            - title: Plot title
            - save_plots: Whether to save the generated plots (default: False)
            - save_dir: Directory to save plots in (default: uses project paths)
            
        Returns:
        --------
        figures : matplotlib.figure.Figure or list of figures
            The generated figures
        """
        if not hasattr(epochs, 'event_id'):
            logging.error("[EpochingStep.generate_plots] Input is not a valid Epochs object")
            return None
            
        # Use provided plot_params or empty dict
        plot_params = plot_params or {}
        
        # Get plotting parameters with defaults
        plot_type = plot_params.get("plot_type", "average")
        event_names = plot_params.get("event_names", None)
        channels = plot_params.get("channels", None)
        time_window = plot_params.get("time_window", None)
        combine = plot_params.get("combine", False)
        title = plot_params.get("title", None)
        
        # Generate plots
        figures = self.plot_epochs(
            epochs, 
            plot_type=plot_type,
            event_names=event_names,
            channels=channels,
            time_window=time_window,
            combine=combine,
            title=title
        )
        
        # If save_plots is specified, save the figures
        save_plots = plot_params.get("save_plots", False)
        if save_plots:
            self._save_figures(figures, plot_params.get("save_dir", None))
            
        return figures

# Example usage of the plotting functionality:
"""
# Example 1: Auto-plotting during epoching
epoching_params = {
    "task_type": "gng",
    "trigger_ids": {"go": 1, "nogo": 2, "response": 3},
    "epoch_params": {"tmin": -0.2, "tmax": 0.8},
    # Enable auto-plotting
    "auto_plot": True,
    "plot_params": {
        "plot_type": "average",  # Options: average, butterfly, image, psd, topo, compare, difference, gfp
        "event_names": ["go", "nogo"],  # Which events to plot (if None, plots all)
        "channels": ["Fz", "Cz", "Pz"],  # Which channels to plot (if None, plots all EEG)
        "time_window": [-0.1, 0.5],  # Optional time window to focus on
        "combine": False,  # Whether to combine channels (for image plots)
        "title": "Go/NoGo Task ERPs",  # Optional custom title
        "save_plots": True,  # Whether to save the plots to disk
        "save_dir": "figures/my_analysis"  # Optional custom save directory
    }
}

# Create and run the epoching step
epoching_step = EpochingStep(params=epoching_params)
epochs = epoching_step.run(raw_data)  # Automatically generates and saves plots

# Example 2: Generate plots after epoching
# If you've already run the epoching step without auto_plot,
# you can still generate plots later:

# Create basic epoching step without auto-plot
basic_params = {
    "task_type": "5pt",
    "trigger_ids": {"trigger_id": 8},
    "epoch_params": {"tmin": -0.2, "tmax": 1.0}
}
epoching_step = EpochingStep(params=basic_params)
epochs = epoching_step.run(raw_data)  # No plots generated yet

# Later, generate different plot types as needed
butterfly_plot = epoching_step.generate_plots(
    epochs,
    plot_params={
        "plot_type": "butterfly",
        "channels": ["Fz", "Cz", "Pz"],
        "save_plots": True
    }
)

# Generate comparison plots between conditions
if "go" in epochs.event_id and "nogo" in epochs.event_id:
    comparison_plots = epoching_step.generate_plots(
        epochs,
        plot_params={
            "plot_type": "compare",
            "event_names": ["go", "nogo"],
            "channels": ["Fz"],  # Look at frontal channels for NoGo effects
            "title": "Go vs. NoGo Comparison",
            "save_plots": True
        }
    )

# Generate topographic plots to see spatial distribution
topo_plots = epoching_step.generate_plots(
    epochs,
    plot_params={
        "plot_type": "topo",
        "event_names": ["onset"],
        "time_window": [0, 0.3],  # Focus on early components
        "save_plots": True
    }
)
"""
