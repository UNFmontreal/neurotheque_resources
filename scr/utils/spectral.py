# File: scr/utils/spectral.py

import logging
import numpy as np
import matplotlib.pyplot as plt
import mne
from typing import Union, List, Dict, Optional, Tuple, Any
import matplotlib


def compute_and_plot_psd(data: Union[mne.io.Raw, mne.Epochs], 
                         fmin: float = 0.0,
                         fmax: Optional[float] = None,
                         tmin: Optional[float] = None,
                         tmax: Optional[float] = None,
                         picks: Any = None,
                         method: str = 'welch',
                         plot_type: str = 'standard',
                         plot_kwargs: Optional[Dict] = None,
                         save_path: Optional[str] = None,
                         show: bool = True,
                         return_data: bool = False) -> Union[plt.Figure, Tuple[plt.Figure, Dict]]:
    """
    Compute and plot power spectral density (PSD) with various options.
    
    Parameters
    ----------
    data : mne.io.Raw or mne.Epochs
        The data to compute PSD from
    fmin : float
        Minimum frequency to include
    fmax : float, optional
        Maximum frequency to include. If None, uses Nyquist frequency
    tmin : float, optional
        Start time for computation (for Epochs or Raw)
    tmax : float, optional
        End time for computation (for Epochs or Raw)
    picks : str, list, slice, None
        Channels to include. E.g. 'eeg', ['Fz', 'Cz', 'Pz'], etc.
    method : str
        Method to compute PSD: 'welch' or 'multitaper'
    plot_type : str
        Type of plot to create:
        - 'standard': Standard MNE plot with all channels
        - 'average': Average across selected channels
        - 'topo': Topographic plot at selected frequencies
        - 'matrix': Matrix plot (time-frequency for epochs)
        - 'bands': Plot average power in classic frequency bands
    plot_kwargs : dict, optional
        Additional kwargs to pass to the plotting function
    save_path : str, optional
        Path to save the figure to
    show : bool
        Whether to show the figure
    return_data : bool
        Whether to return computed PSD data along with figure
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    results : dict, optional
        Dictionary with PSD data if return_data=True
    """
    logging.info(f"Computing PSD using {method} method")
    
    # Set default plot kwargs
    if plot_kwargs is None:
        plot_kwargs = {}
    
    # Ensure fmax doesn't exceed Nyquist frequency
    if fmax is None:
        fmax = data.info['sfreq'] / 2.0
    
    # Define parameters for compute_psd method
    psd_params = {
        'method': method,
        'fmin': fmin,
        'fmax': fmax,
        'picks': picks
    }
    
    # Add method-specific parameters
    if method == 'welch':
        psd_params.update({
            'n_fft': plot_kwargs.pop('n_fft', 2048),
            'n_overlap': plot_kwargs.pop('n_overlap', 0),
            'n_per_seg': plot_kwargs.pop('n_per_seg', None),
            'window': plot_kwargs.pop('window', 'hamming'),
            'average': plot_kwargs.pop('average', 'mean')
        })
    elif method == 'multitaper':
        psd_params.update({
            'bandwidth': plot_kwargs.pop('bandwidth', None),
            'adaptive': plot_kwargs.pop('adaptive', False),
            'low_bias': plot_kwargs.pop('low_bias', True),
            'normalization': plot_kwargs.pop('normalization', 'length')
        })
    
    # Compute PSD using the object's compute_psd method
    if hasattr(data, 'compute_psd'):
        # For time range, different handling depending on data type
        if isinstance(data, mne.io.Raw) and (tmin is not None or tmax is not None):
            # For Raw data, we need to extract the relevant time segment
            if tmin is not None:
                psd_params['tmin'] = tmin
            if tmax is not None:
                psd_params['tmax'] = tmax
                
        # Compute the PSD
        spectrum = data.compute_psd(**psd_params)
        psds = spectrum.get_data()
        freqs = spectrum.freqs
    else:
        raise ValueError("Data object does not support PSD computation. Ensure you're using a compatible MNE object.")
    
    # Log shape info for debugging
    logging.info(f"PSD shape: {psds.shape}, Freqs shape: {freqs.shape}")
    
    # Create appropriate plot based on plot_type
    if plot_type == 'standard':
        # Standard MNE plot - the spectrum object creates its own figure
        # Note: spectrum.plot() doesn't accept figsize directly
        # Create figure first if figsize is specified
        figsize = plot_kwargs.pop('figsize', (10, 6))
        
        # Create the plot
        fig = spectrum.plot(picks=picks, show=False, **plot_kwargs)
        
        # If the returned fig is a list (MNE sometimes returns a list of figures),
        # take the first one
        if isinstance(fig, list):
            fig = fig[0]
            
        # Try to resize the figure if possible
        if hasattr(fig, 'set_size_inches'):
            fig.set_size_inches(figsize)
    
    elif plot_type == 'average':
        # Average across channels
        fig, ax = plt.subplots(figsize=plot_kwargs.pop('figsize', (10, 6)))
        
        # Get mean across channels
        mean_psd = np.mean(psds, axis=0)
        
        # Convert to dB if not already
        if not plot_kwargs.pop('dB', True):
            mean_psd = 10 * np.log10(mean_psd)
            
        ax.plot(freqs, mean_psd, **plot_kwargs)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power Spectral Density (dB)')
        ax.set_title(f'Average PSD across {len(psds)} channels')
        ax.set_xlim([fmin, fmax])
        plt.grid(True)
        
    elif plot_type == 'topo':
        # Topographic plot at specific frequencies
        freq_points = plot_kwargs.pop('freq_points', [6, 10, 20, 30])
        if not hasattr(data, 'info') or not data.info['ch_names']:
            raise ValueError("Data must have channel info for topographic plots")
            
        # Find nearest frequency bin for each frequency point
        freq_indices = [np.argmin(np.abs(freqs - freq)) for freq in freq_points]
        freq_values = [freqs[idx] for idx in freq_indices]
        
        n_freqs = len(freq_indices)
        fig, axes = plt.subplots(1, n_freqs, figsize=plot_kwargs.pop('figsize', (4*n_freqs, 4)))
        if n_freqs == 1:
            axes = [axes]
            
        for ax, freq_idx, freq_val in zip(axes, freq_indices, freq_values):
            power_map = psds[:, freq_idx]
            mne.viz.plot_topomap(power_map, data.info, axes=ax, show=False, **plot_kwargs)
            ax.set_title(f'{freq_val:.1f} Hz')
            
    elif plot_type == 'bands':
        # Plot average power in standard frequency bands
        bands = plot_kwargs.pop('bands', {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        })
        
        # Calculate average power in each band
        band_powers = {}
        for band_name, (band_min, band_max) in bands.items():
            # Find frequency indices for this band
            band_indices = np.where((freqs >= band_min) & (freqs <= band_max))[0]
            if len(band_indices) == 0:
                logging.warning(f"No frequencies found in {band_name} band ({band_min}-{band_max} Hz)")
                continue
                
            # Calculate band power for each channel
            band_psds = psds[:, band_indices]
            band_powers[band_name] = np.mean(band_psds, axis=1)
        
        # Plot
        fig, ax = plt.subplots(figsize=plot_kwargs.pop('figsize', (10, 6)))
        x = np.arange(len(bands))
        width = 0.8
        
        # Plot average band power across channels
        avg_powers = [np.mean(band_powers[band]) for band in bands.keys() if band in band_powers]
        band_names = [band for band in bands.keys() if band in band_powers]
        
        # Convert to dB if needed
        if plot_kwargs.pop('dB', True):
            avg_powers = 10 * np.log10(avg_powers)
            
        bars = ax.bar(x, avg_powers, width, **plot_kwargs)
        ax.set_xlabel('Frequency Band')
        ax.set_ylabel('Power (dB)')
        ax.set_title('Average Power by Frequency Band')
        ax.set_xticks(x)
        ax.set_xticklabels(band_names)
        plt.grid(True, axis='y')
        
    elif plot_type == 'matrix':
        # Matrix plot (mainly for epochs)
        fig, ax = plt.subplots(figsize=plot_kwargs.pop('figsize', (10, 6)))
        
        # For epochs, average across epochs
        if psds.ndim == 3:  # (n_epochs, n_channels, n_freqs)
            psds = np.mean(psds, axis=0)
            
        # Convert to dB if needed
        if plot_kwargs.pop('dB', True):
            psds_db = 10 * np.log10(psds)
        else:
            psds_db = psds
        
        # Plot as a heatmap/matrix
        im = ax.imshow(psds_db, aspect='auto', origin='lower', 
                       extent=[fmin, fmax, 0, psds.shape[0]-1], 
                       **plot_kwargs)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('PSD (dB)')
        
        # Add labels
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Channel')
        ax.set_title('PSD across channels and frequencies')
        
        # Add channel names as yticks if available
        if picks is not None and not isinstance(picks, str):
            channel_names = picks
        else:
            ch_idx = mne.pick_types(data.info, **{p: True for p in ('all',) if p == picks} if picks else {'all': True})
            channel_names = [data.info['ch_names'][i] for i in ch_idx] if hasattr(data, 'info') else None
            
        if channel_names and len(channel_names) == psds.shape[0]:
            # Set yticks at channel positions
            ax.set_yticks(np.arange(len(channel_names)))
            ax.set_yticklabels(channel_names)
        
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")
    
    # Finalize the plot
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"PSD plot saved to {save_path}")
    
    # Show if requested
    if show:
        # Try to use an interactive backend if showing is requested
        backend = matplotlib.get_backend()
        if backend.lower() in ('agg', 'figurecanvasagg'):
            try:
                # First try Qt backends
                for qt_backend in ['Qt5Agg', 'QtAgg', 'TkAgg']:
                    try:
                        matplotlib.use(qt_backend, force=True)
                        logging.info(f"Switched to interactive backend: {qt_backend}")
                        break
                    except ImportError:
                        continue
            except Exception as e:
                logging.warning(f"Could not switch to interactive backend: {str(e)}")
                logging.warning("Plot display might fail. Consider setting matplotlib backend manually.")
        plt.show()
    
    # Return data if requested
    if return_data:
        results = {
            'freqs': freqs,
            'psds': psds,
            'method': method,
            'spectrum': spectrum  # Return the spectrum object for additional processing
        }
        
        if plot_type == 'bands' and 'bands' in locals():
            results['band_powers'] = band_powers
            
        return fig, results
    else:
        return fig


def plot_time_frequency(data: Union[mne.io.Raw, mne.Epochs],
                        picks: Any = None,
                        method: str = 'morlet',
                        freq_range: Tuple[float, float] = (1, 40),
                        n_cycles: Union[float, List[float]] = 7.0,
                        decim: int = 1,
                        average: bool = True,
                        baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
                        plot_type: str = 'power',
                        title: str = 'Time-Frequency Analysis',
                        save_path: Optional[str] = None,
                        show: bool = True,
                        return_data: bool = False) -> Union[plt.Figure, Tuple[plt.Figure, Any]]:
    """
    Compute and plot time-frequency representation of EEG data.
    
    Parameters
    ----------
    data : mne.io.Raw or mne.Epochs
        The data to compute time-frequency from
    picks : str, list, slice, None
        Channels to include. E.g. 'eeg', ['Fz', 'Cz', 'Pz'], etc.
    method : str
        Method to compute TFR: 'morlet' (wavelet) or 'multitaper'
    freq_range : tuple
        (min_freq, max_freq) frequency range to compute TFR
    n_cycles : float or list
        Number of cycles in the wavelet, can be a fixed number or one per frequency
    decim : int
        Decimation factor for time points
    average : bool
        Average across epochs (for Epochs data)
    baseline : tuple
        (min, max) baseline period in seconds for normalization
    plot_type : str
        'power', 'itc' (for inter-trial coherence), or 'both'
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure to
    show : bool
        Whether to show the figure
    return_data : bool
        Whether to return computed TFR data along with figure
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    tfr : instance of AverageTFR or list
        The computed time-frequency data if return_data=True
    """
    logging.info(f"Computing time-frequency analysis using {method} method")
    
    # Set up frequencies
    freqs = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), 
                        num=20)
    
    # Set up n_cycles
    if isinstance(n_cycles, (int, float)):
        n_cycles = n_cycles * np.ones_like(freqs)
    
    # Compute TFR based on method and data type
    if isinstance(data, mne.io.Raw):
        # For Raw, we need to create epochs first
        logging.info("Creating epochs from continuous data for TFR")
        events = mne.make_fixed_length_events(data, id=1, duration=1.0)
        epochs = mne.Epochs(data, events, tmin=0, tmax=1.0, baseline=None, 
                           preload=True, picks=picks)
        
        if method == 'morlet':
            tfr = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles,
                                              use_fft=True, return_itc=True, decim=decim,
                                              average=average)
        elif method == 'multitaper':
            tfr = mne.time_frequency.tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles,
                                                 decim=decim, time_bandwidth=4.0,
                                                 return_itc=True, average=average)
        else:
            raise ValueError(f"Unknown TFR method: {method}")
            
        power, itc = tfr  # Unpack power and ITC
            
    else:  # Epochs
        if method == 'morlet':
            tfr = mne.time_frequency.tfr_morlet(data, freqs=freqs, n_cycles=n_cycles,
                                              use_fft=True, return_itc=True, decim=decim,
                                              average=average, picks=picks)
        elif method == 'multitaper':
            tfr = mne.time_frequency.tfr_multitaper(data, freqs=freqs, n_cycles=n_cycles,
                                                 decim=decim, time_bandwidth=4.0,
                                                 return_itc=True, average=average, picks=picks)
        else:
            raise ValueError(f"Unknown TFR method: {method}")
            
        power, itc = tfr  # Unpack power and ITC
    
    # Apply baseline correction if specified
    if baseline is not None:
        logging.info(f"Applying baseline correction {baseline}")
        power.apply_baseline(baseline, mode='percent')
    
    # Create the plot
    if plot_type == 'power' or plot_type == 'both':
        fig_power = power.plot(picks=picks, title=f'{title} - Power', show=False)
        if save_path and plot_type == 'power':
            power_path = save_path.replace('.png', '_power.png') if '.' in save_path else f"{save_path}_power.png"
            plt.savefig(power_path, dpi=300, bbox_inches='tight')
            logging.info(f"Power TFR plot saved to {power_path}")
    
    if plot_type == 'itc' or plot_type == 'both':
        fig_itc = itc.plot(picks=picks, title=f'{title} - ITC', show=False)
        if save_path and plot_type == 'itc':
            itc_path = save_path.replace('.png', '_itc.png') if '.' in save_path else f"{save_path}_itc.png"
            plt.savefig(itc_path, dpi=300, bbox_inches='tight')
            logging.info(f"ITC TFR plot saved to {itc_path}")
    
    # Show if requested
    if show:
        # Try to use an interactive backend if showing is requested
        backend = matplotlib.get_backend()
        if backend.lower() in ('agg', 'figurecanvasagg'):
            try:
                # First try Qt backends
                for qt_backend in ['Qt5Agg', 'QtAgg', 'TkAgg']:
                    try:
                        matplotlib.use(qt_backend, force=True)
                        logging.info(f"Switched to interactive backend: {qt_backend}")
                        break
                    except ImportError:
                        continue
            except Exception as e:
                logging.warning(f"Could not switch to interactive backend: {str(e)}")
                logging.warning("Plot display might fail. Consider setting matplotlib backend manually.")
        plt.show()
    
    # Return appropriate figure and data
    if return_data:
        if plot_type == 'power':
            return fig_power, power
        elif plot_type == 'itc':
            return fig_itc, itc
        else:  # 'both'
            return (fig_power, fig_itc), (power, itc)
    else:
        if plot_type == 'power':
            return fig_power
        elif plot_type == 'itc':
            return fig_itc
        else:  # 'both'
            return fig_power, fig_itc 