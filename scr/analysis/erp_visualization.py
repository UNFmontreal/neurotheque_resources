#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ERP Visualization Module

This module provides a class for visualizing Event-Related Potentials (ERPs)
using MNE Python. It offers various plotting options and analyses for ERP data.

Author: Claude (Enhanced by Assistant)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from typing import Dict, List, Union, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('erp_visualization')


class ERPVisualizer:
    """
    Class for visualizing and analyzing Event-Related Potentials (ERPs).

    This class provides various methods to visualize ERPs from MNE Epochs objects,
    including condition comparisons, topographic maps, difference waves, and more.
    
    Attributes
    ----------
    epochs : mne.Epochs
        The epochs object containing EEG data.
    output_dir : Optional[str]
        Directory to save plots.
    subject_id : str
        Subject identifier.
    session_id : str
        Session identifier.
    run_id : str
        Run identifier.
    task_id : str
        Task identifier.
    """

    def __init__(
        self, 
        epochs: mne.Epochs,
        output_dir: Optional[str] = None,
        subject_id: Optional[str] = None,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
        task_id: Optional[str] = None
    ):
        """
        Initialize the ERP visualizer with epochs and metadata.
        
        Parameters
        ----------
        epochs : mne.Epochs
            The epochs object containing EEG data.
        output_dir : Optional[str]
            Directory to save plots, by default None.
        subject_id : Optional[str]
            Subject identifier, by default None.
        session_id : Optional[str]
            Session identifier, by default None.
        run_id : Optional[str]
            Run identifier, by default None.
        task_id : Optional[str]
            Task identifier, by default None.
        """
        self.epochs = epochs
        self.output_dir = output_dir
        self.subject_id = subject_id or '01'
        self.session_id = session_id or '001'
        self.run_id = run_id or '01'
        self.task_id = task_id or 'task'

        # Create output directory if provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Initialized ERPVisualizer with {len(epochs)} epochs")
        logger.info(f"Available event types: {epochs.event_id}")

    def get_evokeds(self, 
                    conditions: Union[str, List[str]] = None, 
                    baseline: Optional[Tuple[float, float]] = None
                   ) -> Dict[str, mne.Evoked]:
        """
        Get evoked responses for specified conditions.

        Parameters
        ----------
        conditions : Union[str, List[str]], optional
            Condition(s) to extract; if None uses all available conditions.
        baseline : Optional[Tuple[float, float]]
            Baseline period (start, end) in seconds, by default None.

        Returns
        -------
        Dict[str, mne.Evoked]
            Dictionary mapping condition names to Evoked objects.
        """
        # Process input conditions
        if conditions is None:
            conditions = list(self.epochs.event_id.keys())
        elif isinstance(conditions, str):
            conditions = [conditions]

        # Use a copy of the epochs to avoid modifying the original object
        epochs = self.epochs.copy()
        if baseline is not None:
            logger.info(f"Applying baseline correction: {baseline}")
            epochs.apply_baseline(baseline)

        evokeds = {}
        for condition in conditions:
            try:
                # Extract evoked response using Epochs selection and average
                evoked = epochs[condition].average()
                evokeds[condition] = evoked
                logger.info(f"Extracted evoked response for '{condition}' condition")
            except (ValueError, KeyError) as e:
                logger.warning(f"Could not extract condition '{condition}': {e}")

        return evokeds

    def plot_erps_comparison(self,
                             conditions: List[str],
                             channels: Optional[Union[str, List[str]]] = None,
                             colors: Optional[List[str]] = None,
                             title: str = "ERP Comparison",
                             show: bool = True,
                             save: bool = True,
                             filename: Optional[str] = None,
                             figsize: Tuple[int, int] = (10, 6),
                             baseline: Optional[Tuple[float, float]] = None,
                             ylim: Optional[Tuple[float, float]] = None,
                             spatial_colors: bool = True,
                             gfp: bool = True,
                             custom_style: bool = True
                            ) -> Optional[plt.Figure]:
        """
        Plot ERPs for multiple conditions by comparing evoked responses.

        If custom_style is False, then the function will use MNE's built-in
        plot_compare_evokeds for a cleaner comparison.

        Parameters
        ----------
        conditions : List[str]
            List of condition names to compare.
        channels : Optional[Union[str, List[str]]]
            Channel(s) to plot; if None, all channels are used.
        colors : Optional[List[str]]
            Colors for each condition; if None, a default list is used.
        title : str, optional
            Plot title, by default "ERP Comparison".
        show : bool, optional
            Whether to display the plot, by default True.
        save : bool, optional
            Whether to save the plot, by default True.
        filename : Optional[str], optional
            Custom filename for saved plot, by default None.
        figsize : Tuple[int, int], optional
            Figure size, by default (10, 6).
        baseline : Optional[Tuple[float, float]], optional
            Baseline period for correction, by default None.
        ylim : Optional[Tuple[float, float]], optional
            Y-axis limits, by default None.
        spatial_colors : bool, optional
            Whether to use spatial colors in MNE plots, by default True.
        gfp : bool, optional
            Whether to show global field power (GFP), by default True.
        custom_style : bool, optional
            Whether to apply manual custom styling; if False, MNE's default comparison
            function is used.

        Returns
        -------
        Optional[plt.Figure]
            The figure object containing the plot, or None if no evokeds are extracted.
        """
        # Extract evoked objects
        evokeds = self.get_evokeds(conditions, baseline)
        if not evokeds:
            logger.error(f"No evoked responses could be extracted for conditions: {conditions}")
            return None

        # Set default colors if none provided
        if colors is None:
            colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
            colors = colors[:len(conditions)]

        # If not using custom styling, leverage MNE's built-in function.
        if not custom_style:
            # Use channel selection if specified
            picks = channels if channels is not None else None
            fig = mne.viz.plot_compare_evokeds(evokeds, picks=picks, colors=colors,
                                               show=show, title=title)
            if ylim:
                plt.ylim(ylim)
            if save and self.output_dir:
                if filename is None:
                    conditions_str = '_'.join(conditions)
                    filename = f"{self.subject_id}_{self.session_id}_{conditions_str}_erp.png"
                filepath = os.path.join(self.output_dir, filename)
                try:
                    fig.savefig(filepath, dpi=300)
                    logger.info(f"Saved ERP comparison plot to {filepath}")
                except Exception as e:
                    logger.error(f"Error saving figure: {e}")
            return fig

        # Custom style: manually overlay each evoked
        fig, ax = plt.subplots(figsize=figsize)
        for condition, color in zip(conditions, colors):
            if condition not in evokeds:
                continue

            evoked = evokeds[condition]
            # If channels are specified, pick them from the evoked object
            if channels:
                if isinstance(channels, str):
                    channels_list = [channels]
                else:
                    channels_list = channels
                # Check for valid channels
                valid_chs = [ch for ch in channels_list if ch in evoked.ch_names]
                if not valid_chs:
                    logger.warning(f"No valid channels found for condition {condition}")
                    continue
                evoked = evoked.copy().pick_channels(valid_chs)

            # Plot all channels manually: plotting each channel trace over the same ax
            times = evoked.times
            for idx, ch in enumerate(evoked.ch_names):
                # Label only the first channel per condition
                label = condition if idx == 0 else None
                ax.plot(times, evoked.data[idx, :],
                        color=color, linewidth=1.5, label=label, alpha=0.8)
            if gfp:
                # Optionally plot Global Field Power (GFP) in a thicker line
                gfp_val = evoked.data.std(axis=0)
                ax.plot(times, gfp_val, color=color, linewidth=3, linestyle='--',
                        label=f"{condition} GFP")

        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (μV)")
        if ylim:
            ax.set_ylim(ylim)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        if show:
            plt.show()

        if save and self.output_dir:
            if filename is None:
                conditions_str = '_'.join(conditions)
                filename = f"{self.subject_id}_{self.session_id}_{conditions_str}_erp.png"
            filepath = os.path.join(self.output_dir, filename)
            try:
                fig.savefig(filepath, dpi=300)
                logger.info(f"Saved ERP comparison plot to {filepath}")
            except Exception as e:
                logger.error(f"Error saving figure: {e}")

        return fig
    
    def plot_topo_maps(self,
                        conditions: List[str],
                        times: Optional[List[float]] = None,
                        title: str = "Topographic Maps",
                        show: bool = True,
                        save: bool = True,
                        filename: Optional[str] = None,
                        vmin: Optional[float] = None,
                        vmax: Optional[float] = None,
                        baseline: Optional[Tuple[float, float]] = None
                    ) -> List[plt.Figure]:
        """
        Plot topographic maps for each condition at specified time points.

        Parameters
        ----------
        conditions : List[str]
            List of condition names to plot.
        times : Optional[List[float]], optional
            Time points (in seconds) for which to plot maps; if None defaults are used.
        title : str, optional
            Plot title, by default "Topographic Maps".
        show : bool, optional
            Whether to display the plots, by default True.
        save : bool, optional
            Whether to save the plots, by default True.
        filename : Optional[str], optional
            Custom filename prefix for saved plots.
        vmin : Optional[float], optional
            Minimum value for color scale; not passed directly to plot_topomap.
        vmax : Optional[float], optional
            Maximum value for color scale; not passed directly to plot_topomap.
        baseline : Optional[Tuple[float, float]], optional
            Baseline period for correction, by default None.

        Returns
        -------
        List[plt.Figure]
            List of figure objects for the topographic maps.
        """
        evokeds = self.get_evokeds(conditions, baseline)
        if not evokeds:
            logger.error(f"No evoked responses could be extracted for conditions: {conditions}")
            return []

        if times is None:
            # Default times (adjust as needed based on your ERP components)
            times = [0.1, 0.2, 0.3, 0.4, 0.5]

        figures = []
        for condition in conditions:
            if condition not in evokeds:
                continue
            evoked = evokeds[condition]

            # Prepare keyword arguments without vmin/vmax to avoid error
            topo_kwargs = dict(times=times, show=show, colorbar=True)
            fig = evoked.plot_topomap(**topo_kwargs)

            # If vmin or vmax is provided, update the color limits on the images
            if vmin is not None or vmax is not None:
                for ax in fig.axes:
                    for im in ax.get_images():
                        # Use current clim if limits are not provided
                        current_clim = im.get_clim()
                        new_vmin = vmin if vmin is not None else current_clim[0]
                        new_vmax = vmax if vmax is not None else current_clim[1]
                        im.set_clim(new_vmin, new_vmax)

            if hasattr(fig, 'suptitle'):
                fig.suptitle(f"{title} - {condition}")

            figures.append(fig)

            if save and self.output_dir:
                if filename is None:
                    times_str = '_'.join([f"{t:.2f}".replace('.', '') for t in times])
                    save_filename = f"{self.subject_id}_{self.session_id}_{condition}_topo_{times_str}.png"
                else:
                    save_filename = f"{condition}_{filename}"
                filepath = os.path.join(self.output_dir, save_filename)
                try:
                    fig.savefig(filepath, dpi=300)
                    logger.info(f"Saved topographic map for {condition} to {filepath}")
                except Exception as e:
                    logger.error(f"Error saving figure: {e}")

        return figures

    def plot_butterfly(self,
                       conditions: List[str],
                       title: str = "Butterfly Plot",
                       show: bool = True,
                       save: bool = True,
                       filename: Optional[str] = None,
                       figsize: Tuple[int, int] = (10, 6),
                       baseline: Optional[Tuple[float, float]] = None,
                       spatial_colors: bool = True,
                       gfp: bool = True
                      ) -> Dict[str, plt.Figure]:
        """
        Create butterfly plots for each condition.

        Parameters
        ----------
        conditions : List[str]
            List of condition names to plot.
        title : str, optional
            Plot title, by default "Butterfly Plot".
        show : bool, optional
            Whether to display the plots, by default True.
        save : bool, optional
            Whether to save the plots, by default True.
        filename : Optional[str], optional
            Custom filename prefix for saved plots.
        figsize : Tuple[int, int], optional
            Figure size, by default (10, 6).
        baseline : Optional[Tuple[float, float]], optional
            Baseline period for correction, by default None.
        spatial_colors : bool, optional
            Whether to use spatial colors, by default True.
        gfp : bool, optional
            Whether to show global field power, by default True.

        Returns
        -------
        Dict[str, plt.Figure]
            Dictionary mapping condition names to their corresponding figure objects.
        """
        evokeds = self.get_evokeds(conditions, baseline)
        if not evokeds:
            logger.error(f"No evoked responses could be extracted for conditions: {conditions}")
            return {}

        figures = {}
        for condition, evoked in evokeds.items():
            fig, ax = plt.subplots(figsize=figsize)
            # Plot using MNE's built-in plotting method for an evoked object
            evoked.plot(axes=ax, show=False, spatial_colors=spatial_colors,
                        time_unit='s', gfp=gfp, selectable=False)
            ax.set_title(f"{title} - {condition}")
            plt.tight_layout()
            figures[condition] = fig

            if show:
                plt.show()

            if save and self.output_dir:
                if filename is None:
                    save_filename = f"{self.subject_id}_{self.session_id}_{condition}_butterfly.png"
                else:
                    save_filename = f"{condition}_{filename}"
                filepath = os.path.join(self.output_dir, save_filename)
                try:
                    fig.savefig(filepath, dpi=300)
                    logger.info(f"Saved butterfly plot for {condition} to {filepath}")
                except Exception as e:
                    logger.error(f"Error saving figure: {e}")

        return figures

    def plot_joint(self,
                   conditions: List[str],
                   times: Optional[List[float]] = None,
                   show: bool = True,
                   save: bool = True,
                   filename: Optional[str] = None,
                   baseline: Optional[Tuple[float, float]] = None
                  ) -> Dict[str, plt.Figure]:
        """
        Create joint plots (combining butterfly and topomaps) for each condition.

        Parameters
        ----------
        conditions : List[str]
            List of condition names to plot.
        times : Optional[List[float]], optional
            Time points (in seconds) for the topomaps; if None, defaults are used.
        show : bool, optional
            Whether to display the plots, by default True.
        save : bool, optional
            Whether to save the plots, by default True.
        filename : Optional[str], optional
            Custom filename prefix for saved plots.
        baseline : Optional[Tuple[float, float]], optional
            Baseline period for correction, by default None.

        Returns
        -------
        Dict[str, plt.Figure]
            Dictionary mapping condition names to the joint plot figure objects.
        """
        evokeds = self.get_evokeds(conditions, baseline)
        if not evokeds:
            logger.error(f"No evoked responses could be extracted for conditions: {conditions}")
            return {}

        if times is None:
            times = [0.1, 0.2, 0.3, 0.4, 0.5]

        figures = {}
        for condition, evoked in evokeds.items():
            fig = evoked.plot_joint(times=times, show=show)
            figures[condition] = fig

            if save and self.output_dir:
                times_str = '_'.join([f"{t:.2f}".replace('.', '') for t in times])
                if filename is None:
                    save_filename = f"{self.subject_id}_{self.session_id}_{condition}_joint_{times_str}.png"
                else:
                    save_filename = f"{condition}_{filename}"
                filepath = os.path.join(self.output_dir, save_filename)
                try:
                    fig.savefig(filepath, dpi=300)
                    logger.info(f"Saved joint plot for {condition} to {filepath}")
                except Exception as e:
                    logger.error(f"Error saving figure: {e}")

        return figures

    def plot_difference_wave(self,
                             condition1: str,
                             condition2: str,
                             channels: Optional[Union[str, List[str]]] = None,
                             title: str = "Difference Wave",
                             show: bool = True,
                             save: bool = True,
                             filename: Optional[str] = None,
                             figsize: Tuple[int, int] = (10, 6),
                             baseline: Optional[Tuple[float, float]] = None,
                             ylim: Optional[Tuple[float, float]] = None
                            ) -> Optional[plt.Figure]:
        """
        Plot the difference wave between two conditions.

        Parameters
        ----------
        condition1 : str
            First condition name.
        condition2 : str
            Second condition name.
        channels : Optional[Union[str, List[str]]]
            Channel(s) to plot; if None, all channels are used.
        title : str, optional
            Plot title, by default "Difference Wave".
        show : bool, optional
            Whether to display the plot, by default True.
        save : bool, optional
            Whether to save the plot, by default True.
        filename : Optional[str], optional
            Custom filename for the saved plot, by default None.
        figsize : Tuple[int, int], optional
            Figure size, by default (10, 6).
        baseline : Optional[Tuple[float, float]], optional
            Baseline period for correction, by default None.
        ylim : Optional[Tuple[float, float]], optional
            Y-axis limits, by default None.

        Returns
        -------
        Optional[plt.Figure]
            The figure object containing the plot, or None if extraction fails.
        """
        evokeds = self.get_evokeds([condition1, condition2], baseline)
        if condition1 not in evokeds or condition2 not in evokeds:
            logger.error(f"Could not extract both conditions: {condition1}, {condition2}")
            return None

        # Compute the difference wave
        diff_wave = evokeds[condition1].copy()
        diff_wave.data = evokeds[condition1].data - evokeds[condition2].data
        diff_wave.comment = f"{condition1} - {condition2}"

        # Optionally select channels
        if channels:
            if isinstance(channels, str):
                channels_list = [channels]
            else:
                channels_list = channels
            valid_chs = [ch for ch in channels_list if ch in diff_wave.ch_names]
            if not valid_chs:
                logger.warning(f"No valid channels found for difference plot between {condition1} and {condition2}")
            else:
                diff_wave = diff_wave.copy().pick_channels(valid_chs)

        fig, ax = plt.subplots(figsize=figsize)
        diff_wave.plot(axes=ax, show=False, spatial_colors=True, 
                       time_unit='s', gfp=True, selectable=False)
        ax.set_title(f"{title}: {condition1} - {condition2}")
        if ylim:
            ax.set_ylim(ylim)
        plt.tight_layout()

        if show:
            plt.show()

        if save and self.output_dir:
            if filename is None:
                filename = f"{self.subject_id}_{self.session_id}_{condition1}_minus_{condition2}_diff.png"
            filepath = os.path.join(self.output_dir, filename)
            try:
                fig.savefig(filepath, dpi=300)
                logger.info(f"Saved difference wave plot to {filepath}")
            except Exception as e:
                logger.error(f"Error saving figure: {e}")

        return fig

    def plot_channel_erps(self,
                          conditions: List[str],
                          channel: str,
                          colors: Optional[List[str]] = None,
                          title: Optional[str] = None,
                          show: bool = True,
                          save: bool = True,
                          filename: Optional[str] = None,
                          figsize: Tuple[int, int] = (10, 6),
                          baseline: Optional[Tuple[float, float]] = None,
                          ylim: Optional[Tuple[float, float]] = None
                         ) -> Optional[plt.Figure]:
        """
        Plot ERPs for a specific channel across multiple conditions.

        Parameters
        ----------
        conditions : List[str]
            List of condition names to compare.
        channel : str
            Channel name to plot.
        colors : Optional[List[str]], optional
            Colors for each condition; if None, a default list is used.
        title : Optional[str], optional
            Plot title; if None, a default title is set.
        show : bool, optional
            Whether to display the plot, by default True.
        save : bool, optional
            Whether to save the plot, by default True.
        filename : Optional[str], optional
            Custom filename for the saved plot.
        figsize : Tuple[int, int], optional
            Figure size, by default (10, 6).
        baseline : Optional[Tuple[float, float]], optional
            Baseline period for correction, by default None.
        ylim : Optional[Tuple[float, float]], optional
            Y-axis limits, by default None.

        Returns
        -------
        Optional[plt.Figure]
            The figure object containing the plot, or None if no evoked data is available.
        """
        evokeds = self.get_evokeds(conditions, baseline)
        if not evokeds:
            logger.error(f"No evoked responses could be extracted for conditions: {conditions}")
            return None

        if colors is None:
            colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
            colors = colors[:len(conditions)]

        if title is None:
            title = f"Channel {channel} ERPs"

        fig, ax = plt.subplots(figsize=figsize)
        for condition, color in zip(conditions, colors):
            if condition not in evokeds:
                continue
            evoked = evokeds[condition]
            if channel not in evoked.ch_names:
                logger.warning(f"Channel {channel} not found in condition {condition}")
                continue
            ch_idx = evoked.ch_names.index(channel)
            ax.plot(evoked.times, evoked.data[ch_idx, :],
                    color=color, linewidth=1.5, label=condition)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (μV)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(0, color='k', linestyle='-', alpha=0.2)
        ax.axvline(0, color='k', linestyle='-', alpha=0.2)
        if ylim:
            ax.set_ylim(ylim)
        plt.tight_layout()

        if show:
            plt.show()

        if save and self.output_dir:
            conditions_str = '_'.join(conditions)
            if filename is None:
                filename = f"{self.subject_id}_{self.session_id}_{channel}_{conditions_str}_erp.png"
            filepath = os.path.join(self.output_dir, filename)
            try:
                fig.savefig(filepath, dpi=300)
                logger.info(f"Saved channel ERP plot to {filepath}")
            except Exception as e:
                logger.error(f"Error saving figure: {e}")

        return fig

    def plot_component_analysis(self,
                                conditions: List[str],
                                time_windows: Dict[str, Tuple[float, float]],
                                channel_groups: Optional[Dict[str, List[str]]] = None,
                                show: bool = True,
                                save: bool = True,
                                figsize: Tuple[int, int] = (12, 8),
                                baseline: Optional[Tuple[float, float]] = None
                               ) -> Optional[plt.Figure]:
        """
        Plot ERP component analysis for specified time windows and channel groups.

        Parameters
        ----------
        conditions : List[str]
            List of condition names to compare.
        time_windows : Dict[str, Tuple[float, float]]
            Dictionary mapping component names to (start, end) times.
        channel_groups : Optional[Dict[str, List[str]]], optional
            Dictionary mapping group names to lists of channels; if None, all channels are used.
        show : bool, optional
            Whether to display the plot, by default True.
        save : bool, optional
            Whether to save the plot, by default True.
        figsize : Tuple[int, int], optional
            Figure size, by default (12, 8).
        baseline : Optional[Tuple[float, float]], optional
            Baseline period for correction, by default None.

        Returns
        -------
        Optional[plt.Figure]
            The figure object containing the analysis plot, or None if no evoked data is available.
        """
        evokeds = self.get_evokeds(conditions, baseline)
        if not evokeds:
            logger.error(f"No evoked responses could be extracted for conditions: {conditions}")
            return None

        # If no channel groups provided, use all channels
        if channel_groups is None:
            all_chs = evokeds[list(evokeds.keys())[0]].ch_names
            channel_groups = {"All": all_chs}

        n_components = len(time_windows)
        n_groups = len(channel_groups)
        fig, axes = plt.subplots(n_groups, n_components, figsize=figsize)

        # Ensure axes is a 2D array
        if n_groups == 1 and n_components == 1:
            axes = np.array([[axes]])
        elif n_groups == 1:
            axes = np.array([axes])
        elif n_components == 1:
            axes = np.array([axes]).T

        bar_width = 0.8 / len(conditions)

        for i, (group_name, channels) in enumerate(channel_groups.items()):
            for j, (component_name, (tmin, tmax)) in enumerate(time_windows.items()):
                ax = axes[i, j]
                means = []
                labels = []
                for condition in conditions:
                    if condition not in evokeds:
                        continue
                    evoked = evokeds[condition]
                    valid_channels = [ch for ch in channels if ch in evoked.ch_names]
                    if not valid_channels:
                        logger.warning(f"No valid channels found in group {group_name} for condition {condition}")
                        continue
                    evoked_picked = evoked.copy().pick_channels(valid_channels)
                    time_mask = (evoked.times >= tmin) & (evoked.times <= tmax)
                    mean_amp = evoked_picked.data[:, time_mask].mean()
                    means.append(mean_amp)
                    labels.append(condition)

                # Plot the bars for each condition side by side
                x = np.arange(1)
                for k, (mean_amp, cond_label) in enumerate(zip(means, labels)):
                    ax.bar(x + k * bar_width - 0.4, mean_amp, bar_width, label=cond_label)
                ax.set_title(f"{component_name} ({tmin:.2f}-{tmax:.2f}s)")
                if i == n_groups - 1:
                    ax.set_xlabel("Component")
                if j == 0:
                    ax.set_ylabel(f"{group_name}\nAmplitude (μV)")
                ax.set_xticks([])
                ax.grid(True, linestyle='--', alpha=0.3, axis='y')
                if i == 0 and j == 0:
                    ax.legend()

        plt.tight_layout()

        if show:
            plt.show()

        if save and self.output_dir:
            components_str = '_'.join(time_windows.keys())
            filepath = os.path.join(self.output_dir, f"{self.subject_id}_{self.session_id}_{components_str}_analysis.png")
            try:
                fig.savefig(filepath, dpi=300)
                logger.info(f"Saved component analysis plot to {filepath}")
            except Exception as e:
                logger.error(f"Error saving figure: {e}")

        return fig


# Example usage
if __name__ == "__main__":
    # This block is for demonstration.
    # Replace 'my_epochs' with your actual mne.Epochs object.
    """
    from your_module.erp_visualization import ERPVisualizer

    erp_viz = ERPVisualizer(
        epochs=my_epochs,
        output_dir='figures/erp',
        subject_id='01',
        session_id='001'
    )
    # Plot ERP comparison using built-in MNE function
    erp_viz.plot_erps_comparison(
        conditions=['go', 'nogo'],
        title="Go/NoGo ERPs",
        custom_style=False
    )

    # Plot topographic maps
    erp_viz.plot_topo_maps(
        conditions=['go', 'nogo'],
        times=[0.1, 0.2, 0.3, 0.4]
    )

    # Plot the difference wave between conditions
    erp_viz.plot_difference_wave(
        condition1='nogo',
        condition2='go',
        title="NoGo - Go Difference"
    )
    """

