"""
Utility functions for working with autoreject logs
"""

import os
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def find_autoreject_log(subject_id, session_id=None, task_id=None, run_id=None, 
                        base_dir=None, search_derivatives=True):
    """
    Find the autoreject log file for a given subject/session/task/run.
    
    Parameters
    ----------
    subject_id : str
        Subject ID (without 'sub-' prefix)
    session_id : str, optional
        Session ID (without 'ses-' prefix)
    task_id : str, optional
        Task ID
    run_id : str, optional
        Run ID
    base_dir : str or Path, optional
        Base directory to search in
    search_derivatives : bool
        Whether to also search in derivatives directory
        
    Returns
    -------
    log_file : str or None
        Path to the found autoreject log file, or None if not found
    """
    # Normalize subject_id and session_id format
    if subject_id.startswith('sub-'):
        subject_id = subject_id[4:]
    if session_id and session_id.startswith('ses-'):
        session_id = session_id[4:]
    
    # Define common locations to search
    search_locations = []
    
    # First try the expected autoreject log directory
    if base_dir:
        base_path = Path(base_dir)
        search_locations.extend([
            base_path / 'data' / 'processed' / f'sub-{subject_id}' / f'ses-{session_id}' / 'autoreject',
            base_path / 'data' / 'processed' / f'sub-{subject_id}' / f'ses-{session_id}',
            base_path / 'derivatives' / 'autoreject' / f'sub-{subject_id}' / f'ses-{session_id}',
        ])
    else:
        # If no base_dir, use relative paths
        search_locations.extend([
            Path('data') / 'processed' / f'sub-{subject_id}' / f'ses-{session_id}' / 'autoreject',
            Path('data') / 'processed' / f'sub-{subject_id}' / f'ses-{session_id}',
            Path('derivatives') / 'autoreject' / f'sub-{subject_id}' / f'ses-{session_id}',
        ])
    
    # Define possible filenames to look for
    filenames = []
    
    # Add specific filename patterns based on available parameters
    if task_id and run_id:
        filenames.append(f"sub-{subject_id}_ses-{session_id}_task-{task_id}_run-{run_id}_autoreject_log.pickle")
    elif task_id:
        filenames.append(f"sub-{subject_id}_ses-{session_id}_task-{task_id}_autoreject_log.pickle")
    else:
        filenames.append(f"sub-{subject_id}_ses-{session_id}_autoreject_log.pickle")
    
    # Also try generic filename
    filenames.append("autoreject_log.pickle")
    
    # Search for the file
    for location in search_locations:
        for filename in filenames:
            filepath = location / filename
            if os.path.exists(filepath):
                return str(filepath)
    
    # If not found, return None
    return None

def load_autoreject_log(filepath=None, subject_id=None, session_id=None, task_id=None, run_id=None, 
                         base_dir=None):
    """
    Load an autoreject log from a file.
    
    Parameters
    ----------
    filepath : str, optional
        Path to the autoreject log file. If not provided, will try to find it using find_autoreject_log.
    subject_id : str, optional
        Subject ID (required if filepath not provided)
    session_id : str, optional
        Session ID (required if filepath not provided)
    task_id : str, optional
        Task ID
    run_id : str, optional
        Run ID
    base_dir : str, optional
        Base directory to search in
        
    Returns
    -------
    reject_log : autoreject.RejectLog or None
        The loaded autoreject log, or None if not found or error loading
    """
    if filepath is None:
        if subject_id is None or session_id is None:
            raise ValueError("If filepath is not provided, subject_id and session_id must be provided")
        
        filepath = find_autoreject_log(subject_id, session_id, task_id, run_id, base_dir)
        
        if filepath is None:
            print(f"Could not find autoreject log for sub-{subject_id} ses-{session_id}")
            return None
    
    try:
        with open(filepath, 'rb') as f:
            log = pickle.load(f)
        print(f"Successfully loaded autoreject log from {filepath}")
        return log
    except Exception as e:
        print(f"Error loading autoreject log: {e}")
        return None

def plot_autoreject_summary(reject_log, figsize=(20, 16), save_to=None):
    """
    Create a comprehensive visualization of an autoreject log.
    
    Parameters
    ----------
    reject_log : autoreject.RejectLog
        The autoreject log to visualize
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_to : str, optional
        Path to save the figure to. If None, will just display it.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    if reject_log is None:
        print("No reject_log provided")
        return None
    
    # Extract reject log data
    bad_epochs = reject_log.bad_epochs
    ch_names = reject_log.ch_names
    
    # Get labels from RejectLog (0=good, 1=interpolated, 2=bad)
    n_epochs = len(bad_epochs)
    n_channels = len(ch_names)
    
    # Use the labels attribute directly if available
    if hasattr(reject_log, 'labels') and reject_log.labels is not None:
        mask = reject_log.labels
    else:
        # Fallback if no labels attribute
        mask = np.zeros((n_epochs, n_channels), dtype=int)
        for i, is_bad in enumerate(bad_epochs):
            if is_bad:
                mask[i, :] = 2  # Mark all channels in bad epoch as bad
    
    # Calculate statistics
    n_bad_epochs = np.sum(bad_epochs)
    bad_epoch_percent = (n_bad_epochs / n_epochs) * 100
    
    # Count different types of decisions (0=good, 1=interpolated, 2=bad)
    good_count = np.sum(mask == 0)
    interpolated_count = np.sum(mask == 1)
    bad_count = np.sum(mask == 2)
    total_points = mask.size
    
    good_percent = (good_count / total_points) * 100
    interpolated_percent = (interpolated_count / total_points) * 100
    bad_percent = (bad_count / total_points) * 100
    
    # Calculate problematic channels (>50% rejection rate)
    channel_rejection_rates = np.mean(mask > 0, axis=0) * 100
    problematic_channels = [(ch, rate) for ch, rate in zip(ch_names, channel_rejection_rates) if rate > 50]
    
    # Create a comprehensive figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1], wspace=0.15, hspace=0.25)
    
    # Main heatmap - improved visualization
    ax_main = fig.add_subplot(gs[0, 0])
    im = ax_main.imshow(mask, cmap='viridis', aspect='auto', 
                  interpolation='nearest', vmin=0, vmax=2)
    
    # Improved labeling
    ax_main.set_xlabel('Epochs', fontsize=14, fontweight='bold')
    ax_main.set_ylabel('Channels', fontsize=14, fontweight='bold')
    ax_main.set_title(f'AutoReject Results: {bad_epoch_percent:.1f}% of epochs marked bad', 
                     fontsize=16, fontweight='bold')
    
    # Add colorbar with better labels
    cbar = fig.colorbar(im, ax=ax_main, ticks=[0, 1, 2])
    cbar.set_ticklabels(['Good', 'Interpolated', 'Bad'])
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Channel Status', fontsize=14, fontweight='bold')
    
    # Show all channel names with better visibility
    ax_main.set_yticks(range(len(ch_names)))
    ax_main.set_yticklabels(ch_names, fontsize=10)
    
    # Add grid for better readability
    ax_main.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Show more epoch ticks with better spacing
    n_displayed_epochs = min(40, n_epochs)
    epoch_step = max(1, n_epochs // n_displayed_epochs)
    xtick_positions = range(0, n_epochs, epoch_step)
    ax_main.set_xticks(xtick_positions)
    ax_main.set_xticklabels([str(x) for x in xtick_positions], fontsize=10, rotation=45)
    
    # Add grid for x-axis as well
    ax_main.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Add alternating row colors for better visualization
    for i in range(len(ch_names)):
        if i % 2 == 0:
            ax_main.axhspan(i-0.5, i+0.5, color='white', alpha=0.1)
    
    # Channel rejection rates - improved visualization
    ax_ch = fig.add_subplot(gs[0, 1])
    
    # Use color gradient based on rejection rates
    colors = plt.cm.RdYlGn_r(channel_rejection_rates / 100)
    
    # Create horizontal bar plot with improved colors
    bars = ax_ch.barh(range(n_channels), channel_rejection_rates, color=colors)
    
    # No need to duplicate y-axis labels - they're aligned with main plot
    ax_ch.set_yticks([])
    
    # Add value annotations to bars
    for i, v in enumerate(channel_rejection_rates):
        ax_ch.text(v + 1, i, f"{v:.1f}%", va='center', fontsize=9)
    
    ax_ch.set_xlabel('Rejection Rate (%)', fontsize=12, fontweight='bold')
    ax_ch.set_title('Channel Rejection Rates', fontsize=14, fontweight='bold')
    
    # Add threshold line with better visualization
    threshold_line = ax_ch.axvline(x=50, color='red', linestyle='--', linewidth=2)
    
    # Add problematic zone highlighting
    ax_ch.axvspan(50, 100, alpha=0.1, color='red')
    
    # Add grid for better readability
    ax_ch.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Limit x-axis to 0-100%
    ax_ch.set_xlim(0, 100)
    
    # Epoch rejection status - improved visualization
    ax_ep = fig.add_subplot(gs[1, 0])
    
    # Use red color for bad epochs
    bars = ax_ep.bar(range(n_epochs), [1 if x else 0 for x in bad_epochs], color='red', alpha=0.7)
    
    # Highlight bad epochs with a different color
    for i, is_bad in enumerate(bad_epochs):
        if is_bad:
            bars[i].set_color('darkred')
            bars[i].set_alpha(1.0)
    
    ax_ep.set_xlabel('Epoch Index', fontsize=12, fontweight='bold')
    ax_ep.set_ylabel('Rejected', fontsize=12, fontweight='bold')
    ax_ep.set_yticks([0, 1])
    ax_ep.set_yticklabels(['No', 'Yes'], fontsize=10)
    ax_ep.set_title('Epoch Rejection Status', fontsize=14, fontweight='bold')
    
    # Add grid for better readability
    ax_ep.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Improve x-axis ticks
    ax_ep.set_xticks(xtick_positions)
    ax_ep.set_xticklabels([str(x) for x in xtick_positions], fontsize=10, rotation=45)
    
    # Summary stats with improved layout
    ax_stats = fig.add_subplot(gs[1, 1])
    ax_stats.axis('off')
    
    # Create more detailed stats text
    stats_text = (
        f"AutoReject Summary Statistics\n"
        f"----------------------------\n\n"
        f"Total Epochs: {n_epochs}\n"
        f"Bad Epochs: {n_bad_epochs} ({bad_epoch_percent:.1f}%)\n\n"
        f"Data points classified as:\n"
        f"• Good: {good_count} ({good_percent:.1f}%)\n"
        f"• Interpolated: {interpolated_count} ({interpolated_percent:.1f}%)\n"
        f"• Bad: {bad_count} ({bad_percent:.1f}%)\n\n"
    )
    
    # Add problematic channels if any
    if problematic_channels:
        stats_text += "Problematic Channels (>50% rejection):\n"
        for ch, rate in problematic_channels[:5]:  # Show top 5
            stats_text += f"• {ch}: {rate:.1f}%\n"
        
        if len(problematic_channels) > 5:
            stats_text += f"...and {len(problematic_channels) - 5} more"
    
    # Add the stats text with better formatting
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                 verticalalignment='top', fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))
    
    # Add a legend for clarity
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='purple', label='Good'),
        Patch(facecolor='teal', label='Interpolated'),
        Patch(facecolor='yellow', label='Bad')
    ]
    ax_main.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.suptitle(f'AutoReject Analysis Results', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
    
    if save_to:
        plt.savefig(save_to, dpi=200, bbox_inches='tight')
        print(f"Figure saved to {save_to}")
    
    return fig 

def plot_reject_log(reject_log, orientation='horizontal', figsize=(18, 14), save_to=None):
    """
    Create an enhanced version of the RejectLog's plot method with better visibility
    of channel names and clearer visualization.
    
    Parameters
    ----------
    reject_log : autoreject.RejectLog
        The autoreject log to visualize
    orientation : str
        'horizontal' or 'vertical' orientation for the plot
    figsize : tuple
        Figure size as (width, height) in inches
    save_to : str, optional
        Path to save the figure to. If None, will just display it.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    if reject_log is None:
        print("No reject_log provided")
        return None
    
    # Create the figure with larger size
    plt.figure(figsize=figsize)
    
    # Get the original autoreject plot
    try:
        # Plot with show_names=True to ensure channel names are displayed
        original_fig = reject_log.plot(orientation=orientation, show_names=True, show=False)
        
        # Enhance the figure
        for ax in original_fig.axes:
            # Increase font size for tick labels
            ax.tick_params(axis='both', labelsize=11)
            
            # Ensure y-tick labels are readable
            if orientation == 'horizontal' and ax.get_yticklabels():
                # Increase spacing for y-axis labels
                ax.tick_params(axis='y', pad=10)
                
                # Add a grid to better distinguish rows
                ax.grid(axis='y', linestyle='--', alpha=0.3)
                
                # Add alternating row colors for better readability
                ylim = ax.get_ylim()
                for i in range(int(ylim[0]), int(ylim[1])):
                    if i % 2 == 0:
                        ax.axhspan(i-0.5, i+0.5, color='lightgray', alpha=0.1)
            
            # Add grid for x-axis too
            ax.grid(axis='x', linestyle='--', alpha=0.3)
            
            # Improve tick labels readability
            if orientation == 'horizontal':
                plt.setp(ax.get_yticklabels(), fontweight='bold')
            else:
                plt.setp(ax.get_xticklabels(), fontweight='bold')
        
        # Adjust figure layout
        plt.tight_layout(pad=3.0)
        
        # Add a title and explanatory legend
        plt.suptitle('AutoReject Channel Status', fontsize=16, fontweight='bold', y=0.98)
        
        # Add explanatory text
        legend_text = """
        Color coding:
        - Green: Good data points
        - Blue: Interpolated data points
        - Red: Bad data points
        """
        plt.figtext(0.02, 0.02, legend_text, fontsize=12, 
                   bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))
        
        # Save if requested
        if save_to:
            original_fig.savefig(save_to, dpi=200, bbox_inches='tight')
            print(f"Enhanced RejectLog plot saved to {save_to}")
        
        return original_fig
    
    except Exception as e:
        print(f"Error creating enhanced RejectLog plot: {e}")
        
        # Create a fallback visualization
        plt.figure(figsize=figsize)
        plt.title("RejectLog Visualization (Fallback)", fontsize=16)
        plt.text(0.5, 0.5, "Could not create enhanced RejectLog plot. See detailed visualization instead.",
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=14)
        plt.axis('off')
        
        if save_to:
            plt.savefig(save_to, dpi=150)
            print(f"Fallback plot saved to {save_to}")
        
        return plt.gcf() 

def scan_for_autoreject_logs(base_dir=None):
    """
    Scan a directory and all subdirectories for any autoreject log files.
    This is useful for debugging when logs are not being found in their expected locations.
    
    Parameters
    ----------
    base_dir : str or Path, optional
        Base directory to start the search from. If None, uses the current directory.
        
    Returns
    -------
    list of str
        List of absolute paths to all found autoreject log files
    """
    import glob
    
    if base_dir is None:
        base_dir = os.getcwd()
    
    base_dir = Path(base_dir)
    
    # Common autoreject log filename patterns
    patterns = [
        "**/*rejectlog.pkl",
        "**/*autoreject_log.pickle",
        "**/autoreject/*.pickle",
        "**/autoreject/**/*.pickle",
        "**/processed/**/*rejectlog.pkl",
        "**/processed/**/*autoreject_log.pickle"
    ]
    
    # Collect all matching files
    found_files = []
    for pattern in patterns:
        matches = list(base_dir.glob(pattern))
        found_files.extend([str(path) for path in matches])
    
    # Remove duplicates and sort
    found_files = sorted(set(found_files))
    
    print(f"Found {len(found_files)} potential autoreject log files:")
    for i, file_path in enumerate(found_files):
        # Get file size to confirm it's a valid file
        try:
            size = os.path.getsize(file_path)
            print(f"{i+1}. {file_path} ({size} bytes)")
        except Exception as e:
            print(f"{i+1}. {file_path} (Error: {e})")
    
    return found_files 