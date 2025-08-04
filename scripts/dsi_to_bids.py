import click
import mne
from mne_bids import write_raw_bids, BIDSPath
from pathlib import Path

@click.command()
@click.argument('dsi_file', type=click.Path(exists=True))
@click.argument('bids_root', type=click.Path(file_okay=False))
@click.option('--subject', required=True, help='Subject ID (e.g., "01")')
@click.option('--session', required=True, help='Session ID (e.g., "001")')
@click.option('--task', required=True, help='Task name (e.g., "rest")')
@click.option('--run', type=int, default=1, help='Run number')
def main(dsi_file, bids_root, subject, session, task, run):
    """
    Converts a DSI-24 recording to BIDS format.
    """
    click.echo(f"Reading DSI file: {dsi_file}")
    raw = mne.io.read_raw_dsi(dsi_file)
    
    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        run=run,
        root=bids_root
    )
    
    click.echo(f"Writing to BIDS path: {bids_path}")
    write_raw_bids(raw, bids_path, overwrite=True)
    
    click.echo("Conversion to BIDS complete.")

if __name__ == '__main__':
    main()
