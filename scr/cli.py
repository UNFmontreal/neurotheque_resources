import click
import logging
from pathlib import Path

from scr.pipeline import Pipeline
from scr.config_generator import main as generate_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@click.group()
def cli():
    """A command-line interface for the neuroflow processing pipeline."""
    pass


@cli.command(name="run")
@click.argument("config_file", type=click.Path(exists=True, dir_okay=False))
def run_pipeline(config_file):
    """Runs the pipeline from a given YAML configuration file."""
    try:
        pipeline = Pipeline(config_file=config_file)
        pipeline.run()
        logging.info("Pipeline executed successfully.")
    except Exception as e:
        logging.error(f"Failed to run pipeline: {e}")
        raise


@cli.command(name="new-config")
def new_config():
    """Launches an interactive wizard to create a new configuration file."""
    generate_config()


if __name__ == "__main__":
    cli()
