__all__ = ["__version__", "run_pipeline"]

__version__ = "0.1.0"


def run_pipeline(config_path: str, validate: bool = True) -> None:
    """Programmatic entry to run the pipeline from a config file."""
    from .pipeline import Pipeline

    pipe = Pipeline(config_file=config_path, validate_config=validate)
    pipe.run()
