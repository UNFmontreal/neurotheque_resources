import os
from pathlib import Path

from scr.pipeline import Pipeline


def test_pipeline_smoke(tmp_path: Path):
    cfg = {
        "auto_save": False,
        "default_subject": "01",
        "default_session": "001",
        "default_run": "01",
        "directory": {
            "root": str(tmp_path),
            "raw_data_dir": "data/raw",
            "processed_dir": "data/processed",
            "reports_dir": "reports",
            "derivatives_dir": "derivatives",
        },
        # Single-subject: no file_path_pattern
        "pipeline": {
            "steps": [
                {"name": "SyntheticRawStep", "params": {"duration_sec": 1.0, "sfreq": 100.0}},
                {"name": "FilterStep", "params": {"l_freq": 1.0, "h_freq": 40.0}},
                {"name": "SaveCheckpoint", "params": {"checkpoint_key": "after_filter"}},
            ]
        },
    }

    pipe = Pipeline(config_dict=cfg, validate_config=True)
    pipe.run()

    # Expect a checkpoint under processed/sub-01/ses-001 with suffix after_filter
    proc = Path(cfg["directory"]["root"]) / cfg["directory"]["processed_dir"]
    out_dir = proc / "sub-01" / "ses-001"
    candidates = list(out_dir.glob("*after_filter.fif"))
    assert candidates, f"No checkpoint written in {out_dir}"

