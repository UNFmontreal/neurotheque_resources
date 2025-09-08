import json
from pathlib import Path

from scr.pipeline import Pipeline


def test_json_config_smoke(tmp_path: Path):
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
        "pipeline": {
            "steps": [
                {"name": "SyntheticRawStep", "params": {"duration_sec": 0.5, "sfreq": 100.0}},
                {"name": "FilterStep", "params": {"l_freq": 1.0, "h_freq": 40.0}},
                {"name": "SaveCheckpoint", "params": {"checkpoint_key": "after_filter"}},
            ]
        },
    }
    config_file = tmp_path / "smoke.json"
    config_file.write_text(json.dumps(cfg), encoding="utf-8")

    pipe = Pipeline(config_file=str(config_file), validate_config=True)
    pipe.run()

    proc = Path(cfg["directory"]["root"]) / cfg["directory"]["processed_dir"]
    out_dir = proc / "sub-01" / "ses-001"
    candidates = list(out_dir.glob("*after_filter.fif"))
    assert candidates, f"No checkpoint written in {out_dir}"
