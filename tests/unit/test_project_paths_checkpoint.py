from pathlib import Path
from scr.steps.project_paths import ProjectPaths


def test_get_checkpoint_path_naming(tmp_path: Path):
    cfg = {
        "directory": {
            "root": str(tmp_path),
            "raw_data_dir": "raw",
            "processed_dir": "processed",
            "reports_dir": "reports",
            "derivatives_dir": "derivatives",
        }
    }
    paths = ProjectPaths(cfg)
    p = paths.get_checkpoint_path("01", "001", task_id="gng", run_id="01", checkpoint_name="after_filter")
    # Verify naming convention and location
    assert p.name.endswith("after_filter.fif")
    assert "sub-01_ses-001_task-gng_run-01_after_filter.fif" in str(p)
