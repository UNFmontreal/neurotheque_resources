import pytest
from pathlib import Path
from scr.steps.project_paths import ProjectPaths

@pytest.fixture
def mock_config():
    return {
        "directory": {
            "root": "test_data",
            "raw_data_dir": "raw",
            "processed_dir": "processed",
            "reports_dir": "reports",
            "derivatives_dir": "derivatives"
        }
    }

@pytest.fixture
def project_paths(tmp_path, mock_config):
    # Create a dummy project root
    project_root = tmp_path
    (project_root / "test_data").mkdir()
    return ProjectPaths(mock_config, project_root=project_root)

def test_project_paths_creation(project_paths, tmp_path):
    assert project_paths.base_dir == (tmp_path / "test_data").resolve()
    assert project_paths.raw_data_dir.exists()
    assert project_paths.processed_dir.exists()
    assert project_paths.reports_dir.exists()
    assert project_paths.derivatives_dir.exists()

def test_get_raw_eeg_path(project_paths):
    path = project_paths.get_raw_eeg_path("sub-01", "ses-01", "test_task", "01")
    assert isinstance(path, Path)
    expected_path = "raw/sub-01/ses-01/eeg/sub-01_ses-01_task-test_task_run-01_eeg.fif"
    assert expected_path in str(path)

def test_get_derivative_path(project_paths):
    path = project_paths.get_derivative_path("01", "001", "test_task", "01", "preprocessed")
    assert isinstance(path, Path)
    expected_path = "processed/sub-01/ses-001/sub-01_ses-001_task-test_task_run-01_preprocessed.fif"
    assert expected_path in str(path)
