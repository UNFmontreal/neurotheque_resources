# File: scr/project_paths.py

from pathlib import Path

class ProjectPaths:
    """
    Central class for all file and directory paths.
    Each method here returns the *final* path for reading/writing,
    based on subject/session and your folder conventions.
    """

    def __init__(self, base_dir):
        self.base_dir = Path(base_dir).resolve()

    def get_raw_input_file(self, subject_id, session_id):
        """
        Return the EDF or FIF raw input path for the pipeline to load.
        E.g., "data/pilot_data/sub-{subject_id}_ses-{session_id}_raw.edf"
        """
        # You can define an actual search if you want, or a direct template:
        return self.base_dir / "data" / "pilot_data" / f"sub-{subject_id}_ses-{session_id}_raw.edf"

    def get_checkpoint_file(self, subject_id, session_id, checkpoint_key="after_autoreject"):
        """
        Where we store the 'checkpoint' for that subject & session.
        E.g. "data/pilot_data/sub-01/ses-001/after_autoreject-raw.fif"
        """
        return (
            self.base_dir
            / "data" / "pilot_data"
            / f"sub-{subject_id}" / f"ses-{session_id}"
            / f"{checkpoint_key}-raw.fif"
        )

    def get_ica_folder(self, subject_id, session_id):
        """
        Where we might store ICA plots or QA.
        e.g., "reports/ica/sub-01/ses-001"
        """
        return (
            self.base_dir
            / "reports" / "ica"
            / f"sub-{subject_id}" / f"ses-{session_id}"
        )

    def ensure_parent(self, path: Path):
        """Convenience method to create the parent folder if missing."""
        path.parent.mkdir(parents=True, exist_ok=True)
