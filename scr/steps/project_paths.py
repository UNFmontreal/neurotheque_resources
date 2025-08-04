import os
from pathlib import Path
import re
import logging
from mne_bids import BIDSPath, get_bids_path_from_fname


class ProjectPaths:
    """
    Central class for all file and directory paths, ensuring BIDS compliance.
    """

    def __init__(self, config, project_root=None):
        if project_root is None:
            project_root = Path.cwd()
        else:
            project_root = Path(project_root)

        root_path_str = config["directory"]["root"]

        if root_path_str.startswith("~"):
            self.bids_root = Path(root_path_str).expanduser()
        elif os.path.isabs(root_path_str):
            self.bids_root = Path(root_path_str)
        else:
            self.bids_root = project_root / root_path_str

        self.bids_root = self.bids_root.resolve()

        # Derivatives directory is a sub-directory of the BIDS root
        self.derivatives_dir = self.bids_root / "derivatives"

        # Create base directories
        self.bids_root.mkdir(parents=True, exist_ok=True)
        self.derivatives_dir.mkdir(parents=True, exist_ok=True)

    def get_bids_path(
        self,
        subject,
        session,
        task=None,
        run=None,
        processing=None,
        suffix=None,
        extension=None,
    ):
        """
        Constructs a BIDS-compliant path using mne-bids.
        """
        bids_path = BIDSPath(
            subject=subject,
            session=session,
            task=task,
            run=run,
            processing=processing,
            suffix=suffix,
            extension=extension,
            root=self.derivatives_dir,
        )
        bids_path.directory.mkdir(parents=True, exist_ok=True)
        return bids_path

    @staticmethod
    def get_bids_entities_from_file(file_path):
        """
        Extracts BIDS entities from a filename.
        """
        return get_bids_path_from_fname(file_path, check=False)

    # --- Deprecated Methods for Backward Compatibility ---

    def _deprecation_warning(self, old_method, new_method):
        logging.warning(
            f"`{old_method}` is deprecated and will be removed in a future version. "
            f"Please use `{new_method}` instead."
        )

    def get_derivative_path(
        self, subject_id, session_id, task_id=None, run_id=None, stage=None
    ):
        self._deprecation_warning("get_derivative_path", "get_bids_path")
        return self.get_bids_path(
            subject=subject_id,
            session=session_id,
            task=task_id,
            run=run_id,
            processing=stage,
            suffix="eeg",
            extension=".fif",
        )

    def get_report_path(
        self, report_type, subject_id, session_id, task_id=None, run_id=None, name=None
    ):
        self._deprecation_warning("get_report_path", "get_bids_path")
        # Note: BIDSPath doesn't have a direct equivalent for arbitrary report names,
        # so we construct it manually but keep it in a BIDS-like structure.
        path = (
            self.derivatives_dir
            / "reports"
            / report_type
            / f"sub-{subject_id}"
            / f"ses-{session_id}"
        )
        if task_id:
            path = path / f"task-{task_id}"
        if run_id:
            path = path / f"run-{run_id}"
        path.mkdir(parents=True, exist_ok=True)
        return path / (name if name else "report.html")

    def get_checkpoint_path(
        self, subject_id, session_id, task_id=None, run_id=None, checkpoint_name=None
    ):
        self._deprecation_warning("get_checkpoint_path", "get_bids_path")
        return self.get_bids_path(
            subject=subject_id,
            session=session_id,
            task=task_id,
            run=run_id,
            processing=checkpoint_name,
            suffix="eeg",
            extension=".fif",
        )
