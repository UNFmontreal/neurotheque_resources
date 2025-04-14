# File: scr/project_paths.py

from pathlib import Path
import re
class ProjectPaths:
    """
    Central class for all file and directory paths.
    Each method here returns the *final* path for reading/writing,
    based on subject/session and your folder conventions.
    """

    def __init__(self, config):
        self.base_dir = Path(config["directory"]["root"]).resolve()
        self.raw_data_dir = self.base_dir / config["directory"]["raw_data_dir"]
        self.processed_dir = self.base_dir / config["directory"]["processed_dir"]
        self.reports_dir = self.base_dir / config["directory"]["reports_dir"]
        self.derivatives_dir = self.base_dir / config["directory"]["derivatives_dir"]
        
        # Create base directories
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.derivatives_dir.mkdir(parents=True, exist_ok=True)
        
    def validate_id(self, identifier, id_type):
        """BIDS-compliant ID validation"""
        if not re.match(rf"^{id_type}-\d+$", identifier):
            raise ValueError(f"Invalid {id_type} format: {identifier}. Expected format: {id_type}-<number>")
        return identifier
    
    def get_subject_session_path(self, subject_id, session_id):
        """Base path for subject/session"""
        sub = self.validate_id(subject_id, "sub")
        ses = self.validate_id(session_id, "ses")
        return self.raw_data_dir / sub / ses
    
    def get_raw_eeg_path(self, subject_id, session_id, task_id, run_id=None):
        """Raw EEG data path following BIDS"""
        base_path = self.get_subject_session_path(subject_id, session_id)
        filename = f"{subject_id}_{session_id}_task-{task_id}"
        if run_id:
            filename += f"_run-{run_id}"
        filename += "_eeg.fif"
        return base_path / "eeg" / filename
        
    def get_derivative_path(self, subject_id, session_id, task_id=None, run_id=None, stage=None):
        """Processed data path following BIDS derivatives"""
        sub = f'sub-{subject_id}'    
        ses = f'ses-{session_id}'
        filename = f"{sub}_{ses}"
        if task_id:
            filename += f"_task-{task_id}"
        if run_id:
            filename += f"_run-{run_id}"
        if stage:
            filename += f"_{stage}"
        filename += ".fif"
        
        path = self.processed_dir / sub / ses
        path.mkdir(parents=True, exist_ok=True)
        return path / filename
        
    def get_split_task_path(self, subject_id, session_id, task_id=None, run_id=None):
        """Processed data path following BIDS derivatives"""
        sub = f'sub-{subject_id}'    
        ses = f'ses-{session_id}'
        filename = f"{sub}_{ses}"
        if task_id:
            filename += f"_task-{task_id}"
        if run_id:
            filename += f"_run-{run_id}"
        filename += "_split.fif"
        
        path = self.processed_dir / sub / ses
        path.mkdir(parents=True, exist_ok=True)
        return path / filename
        
    def get_report_path(self, report_type, subject_id, session_id, task_id=None, run_id=None, name=None):
        """Standardized report paths"""
        sub = self.validate_id(subject_id, "sub")
        ses = self.validate_id(session_id, "ses")
        path = self.reports_dir / report_type / sub / ses
        
        if task_id:
            path = path / f"task-{task_id}"
        if run_id:
            path = path / f"run-{run_id}"
            
        path.mkdir(parents=True, exist_ok=True)
        
        if name:
            return path / name
        return path
    
    def get_auto_reject_log_path(self, subject_id, session_id, task_id=None, run_id=None):
        """
        DEPRECATED: This function is kept for backward compatibility only.
        
        AutoReject now stores bad epochs as annotations in the data instead of log files.
        This makes pipelines simpler and more reliable as annotations are saved with the data.
        
        Returns a placeholder path that maintains compatibility with existing code.
        """
        import logging
        logging.warning("get_auto_reject_log_path is deprecated - AutoReject now uses annotations instead of log files")
        
        sub = f'sub-{subject_id}'    
        ses = f'ses-{session_id}'
        
        # Create a basic path for compatibility
        path = self.processed_dir / sub / ses
        path.mkdir(parents=True, exist_ok=True)
        return path
        
    def get_checkpoint_path(self, subject_id, session_id, task_id=None, run_id=None, checkpoint_name=None):
        """Standardized checkpoint pathing"""
        sub = f'sub-{subject_id}'    
        ses = f'ses-{session_id}'
        filename = f"{sub}_{ses}"
        if task_id:
            filename += f"_task-{task_id}"
        if run_id:
            filename += f"_run-{run_id}"
        if checkpoint_name:
            filename += f"_{checkpoint_name}"
        filename += ".fif"
        
        path = self.processed_dir / sub / ses
        path.mkdir(parents=True, exist_ok=True)
        return path / filename

    def get_epoched_file_path(self, subject_id, session_id, task_id=None, run_id=None):
        """
        Get the exact path for epoched files in the format required by other scripts.
        Format: sub-{sub_id}_ses-{ses_id}_task-{task_id}_run-{run_id}_preprocessed-epoched.fif
        """
        sub = f'sub-{subject_id}'    
        ses = f'ses-{session_id}'
        
        # Create the filename with the exact format required
        filename = f"{sub}_{ses}"
        if task_id:
            filename += f"_task-{task_id}"
        if run_id:
            filename += f"_run-{run_id}"
        filename += "_preprocessed-epoched.fif"
        
        path = self.processed_dir / sub / ses
        path.mkdir(parents=True, exist_ok=True)
        return path / filename

    def get_autoreject_report_dir(self, subject_id, session_id, task_id=None, run_id=None):
        """Standardized report directory with auto-creation"""
        sub = f'sub-{subject_id}'    
        ses = f'ses-{session_id}'        
        path = self.reports_dir / "autoreject" / sub / ses
        
        if task_id:
            path = path / f"task-{task_id}"
        if run_id:
            path = path / f"run-{run_id}"
            
        path.mkdir(parents=True, exist_ok=True)
        return path    
        
    def get_ica_report_dir(self, subject_id, session_id, task_id=None, run_id=None):
        """Report directory for ICA results"""
        path = self.reports_dir / "ica" / subject_id / session_id
        
        if task_id:
            path = path / f"task-{task_id}"
        if run_id:
            path = path / f"run-{run_id}"
            
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_parent(self, path: Path):
        """Convenience method to create the parent folder if missing."""
        path.parent.mkdir(parents=True, exist_ok=True)
