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
    
    def get_raw_eeg_path(self, subject_id, session_id, task):
        """Raw EEG data path following BIDS"""
        base_path = self.get_subject_session_path(subject_id, session_id)
        return base_path / "eeg" / f"{subject_id}_{session_id}_task-{task}_eeg.fif"
    def get_derivative_path(self, subject_id, session_id):
        """Processed data path following BIDS derivatives"""
        sub = f'sub{subject_id}'    
        ses = f'ses{session_id}'
        return (
            self.processed_dir
            
        )
    def get_split_task_path(self, subject_id, session_id):
        """Processed data path following BIDS derivatives"""
        sub = f'sub{subject_id}'    
        ses = f'ses{session_id}'
        return (
            self.processed_dir
        )
    def get_report_path(self, report_type, subject_id, session_id, name):
        """Standardized report paths"""
        sub = self.validate_id(subject_id, "sub")
        ses = self.validate_id(session_id, "ses")
        path = self.reports_dir / report_type / sub / ses 
        path.mkdir(parents=True, exist_ok=True)
        return path / name        
    
    def get_auto_reject_log_path(self, subject_id, session_id):
        """Standardized path for autoreject log"""
        sub = f'sub{subject_id}'    
        ses = f'ses{session_id}'
        path= (
            self.processed_dir
        )    
        return path
    def get_checkpoint_path(self, subject_id, session_id, checkpoint_name):
        """Standardized checkpoint pathing"""
        sub = f'sub{subject_id}'    
        ses = f'ses{session_id}'
        path = (
            self.processed_dir 
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_autoreject_report_dir(self, subject_id, session_id):
        """Standardized report directory with auto-creation"""
        sub = f'sub{subject_id}'    
        ses = f'ses{session_id}'        
        path = (
            self.reports_dir
            / "autoreject"
            / sub
            / ses
        )
        path.mkdir(parents=True, exist_ok=True)
        return path    
    def get_ica_report_dir(self, subject_id, session_id):
        path = (
            self.reports_dir
            / "ica"
            / subject_id
            / session_id
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_parent(self, path: Path):
        """Convenience method to create the parent folder if missing."""
        path.parent.mkdir(parents=True, exist_ok=True)
