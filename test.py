import sys
sys.path.append('.')
import os, mne
from pathlib import Path
from scr.steps.ica import ICAStep
raw = mne.io.read_raw_fif('data/processed/sub-01/ses-001/sub-01_ses-001_task-5pt_run-01_after_autoreject.fif', preload=True)
class PathsObj: pass
paths = PathsObj()
paths.get_ica_report_dir = lambda sub_id, ses_id: Path(f'data/processed/sub-01/ses-001/ica_test')
paths.get_derivative_path = lambda subject_id, session_id, **kwargs: Path(f'data/processed/sub-{subject_id}/ses-{session_id}')
ica_step = ICAStep({'subject_id': '01', 'session_id': '001', 'task_id': '5pt', 'run_id': '01', 'interactive': False, 'paths': paths})
os.makedirs('data/processed/sub-01/ses-001/ica_test', exist_ok=True)
print('Running ICA step...')
ica_step.run(raw)
print('ICA step completed successfully!')
print('Files in ICA report directory:')
print(list(Path('data/processed/sub-01/ses-001/ica_test').glob('*')))
