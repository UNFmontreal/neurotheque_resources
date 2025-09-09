import unittest
import os
import tempfile
import shutil
import yaml
import subprocess
import sys


class TestCli(unittest.TestCase):
    """Integration tests for the real CLI entrypoint (scr.pipeline)."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def test_run_with_invalid_config(self):
        """Invalid config should fail validation and return non-zero."""
        invalid_config = {
            "directory": {
                "root": self.temp_dir,
                "raw_data_dir": "raw",
                "processed_dir": "processed",
                "reports_dir": "reports",
                # missing derivatives_dir and required top-level keys
            }
        }
        config_path = os.path.join(self.temp_dir, "invalid_config.yml")
        with open(config_path, "w") as f:
            yaml.dump(invalid_config, f)

        result = subprocess.run(
            [sys.executable, "-m", "scr.pipeline", "--config", config_path],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        # Error is logged to stdout by our logger configuration
        combined = (result.stdout or "") + (result.stderr or "")
        self.assertIn("Configuration validation failed", combined)

    def test_dry_run_smoke(self):
        """Dry-run prints plan and exits cleanly."""
        minimal_cfg = {
            "auto_save": False,
            "default_subject": "01",
            "default_session": "001",
            "default_run": "01",
            "directory": {
                "root": self.temp_dir,
                "raw_data_dir": "raw",
                "processed_dir": "processed",
                "reports_dir": "reports",
                "derivatives_dir": "derivatives",
            },
            "pipeline": {
                "steps": [
                    {"name": "SyntheticRawStep", "params": {"duration_sec": 0.1, "sfreq": 50.0}},
                    {"name": "FilterStep", "params": {"l_freq": 1.0, "h_freq": 20.0}},
                ]
            },
        }
        cfg_path = os.path.join(self.temp_dir, "dryrun.yml")
        with open(cfg_path, "w") as f:
            yaml.dump(minimal_cfg, f)

        result = subprocess.run(
            [sys.executable, "-m", "scr.pipeline", "--dry-run", "--config", cfg_path],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        out = (result.stdout or "") + (result.stderr or "")
        self.assertIn("Pipeline Plan", out)

if __name__ == '__main__':
    unittest.main()
