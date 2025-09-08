import unittest
import os
import tempfile
import shutil
import yaml
import subprocess
import sys

class TestCli(unittest.TestCase):
    """Integration tests for the command-line interface."""

    @classmethod
    def setUpClass(cls):
        """Set up a temporary directory."""
        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        """Clean up the temporary directory."""
        shutil.rmtree(cls.temp_dir)

    def test_run_with_invalid_config(self):
        """Test that the CLI returns a non-zero exit code with a bad config."""
        # Create an invalid config file (missing 'pipeline' key)
        invalid_config = {
            "directory": {
                "root": self.temp_dir,
                "raw_data_dir": "raw",
                "processed_dir": "processed",
                "reports_dir": "reports"
            }
        }
        config_path = os.path.join(self.temp_dir, "invalid_config.yml")
        with open(config_path, 'w') as f:
            yaml.dump(invalid_config, f)

        # Run the CLI command
        result = subprocess.run(
            [sys.executable, "-m", "scr.cli", "run", config_path],
            capture_output=True,
            text=True
        )

        # Assert that it failed (non-zero exit code)
        self.assertNotEqual(result.returncode, 0)
        # Check for the expected error message in stderr
        self.assertIn("Failed to run pipeline", result.stderr)
        self.assertIn("is a required property", result.stderr)


    def test_new_config_command(self):
        """Test that the 'new-config' command runs without error."""
        # Use subprocess to run the command and pipe 'q' to quit the prompts
        process = subprocess.Popen(
            [sys.executable, "-m", "scr.cli", "new-config"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # This is a simple test to see if it starts; we'll just close it.
        try:
            # We send multiple newlines to get through the prompts with defaults
            outs, errs = process.communicate(input='\\n\\n\\n\\n\\n\\n\\n', timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            outs, errs = process.communicate()
        
        self.assertEqual(process.returncode, 0)
        self.assertIn("Welcome to the Neuroflow Configuration Wizard!", outs)

if __name__ == '__main__':
    unittest.main()
