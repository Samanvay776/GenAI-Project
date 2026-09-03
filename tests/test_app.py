"""
Unit tests for RepoViva Streamlit Web Application integration.
Uses Python standard library `unittest`.
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestAppIntegration(unittest.TestCase):

    def test_app_file_exists(self):
        """Verify app.py entry point exists."""
        app_path = PROJECT_ROOT / "app.py"
        self.assertTrue(app_path.exists(), "app.py entry point must exist")

    def test_requirements_file_exists(self):
        """Verify requirements.txt exists for production deployment."""
        req_path = PROJECT_ROOT / "requirements.txt"
        self.assertTrue(req_path.exists(), "requirements.txt must exist")
        content = req_path.read_text()
        self.assertIn("streamlit", content)
        self.assertIn("chromadb", content)


if __name__ == "__main__":
    unittest.main()
