"""
Unit tests for RepoViva Milestone 1 & Production: Repository Ingestion & Authentication
Uses Python standard library `unittest` for zero-dependency test execution.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion import (
    validate_github_url,
    parse_github_owner_repo,
    sanitize_token_text,
    detect_language,
    is_ignored_path,
    is_supported_file,
    scan_repository,
    clone_repository,
    SourceFile,
    IngestionResult
)


class TestRepositoryIngestion(unittest.TestCase):

    def test_validate_github_url_valid(self):
        """Test valid GitHub repository URLs."""
        valid_urls = [
            "https://github.com/octocat/Hello-World",
            "https://github.com/octocat/Hello-World.git",
            "https://www.github.com/pallets/flask",
            "https://github.com/a_user-name/repo.name",
        ]
        for url in valid_urls:
            with self.subTest(url=url):
                self.assertTrue(validate_github_url(url), f"Failed for valid URL: {url}")

    def test_validate_github_url_invalid(self):
        """Test invalid GitHub URLs or non-GitHub URLs."""
        invalid_urls = [
            "",
            "not-a-url",
            "http://github.com/octocat/Hello-World",  # Requires HTTPS
            "https://gitlab.com/owner/repo",  # Non-GitHub
            "https://github.com/",
            "https://github.com/owner",
            "ftp://github.com/owner/repo",
        ]
        for url in invalid_urls:
            with self.subTest(url=url):
                self.assertFalse(validate_github_url(url), f"Should fail for invalid URL: {url}")

    def test_parse_github_owner_repo(self):
        """Test owner and repository name parsing."""
        owner, repo = parse_github_owner_repo("https://github.com/octocat/Hello-World.git")
        self.assertEqual(owner, "octocat")
        self.assertEqual(repo, "Hello-World")

    def test_sanitize_token_text(self):
        """Test token redaction to prevent secret leakage in logs and exceptions."""
        raw_secret = "ghp_12345abcdefghijklmnopqrstuvwxyz6789"
        error_msg = f"Failed to authenticate with token {raw_secret} for repo"
        sanitized = sanitize_token_text(error_msg, token=raw_secret)
        self.assertNotIn(raw_secret, sanitized)
        self.assertIn("***GITHUB_TOKEN***", sanitized)

    def test_detect_language(self):
        """Test language detection based on extension."""
        self.assertEqual(detect_language(Path("main.py")), "Python")
        self.assertEqual(detect_language(Path("app.jsx")), "React JSX")
        self.assertEqual(detect_language(Path("README.md")), "Markdown")
        self.assertEqual(detect_language(Path("config.json")), "JSON")
        self.assertEqual(detect_language(Path("unknown.xyz")), "Unknown / Plain Text")

    def test_is_ignored_path(self):
        """Test path ignoring rules."""
        self.assertTrue(is_ignored_path(Path(".git/config")))
        self.assertTrue(is_ignored_path(Path("node_modules/express/index.js")))
        self.assertTrue(is_ignored_path(Path("src/__pycache__/app.cpython-39.pyc")))
        self.assertTrue(is_ignored_path(Path("package-lock.json")))
        
        self.assertFalse(is_ignored_path(Path("src/main.py")))
        self.assertFalse(is_ignored_path(Path("docs/README.md")))

    def test_is_supported_file(self):
        """Test extension support matching."""
        self.assertTrue(is_supported_file(Path("main.py")))
        self.assertTrue(is_supported_file(Path("script.ts")))
        self.assertTrue(is_supported_file(Path("Dockerfile")))
        
        self.assertFalse(is_supported_file(Path("image.png")))
        self.assertFalse(is_supported_file(Path("binary.exe")))

    def test_scan_repository_mock_directory(self):
        """Test repository scanning on a mock directory structure."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create standard files
            (tmp_path / "main.py").write_text("print('hello world')\nx = 10\n", encoding="utf-8")
            (tmp_path / "README.md").write_text("# Project Title\n\nDocs here.", encoding="utf-8")
            
            # Create ignored directory and file
            git_dir = tmp_path / ".git"
            git_dir.mkdir()
            (git_dir / "HEAD").write_text("ref: refs/heads/main", encoding="utf-8")
            
            node_dir = tmp_path / "node_modules"
            node_dir.mkdir()
            (node_dir / "package.js").write_text("console.log('ignored');", encoding="utf-8")
            
            # Create ignored extension file
            (tmp_path / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            
            # Perform scan
            res: IngestionResult = scan_repository(tmp_dir)
            
            self.assertEqual(res.total_files_scanned, 2)
            self.assertEqual(res.skipped_files_count, 1)  # logo.png skipped
            
            file_paths = {f.relative_path for f in res.files}
            self.assertIn("main.py", file_paths)
            self.assertIn("README.md", file_paths)
            self.assertNotIn(".git/HEAD", file_paths)
            self.assertNotIn("node_modules/package.js", file_paths)
            self.assertNotIn("logo.png", file_paths)

    def test_clone_repository_invalid_url_raises(self):
        """Test cloning with an invalid URL raises ValueError."""
        with self.assertRaises(ValueError):
            clone_repository("https://invalid-url.com", "/tmp/dummy")

    def test_clone_repository_private_missing_token_handling(self):
        """Test accessing non-existent/private repo without token raises clear GITHUB_TOKEN prompt error."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(RuntimeError) as ctx:
                clone_repository("https://github.com/nonexistent-user-12345/nonexistent-private-repo-99", tmp_dir, token=None)
            self.assertIn("GITHUB_TOKEN", str(ctx.exception))

    def test_clone_repository_private_invalid_token_sanitization(self):
        """Test private repo access with invalid token raises error and redacts token."""
        dummy_token = "ghp_dummysecrettoken9876543210"
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(RuntimeError) as ctx:
                clone_repository("https://github.com/nonexistent-user-12345/nonexistent-private-repo-99", tmp_dir, token=dummy_token)
            err_str = str(ctx.exception)
            self.assertNotIn(dummy_token, err_str)


if __name__ == "__main__":
    unittest.main()
