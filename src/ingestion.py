"""
RepoViva - Milestone 1: Repository Ingestion Module

This module provides functionality to validate, clone, and scan GitHub repositories,
extracting source code and documentation files along with rich metadata while ignoring
unnecessary binaries and build artifacts.
"""

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set, Dict


# Map of common extensions to human-readable language names
LANGUAGE_MAP: Dict[str, str] = {
    ".py": "Python",
    ".js": "JavaScript",
    ".jsx": "React JSX",
    ".ts": "TypeScript",
    ".tsx": "React TSX",
    ".java": "Java",
    ".cpp": "C++",
    ".c": "C",
    ".h": "C/C++ Header",
    ".hpp": "C++ Header",
    ".cs": "C#",
    ".go": "Go",
    ".rs": "Rust",
    ".php": "PHP",
    ".rb": "Ruby",
    ".html": "HTML",
    ".css": "CSS",
    ".scss": "SCSS",
    ".sql": "SQL",
    ".sh": "Shell Script",
    ".bash": "Bash Script",
    ".md": "Markdown",
    ".txt": "Text",
    ".json": "JSON",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".toml": "TOML",
    ".xml": "XML",
    ".ini": "INI Config",
}

# Supported file extensions for source code and documentation
SUPPORTED_EXTENSIONS: Set[str] = set(LANGUAGE_MAP.keys())

# Default directory patterns and file names to ignore during scanning
IGNORED_DIRS: Set[str] = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "build",
    "dist",
    ".idea",
    ".vscode",
    "coverage",
    ".next",
    ".nuxt",
    "out",
    "vendor",
}

IGNORED_FILES: Set[str] = {
    ".ds_store",
    "thumbs.db",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "cargo.lock",
}


@dataclass
class SourceFile:
    """Represents a single ingested source code or documentation file."""
    relative_path: str
    file_name: str
    extension: str
    language: str
    line_count: int
    content: str


@dataclass
class IngestionResult:
    """Represents the complete result of a repository ingestion run."""
    repo_url: str
    repo_dir: str
    files: List[SourceFile] = field(default_factory=list)
    total_files_scanned: int = 0
    skipped_files_count: int = 0
    errors: List[str] = field(default_factory=list)


def validate_github_url(url: str) -> bool:
    """
    Validates whether the provided string is a valid HTTPS GitHub repository URL.
    
    Examples of valid URLs:
    - https://github.com/owner/repository
    - https://github.com/owner/repository.git
    """
    if not url or not isinstance(url, str):
        return False
    
    url = url.strip()
    pattern = r"^https:\/\/(www\.)?github\.com\/[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+(\.git)?\/?$"
    return bool(re.match(pattern, url))


def detect_language(file_path: Path) -> str:
    """Detects human-readable programming language or format from file extension."""
    ext = file_path.suffix.lower()
    return LANGUAGE_MAP.get(ext, "Unknown / Plain Text")


def is_ignored_path(relative_path: Path) -> bool:
    """Determines whether a file or directory path should be ignored."""
    parts = set(relative_path.parts)
    if parts.intersection(IGNORED_DIRS):
        return True
    
    if relative_path.name.lower() in IGNORED_FILES:
        return True
    
    return False


def is_supported_file(file_path: Path) -> bool:
    """Checks if a file extension is in the supported text/code extensions."""
    ext = file_path.suffix.lower()
    if ext in SUPPORTED_EXTENSIONS:
        return True
    
    # Handle special extensionless files like Dockerfile, Makefile
    if file_path.name.lower() in {"dockerfile", "makefile"}:
        return True
        
    return False


def clone_repository(repo_url: str, target_dir: str) -> None:
    """
    Clones a remote GitHub repository into target_dir using shallow clone (--depth 1).
    Raises ValueError for invalid URLs and RuntimeError if cloning fails.
    """
    if not validate_github_url(repo_url):
        raise ValueError(f"Invalid GitHub URL: '{repo_url}'. Expected format: https://github.com/owner/repo")
    
    # Normalize URL by ensuring .git suffix for git clone command
    clone_url = repo_url.strip()
    if not clone_url.endswith(".git"):
        clone_url = f"{clone_url}.git"
        
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", clone_url, target_dir],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.strip() or e.stdout.strip() or "Unknown git error"
        raise RuntimeError(f"Failed to clone repository '{repo_url}'. Git error: {err_msg}")
    except FileNotFoundError:
        raise RuntimeError("Git CLI is not installed or not available in PATH.")


def scan_repository(repo_dir: str) -> IngestionResult:
    """
    Recursively scans a local directory for source code and documentation files.
    Returns an IngestionResult containing all matched files and metadata.
    """
    base_path = Path(repo_dir).resolve()
    if not base_path.exists() or not base_path.is_dir():
        raise ValueError(f"Target directory does not exist or is not a directory: '{repo_dir}'")
        
    result = IngestionResult(repo_url="local", repo_dir=str(base_path))
    
    for root, dirs, files in os.walk(base_path):
        # Prune ignored directory names in-place so os.walk doesn't descend into them
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
        
        for file in files:
            full_path = Path(root) / file
            try:
                rel_path = full_path.relative_to(base_path)
            except ValueError:
                continue
                
            if is_ignored_path(rel_path):
                result.skipped_files_count += 1
                continue
                
            if not is_supported_file(full_path):
                result.skipped_files_count += 1
                continue
                
            result.total_files_scanned += 1
            
            # Read file content safely
            try:
                content = None
                for encoding in ["utf-8", "latin-1"]:
                    try:
                        with open(full_path, "r", encoding=encoding) as f:
                            content = f.read()
                        break
                    except UnicodeDecodeError:
                        continue
                        
                if content is None:
                    result.errors.append(f"Could not decode file '{rel_path}' (likely binary).")
                    result.skipped_files_count += 1
                    continue
                    
                lines = content.splitlines()
                source_file = SourceFile(
                    relative_path=str(rel_path),
                    file_name=full_path.name,
                    extension=full_path.suffix.lower(),
                    language=detect_language(full_path),
                    line_count=len(lines),
                    content=content
                )
                result.files.append(source_file)
                
            except Exception as e:
                result.errors.append(f"Error reading file '{rel_path}': {str(e)}")
                result.skipped_files_count += 1

    return result


def ingest_repository(repo_url: str, cleanup: bool = True) -> IngestionResult:
    """
    High-level function: validates URL, clones repository to a temp folder, and scans files.
    """
    if not validate_github_url(repo_url):
        raise ValueError(f"Invalid GitHub URL: '{repo_url}'")
        
    temp_dir = tempfile.mkdtemp(prefix="repoviva_")
    try:
        clone_repository(repo_url, temp_dir)
        res = scan_repository(temp_dir)
        res.repo_url = repo_url
        return res
    finally:
        if cleanup and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/ingestion.py <github_repo_url>")
        sys.exit(1)
        
    target_url = sys.argv[1]
    print(f"Ingesting repository: {target_url} ...")
    try:
        ingest_res = ingest_repository(target_url, cleanup=True)
        print(f"\n--- Ingestion Summary ---")
        print(f"Repository URL    : {ingest_res.repo_url}")
        print(f"Accepted Files    : {len(ingest_res.files)}")
        print(f"Skipped Files     : {ingest_res.skipped_files_count}")
        print(f"Total Lines Read  : {sum(f.line_count for f in ingest_res.files)}")
        if ingest_res.errors:
            print(f"Warnings/Errors   : {len(ingest_res.errors)}")
            for err in ingest_res.errors[:5]:
                print(f"  - {err}")
        print("\nSample Files Ingested:")
        for f in ingest_res.files[:10]:
            print(f"  - [{f.language}] {f.relative_path} ({f.line_count} lines)")
    except Exception as err:
        print(f"Ingestion failed: {err}")
        sys.exit(1)
