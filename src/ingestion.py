"""
RepoViva - Milestone 1 & Production: Repository Ingestion Module

This module provides functionality to validate, download/clone, and scan GitHub repositories
(supporting both public and private repositories securely via GitHub API zipball extraction
and authenticated Git headers), extracting source code and documentation files along with rich metadata.
"""

import os
import re
import io
import shutil
import zipfile
import subprocess
import tempfile
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set, Dict, Tuple, Union


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


def parse_github_owner_repo(repo_url: str) -> Tuple[str, str]:
    """Extracts (owner, repo_name) from a GitHub repository URL."""
    if not validate_github_url(repo_url):
        raise ValueError(f"Invalid GitHub URL: '{repo_url}'")
    
    clean_url = repo_url.strip().rstrip("/")
    if clean_url.endswith(".git"):
        clean_url = clean_url[:-4]
        
    parts = clean_url.split("/")
    return parts[-2], parts[-1]


def sanitize_token_text(text: str, token: Optional[str] = None) -> str:
    """
    Sanitizes string output to ensure GitHub tokens are never exposed in error messages or logs.
    """
    if not text:
        return ""
    if token and isinstance(token, str) and token.strip():
        clean_token = token.strip()
        text = text.replace(clean_token, "***GITHUB_TOKEN***")
    # Redact any accidental inline token patterns (ghp_, gho_, github_pat_)
    text = re.sub(r"(ghp|gho|github_pat)_[A-Za-z0-9_]+", "***GITHUB_TOKEN***", text)
    return text


def validate_github_token_permissions(token: str) -> Tuple[bool, str]:
    """
    Validates a GitHub Personal Access Token against GitHub REST API (/user endpoint).
    Returns (is_valid: bool, status_message: str).
    Ensures tokens are active and have valid access without logging the raw token.
    """
    if not token or not isinstance(token, str) or not token.strip():
        return False, "Token string is empty or invalid."

    clean_token = token.strip()
    req = urllib.request.Request("https://api.github.com/user")
    req.add_header("User-Agent", "RepoViva-Ingestor")
    req.add_header("Authorization", f"Bearer {clean_token}")

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                scopes = resp.headers.get("x-oauth-scopes", "")
                scope_str = f" (Scopes: {scopes})" if scopes else ""
                return True, f"Token active and validated successfully{scope_str}."
            return True, "Token active and validated successfully."
    except urllib.error.HTTPError as err:
        if err.code in (401, 403):
            return False, "Invalid or expired GitHub token. Please verify your token has read-only Contents access."
        return False, f"GitHub API returned HTTP status {err.code} during token validation."
    except Exception as e:
        safe_msg = sanitize_token_text(str(e), clean_token)
        return False, f"Token validation failed: {safe_msg}"


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


def clone_repository(repo_url: str, target_dir: str, token: Optional[str] = None) -> None:
    """
    Clones or downloads a remote GitHub repository into target_dir.
    Supports both public and private repositories securely via authenticated GitHub API zipball
    extraction and authenticated Git headers without exposing credentials.
    """
    if not validate_github_url(repo_url):
        raise ValueError(f"Invalid GitHub URL: '{repo_url}'. Expected format: https://github.com/owner/repo")

    # Resolve token safely ensuring string type
    candidate_token = token if isinstance(token, str) and token.strip() else os.environ.get("GITHUB_TOKEN")
    active_token = candidate_token.strip() if candidate_token and isinstance(candidate_token, str) and candidate_token.strip() else None

    owner, repo = parse_github_owner_repo(repo_url)

    # 1. Attempt Secure GitHub REST API Zipball Archive Download
    zip_url = f"https://api.github.com/repos/{owner}/{repo}/zipball"
    req = urllib.request.Request(zip_url)
    req.add_header("User-Agent", "RepoViva-Ingestor")

    if active_token:
        req.add_header("Authorization", f"Bearer {active_token}")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            zip_bytes = resp.read()
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                # Zipball archive contains a top-level root directory (e.g., owner-repo-commit_sha/)
                namelist = zf.namelist()
                root_prefix = namelist[0].split("/")[0] + "/" if namelist else ""
                
                for member in zf.infolist():
                    member_path = member.filename
                    if root_prefix and member_path.startswith(root_prefix):
                        target_subpath = member_path[len(root_prefix):]
                    else:
                        target_subpath = member_path

                    if not target_subpath or member.is_dir():
                        continue

                    dest_path = Path(target_dir) / target_subpath
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with zf.open(member) as src, open(dest_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                        
                return  # Download & extraction successful!
                
    except urllib.error.HTTPError as http_err:
        if http_err.code in (401, 403, 404):
            if not active_token:
                raise RuntimeError(
                    f"Failed to access repository '{repo_url}'. "
                    f"If this is a private repository, a GitHub Personal Access Token (with read-only Contents permission) is required."
                )
            else:
                raise RuntimeError(
                    f"Failed to authenticate with GitHub for private repository '{repo_url}'. "
                    f"Please check that your GitHub Personal Access Token has permission to access this repository."
                )
    except Exception:
        pass  # Fall back to git clone method if zipball API is unavailable

    # 2. Fallback: Subprocess Git Clone with Authenticated Header
    clone_url = repo_url.strip()
    if not clone_url.endswith(".git"):
        clone_url = f"{clone_url}.git"

    cmd = ["git"]
    if active_token:
        cmd.extend(["-c", f"http.extraHeader=Authorization: Bearer {active_token}"])
    cmd.extend(["clone", "--depth", "1", clone_url, target_dir])

    try:
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        raw_err = e.stderr.strip() or e.stdout.strip() or "Unknown git error"
        safe_err = sanitize_token_text(raw_err, active_token)
        
        if any(err_term in safe_err for err_term in ["could not read Username", "Authentication failed", "Repository not found", "unable to access", "Could not resolve host"]):
            if not active_token:
                raise RuntimeError(
                    f"Failed to clone repository '{repo_url}'. "
                    f"If this is a private repository, please supply a GitHub Personal Access Token (GITHUB_TOKEN) in the Private Repo Authentication settings."
                )
            else:
                raise RuntimeError(
                    f"Failed to access private repository '{repo_url}'. "
                    f"Please check that your GitHub Personal Access Token has access permissions to this repository."
                )
        raise RuntimeError(f"Failed to clone repository '{repo_url}'. Git error: {safe_err}")
    except FileNotFoundError:
        raise RuntimeError("Git CLI is not installed or available in PATH.")


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


def ingest_repository(repo_url: str, token: Optional[Union[str, bool]] = None, cleanup: bool = True) -> IngestionResult:
    """
    High-level function: validates URL, clones/downloads repository to a temp folder, and scans files.
    
    Supports token parameter as keyword argument or positional parameter:
    - ingest_repository(repo_url)
    - ingest_repository(repo_url, token="ghp_...")
    - ingest_repository(repo_url, token=None, cleanup=True)
    - ingest_repository(repo_url, True) -> backwards-compatible call where second arg is cleanup boolean
    """
    # Handle backwards-compatible positional calls where boolean cleanup is passed as 2nd parameter
    if isinstance(token, bool):
        cleanup = token
        actual_token = None
    else:
        actual_token = token

    if not validate_github_url(repo_url):
        raise ValueError(f"Invalid GitHub URL: '{repo_url}'")
        
    temp_dir = tempfile.mkdtemp(prefix="repoviva_")
    try:
        clone_repository(repo_url, temp_dir, token=actual_token)
        res = scan_repository(temp_dir)
        res.repo_url = repo_url
        return res
    finally:
        if cleanup and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/ingestion.py <github_repo_url> [github_token]")
        sys.exit(1)
        
    target_url = sys.argv[1]
    input_token = sys.argv[2] if len(sys.argv) > 2 else None
    print(f"Ingesting repository: {target_url} ...")
    try:
        ingest_res = ingest_repository(target_url, token=input_token, cleanup=True)
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
