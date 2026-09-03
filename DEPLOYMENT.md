# RepoViva Deployment Guide 🌐

This document explains how to run RepoViva locally and deploy it to free public cloud platforms, including configuring private GitHub repository access for multiple users securely.

---

## 1. Local Development Execution

### Prerequisites
- Python 3.9+ installed on your machine.
- Git CLI installed.

### Setup Instructions
1. Clone the GitHub repository:
   ```bash
   git clone https://github.com/Samanvay776/GenAI-Project.git
   cd GenAI-Project
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch the Streamlit Web Application:
   ```bash
   streamlit run app.py
   ```
   The application will automatically open in your browser at `http://localhost:8501`.

---

## 2. Public Cloud Deployment Options

RepoViva is designed to run seamlessly on free public cloud hosting platforms.

### Option A: Streamlit Community Cloud (Recommended)
1. Fork or push the project repository to your GitHub account (`https://github.com/Samanvay776/GenAI-Project`).
2. Sign in to [share.streamlit.io](https://share.streamlit.io) using your GitHub account.
3. Click **New App** and select:
   - **Repository**: `Samanvay776/GenAI-Project`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **Deploy**. Streamlit Cloud will install dependencies from `requirements.txt` and launch the app under a public URL (e.g. `https://repoviva.streamlit.app`).

---

## 3. Multi-User Private Repository Authentication Model

RepoViva supports both **public** and **private** GitHub repositories for multiple concurrent users safely.

### Public Repositories
Public repositories require no authentication or tokens. Simply enter any public GitHub URL in the sidebar and click **Ingest & Index Repository**.

### Private Repositories (Per-Session Security)
For multi-user safety, RepoViva uses ephemeral, per-session authentication:

1. **Per-Session User Credentials (UI Input)**:
   - Any user visiting the app can open **🔐 Private Repo Authentication (Per-Session)** in the sidebar.
   - Enter a GitHub Personal Access Token generated with **read-only Contents** scope (`Contents: Read-only`).
   - The token is stored strictly in browser memory (`st.session_state`), used during in-memory zipball download HTTP headers, and **never saved to disk, logged, or shared across sessions**.

2. **Optional Server Default (Streamlit Secrets)**:
   - App maintainers can optionally configure a default `GITHUB_TOKEN` in Streamlit Secrets (`.streamlit/secrets.toml`) for server-side testing:
     ```toml
     GITHUB_TOKEN = "ghp_your_personal_token_here"
     ```
   - User-supplied tokens in the UI always take precedence over server secrets.

---

## 4. Multi-User Security & Privacy Guarantees

- **Zero Token Storage**: User-supplied tokens are held only in RAM during the active browser session.
- **Zero Token Leakage**: Tokens are sent exclusively via encrypted HTTPS headers (`Authorization: Bearer <token>`). Raw tokens are stripped and redacted to `***GITHUB_TOKEN***` in all logs, exceptions, and UI error outputs.
- **Pre-flight Permission Validation**: Automatically checks token activity against `https://api.github.com/user` without storing secrets.
- **Zero Paid APIs**: Runs 100% free using local CPU Hugging Face embeddings (`sentence-transformers/all-MiniLM-L6-v2`) and ChromaDB vector indexing.
