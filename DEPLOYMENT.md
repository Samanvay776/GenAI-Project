# RepoViva Deployment Guide 🌐

This document explains how to run RepoViva locally and deploy it to free public cloud platforms, including configuring private GitHub repository access.

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

## 3. Configuring Private GitHub Repository Access

RepoViva supports both **public** and **private** GitHub repositories.

### Public Repositories
Public repositories require no authentication or tokens. Simply enter any public GitHub URL in the sidebar and click **Ingest & Index Repository**.

### Private Repositories
To allow RepoViva to ingest private repositories on Streamlit Community Cloud:

1. **Generate a GitHub Personal Access Token**:
   - Go to GitHub Settings -> Developer Settings -> Personal Access Tokens -> Tokens (classic) or Fine-grained tokens.
   - Generate a token with `repo` (Read access to code) scope.

2. **Configure Streamlit Community Cloud Secrets**:
   - In your Streamlit Cloud app dashboard, click **App Settings** -> **Secrets**.
   - Add your token as `GITHUB_TOKEN`:
     ```toml
     GITHUB_TOKEN = "ghp_your_github_personal_access_token_here"
     ```
   - Save the secret. RepoViva will automatically use this token to securely access private repositories.

3. **On-Demand Token Entry in UI**:
   - Users can also enter a token on-demand under the **🔐 Private Repo Authentication** expander in the sidebar without storing it permanently.

---

## 4. Architecture & Security Guarantee

- **Zero Token Leakage**: Tokens are sent via secure HTTPS headers (`Authorization: Bearer <token>`). Tokens are never embedded in Git clone URLs or logged in exception messages.
- **Zero Paid APIs**: Runs 100% free using CPU Hugging Face embeddings and local ChromaDB vector indexing.
