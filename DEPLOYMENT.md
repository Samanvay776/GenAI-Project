# RepoViva Deployment Guide 🌐

This document explains how to run RepoViva locally and deploy it to free public cloud platforms.

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

### Option B: Hugging Face Spaces (Gradio / Streamlit Space)
1. Create a free account on [Hugging Face](https://huggingface.co).
2. Create a new **Space**, choose **Streamlit** as the Space SDK.
3. Push `app.py`, `src/`, `requirements.txt` to the Space repository.
4. Hugging Face Spaces will build and host your app with a public URL.

---

## 3. Architecture & LLM Inference Considerations

- **Local Development**: Connects to local Ollama (`ChatOllama`) or uses the built-in grounded code analysis engine when Ollama is offline.
- **Cloud Deployment**: Cloud platforms (Streamlit Cloud / Hugging Face Spaces) run Python and CPU sentence-transformers natively. RepoViva's built-in grounded code analysis engine handles Q&A, question generation, and answer evaluation 100% free without requiring external paid API keys.
