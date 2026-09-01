# RepoViva Implementation Plan

RepoViva is a code-grounded technical interview assistant that uses LangChain, local ChromaDB, Hugging Face embeddings, and Gemini LLM. It helps developers test their understanding of their codebase through a interactive mock technical interview.

## User Review Required

> [!IMPORTANT]
> - **Google Gemini API Key**: The system requires a `GEMINI_API_KEY` set in a `.env` file for LangChain to connect with Gemini.
> - **Local Virtual Environment**: We will initialize a local Python virtual environment `.venv` inside `GenAI-Project` to keep dependencies clean and isolated.
> - **No Project File Deletions**: We must avoid editing Git configs or other repo files.

## Milestones

Here is the step-by-step roadmap for RepoViva:

---

### Milestone 1: Project Setup & Repository Ingestor (Current Milestone)
* **Goal**: Establish the base directory structure, local Python virtual environment, dependencies, and write a module to clone/download a target repository and scan its files.
* **Details**:
  * Set up a python virtual environment (`.venv`) and install `python-dotenv`.
  * Implement `src/ingestion.py` which takes a GitHub repository URL, clones it to a temporary directory, and lists files while ignoring standard git/dependencies paths.
  * Output a summary of scanned files and their structures.
* **Verification**: Run `python src/ingestion.py` with a sample GitHub URL and verify it clones and parses files correctly.

---

### Milestone 2: Intelligent Chunking & Parsing
* **Goal**: Read code and document files, parse them using language-specific rules, and divide them into semantic chunks.
* **Details**:
  * Implement `src/parser.py` using LangChain's `RecursiveCharacterTextSplitter.from_language`.
  * Preserve file path, filename, and line numbers in the chunk metadata.
* **Verification**: Run parsing on Python and Markdown files and check that chunk sizes are within parameters and metadata is fully populated.

---

### Milestone 3: Hugging Face Embeddings & ChromaDB Vector Store
* **Goal**: Generate vector embeddings for the chunks and store them in a persistent local ChromaDB database.
* **Details**:
  * Integrate Hugging Face local embeddings (e.g. `sentence-transformers/all-MiniLM-L6-v2`).
  * Implement `src/vector_store.py` to create or update local ChromaDB collections and index the parsed chunks.
* **Verification**: Run a semantic similarity search script against ChromaDB and verify that relevant source files are retrieved for a technical query.

---

### Milestone 4: LangChain RAG & Gemini Integration
* **Goal**: Connect ChromaDB retrievers with Gemini LLM using LangChain expression language (LCEL).
* **Details**:
  * Set up the LangChain QA chain using `ChatGoogleGenerativeAI`.
  * Draft system prompt templates to ground Gemini's answers strictly on retrieved code context.
* **Verification**: Query the system about specific code implementation details and confirm it responds using actual codebase information.

---

### Milestone 5: Technical Interview Question Generator
* **Goal**: Generate a set of targeted technical interview questions based on the overall repository content.
* **Details**:
  * Implement a generator module that queries the codebase structure and creates 3-5 open-ended conceptual and coding questions about the repository (e.g. state management, API routes, database models).
* **Verification**: Feed a repo to the generator and verify it outputs logically sound, codebase-relevant questions.

---

### Milestone 6: Evidence-Based Evaluation & Follow-ups
* **Goal**: Evaluate user answers using retrieved code fragments as evidence and generate adaptive follow-up questions.
* **Details**:
  * Create an evaluation chain that takes a user's answer, retrieves relevant code files from ChromaDB, and uses Gemini to verify correctness, completeness, and evidence.
  * Formulate follow-up questions if the user missed details or misunderstood something.
* **Verification**: Test with correct and intentionally incorrect answers, verifying that the feedback accurately references code evidence.

---

### Milestone 7: Streamlit Interface & Polish
* **Goal**: Implement a fully responsive Streamlit dashboard.
* **Details**:
  * Build a GUI for repository URL input, ingestion status visualization, interview portal, interactive question-answer chat, and final grading report.
  * Style the page using premium dark-mode styling and glassmorphism.
* **Verification**: End-to-end interactive manual test of the entire application.

---

## Verification Plan

### Automated Verification
* Unit tests for ingestion, chunking, and database queries.

### Manual Verification
* Run command line scripts for Milestones 1-6.
* Open the Streamlit web application on local port 8501 for Milestone 7.
