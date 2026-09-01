# RepoViva 🚀

RepoViva is a Generative AI developer tool built with LangChain, Vector Databases, and RAG. It conducts grounded technical interviews based on actual GitHub codebases.

---

## Milestone 1: Repository Ingestion

Milestone 1 implements the repository ingestion and scanning module under `src/ingestion.py`. It accepts a public GitHub repository URL, clones the repository into a temporary directory, scans supported source code and documentation files, extracts rich metadata, and ignores unwanted build artifacts/binaries.

---

## Milestone 2: Intelligent Code & Documentation Chunking

Milestone 2 implements language-aware text and code chunking under `src/parser.py`. It parses source code and documentation files into semantic chunks while generating LangChain-compatible `Document` structures enriched with precise line numbers and file metadata.

---

## Milestone 3: Hugging Face Embeddings & ChromaDB Vector Store

Milestone 3 integrates local Hugging Face Sentence Transformers embeddings (`sentence-transformers/all-MiniLM-L6-v2`) and persistent vector database indexing using ChromaDB under `src/vector_store.py`.

---

## Milestone 4: LangChain RAG Pipeline & Ollama Integration

Milestone 4 implements the grounded Retrieval-Augmented Generation (RAG) pipeline under `src/rag_pipeline.py`. It queries ChromaDB vector retrievers, injects retrieved code chunks into a grounded system prompt template, and invokes local Ollama models (`ChatOllama`) to answer technical questions backed strictly by code evidence.

---

## Milestone 5: Technical Interview Question Generation Engine

Milestone 5 implements the technical question generation engine under `src/interview_generator.py`. It analyzes the indexed repository vector store, identifies key modules and architectural patterns, and formulates codebase-grounded interview questions paired with target files and expected technical concepts.

---

## Milestone 6: Evidence-Based Answer Evaluation & Adaptive Follow-up Generator

Milestone 6 implements the answer evaluation and adaptive follow-up generator under `src/evaluator.py`. It compares candidate responses against retrieved codebase ground-truth snippets, computes accuracy scores (0–100), details strengths and missed concepts, cites file paths and line ranges, and formulates adaptive follow-up questions.

### Features
- **Ground-Truth Code Evidence Retrieval**: Automatically fetches matching source code chunks from ChromaDB for evidence-based grading.
- **Detailed Candidate Feedback**: Highlights exact technical concepts explained (`strengths`) versus omitted (`missed_concepts`).
- **Adaptive Follow-up Generation**: Dynamically formulates the next question: probing missed concepts for partial answers, or posing advanced scaling/optimization questions for high-scoring answers.

---

### How to Run & Verify All Milestones

#### 1. Run Complete Automated Test Suite (31/31 Tests Passing)
```bash
python3 -m unittest discover -s tests
```

#### 2. Run Answer Evaluation & Adaptive Follow-up CLI
Run `src/evaluator.py` with a public GitHub repository URL:
```bash
python3 src/evaluator.py https://github.com/octocat/Hello-World
```

#### Expected Terminal Output:
```text
1. Ingesting: https://github.com/octocat/Hello-World ...
2. Chunking & Indexing ...
3. Generating Question ...

Evaluating Sample Answer for Question #1 ...

==================== EVALUATION REPORT ====================
Question       : How does README handle data flow and primary responsibilities?
User Answer    : The module uses standard text and handles data flow across the application.
Score          : 80/100 (Correct: True)
Feedback       : Good answer! You explained the core implementation details accurately.
Strengths      : Data Flow
Missed Concepts: Function Definitions
Follow-up      : To probe deeper into README (Lines 1-1): Can you explain how 'Function Definitions' is specifically handled in this part of the code?
===========================================================
```
