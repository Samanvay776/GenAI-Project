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

### Features
- **Codebase-Grounded Technical Questions**: Generates 3-5 open-ended interview questions tailored to actual repository modules.
- **Categorized Question Topics**: Categorizes questions into topics such as *Architecture & Flow*, *Implementation Logic*, *Error Handling & Edge Cases*, and *Integration & Dependencies*.
- **Target File Citations & Expected Concepts**: Every question lists target file paths and key technical concepts expected in a complete answer.

---

### How to Run & Verify All Milestones

#### 1. Run Complete Automated Test Suite (27/27 Tests Passing)
```bash
python3 -m unittest discover -s tests
```

#### 2. Run Technical Interview Question Generator CLI
Run `src/interview_generator.py` with a public GitHub repository URL:
```bash
python3 src/interview_generator.py https://github.com/octocat/Hello-World
```

#### Expected Terminal Output:
```text
1. Ingesting repository: https://github.com/octocat/Hello-World ...
2. Parsing & Chunking 1 files ...
3. Indexing in Vector Store ...
4. Generating Technical Interview Questions ...

==================== GENERATED TECHNICAL INTERVIEW ====================

Question #1 [Architecture & Flow]
  Prompt          : How does data flow through README, and what is its primary responsibility in the application?
  Target File(s)  : README
  Expected Concepts: Function Definitions, Data Flow

Question #2 [Implementation Logic]
  Prompt          : Explain the core functions or classes defined in README. How are key operations handled?
  Target File(s)  : README
  Expected Concepts: Function Definitions, Data Flow
=======================================================================
```
