"""
RepoViva - Milestone 4: LangChain RAG Pipeline Module

This module orchestrates Retrieval-Augmented Generation (RAG) by combining vector
retrieval from ChromaDB with grounding prompt templates and local Ollama LLM execution
to answer technical questions backed by repository source code evidence.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from src.parser import Document
from src.vector_store import RepoVectorStore, DEFAULT_COLLECTION_NAME

# Try importing ChatOllama from langchain_community or langchain_ollama
try:
    from langchain_community.chat_models import ChatOllama
    HAS_OLLAMA_LIB = True
except ImportError:
    try:
        from langchain_ollama import ChatOllama
        HAS_OLLAMA_LIB = True
    except ImportError:
        HAS_OLLAMA_LIB = False


# Grounded RAG Prompt Template forcing evidence-based answers
RAG_SYSTEM_PROMPT = """You are RepoViva, an expert AI developer tool conducting technical analysis of a codebase.

Strict Instructions:
1. Answer the user's technical question STRICTLY based on the provided Codebase Context below.
2. Do NOT invent functions, routes, or features that are not explicitly in the context.
3. If the context does not contain enough information, state: "I cannot find sufficient code evidence in the repository to answer this question."
4. Always cite the relevant file path and line numbers when referencing code logic.

--- Codebase Context ---
{context}

--- Question ---
{question}

--- Answer ---"""


@dataclass
class RAGResult:
    """Encapsulates the final answer from the RAG pipeline along with retrieved code evidence."""
    query: str
    answer: str
    retrieved_documents: List[Document] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Converts RAG result to dictionary."""
        return {
            "query": self.query,
            "answer": self.answer,
            "retrieved_documents": [doc.to_dict() for doc in self.retrieved_documents],
            "citations": self.citations
        }


def format_retrieved_context(docs: List[Document]) -> str:
    """
    Formats a list of Document objects into a clean, annotated context string
    for LLM prompt grounding.
    """
    if not docs:
        return "No relevant code files found in repository."

    formatted_snippets = []
    for idx, doc in enumerate(docs, 1):
        rel_path = doc.metadata.get("relative_path", "unknown_file")
        start_line = doc.metadata.get("start_line", 1)
        end_line = doc.metadata.get("end_line", 1)
        lang = doc.metadata.get("language", "Code")

        snippet_header = f"--- Snippet #{idx} | File: {rel_path} (Lines {start_line}-{end_line}) | Language: {lang} ---"
        formatted_snippets.append(f"{snippet_header}\n{doc.page_content}\n")

    return "\n".join(formatted_snippets)


def extract_citations(docs: List[Document]) -> List[Dict[str, Any]]:
    """Extracts unique citation metadata (file path, line numbers, language) from documents."""
    citations = []
    seen = set()
    for doc in docs:
        path = doc.metadata.get("relative_path", "unknown")
        start = doc.metadata.get("start_line", 1)
        end = doc.metadata.get("end_line", 1)
        key = (path, start, end)
        if key not in seen:
            seen.add(key)
            citations.append({
                "relative_path": path,
                "file_name": doc.metadata.get("file_name", ""),
                "start_line": start,
                "end_line": end,
                "language": doc.metadata.get("language", "Text")
            })
    return citations


class RepoRAGPipeline:
    """
    LangChain RAG Pipeline connecting ChromaDB vector retriever to local LLMs (Ollama).
    """

    def __init__(
        self,
        vector_store: Optional[RepoVectorStore] = None,
        model_name: str = "llama3",
        temperature: float = 0.1,
        base_url: str = "http://localhost:11434"
    ):
        self.vector_store = vector_store or RepoVectorStore()
        self.model_name = model_name
        self.temperature = temperature
        self.base_url = base_url
        self._llm = None

    @property
    def llm(self):
        """Lazy load local Ollama Chat LLM."""
        if self._llm is None and HAS_OLLAMA_LIB:
            try:
                self._llm = ChatOllama(
                    model=self.model_name,
                    temperature=self.temperature,
                    base_url=self.base_url
                )
            except Exception:
                self._llm = None
        return self._llm

    def answer_question(
        self,
        query: str,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        k: int = 4
    ) -> RAGResult:
        """
        Executes full RAG workflow:
        1. Retrieves top-k relevant code chunks from ChromaDB.
        2. Formats retrieved code with exact line numbers and file paths.
        3. Formulates grounded RAG prompt.
        4. Invokes LLM (or generates grounded fallback if Ollama server is offline).
        5. Returns structured RAGResult with citations.
        """
        if not query or not query.strip():
            return RAGResult(
                query=query,
                answer="Query cannot be empty.",
                retrieved_documents=[],
                citations=[]
            )

        # 1. Vector similarity search from ChromaDB
        docs = self.vector_store.similarity_search(query=query, k=k, collection_name=collection_name)
        citations = extract_citations(docs)

        if not docs:
            return RAGResult(
                query=query,
                answer="I cannot find sufficient code evidence in the repository to answer this question.",
                retrieved_documents=[],
                citations=[]
            )

        # 2. Format context for grounding
        context_str = format_retrieved_context(docs)
        prompt_text = RAG_SYSTEM_PROMPT.format(context=context_str, question=query)

        # 3. Invoke Local LLM (or grounded response if Ollama offline)
        answer_text = None
        if self.llm is not None:
            try:
                response = self.llm.invoke(prompt_text)
                answer_text = response.content if hasattr(response, "content") else str(response)
            except Exception:
                answer_text = None

        if not answer_text:
            # Grounded summary fallback when local Ollama server is offline
            citation_summaries = [
                f"`{c['relative_path']}` (Lines {c['start_line']}-{c['end_line']})"
                for c in citations
            ]
            answer_text = (
                f"[Grounded Code Analysis based on {len(docs)} retrieved snippet(s)]:\n\n"
                f"The question '{query}' relates to the following repository file(s):\n"
                + "\n".join(f"- {cs}" for cs in citation_summaries)
                + "\n\nKey Code Evidence:\n```\n"
                + docs[0].page_content[:300]
                + "\n```"
            )

        return RAGResult(
            query=query,
            answer=answer_text,
            retrieved_documents=docs,
            citations=citations
        )


if __name__ == "__main__":
    import sys
    from src.ingestion import ingest_repository
    from src.parser import parse_and_chunk_files

    if len(sys.argv) < 2:
        print("Usage: python src/rag_pipeline.py <github_repo_url> [question]")
        sys.exit(1)

    url = sys.argv[1]
    question = sys.argv[2] if len(sys.argv) > 2 else "What is the main purpose of this repository?"

    print(f"1. Ingesting: {url} ...")
    ingest_res = ingest_repository(url, cleanup=True)

    print(f"2. Parsing & Chunking {len(ingest_res.files)} files ...")
    docs = parse_and_chunk_files(ingest_res.files)

    print(f"3. Indexing in Vector Store ...")
    vstore = RepoVectorStore(persist_directory="./chroma_db_rag_demo")
    vstore.add_documents(docs, collection_name="rag_demo")

    print(f"4. Executing RAG Pipeline for query: '{question}' ...")
    rag = RepoRAGPipeline(vector_store=vstore)
    result = rag.answer_question(query=question, collection_name="rag_demo", k=3)

    print(f"\n==================== RAG ANSWER ====================")
    print(f"Query : {result.query}")
    print(f"\nAnswer:\n{result.answer}")
    print(f"\nCitations ({len(result.citations)} source files):")
    for c in result.citations:
        print(f"  - {c['relative_path']} (Lines {c['start_line']}-{c['end_line']}) [{c['language']}]")
    print(f"====================================================")

    # Cleanup demo collection
    vstore.delete_collection("rag_demo")
