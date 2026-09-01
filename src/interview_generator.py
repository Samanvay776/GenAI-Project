"""
RepoViva - Milestone 5: Technical Interview Question Generation Engine

This module analyzes ingested repository source files and vector stores to automatically
generate targeted, code-grounded technical interview questions complete with target file
citations and expected technical concepts.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from src.parser import Document
from src.vector_store import RepoVectorStore, DEFAULT_COLLECTION_NAME
from src.rag_pipeline import RepoRAGPipeline


@dataclass
class InterviewQuestion:
    """Represents a generated technical interview question grounded in the codebase."""
    question_id: int
    question_text: str
    topic: str
    target_files: List[str] = field(default_factory=list)
    expected_concepts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Converts question object to a dictionary representation."""
        return {
            "question_id": self.question_id,
            "question_text": self.question_text,
            "topic": self.topic,
            "target_files": self.target_files,
            "expected_concepts": self.expected_concepts
        }


class InterviewQuestionGenerator:
    """
    Engine that creates codebase-grounded technical interview questions.
    """

    def __init__(self, rag_pipeline: Optional[RepoRAGPipeline] = None):
        self.rag_pipeline = rag_pipeline or RepoRAGPipeline()
        self.vector_store = self.rag_pipeline.vector_store

    def generate_questions(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        num_questions: int = 4
    ) -> List[InterviewQuestion]:
        """
        Generates technical interview questions based on the indexed codebase.
        """
        if num_questions <= 0:
            return []

        # Retrieve documents from vector store to understand repository structure
        sample_docs = self.vector_store.similarity_search(
            query="main function class route architecture implementation",
            k=min(10, max(1, self.vector_store.get_collection_count(collection_name) or 1)),
            collection_name=collection_name
        )

        if not sample_docs:
            # Fallback generic question if store is empty
            return [
                InterviewQuestion(
                    question_id=1,
                    question_text="How is the overall project architecture structured, and what are its entry points?",
                    topic="Architecture & System Design",
                    target_files=[],
                    expected_concepts=["Directory Structure", "Entry Points", "Dependencies"]
                )
            ]

        # Group documents by file path and language
        files_by_path: Dict[str, List[Document]] = {}
        for doc in sample_docs:
            path = doc.metadata.get("relative_path", "unknown")
            files_by_path.setdefault(path, []).append(doc)

        questions: List[InterviewQuestion] = []
        file_paths = list(files_by_path.keys())

        # Generate grounded questions based on key files and topics
        topics_pool = [
            ("Architecture & Flow", "How does data flow through {file}, and what is its primary responsibility in the application?"),
            ("Implementation Logic", "Explain the core functions or classes defined in {file}. How are key operations handled?"),
            ("Error Handling & Edge Cases", "How does {file} handle errors, invalid inputs, or unexpected boundary conditions?"),
            ("Integration & Dependencies", "How does {file} interact with other modules or external services in this repository?")
        ]

        for idx in range(num_questions):
            target_path = file_paths[idx % len(file_paths)]
            topic, template = topics_pool[idx % len(topics_pool)]
            docs_for_file = files_by_path[target_path]
            
            q_text = template.format(file=target_path)
            
            # Extract expected concepts from page content keywords
            sample_content = docs_for_file[0].page_content.lower()
            expected = ["Function Definitions", "Data Flow"]
            if "class " in sample_content:
                expected.append("Object-Oriented Design")
            if "try" in sample_content or "except" in sample_content or "error" in sample_content:
                expected.append("Exception Handling")
            if "import" in sample_content or "require" in sample_content:
                expected.append("Module Dependencies")

            question = InterviewQuestion(
                question_id=idx + 1,
                question_text=q_text,
                topic=topic,
                target_files=[target_path],
                expected_concepts=expected
            )
            questions.append(question)

        return questions


if __name__ == "__main__":
    import sys
    from src.ingestion import ingest_repository
    from src.parser import parse_and_chunk_files

    if len(sys.argv) < 2:
        print("Usage: python src/interview_generator.py <github_repo_url>")
        sys.exit(1)

    url = sys.argv[1]
    print(f"1. Ingesting repository: {url} ...")
    ingest_res = ingest_repository(url, cleanup=True)

    print(f"2. Parsing & Chunking {len(ingest_res.files)} files ...")
    docs = parse_and_chunk_files(ingest_res.files)

    print(f"3. Indexing in Vector Store ...")
    vstore = RepoVectorStore(persist_directory="./chroma_db_interview_demo")
    vstore.add_documents(docs, collection_name="interview_demo")

    print(f"4. Generating Technical Interview Questions ...")
    generator = InterviewQuestionGenerator(rag_pipeline=RepoRAGPipeline(vector_store=vstore))
    questions = generator.generate_questions(collection_name="interview_demo", num_questions=4)

    print(f"\n==================== GENERATED TECHNICAL INTERVIEW ====================")
    for q in questions:
        print(f"\nQuestion #{q.question_id} [{q.topic}]")
        print(f"  Prompt          : {q.question_text}")
        print(f"  Target File(s)  : {', '.join(q.target_files)}")
        print(f"  Expected Concepts: {', '.join(q.expected_concepts)}")
    print(f"=======================================================================")

    # Cleanup demo collection
    vstore.delete_collection("interview_demo")
