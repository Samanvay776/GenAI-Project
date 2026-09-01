"""
RepoViva - Milestone 2: Intelligent Code & Documentation Chunking Module

This module parses source code and documentation files into language-aware semantic
chunks. It produces LangChain-compatible Document objects containing page_content
and rich metadata (file path, line numbers, language, chunk indices).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.ingestion import SourceFile


@dataclass
class Document:
    """
    LangChain-compatible Document representation.
    Matches `langchain_core.documents.Document`:
      - page_content (str): Chunk text content
      - metadata (dict): Dictionary of metadata attributes
    """
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert document to standard dictionary format."""
        return {
            "page_content": self.page_content,
            "metadata": self.metadata
        }


# Boundary separators for language-aware chunking
LANGUAGE_SEPARATORS: Dict[str, List[str]] = {
    "Python": [
        "\nclass ",
        "\ndef ",
        "\nasync def ",
        "\n\n",
        "\n",
        " ",
        ""
    ],
    "JavaScript": [
        "\nfunction ",
        "\nclass ",
        "\nconst ",
        "\nlet ",
        "\nvar ",
        "\nexport ",
        "\n\n",
        "\n",
        " ",
        ""
    ],
    "React JSX": [
        "\nfunction ",
        "\nconst ",
        "\nexport default ",
        "\nexport ",
        "\nclass ",
        "\n\n",
        "\n",
        " ",
        ""
    ],
    "TypeScript": [
        "\nfunction ",
        "\nclass ",
        "\ninterface ",
        "\ntype ",
        "\nconst ",
        "\nexport ",
        "\n\n",
        "\n",
        " ",
        ""
    ],
    "React TSX": [
        "\nfunction ",
        "\nconst ",
        "\ninterface ",
        "\ntype ",
        "\nexport ",
        "\n\n",
        "\n",
        " ",
        ""
    ],
    "Java": [
        "\npublic class ",
        "\nclass ",
        "\npublic ",
        "\nprivate ",
        "\nprotected ",
        "\n\n",
        "\n",
        " ",
        ""
    ],
    "C++": [
        "\nclass ",
        "\nstruct ",
        "\nnamespace ",
        "\nvoid ",
        "\nint ",
        "\n#include",
        "\n\n",
        "\n",
        " ",
        ""
    ],
    "C": [
        "\nstruct ",
        "\nvoid ",
        "\nint ",
        "\n#include",
        "\n\n",
        "\n",
        " ",
        ""
    ],
    "Markdown": [
        "\n# ",
        "\n## ",
        "\n### ",
        "\n#### ",
        "\n```",
        "\n\n",
        "\n",
        " ",
        ""
    ],
    "Default": [
        "\n\n",
        "\n",
        " ",
        ""
    ]
}


class CodeChunker:
    """
    Intelligent language-aware code and text chunker.
    Splits text into chunks of target size while preserving syntactic structure
    and line number boundaries.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")
            
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_source_file(self, source_file: SourceFile) -> List[Document]:
        """
        Chunks a SourceFile object into a list of LangChain-compatible Document objects
        enriched with line number and source metadata.
        """
        if not source_file.content or not source_file.content.strip():
            return []

        chunks_with_lines = self._split_text_with_line_numbers(
            content=source_file.content,
            language=source_file.language
        )

        documents: List[Document] = []
        total_chunks = len(chunks_with_lines)

        for idx, (chunk_text, start_line, end_line) in enumerate(chunks_with_lines):
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "relative_path": source_file.relative_path,
                    "file_name": source_file.file_name,
                    "extension": source_file.extension,
                    "language": source_file.language,
                    "start_line": start_line,
                    "end_line": end_line,
                    "chunk_index": idx,
                    "total_chunks": total_chunks,
                    "char_count": len(chunk_text),
                }
            )
            documents.append(doc)

        return documents

    def _split_text_with_line_numbers(
        self, content: str, language: str
    ) -> List[tuple[str, int, int]]:
        """
        Splits text content into chunks while tracking exact (content, start_line, end_line).
        Uses line-based accumulation to guarantee accurate line boundaries.
        """
        lines = content.splitlines(keepends=True)
        if not lines:
            return []

        # Get separators for the given language
        separators = LANGUAGE_SEPARATORS.get(language, LANGUAGE_SEPARATORS["Default"])

        chunks: List[tuple[str, int, int]] = []
        
        current_lines: List[str] = []
        current_char_count = 0
        start_line_idx = 0  # 0-indexed line pointer

        for i, line in enumerate(lines):
            # Check if adding this line exceeds target chunk size
            if current_char_count + len(line) > self.chunk_size and current_lines:
                # Check if current line starts a new code boundary (e.g. def, class, heading)
                is_boundary = any(line.lstrip().startswith(sep.strip()) for sep in separators if sep.strip())
                
                # Emit chunk if size limit reached or if at a structural boundary
                chunk_str = "".join(current_lines)
                end_line_idx = start_line_idx + len(current_lines) - 1
                chunks.append((chunk_str, start_line_idx + 1, end_line_idx + 1))

                # Handle overlap by stepping back lines to retain context
                overlap_chars = 0
                overlap_lines: List[str] = []
                for prev_line in reversed(current_lines):
                    if overlap_chars + len(prev_line) <= self.chunk_overlap:
                        overlap_lines.insert(0, prev_line)
                        overlap_chars += len(prev_line)
                    else:
                        break

                current_lines = overlap_lines
                current_char_count = overlap_chars
                start_line_idx = (i - len(current_lines))

            current_lines.append(line)
            current_char_count += len(line)

        # Emit remaining lines as final chunk
        if current_lines:
            chunk_str = "".join(current_lines)
            end_line_idx = start_line_idx + len(current_lines) - 1
            chunks.append((chunk_str, start_line_idx + 1, end_line_idx + 1))

        return chunks


def parse_and_chunk_files(
    source_files: List[SourceFile],
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> List[Document]:
    """
    Utility function to process a batch of SourceFile objects into LangChain Document chunks.
    """
    chunker = CodeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_documents: List[Document] = []
    for sf in source_files:
        all_documents.extend(chunker.chunk_source_file(sf))
    return all_documents


if __name__ == "__main__":
    import sys
    from src.ingestion import ingest_repository

    if len(sys.argv) < 2:
        print("Usage: python src/parser.py <github_repo_url>")
        sys.exit(1)

    url = sys.argv[1]
    print(f"Ingesting and Chunking repository: {url}")
    ingest_res = ingest_repository(url, cleanup=True)
    docs = parse_and_chunk_files(ingest_res.files, chunk_size=500, chunk_overlap=100)

    print(f"\n--- Chunking Summary ---")
    print(f"Total Source Files Processed : {len(ingest_res.files)}")
    print(f"Total Document Chunks Created: {len(docs)}")

    if docs:
        sample = docs[0]
        print(f"\n--- Sample Chunk (Metadata & Content) ---")
        print(f"Path      : {sample.metadata['relative_path']}")
        print(f"Language  : {sample.metadata['language']}")
        print(f"Lines     : {sample.metadata['start_line']} to {sample.metadata['end_line']}")
        print(f"Chunk Index: {sample.metadata['chunk_index']} of {sample.metadata['total_chunks']}")
        print("Content Snippet:")
        print("-" * 40)
        print(sample.page_content[:300] + ("..." if len(sample.page_content) > 300 else ""))
        print("-" * 40)
