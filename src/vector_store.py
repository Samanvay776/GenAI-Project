"""
RepoViva - Milestone 3: Vector Database & Hugging Face Embeddings Module

This module provides persistent vector storage and semantic search capabilities
using local Hugging Face Sentence Transformers embeddings and ChromaDB, with a built-in
in-memory fallback vector store for zero-dependency test execution.
"""

import os
import math
import uuid
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Import Document from parser
from src.parser import Document

# Check for ChromaDB
try:
    import chromadb
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

# Check for Hugging Face Embeddings
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HAS_HUGGINGFACE = True
except ImportError:
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        HAS_HUGGINGFACE = True
    except ImportError:
        HAS_HUGGINGFACE = False


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DB_DIR = "./chroma_db"
DEFAULT_COLLECTION_NAME = "repoviva_codebase"


class SimpleFallbackEmbeddings:
    """
    Lightweight, deterministic feature-hash embedding generator used when
    sentence-transformers is not installed. Produces 64-dimensional normalized vectors.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _hash_token(self, token: str) -> int:
        hash_val = 5381
        for char in token:
            hash_val = ((hash_val << 5) + hash_val) + ord(char)
        return hash_val

    def embed_text(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = text.lower().split()
        if not tokens:
            return vec
        
        for token in tokens:
            idx = abs(self._hash_token(token)) % self.dim
            vec[idx] += 1.0

        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_text(text)


class EmbeddingManager:
    """
    Manages embedding generation. Uses HuggingFaceEmbeddings if installed,
    otherwise falls back to SimpleFallbackEmbeddings.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self._embeddings = None

    @property
    def embeddings(self):
        """Lazy load HuggingFaceEmbeddings model or fallback."""
        if self._embeddings is None:
            if HAS_HUGGINGFACE:
                try:
                    self._embeddings = HuggingFaceEmbeddings(
                        model_name=self.model_name,
                        model_kwargs={"device": "cpu"},
                        encode_kwargs={"normalize_embeddings": True}
                    )
                except Exception:
                    self._embeddings = SimpleFallbackEmbeddings()
            else:
                self._embeddings = SimpleFallbackEmbeddings()
        return self._embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)


class InMemoryVectorStore:
    """
    In-memory vector store implementation used when ChromaDB is not installed.
    Computes exact cosine distance matching for testing and fallback environments.
    """

    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.collections: Dict[str, List[Tuple[str, List[float], Dict[str, Any]]]] = {}

    def add_documents(self, documents: List[Document], collection_name: str) -> List[str]:
        if collection_name not in self.collections:
            self.collections[collection_name] = []

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        embeddings = self.embedding_manager.embed_documents(texts)
        ids = [str(uuid.uuid4()) for _ in documents]

        for text, meta, emb, doc_id in zip(texts, metadatas, embeddings, ids):
            self.collections[collection_name].append((text, emb, meta))

        return ids

    def similarity_search_with_score(self, query: str, k: int, collection_name: str) -> List[Tuple[Document, float]]:
        if collection_name not in self.collections or not self.collections[collection_name]:
            return []

        query_emb = self.embedding_manager.embed_query(query)
        scored_docs: List[Tuple[Document, float]] = []

        for text, emb, meta in self.collections[collection_name]:
            # Compute cosine similarity
            dot_product = sum(q * e for q, e in zip(query_emb, emb))
            norm_q = math.sqrt(sum(q * q for q in query_emb))
            norm_e = math.sqrt(sum(e * e for e in emb))
            
            sim = dot_product / (norm_q * norm_e) if (norm_q * norm_e) > 0 else 0.0
            distance = 1.0 - sim  # Cosine distance
            
            doc = Document(page_content=text, metadata=meta)
            scored_docs.append((doc, distance))

        scored_docs.sort(key=lambda x: x[1])
        return scored_docs[:k]

    def get_collection_count(self, collection_name: str) -> int:
        return len(self.collections.get(collection_name, []))

    def delete_collection(self, collection_name: str) -> None:
        self.collections.pop(collection_name, None)


class RepoVectorStore:
    """
    Modular Vector Store wrapper.
    Uses ChromaDB persistent store when available, or InMemoryVectorStore fallback.
    """

    def __init__(
        self,
        persist_directory: str = DEFAULT_DB_DIR,
        embedding_manager: Optional[EmbeddingManager] = None
    ):
        self.persist_directory = os.path.abspath(persist_directory)
        self.embedding_manager = embedding_manager or EmbeddingManager()
        
        if HAS_CHROMADB:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.backend = "chromadb"
        else:
            self.in_memory_store = InMemoryVectorStore(self.embedding_manager)
            self.backend = "in_memory"

    def get_or_create_collection(self, collection_name: str = DEFAULT_COLLECTION_NAME):
        """Retrieves or creates a ChromaDB collection."""
        if self.backend == "chromadb":
            return self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        return None

    def add_documents(
        self,
        documents: List[Document],
        collection_name: str = DEFAULT_COLLECTION_NAME
    ) -> List[str]:
        """Embeds and stores Document objects."""
        if not documents:
            return []

        if self.backend == "chromadb":
            collection = self.get_or_create_collection(collection_name)
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            ids = [str(uuid.uuid4()) for _ in documents]
            embeddings = self.embedding_manager.embed_documents(texts)

            collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            return ids
        else:
            return self.in_memory_store.add_documents(documents, collection_name)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        collection_name: str = DEFAULT_COLLECTION_NAME
    ) -> List[Document]:
        """Performs vector similarity search against vector store and returns top-k matching Documents."""
        results_with_scores = self.similarity_search_with_score(query=query, k=k, collection_name=collection_name)
        return [doc for doc, _score in results_with_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        collection_name: str = DEFAULT_COLLECTION_NAME
    ) -> List[Tuple[Document, float]]:
        """Performs vector similarity search and returns tuple of (Document, similarity_distance_score)."""
        if not query or not query.strip():
            return []

        if self.backend == "chromadb":
            collection = self.get_or_create_collection(collection_name)
            query_embedding = self.embedding_manager.embed_query(query)

            count = collection.count()
            if count == 0:
                return []

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, count)
            )

            matching_docs: List[Tuple[Document, float]] = []
            if results and results.get("documents") and results["documents"][0]:
                docs = results["documents"][0]
                metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
                distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

                for text, meta, dist in zip(docs, metas, distances):
                    doc = Document(page_content=text, metadata=meta or {})
                    matching_docs.append((doc, dist))

            return matching_docs
        else:
            return self.in_memory_store.similarity_search_with_score(query, k, collection_name)

    def get_collection_count(self, collection_name: str = DEFAULT_COLLECTION_NAME) -> int:
        """Returns the total number of document chunks indexed in a collection."""
        if self.backend == "chromadb":
            collection = self.get_or_create_collection(collection_name)
            return collection.count()
        else:
            return self.in_memory_store.get_collection_count(collection_name)

    def delete_collection(self, collection_name: str = DEFAULT_COLLECTION_NAME) -> None:
        """Deletes a collection and all stored vectors."""
        if self.backend == "chromadb":
            try:
                self.client.delete_collection(name=collection_name)
            except Exception:
                pass
        else:
            self.in_memory_store.delete_collection(collection_name)


if __name__ == "__main__":
    import sys
    from src.ingestion import ingest_repository
    from src.parser import parse_and_chunk_files

    if len(sys.argv) < 2:
        print("Usage: python src/vector_store.py <github_repo_url>")
        sys.exit(1)

    url = sys.argv[1]
    print(f"1. Ingesting repository: {url} ...")
    ingest_res = ingest_repository(url, cleanup=True)

    print(f"2. Chunking {len(ingest_res.files)} files ...")
    docs = parse_and_chunk_files(ingest_res.files)

    print(f"3. Initializing vector store and embedding chunks ...")
    vstore = RepoVectorStore(persist_directory="./chroma_db_demo")
    doc_ids = vstore.add_documents(docs, collection_name="demo_repo")

    print(f"\n--- Storage Summary ---")
    print(f"Vector Backend     : {vstore.backend}")
    print(f"Total Chunks Stored: {vstore.get_collection_count('demo_repo')}")

    # Perform a sample search query
    query = "function class definition or main entry point"
    print(f"\n--- Testing Similarity Search for: '{query}' ---")
    results = vstore.similarity_search_with_score(query=query, k=2, collection_name="demo_repo")

    for idx, (doc, score) in enumerate(results, 1):
        print(f"\nMatch #{idx} (Distance Score: {score:.4f}):")
        print(f"  Path    : {doc.metadata.get('relative_path')}")
        print(f"  Lines   : {doc.metadata.get('start_line')} - {doc.metadata.get('end_line')}")
        print(f"  Language: {doc.metadata.get('language')}")
        print(f"  Snippet : {doc.page_content[:150]}...")

    # Cleanup demo collection
    vstore.delete_collection("demo_repo")
