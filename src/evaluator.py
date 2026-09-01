"""
RepoViva - Milestone 6: Evidence-Based Answer Evaluation & Adaptive Follow-up Generator

This module evaluates user answers to technical interview questions against ground-truth
code evidence retrieved from ChromaDB. It calculates scores, highlights strengths and
missed concepts, cites source file line numbers, and generates adaptive follow-up questions.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from src.parser import Document
from src.vector_store import RepoVectorStore, DEFAULT_COLLECTION_NAME
from src.rag_pipeline import RepoRAGPipeline, extract_citations
from src.interview_generator import InterviewQuestion


@dataclass
class AnswerEvaluation:
    """Represents the evidence-based evaluation of a candidate's answer."""
    question_id: int
    question_text: str
    user_answer: str
    score: int  # Score from 0 to 100
    is_correct: bool
    feedback: str
    strengths: List[str] = field(default_factory=list)
    missed_concepts: List[str] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    follow_up_question: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts evaluation object to dictionary format for API/UI representation."""
        return {
            "question_id": self.question_id,
            "question_text": self.question_text,
            "user_answer": self.user_answer,
            "score": self.score,
            "is_correct": self.is_correct,
            "feedback": self.feedback,
            "strengths": self.strengths,
            "missed_concepts": self.missed_concepts,
            "citations": self.citations,
            "follow_up_question": self.follow_up_question
        }


class AnswerEvaluator:
    """
    Evaluates candidate responses using retrieved source code ground-truth evidence.
    """

    def __init__(self, rag_pipeline: Optional[RepoRAGPipeline] = None):
        self.rag_pipeline = rag_pipeline or RepoRAGPipeline()
        self.vector_store = self.rag_pipeline.vector_store

    def evaluate_answer(
        self,
        question: InterviewQuestion,
        user_answer: str,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        k: int = 3
    ) -> AnswerEvaluation:
        """
        Evaluates a candidate's answer against codebase evidence and expected concepts.
        """
        if not user_answer or not user_answer.strip():
            return AnswerEvaluation(
                question_id=question.question_id,
                question_text=question.question_text,
                user_answer="",
                score=0,
                is_correct=False,
                feedback="No answer was provided.",
                strengths=[],
                missed_concepts=question.expected_concepts,
                citations=[],
                follow_up_question=f"Would you like to attempt answering '{question.question_text}'?"
            )

        # 1. Retrieve ground-truth code evidence from ChromaDB
        search_query = f"{question.question_text} {' '.join(question.target_files)}"
        evidence_docs = self.vector_store.similarity_search(
            query=search_query,
            k=k,
            collection_name=collection_name
        )
        citations = extract_citations(evidence_docs)

        # 2. Concept matching & scoring
        answer_lower = user_answer.lower()
        matched_strengths: List[str] = []
        missed_concepts: List[str] = []

        for concept in question.expected_concepts:
            # Check concept name or individual words in user answer
            words = [w.lower() for w in concept.split()]
            if any(w in answer_lower for w in words):
                matched_strengths.append(concept)
            else:
                missed_concepts.append(concept)

        # Base score calculation based on matched concepts and answer length
        total_concepts = len(question.expected_concepts) or 1
        concept_score = (len(matched_strengths) / total_concepts) * 80.0
        length_bonus = min(20.0, (len(user_answer.split()) / 20.0) * 20.0)
        
        final_score = int(min(100.0, concept_score + length_bonus))
        is_correct = final_score >= 60

        # 3. Formulate feedback text
        if is_correct:
            if final_score >= 90:
                feedback = "Excellent answer! You demonstrated a comprehensive understanding of the codebase logic."
            else:
                feedback = "Good answer! You explained the core implementation details accurately."
        else:
            feedback = "Your answer partially covers the code logic, but misses key technical details present in the codebase."

        # 4. Generate adaptive follow-up question based on score & missed concepts
        follow_up = self._generate_adaptive_follow_up(
            question=question,
            score=final_score,
            missed_concepts=missed_concepts,
            citations=citations
        )

        return AnswerEvaluation(
            question_id=question.question_id,
            question_text=question.question_text,
            user_answer=user_answer,
            score=final_score,
            is_correct=is_correct,
            feedback=feedback,
            strengths=matched_strengths,
            missed_concepts=missed_concepts,
            citations=citations,
            follow_up_question=follow_up
        )

    def _generate_adaptive_follow_up(
        self,
        question: InterviewQuestion,
        score: int,
        missed_concepts: List[str],
        citations: List[Dict[str, Any]]
    ) -> str:
        """
        Generates an adaptive follow-up question tailored to the candidate's performance.
        """
        target_file_str = citations[0]["relative_path"] if citations else (question.target_files[0] if question.target_files else "the codebase")
        lines_str = f" (Lines {citations[0]['start_line']}-{citations[0]['end_line']})" if citations else ""

        if score >= 85:
            return (
                f"Great job on {target_file_str}! As an advanced follow-up: "
                f"How would you optimize or scale this component if file sizes or query volume increased significantly?"
            )
        elif missed_concepts:
            primary_missed = missed_concepts[0]
            return (
                f"To probe deeper into {target_file_str}{lines_str}: "
                f"Can you explain how '{primary_missed}' is specifically handled in this part of the code?"
            )
        else:
            return (
                f"Following up on {target_file_str}: "
                f"Can you walk through an edge case or potential failure scenario for this implementation?"
            )


if __name__ == "__main__":
    import sys
    from src.ingestion import ingest_repository
    from src.parser import parse_and_chunk_files
    from src.interview_generator import InterviewQuestionGenerator

    if len(sys.argv) < 2:
        print("Usage: python src/evaluator.py <github_repo_url>")
        sys.exit(1)

    url = sys.argv[1]
    print(f"1. Ingesting: {url} ...")
    ingest_res = ingest_repository(url, cleanup=True)

    print(f"2. Chunking & Indexing ...")
    docs = parse_and_chunk_files(ingest_res.files)
    vstore = RepoVectorStore(persist_directory="./chroma_db_eval_demo")
    vstore.add_documents(docs, collection_name="eval_demo")

    print(f"3. Generating Question ...")
    rag = RepoRAGPipeline(vector_store=vstore)
    gen = InterviewQuestionGenerator(rag_pipeline=rag)
    questions = gen.generate_questions(collection_name="eval_demo", num_questions=1)

    if questions:
        q = questions[0]
        evaluator = AnswerEvaluator(rag_pipeline=rag)
        
        sample_answer = "The module uses standard function definitions and handles data flow across the application."
        print(f"\nEvaluating Sample Answer for Question #{q.question_id} ...")
        eval_res = evaluator.evaluate_answer(q, user_answer=sample_answer, collection_name="eval_demo")

        print(f"\n==================== EVALUATION REPORT ====================")
        print(f"Question       : {eval_res.question_text}")
        print(f"User Answer    : {eval_res.user_answer}")
        print(f"Score          : {eval_res.score}/100 (Correct: {eval_res.is_correct})")
        print(f"Feedback       : {eval_res.feedback}")
        print(f"Strengths      : {', '.join(eval_res.strengths) or 'None'}")
        print(f"Missed Concepts: {', '.join(eval_res.missed_concepts) or 'None'}")
        print(f"Follow-up      : {eval_res.follow_up_question}")
        print(f"===========================================================")

    vstore.delete_collection("eval_demo")
