"""
RepoViva - Interactive Streamlit Web Application

RepoViva is a Generative AI developer tool built with LangChain, Vector Databases, and RAG.
It conducts grounded technical interviews based on actual GitHub codebases.
"""

import os
import json
import streamlit as st
from typing import List, Dict, Any, Optional

from src.ingestion import (
    ingest_repository,
    validate_github_url,
    validate_github_token_permissions,
    IngestionResult
)
from src.parser import parse_and_chunk_files, Document
from src.vector_store import RepoVectorStore
from src.rag_pipeline import RepoRAGPipeline, RAGResult
from src.interview_generator import InterviewQuestionGenerator, InterviewQuestion
from src.evaluator import AnswerEvaluator, AnswerEvaluation

# -----------------------------------------------------------------------------
# Page Configuration & UI Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="RepoViva 🚀 | Codebase AI Technical Interviewer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and glassmorphism styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #9CA3AF;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: rgba(31, 41, 55, 0.6);
        border: 1px solid rgba(75, 85, 99, 0.4);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .citation-box {
        background-color: #1E293B;
        border-left: 4px solid #3B82F6;
        padding: 10px 15px;
        border-radius: 4px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "repo_url" not in st.session_state:
    st.session_state.repo_url = ""
if "ingest_result" not in st.session_state:
    st.session_state.ingest_result = None
if "documents" not in st.session_state:
    st.session_state.documents = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "questions" not in st.session_state:
    st.session_state.questions = []
if "evaluations" not in st.session_state:
    st.session_state.evaluations = {}
if "current_question_idx" not in st.session_state:
    st.session_state.current_question_idx = 0
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def get_configured_github_token() -> Optional[str]:
    """Retrieves GITHUB_TOKEN from Streamlit Secrets or environment variables."""
    if hasattr(st, "secrets") and "GITHUB_TOKEN" in st.secrets:
        return str(st.secrets["GITHUB_TOKEN"]).strip()
    if "GITHUB_TOKEN" in os.environ:
        return os.environ["GITHUB_TOKEN"].strip()
    return None


def initialize_repository_pipeline(url: str, token: Optional[str] = None, collection_name: str = "streamlit_session"):
    """Ingests, chunks, and indexes a repository in ChromaDB."""
    with st.spinner("Step 1/3: Ingesting repository files..."):
        ingest_res = ingest_repository(url, token=token, cleanup=True)
        st.session_state.ingest_result = ingest_res

    with st.spinner(f"Step 2/3: Parsing & chunking {len(ingest_res.files)} source files..."):
        docs = parse_and_chunk_files(ingest_res.files)
        st.session_state.documents = docs

    with st.spinner("Step 3/3: Indexing embeddings in local ChromaDB vector store..."):
        db_path = f"./chroma_db_{abs(hash(url)) % 10000}"
        vstore = RepoVectorStore(persist_directory=db_path)
        vstore.add_documents(docs, collection_name=collection_name)
        st.session_state.vector_store = vstore
        st.session_state.rag_pipeline = RepoRAGPipeline(vector_store=vstore)

    st.session_state.repo_url = url
    st.session_state.questions = []
    st.session_state.evaluations = {}
    st.session_state.current_question_idx = 0


# -----------------------------------------------------------------------------
# Sidebar Navigation & Settings
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/gradient-geometry/100/code-fork.png", width=64)
    st.title("RepoViva Dashboard")
    st.markdown("---")
    
    st.subheader("📌 Target Repository")
    input_url = st.text_input(
        "GitHub Repository URL",
        value=st.session_state.repo_url or "https://github.com/octocat/Hello-World",
        placeholder="https://github.com/owner/repo"
    )

    # Multi-user per-session private token handling
    secret_token = get_configured_github_token()
    with st.expander("🔐 Private Repo Authentication (Per-Session)"):
        st.caption(
            "🔒 **Privacy Notice**: For private repositories, supply a GitHub Personal Access Token with **read-only Contents** scope (`Contents: Read-only`). "
            "Your token is processed in-memory for your browser session only and is **never saved, logged, or shared**."
        )
        user_token_input = st.text_input(
            "Per-Session GitHub Access Token",
            type="password",
            help="Fine-grained token with read-only Contents permission for private repos."
        )
        if user_token_input.strip():
            is_valid, msg = validate_github_token_permissions(user_token_input.strip())
            if is_valid:
                st.caption(f"🟢 {msg}")
            else:
                st.caption(f"🔴 {msg}")
        elif secret_token:
            st.caption("ℹ️ Optional server GITHUB_TOKEN detected in Streamlit Secrets.")

    # Precedence: User-supplied per-session token -> Server Secrets token -> None
    active_token = user_token_input.strip() if user_token_input.strip() else secret_token

    if st.button("⚡ Ingest & Index Repository", type="primary", use_container_width=True):
        if not validate_github_url(input_url):
            st.error("Please enter a valid HTTPS GitHub repository URL.")
        else:
            try:
                initialize_repository_pipeline(input_url, token=active_token)
                st.success("Repository successfully indexed!")
            except Exception as err:
                st.error(f"Ingestion failed: {err}")

    st.markdown("---")
    st.subheader("⚙️ System Status")
    if st.session_state.ingest_result:
        res: IngestionResult = st.session_state.ingest_result
        st.success(f"Connected: `{st.session_state.repo_url}`")
        st.metric("Files Ingested", len(res.files))
        st.metric("Document Chunks", len(st.session_state.documents))
        if st.session_state.vector_store:
            st.caption(f"Vector Backend: `{st.session_state.vector_store.backend}`")
    else:
        st.info("No repository loaded yet. Enter a GitHub URL above.")


# -----------------------------------------------------------------------------
# Main Application Content
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">RepoViva Developer Tool</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Generative AI Codebase Analysis & Evidence-Based Technical Interview Assistant</div>', unsafe_allow_html=True)

# Main Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Repository Overview",
    "🔍 Codebase Q&A (RAG)",
    "🎓 Technical Interview",
    "📊 Evaluation Analytics"
])


# -----------------------------------------------------------------------------
# Tab 1: Repository Overview & Ingestion Explorer
# -----------------------------------------------------------------------------
with tab1:
    st.header("Repository Summary & Scanned Source Files")
    if not st.session_state.ingest_result:
        st.warning("Please ingest a GitHub repository from the sidebar to view file details.")
    else:
        res: IngestionResult = st.session_state.ingest_result
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accepted Source Files", len(res.files))
        col2.metric("Total Lines Scanned", sum(f.line_count for f in res.files))
        col3.metric("Skipped Artifacts", res.skipped_files_count)
        col4.metric("Document Chunks", len(st.session_state.documents))

        st.markdown("---")
        st.subheader("📁 Ingested Files List")
        
        file_data = [
            {
                "Relative Path": f.relative_path,
                "Language": f.language,
                "Lines": f.line_count,
                "Extension": f.extension
            }
            for f in res.files
        ]
        st.dataframe(file_data, use_container_width=True)

        if st.checkbox("Show Sample Chunk Metadata"):
            if st.session_state.documents:
                sample_doc = st.session_state.documents[0]
                st.json(sample_doc.to_dict())


# -----------------------------------------------------------------------------
# Tab 2: Codebase Q&A (RAG Pipeline)
# -----------------------------------------------------------------------------
with tab2:
    st.header("Codebase Grounded Technical Q&A")
    st.caption("Ask any question about the architecture, implementation, or data flow of the repository.")

    if not st.session_state.rag_pipeline:
        st.warning("Please ingest a GitHub repository first from the sidebar.")
    else:
        query_input = st.text_input("Enter your technical question:", placeholder="e.g. How does error handling or user authentication work?")
        top_k = st.slider("Number of Context Chunks (Top-K):", min_value=1, max_value=6, value=3)

        if st.button("Ask RepoViva", type="primary"):
            if not query_input.strip():
                st.error("Please enter a valid question.")
            else:
                with st.spinner("Retrieving ground-truth code snippets and generating answer..."):
                    rag_res: RAGResult = st.session_state.rag_pipeline.answer_question(
                        query=query_input,
                        collection_name="streamlit_session",
                        k=top_k
                    )
                    st.session_state.qa_history.append(rag_res)

        if st.session_state.qa_history:
            st.markdown("---")
            latest_qa: RAGResult = st.session_state.qa_history[-1]
            
            st.subheader("💡 Answer")
            st.markdown(latest_qa.answer)

            st.subheader("📌 Code Citations")
            if latest_qa.citations:
                for cite in latest_qa.citations:
                    st.markdown(
                        f"<div class='citation-box'>"
                        f"📄 <b>{cite['relative_path']}</b> (Lines {cite['start_line']}-{cite['end_line']}) "
                        f"| Language: <code>{cite['language']}</code>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.caption("No specific file citations generated.")

            with st.expander("Show Retrieved Source Code Chunks"):
                for doc in latest_qa.retrieved_documents:
                    st.markdown(f"**File:** `{doc.metadata.get('relative_path')}` (Lines {doc.metadata.get('start_line')}-{doc.metadata.get('end_line')})")
                    st.code(doc.page_content, language=doc.metadata.get('language', 'text').lower())


# -----------------------------------------------------------------------------
# Tab 3: Interactive Technical Interview Portal
# -----------------------------------------------------------------------------
with tab3:
    st.header("Interactive Technical Interview")
    st.caption("RepoViva will generate code-grounded questions and evaluate your answers against actual source files.")

    if not st.session_state.rag_pipeline:
        st.warning("Please ingest a GitHub repository from the sidebar to unlock the interview portal.")
    else:
        col_gen1, col_gen2 = st.columns([3, 1])
        num_q = col_gen1.number_input("Number of Interview Questions to Generate:", min_value=1, max_value=8, value=3)
        
        if col_gen2.button("🎲 Generate Interview", type="primary", use_container_width=True):
            with st.spinner("Analyzing codebase structure and formulating interview questions..."):
                gen = InterviewQuestionGenerator(rag_pipeline=st.session_state.rag_pipeline)
                st.session_state.questions = gen.generate_questions(
                    collection_name="streamlit_session",
                    num_questions=num_q
                )
                st.session_state.evaluations = {}
                st.session_state.current_question_idx = 0
                st.success(f"Generated {len(st.session_state.questions)} interview questions!")

        if st.session_state.questions:
            q_idx = st.session_state.current_question_idx
            q: InterviewQuestion = st.session_state.questions[q_idx]

            st.markdown("---")
            st.subheader(f"Question {q.question_id} of {len(st.session_state.questions)}")
            
            c1, c2 = st.columns([3, 1])
            c1.info(f"**Topic**: {q.topic}\n\n**Question**: {q.question_text}")
            c2.markdown(f"**Target Files**:\n" + "\n".join(f"- `{f}`" for f in q.target_files))
            c2.markdown(f"**Expected Concepts**:\n" + "\n".join(f"- {c}" for c in q.expected_concepts))

            # Candidate answer input
            user_ans_key = f"ans_input_{q.question_id}"
            user_ans = st.text_area(
                "Your Answer:",
                height=150,
                key=user_ans_key,
                placeholder="Explain the implementation details, functions, or architectural logic..."
            )

            if st.button("Submit Answer for Evaluation", type="secondary"):
                if not user_ans.strip():
                    st.error("Please enter an answer before submitting.")
                else:
                    with st.spinner("Evaluating answer against ground-truth source code..."):
                        evaluator = AnswerEvaluator(rag_pipeline=st.session_state.rag_pipeline)
                        eval_res = evaluator.evaluate_answer(
                            question=q,
                            user_answer=user_ans,
                            collection_name="streamlit_session"
                        )
                        st.session_state.evaluations[q.question_id] = eval_res

            # Display evaluation report if available
            if q.question_id in st.session_state.evaluations:
                ev: AnswerEvaluation = st.session_state.evaluations[q.question_id]
                st.markdown("### 📝 Evaluation Report")
                
                score_col, status_col = st.columns(2)
                score_col.metric("Candidate Score", f"{ev.score} / 100")
                if ev.is_correct:
                    status_col.success("Status: PASSED / CORRECT")
                else:
                    status_col.error("Status: NEEDS IMPROVEMENT")

                st.markdown(f"**Feedback**: {ev.feedback}")

                if ev.strengths:
                    st.markdown("**Demonstrated Strengths**:")
                    for s in ev.strengths:
                        st.markdown(f"- ✅ {s}")

                if ev.missed_concepts:
                    st.markdown("**Missed Concepts**:")
                    for m in ev.missed_concepts:
                        st.markdown(f"- ❌ {m}")

                if ev.follow_up_question:
                    st.warning(f"**Adaptive Follow-up Question**:\n\n{ev.follow_up_question}")

            # Navigation buttons
            st.markdown("---")
            nav_col1, nav_col2, _ = st.columns([1, 1, 4])
            if q_idx > 0:
                if nav_col1.button("⬅️ Previous Question"):
                    st.session_state.current_question_idx -= 1
                    st.rerun()
            if q_idx < len(st.session_state.questions) - 1:
                if nav_col2.button("Next Question ➡️"):
                    st.session_state.current_question_idx += 1
                    st.rerun()


# -----------------------------------------------------------------------------
# Tab 4: Evaluation Analytics & Report Summary
# -----------------------------------------------------------------------------
with tab4:
    st.header("Interview Performance & Evaluation Summary")

    if not st.session_state.evaluations:
        st.info("No completed question evaluations yet. Complete questions in the 'Technical Interview' tab.")
    else:
        evals = list(st.session_state.evaluations.values())
        avg_score = sum(e.score for e in evals) / len(evals)
        passed_count = sum(1 for e in evals if e.is_correct)

        col_a1, col_a2, col_a3 = st.columns(3)
        col_a1.metric("Questions Evaluated", len(evals))
        col_a2.metric("Average Score", f"{avg_score:.1f} / 100")
        col_a3.metric("Passed Questions", f"{passed_count} / {len(evals)}")

        st.markdown("---")
        st.subheader("Detailed Question Score Breakdown")
        
        summary_table = [
            {
                "QID": e.question_id,
                "Question": e.question_text[:60] + "...",
                "Score": e.score,
                "Passed": "YES" if e.is_correct else "NO",
                "Strengths": len(e.strengths),
                "Missed": len(e.missed_concepts)
            }
            for e in evals
        ]
        st.dataframe(summary_table, use_container_width=True)

        # Downloadable JSON report
        report_dict = {
            "repository": st.session_state.repo_url,
            "average_score": avg_score,
            "passed_count": passed_count,
            "evaluations": [e.to_dict() for e in evals]
        }
        st.download_button(
            label="📥 Download Evaluation Report (JSON)",
            data=json.dumps(report_dict, indent=2),
            file_name="repoviva_interview_report.json",
            mime="application/json"
        )
