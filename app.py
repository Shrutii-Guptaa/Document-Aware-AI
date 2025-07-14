import os
import io
import re
from typing import List, Tuple

import streamlit as st
from transformers import pipeline
import PyPDF2

# --------------------------------------------------
# 🛠️  Basic configuration & environment variables
# --------------------------------------------------
st.set_page_config(
    page_title="Document‑Aware Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --------------------------------------------------
# 🎨  Sidebar (now cleaner & collapsible)
# --------------------------------------------------

# --------------------------------------------------
# 🏷️  Main title & description
# --------------------------------------------------
st.title("📄 Document‑Aware Assistant 🔍")

st.markdown(
    """
    This assistant **reads your uploaded PDF or TXT document**, produces a *≤150‑word* summary, answers your questions with paragraph‑level justification, **generates logic‑based questions**, and evaluates your responses.
    """
)

# --------------------------------------------------
# 🚀  Load Hugging Face pipelines (cached)
# --------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_models():
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    qa = pipeline("question-answering", model="deepset/minilm-uncased-squad2")
    return summarizer, qa

summarizer, qa_pipeline = load_models()

# --------------------------------------------------
# 📜  Utility functions
# --------------------------------------------------

def extract_text_from_pdf(file: io.BytesIO) -> str:
    reader = PyPDF2.PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_text(upload) -> str:
    name = upload.name.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(upload)
    if name.endswith(".txt"):
        return upload.read().decode("utf-8", errors="ignore")
    return ""


def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.replace("\n", " ").strip() for s in sentences if s.strip()]


def chunk_text(text: str, max_tokens: int = 450) -> List[str]:
    sentences = split_into_sentences(text)
    chunks, current = [], []
    tokens = 0
    for sent in sentences:
        n = len(sent.split())
        if tokens + n > max_tokens and current:
            chunks.append(" ".join(current))
            current, tokens = [], 0
        current.append(sent)
        tokens += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def summarize_large_document(text: str, chunk_size: int = 800, max_words: int = 150) -> str:
    paragraphs = chunk_text(text, max_tokens=chunk_size)
    partial = []
    for para in paragraphs:
        try:
            summary = summarizer(para, max_length=80, min_length=20, do_sample=False)[0][
                "summary_text"
            ]
            partial.append(summary)
        except Exception:
            continue
    combined = " ".join(partial)
    try:
        return summarizer(
            combined[:4000], max_length=max_words, min_length=40, do_sample=False
        )[0]["summary_text"]
    except Exception:
        return combined[: max_words * 5]


def get_best_answer(question: str, chunks: List[str]) -> Tuple[str, int, int, float, str]:
    best = {"score": -float("inf")}
    for chunk in chunks:
        try:
            answer = qa_pipeline(question=question, context=chunk)
            if answer["score"] > best["score"] and answer["answer"].strip():
                best = {
                    "answer": answer["answer"],
                    "score": answer["score"],
                    "start": answer["start"],
                    "end": answer["end"],
                    "context": chunk,
                }
        except Exception:
            continue
    if best["score"] == -float("inf"):
        return "", 0, 0, 0.0, ""
    return best["answer"], best["start"], best["end"], best["score"], best["context"]


def highlight_answer(context: str, start: int, end: int) -> str:
    return context[:start] + " **" + context[start:end] + "** " + context[end:]


def generate_static_questions() -> List[str]:
    return [
        "What is the main topic of the document?",
        "Summarize the methodology described.",
        "What are the key findings or conclusions?",
    ]

# --------------------------------------------------
# 📂  File uploader & processing
# --------------------------------------------------
uploaded = st.file_uploader(
    "📤 Upload PDF or TXT Document",
    type=["pdf", "txt"],
    help="Maximum file size: 100 MB",
)

if not uploaded:
    st.info("⬆️  Upload a document to get started.")
    st.stop()

# Extract & display document stats
with st.spinner("Extracting text …"):
    doc_text = extract_text(uploaded)

st.success(f"✅ Loaded *{uploaded.name}* ({len(doc_text.split()):,} words)")

# Cache chunks & summary in session state to avoid recomputation
if "chunks" not in st.session_state:
    st.session_state.chunks = chunk_text(doc_text)
if "summary" not in st.session_state:
    with st.spinner("Generating summary …"):
        st.session_state.summary = summarize_large_document(doc_text)

# --------------------------------------------------
# 🗂️  Tabs for interaction modes
# --------------------------------------------------
summary_tab, qa_tab, challenge_tab = st.tabs([
    "📝 Summary",
    "💬 Ask Anything",
    "🎯 Challenge Me",
])

# ---------- Summary Tab ----------
with summary_tab:
    st.subheader("Auto‑generated Summary (≤ 150 words)")
    st.write(st.session_state.summary)

# ---------- Q&A Tab -------------
with qa_tab:
    st.subheader("Ask a Question about the Document")
    question = st.text_input("Type your question and press Enter …", key="user_question")
    if question:
        with st.spinner("Searching for answer …"):
            ans, start, end, score, ctx = get_best_answer(question, st.session_state.chunks)
        if ans:
            st.markdown(f"**Answer:** {ans}")
            st.caption(f"Confidence: {score:.3f}")
            st.markdown("----")
            st.caption("_Justification (excerpt):_")
            st.write(f"…{highlight_answer(ctx, start, end)[:300]}…")
        else:
            st.warning("Sorry, I couldn't find an answer in the document.")

# ---------- Challenge Me Tab ------
with challenge_tab:
    st.subheader("Logic‑Based Questions")
    if "logic_qs" not in st.session_state:
        st.session_state.logic_qs = generate_static_questions()
    user_answers = {}
    for i, q in enumerate(st.session_state.logic_qs, 1):
        user_answers[q] = st.text_input(f"Q{i}: {q}")

    if st.button("Submit Answers"):
        st.markdown("---")
        for i, (q, user_ans) in enumerate(user_answers.items(), 1):
            correct, start, end, score, ctx = get_best_answer(q, st.session_state.chunks)
            st.markdown(f"**Q{i} Evaluation**")
            st.write(f"*Your Answer*: {user_ans or '—'}")
            st.write(f"*Expected Answer*: {correct or 'Not found'}")
            if correct:
                st.caption(f"Confidence Score: {score:.3f}")
                st.caption("_Justification:_")
                st.write(f"…{highlight_answer(ctx, start, end)[:300]}…")
            st.markdown("---")
