"""
SMIT AI Teaching Assistant - Streamlit Web UI
==============================================

Run locally:
    export ANTHROPIC_API_KEY=sk-ant-...
    streamlit run app.py

Deploy for free on Streamlit Community Cloud:
    1. Push this whole folder to a GitHub repo
    2. Go to https://share.streamlit.io -> "New app" -> pick the repo -> main file = app.py
    3. In the app's "Secrets" settings, add:
           ANTHROPIC_API_KEY = "sk-ant-..."
    4. Deploy -> you get a public https://<yourapp>.streamlit.app link
"""

import os
import json
import streamlit as st

from vector_store import retrieve_context, get_collection, build_knowledge_base
from agents import CodeReviewAgent, TutorAgent, AssignmentRubricAgent, FeedbackAgent

st.set_page_config(page_title="SMIT AI Teaching Assistant", page_icon="🎓", layout="wide")


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def ensure_knowledge_base():
    db_dir = os.path.join(os.path.dirname(__file__), "vector_db")
    if not os.path.exists(db_dir):
        build_knowledge_base(reset=True)
    else:
        try:
            get_collection()
        except Exception:
            build_knowledge_base(reset=True)
    return True


def get_api_key_status():
    zen = st.secrets.get("OPENCODE_ZEN_API_KEY") or os.environ.get("OPENCODE_ZEN_API_KEY")
    anthropic_key = st.secrets.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if zen:
        return True, "OpenCode Zen key found ✅"
    if anthropic_key:
        return True, "Anthropic API key found ✅"
    return False, "No API key set. Add OPENCODE_ZEN_API_KEY or ANTHROPIC_API_KEY in secrets/env."


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🎓 SMIT AI TA")
    st.caption("AI Teaching Assistant for SMIT / Programming Classes")
    st.markdown("---")
    st.markdown(
        "**Pipeline**\n"
        "1. RAG retrieval (ChromaDB)\n"
        "2. Code Review Agent\n"
        "3. Tutor Agent (Roman Urdu/English)\n"
        "4. Rubric Agent\n"
        "5. Feedback Agent"
    )
    st.markdown("---")
    if st.button("🔄 Rebuild knowledge base"):
        with st.spinner("Rebuilding RAG index..."):
            build_knowledge_base(reset=True)
        st.success("Knowledge base rebuilt.")

    api_key_ok, api_key_status = get_api_key_status()
    # Make secrets available to agents.py (which reads from os.environ)
    for key in ("OPENCODE_ZEN_API_KEY", "ANTHROPIC_API_KEY"):
        if key in st.secrets and key not in os.environ:
            os.environ[key] = st.secrets[key]

    if api_key_ok:
        st.success(api_key_status)
    else:
        st.error(api_key_status)


ensure_knowledge_base()

st.title("SMIT AI Teaching Assistant")
st.write("Upload a student's code file (e.g. `index.js`, `app.js`) and get an instant code review, "
         "beginner-friendly explanations, a rubric-based score, and corrected code.")

tab1, tab2 = st.tabs(["📤 Grade a submission", "📚 Knowledge base"])

# ---------------------------------------------------------------------------
# TAB 1: Grade submission
# ---------------------------------------------------------------------------
with tab1:
    uploaded = st.file_uploader(
        "Upload student code file", type=["js", "py", "html", "css", "txt", "java", "ts"]
    )
    pasted_code = st.text_area("...or paste code directly", height=200,
                                placeholder="// paste student code here")

    run = st.button("🚀 Run Teaching Assistant", type="primary")

    if run:
        if not api_key_ok:
            st.error("Please set OPENCODE_ZEN_API_KEY or ANTHROPIC_API_KEY first (see sidebar).")
            st.stop()

        if uploaded is not None:
            code = uploaded.read().decode("utf-8", errors="ignore")
            filename = uploaded.name
        elif pasted_code.strip():
            code = pasted_code
            filename = "submission.js"
        else:
            st.warning("Please upload a file or paste some code.")
            st.stop()

        with st.spinner("Step 1/5 — Retrieving relevant rubric, common mistakes & class notes (RAG)..."):
            context_chunks = retrieve_context(code, n_results=6)

        with st.expander("🔎 RAG retrieval results", expanded=False):
            for c in context_chunks:
                st.markdown(f"**[{c['metadata']['type']}]** (distance={c['distance']:.3f})")
                st.code(c["text"][:400])

        with st.spinner("Step 2/5 — Code Review Agent analyzing code..."):
            review = CodeReviewAgent().review(filename, code, context_chunks)

        with st.spinner("Step 3/5 — Tutor Agent writing Roman Urdu/English explanations..."):
            explanations = TutorAgent().explain(review, context_chunks)

        with st.spinner("Step 4/5 — Rubric Agent scoring submission..."):
            rubric_result = AssignmentRubricAgent().score(code, review, context_chunks)

        with st.spinner("Step 5/5 — Feedback Agent compiling final report..."):
            feedback = FeedbackAgent().compile(code, review, explanations, rubric_result)

        st.success("Done!")

        # ---- Summary ----
        st.header("📋 Summary")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Score", feedback.get("score_line") or "N/A")
        with col2:
            st.write(feedback.get("summary") or "")

        if feedback.get("top_priority_fixes"):
            st.subheader("⚡ Top priority fixes")
            for fix in feedback["top_priority_fixes"]:
                st.markdown(f"- {fix}")

        # ---- Code Review ----
        st.header("🔍 Code Review")
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Syntax issues**")
            for i in review.get("syntax_issues") or []:
                st.markdown(f"- {i}")
            st.markdown("**Logic issues**")
            for i in review.get("logic_issues") or []:
                st.markdown(f"- {i}")
        with cols[1]:
            st.markdown("**Structure issues**")
            for i in review.get("structure_issues") or []:
                st.markdown(f"- {i}")
            st.markdown("**Best practice issues**")
            for i in review.get("best_practice_issues") or []:
                st.markdown(f"- {i}")
        if review.get("positives"):
            st.markdown("**👍 Positives**")
            for i in review["positives"]:
                st.markdown(f"- {i}")

        # ---- Tutor explanations ----
        st.header("🗣️ Tutor Explanations (Roman Urdu/English)")
        for e in explanations.get("explanations") or []:
            st.markdown(f"**{e.get('issue')}**")
            st.write(e.get("explanation_roman_urdu_english"))
            st.markdown("---")

        # ---- Rubric scoring ----
        st.header("📊 Rubric Score")
        st.write(f"Rubric used: **{rubric_result.get('rubric_used') or 'N/A'}**")
        for c in rubric_result.get("criteria_scores") or []:
            st.markdown(
                f"- **{c.get('criterion')}**: {c.get('awarded_marks')} / {c.get('max_marks')} "
                f"— {c.get('justification')}"
            )
        st.write(
            f"**Total: {rubric_result.get('total_awarded')} / {rubric_result.get('total_possible')} "
            f"— {(rubric_result.get('pass_fail') or '').upper()}**"
        )

        # ---- Corrected code ----
        st.header("✅ Corrected Code")
        st.code(feedback.get("corrected_code") or "// no corrected code generated",
                language="javascript")
        for w in feedback.get("what_changed") or []:
            st.markdown(f"- {w}")

        # ---- Download full report ----
        st.download_button(
            "⬇️ Download full JSON report",
            data=json.dumps({
                "code_review": review,
                "tutor_explanations": explanations,
                "rubric_score": rubric_result,
                "final_feedback": feedback,
            }, indent=2, ensure_ascii=False),
            file_name=f"{filename}_report.json",
            mime="application/json",
        )

# ---------------------------------------------------------------------------
# TAB 2: Knowledge base
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Rubrics")
    with open("data/rubrics.json") as f:
        st.json(json.load(f))

    st.subheader("Common mistakes")
    with open("data/common_mistakes.json") as f:
        st.json(json.load(f))

    st.subheader("Class notes")
    with open("data/class_notes.md") as f:
        st.markdown(f.read())

    st.info("Edit these files in `data/` and click 'Rebuild knowledge base' in the sidebar "
            "to update the RAG index with your real class content.")
