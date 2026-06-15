"""
SMIT AI Teaching Assistant - Main Pipeline
===========================================

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 main.py submissions/index.js

What happens:
    1. RAG STEP: retrieve relevant rubric, common-mistake explanations, and
       class notes from the ChromaDB knowledge base based on the submitted code.
    2. Code Review Agent  -> finds syntax/logic/structure/best-practice issues
    3. Tutor Agent        -> explains issues in beginner Roman Urdu/English
    4. Rubric Agent       -> scores the submission against the retrieved rubric
    5. Feedback Agent     -> compiles final report + corrected code

Output:
    - Printed summary to console
    - Full JSON report saved to reports/<filename>_report.json
    - Corrected code saved to reports/<filename>_corrected.<ext>
"""

import sys
import os
import json

from vector_store import retrieve_context, get_collection, build_knowledge_base
from agents import CodeReviewAgent, TutorAgent, AssignmentRubricAgent, FeedbackAgent

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")


def ensure_knowledge_base():
    """Build the RAG index on first run if it doesn't exist yet."""
    db_dir = os.path.join(os.path.dirname(__file__), "vector_db")
    if not os.path.exists(db_dir):
        print("[setup] No knowledge base found - building it now...")
        build_knowledge_base(reset=True)
    else:
        # Quick check the collection actually exists
        try:
            get_collection()
        except Exception:
            print("[setup] Knowledge base incomplete - rebuilding...")
            build_knowledge_base(reset=True)


def run_pipeline(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()
    filename = os.path.basename(filepath)

    print(f"\n=== SMIT AI Teaching Assistant ===")
    print(f"Submission: {filename}\n")

    # ---------- RAG STEP ----------
    print("[1/5] Retrieving relevant rubric / common mistakes / class notes (RAG)...")
    context_chunks = retrieve_context(code, n_results=6)
    for c in context_chunks:
        print(f"   - retrieved [{c['metadata']['type']}] (distance={c['distance']:.3f})")

    # ---------- AGENT 1: Code Review ----------
    print("\n[2/5] Code Review Agent analyzing code...")
    review = CodeReviewAgent().review(filename, code, context_chunks)

    # ---------- AGENT 2: Tutor ----------
    print("[3/5] Tutor Agent generating beginner-friendly explanations...")
    explanations = TutorAgent().explain(review, context_chunks)

    # ---------- AGENT 3: Rubric / Marking ----------
    print("[4/5] Assignment Rubric Agent scoring submission...")
    rubric_result = AssignmentRubricAgent().score(code, review, context_chunks)

    # ---------- AGENT 4: Feedback ----------
    print("[5/5] Feedback Agent compiling final report + corrected code...")
    feedback = FeedbackAgent().compile(code, review, explanations, rubric_result)

    # ---------- SAVE OUTPUT ----------
    os.makedirs(REPORTS_DIR, exist_ok=True)
    base, ext = os.path.splitext(filename)
    ext = ext or ".txt"

    report = {
        "filename": filename,
        "retrieved_context": [
            {"type": c["metadata"]["type"], "distance": c["distance"]} for c in context_chunks
        ],
        "code_review": review,
        "tutor_explanations": explanations,
        "rubric_score": rubric_result,
        "final_feedback": feedback,
    }
    report_path = os.path.join(REPORTS_DIR, f"{base}_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    corrected_path = os.path.join(REPORTS_DIR, f"{base}_corrected{ext}")
    corrected_code = feedback.get("corrected_code") or "// (no corrected code generated)"
    with open(corrected_path, "w", encoding="utf-8") as f:
        f.write(corrected_code)

    # ---------- PRINT SUMMARY ----------
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Score: {feedback.get('score_line', 'N/A')}")
    print(f"\n{feedback.get('summary', '')}")
    print("\nTop priority fixes:")
    for fix in feedback.get("top_priority_fixes") or []:
        print(f"  - {fix}")
    print(f"\nFull report saved to: {report_path}")
    print(f"Corrected code saved to: {corrected_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 main.py <path-to-student-submission>")
        sys.exit(1)

    ensure_knowledge_base()
    run_pipeline(sys.argv[1])
