# SMIT AI Teaching Assistant

A multi-agent AI grading/tutoring system with a working RAG pipeline, built for
SMIT-style programming classes (beginner JS/HTML/CSS assignments).

## Architecture

```
Student submits code (app.js / index.js)
        |
        v
 [RAG: vector_store.py]  --> ChromaDB (local, persistent)
   retrieves relevant:
     - grading rubric for this assignment type
     - common student mistakes + Roman Urdu/English explanations
     - class notes
        |
        v
 [Code Review Agent]  -> finds syntax/logic/structure/best-practice issues
        |
        v
 [Tutor Agent]        -> explains each issue in simple Roman Urdu/English
        |
        v
 [Rubric Agent]       -> scores against the retrieved rubric (marks + justification)
        |
        v
 [Feedback Agent]     -> final report + corrected code
        |
        v
 reports/<file>_report.json
 reports/<file>_corrected.js
```

## Vector Database

Uses **ChromaDB** (runs locally, persisted to `./chroma_db/`, no account/API
key needed for the DB itself). Embeddings are generated with a local
TF-IDF + SVD pipeline (scikit-learn) so the whole RAG index can be built
fully offline.

> To use Qdrant or Pinecone instead: swap `vector_store.py`'s
> `chromadb.PersistentClient` for `QdrantClient` / Pinecone's client and keep
> the same `retrieve_context()` interface — the rest of the system (agents,
> main.py) doesn't need to change.

> To use higher-quality embeddings (recommended for larger knowledge bases):
> replace `LocalTfidfEmbedding` with an embedding function that calls
> OpenAI/Voyage/Cohere embeddings, or `sentence-transformers` (requires model
> download / internet access).

## Knowledge Base (data/)

- `rubrics.json` — your class's grading rubrics per assignment type
- `common_mistakes.json` — recurring student mistakes + bilingual explanations
- `class_notes.md` — your class notes, chunked by `##` heading

Edit/extend these with your own class content, then rebuild the index:

```bash
python3 vector_store.py
```

## Setup

```bash
pip install -r requirements.txt

# Option A (recommended): OpenCode Zen - curated models for coding agents
#   Get a key: run `/connect` in the OpenCode TUI and pick "OpenCode Zen",
#   or get one at https://opencode.ai/auth
export OPENCODE_ZEN_API_KEY=...

# Option B: direct Anthropic API
export ANTHROPIC_API_KEY=sk-ant-...
```

Only one of the two is required — the agents auto-detect which is set
(Zen takes priority if both are present). Override the model with
`ZEN_MODEL` (default `opencode/claude-sonnet-4-5`) or `ANTHROPIC_MODEL`
(default `claude-sonnet-4-6`).

## Usage

```bash
# 1. Build the RAG knowledge base (run once, or after editing data/*)
python3 vector_store.py

# 2. Grade a submission
python3 main.py submissions/index.js
```

Output:
- Console summary (score, top fixes)
- `reports/index_report.json` — full structured report from all 4 agents
- `reports/index_corrected.js` — AI-corrected version of the student's code

## Agents (agents.py)

| Agent | File | Responsibility |
|---|---|---|
| Code Review Agent | `CodeReviewAgent` | Syntax, logic, naming, structure, best practices |
| Tutor Agent | `TutorAgent` | Beginner-friendly Roman Urdu/English explanations |
| Assignment Rubric Agent | `AssignmentRubricAgent` | Marks against your rubric, with justifications |
| Feedback Agent | `FeedbackAgent` | Final report + corrected code |

## Customizing for your class

1. Add your real rubrics to `data/rubrics.json` (one entry per assignment type,
   with a descriptive `topic` field — this is what RAG matches against).
2. Add recurring mistakes you see in `data/common_mistakes.json`.
3. Paste your actual class notes into `data/class_notes.md`.
4. Rebuild: `python3 vector_store.py`
5. Run `main.py` on student submissions (loop over a folder for batch grading).

## Web UI (Streamlit)

A full web UI is included in `app.py` — upload a file or paste code, and get
the review/score/feedback rendered in the browser.

### Run locally
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
streamlit run app.py
```
Opens at `http://localhost:8501`.

### Deploy for free (get a public link) — Streamlit Community Cloud
1. Push this folder to a **public or private GitHub repo**.
2. Go to https://share.streamlit.io → "New app" → select the repo/branch → set
   **Main file path** to `app.py`.
3. In the app's **Settings → Secrets**, add one of:
   ```toml
   OPENCODE_ZEN_API_KEY = "..."
   ```
   or
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
4. Click **Deploy**. You'll get a public link like
   `https://your-app-name.streamlit.app` that anyone (e.g. your students) can open.

## Running with Claude Code / OpenCode

This project is plain Python with no special framework — both Claude Code and
OpenCode can run it directly (`python3 main.py <file>`), and you can ask either
tool to extend it (e.g. add a web UI, batch-grade a whole folder of
submissions, plug in a different vector DB, or add a 5th agent for plagiarism
detection across the class's submission history).
