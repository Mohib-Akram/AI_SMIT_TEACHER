"""
RAG Vector Store for the SMIT Teaching Assistant
=================================================

A small, self-contained vector store (numpy + pickle) — no external vector
database service required. This avoids a known dependency conflict where
`chromadb` pulls in `opentelemetry`/`grpc`/`protobuf` packages that crash on
newer Python versions (e.g. Python 3.14 on Streamlit Cloud).

Embeddings are generated with scikit-learn's TF-IDF + SVD, fully offline --
no API key / model download needed.

`retrieve_context()` keeps the same interface as before, so nothing else in
the app (agents.py, main.py, app.py) needs to change.

--------------------------------------------------------------------------
Swapping in a real vector DB (Qdrant / Pinecone / ChromaDB) for production:
  - Keep `LocalTfidfEmbedding` (or swap for OpenAI/Voyage embeddings)
  - Replace `SimpleVectorStore` with calls to your DB's client
  - Keep the same `retrieve_context(query, n_results)` -> list[dict] interface:
        [{"text": ..., "metadata": {...}, "distance": float}, ...]
--------------------------------------------------------------------------
"""

import json
import os
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_DIR = os.path.join(os.path.dirname(__file__), "vector_db")
STORE_PATH = os.path.join(DB_DIR, "store.pkl")

EMBED_DIM = 64  # dimensionality after SVD reduction


class LocalTfidfEmbedding:
    """TF-IDF -> TruncatedSVD (dense vectors). Fit once on the corpus."""

    def __init__(self):
        self.vectorizer = None
        self.svd = None

    def fit(self, corpus: list[str]):
        n_components = min(EMBED_DIM, max(2, len(corpus) - 1))
        self.vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.svd.fit(tfidf_matrix)

    def transform(self, texts: list[str]) -> np.ndarray:
        if self.vectorizer is None or self.svd is None:
            raise RuntimeError("Embedding model not fitted yet.")
        tfidf_matrix = self.vectorizer.transform(texts)
        return self.svd.transform(tfidf_matrix)


class SimpleVectorStore:
    """Minimal vector store: in-memory numpy array + cosine distance search."""

    def __init__(self):
        self.embedder = LocalTfidfEmbedding()
        self.documents: list = []
        self.metadatas: list = []
        self.ids: list = []
        self.embeddings = None

    def build(self, documents, metadatas, ids):
        self.documents = documents
        self.metadatas = metadatas
        self.ids = ids
        self.embedder.fit(documents)
        self.embeddings = self.embedder.transform(documents)

    def query(self, query_text: str, n_results: int = 4, where: dict | None = None):
        query_vec = self.embedder.transform([query_text])[0]

        # Optional metadata filter (e.g. {"type": "rubric"})
        candidate_idx = list(range(len(self.documents)))
        if where:
            candidate_idx = [
                i for i in candidate_idx
                if all(self.metadatas[i].get(k) == v for k, v in where.items())
            ]
        if not candidate_idx:
            return []

        embs = self.embeddings[candidate_idx]

        # Cosine distance = 1 - cosine similarity
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10)
        similarities = embs_norm @ query_norm
        distances = 1 - similarities

        order = np.argsort(distances)[:n_results]

        results = []
        for rank in order:
            i = candidate_idx[rank]
            results.append({
                "text": self.documents[i],
                "metadata": self.metadatas[i],
                "distance": float(distances[rank]),
            })
        return results

    def save(self, path: str = STORE_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str = STORE_PATH) -> "SimpleVectorStore":
        with open(path, "rb") as f:
            return pickle.load(f)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _chunk_markdown(md_text: str):
    """Split class notes markdown into chunks by '##' headings."""
    chunks = []
    current = []
    for line in md_text.splitlines():
        if line.startswith("## ") and current:
            chunks.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        chunks.append("\n".join(current).strip())
    return [c for c in chunks if c.strip()]


def build_knowledge_base(reset: bool = False) -> SimpleVectorStore:
    """
    Builds (or rebuilds) the vector store from:
      - data/rubrics.json          (grading rubrics)
      - data/common_mistakes.json  (common student mistakes + explanations)
      - data/class_notes.md        (class notes, chunked by heading)
    """
    if reset and os.path.exists(DB_DIR):
        import shutil
        shutil.rmtree(DB_DIR)

    rubrics = _load_json(os.path.join(DATA_DIR, "rubrics.json"))
    mistakes = _load_json(os.path.join(DATA_DIR, "common_mistakes.json"))
    with open(os.path.join(DATA_DIR, "class_notes.md"), "r", encoding="utf-8") as f:
        notes_chunks = _chunk_markdown(f.read())

    documents, metadatas, ids = [], [], []

    for r in rubrics:
        text = f"RUBRIC: {r['assignment']}\nTopic: {r['topic']}\n" + "\n".join(
            f"- {c['name']} ({c['marks']} marks)" for c in r["criteria"]
        )
        documents.append(text)
        metadatas.append({"type": "rubric", "source_id": r["id"], "assignment": r["assignment"]})
        ids.append(r["id"])

    for m in mistakes:
        text = (
            f"COMMON MISTAKE: {m['mistake']}\nTopic: {m['topic']}\n"
            f"English explanation: {m['explanation_en']}\n"
            f"Roman Urdu explanation: {m['explanation_roman_urdu']}"
        )
        documents.append(text)
        metadatas.append({"type": "common_mistake", "source_id": m["id"]})
        ids.append(m["id"])

    for i, chunk in enumerate(notes_chunks):
        documents.append(f"CLASS NOTES:\n{chunk}")
        metadatas.append({"type": "class_notes", "source_id": f"notes_{i}"})
        ids.append(f"notes_{i}")

    store = SimpleVectorStore()
    store.build(documents, metadatas, ids)
    store.save()
    return store


def get_collection() -> SimpleVectorStore:
    """Load the existing vector store (assumes build_knowledge_base() was run before)."""
    if not os.path.exists(STORE_PATH):
        raise FileNotFoundError("Vector store not built yet. Run build_knowledge_base().")
    return SimpleVectorStore.load()


def retrieve_context(query: str, n_results: int = 4, where: dict | None = None):
    """
    Retrieve the most relevant chunks (rubric criteria, common mistakes, class notes)
    for a given query (e.g. the student's code, or a description of the assignment).
    """
    store = get_collection()
    return store.query(query, n_results=n_results, where=where)


if __name__ == "__main__":
    print("Building knowledge base (RAG index) ...")
    build_knowledge_base(reset=True)
    print("Done. Test query:")
    for r in retrieve_context("var hoisting infinite loop", n_results=3):
        print(f"- [{r['metadata']['type']}] dist={r['distance']:.3f}")
        print("  " + r["text"].splitlines()[0])
