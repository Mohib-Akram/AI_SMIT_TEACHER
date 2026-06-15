"""
RAG Vector Store for the SMIT Teaching Assistant
=================================================

Uses ChromaDB as the vector database (persistent, local).
Embeddings are generated with scikit-learn's TF-IDF vectorizer so the whole
pipeline runs fully offline -- no API key / model download needed for embedding.

In production you could swap `LocalTfidfEmbedding` for OpenAI / Voyage / Cohere
embeddings, or sentence-transformers, by implementing the same `__call__` interface
that chromadb.EmbeddingFunction expects. Everything else (collections, querying,
persistence) stays the same.
"""

import json
import os
import pickle

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
VECTORIZER_PATH = os.path.join(DB_DIR, "tfidf_vectorizer.pkl")
SVD_PATH = os.path.join(DB_DIR, "svd.pkl")

EMBED_DIM = 64  # dimensionality after SVD reduction


class LocalTfidfEmbedding(EmbeddingFunction):
    """
    A self-contained embedding function: TF-IDF -> TruncatedSVD (dense vectors).
    The vectorizer/SVD are fit once on the knowledge base corpus and saved to disk
    so that the same transformation is used for both indexing and querying.
    """

    def __init__(self):
        self.vectorizer = None
        self.svd = None
        self._load_if_exists()

    def _load_if_exists(self):
        if os.path.exists(VECTORIZER_PATH) and os.path.exists(SVD_PATH):
            with open(VECTORIZER_PATH, "rb") as f:
                self.vectorizer = pickle.load(f)
            with open(SVD_PATH, "rb") as f:
                self.svd = pickle.load(f)

    def fit(self, corpus: list[str]):
        os.makedirs(DB_DIR, exist_ok=True)
        n_components = min(EMBED_DIM, max(2, len(corpus) - 1))
        self.vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.svd.fit(tfidf_matrix)
        with open(VECTORIZER_PATH, "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(SVD_PATH, "wb") as f:
            pickle.dump(self.svd, f)

    def __call__(self, input: Documents) -> Embeddings:
        if self.vectorizer is None or self.svd is None:
            raise RuntimeError(
                "Embedding model not fitted yet. Run build_knowledge_base() first."
            )
        tfidf_matrix = self.vectorizer.transform(input)
        dense = self.svd.transform(tfidf_matrix)
        return dense.tolist()


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


def build_knowledge_base(reset: bool = False):
    """
    Builds (or rebuilds) the ChromaDB collection 'smit_knowledge_base' from:
      - data/rubrics.json          (grading rubrics)
      - data/common_mistakes.json  (common student mistakes + explanations)
      - data/class_notes.md        (class notes, chunked by heading)

    Returns the ChromaDB collection, ready for querying.
    """
    if reset and os.path.exists(DB_DIR):
        import shutil
        shutil.rmtree(DB_DIR)

    os.makedirs(DB_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=DB_DIR)

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

    embedder = LocalTfidfEmbedding()
    embedder.fit(documents)  # fit on the whole corpus, then persist

    # Drop old collection if present so we don't get duplicate-id errors on rebuild
    try:
        client.delete_collection("smit_knowledge_base")
    except Exception:
        pass

    collection = client.create_collection(
        name="smit_knowledge_base",
        embedding_function=embedder,
    )
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    return collection


def get_collection():
    """Get the existing collection (assumes build_knowledge_base() was run before)."""
    client = chromadb.PersistentClient(path=DB_DIR)
    embedder = LocalTfidfEmbedding()
    return client.get_collection(name="smit_knowledge_base", embedding_function=embedder)


def retrieve_context(query: str, n_results: int = 4, where: dict | None = None):
    """
    Retrieve the most relevant chunks (rubric criteria, common mistakes, class notes)
    for a given query (e.g. the student's code, or a description of the assignment).
    """
    collection = get_collection()
    kwargs = {"query_texts": [query], "n_results": n_results}
    if where:
        kwargs["where"] = where
    results = collection.query(**kwargs)
    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        retrieved.append({"text": doc, "metadata": meta, "distance": dist})
    return retrieved


if __name__ == "__main__":
    print("Building knowledge base (RAG index) ...")
    build_knowledge_base(reset=True)
    print("Done. Test query:")
    for r in retrieve_context("var hoisting infinite loop", n_results=3):
        print(f"- [{r['metadata']['type']}] dist={r['distance']:.3f}")
        print("  " + r["text"].splitlines()[0])
