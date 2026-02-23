import os
import requests

OLLAMA_EMBED_URL = os.getenv("OLLAMA_EMBED_URL", "http://localhost:11434/api/embeddings")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Calls Ollama /api/embeddings for each text.
    (Batching is possible later; keep it simple for Week 2.)
    """
    out: list[list[float]] = []
    for t in texts:
        r = requests.post(OLLAMA_EMBED_URL, json={"model": EMBEDDING_MODEL, "prompt": t})
        r.raise_for_status()
        out.append(r.json()["embedding"])
    return out