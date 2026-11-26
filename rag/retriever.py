# rag/retriever.py
"""Local & Server-compatible Qdrant retrieval wrapper.

Automatically connects to Qdrant server (QDRANT_HOST:QDRANT_PORT) when available,
or falls back to embedded local disk storage (QDRANT_PATH) so RAG runs locally
with zero external container requirements.
"""

import os
import socket
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "documents")
QDRANT_PATH = os.getenv("QDRANT_PATH", str(PROJECT_ROOT / "qdrant_storage"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

_client: Optional[QdrantClient] = None
_client_mode: str = "uninitialized"
_model: Optional[SentenceTransformer] = None


def _is_port_open(host: str, port: int, timeout: float = 0.4) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def get_qdrant_client() -> QdrantClient:
    """Return a cached QdrantClient (remote server if reachable, else local disk)."""
    global _client, _client_mode
    if _client is not None:
        return _client

    if _is_port_open(QDRANT_HOST, QDRANT_PORT):
        _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)
        _client_mode = f"server ({QDRANT_HOST}:{QDRANT_PORT})"
    else:
        os.makedirs(QDRANT_PATH, exist_ok=True)
        _client = QdrantClient(path=QDRANT_PATH)
        _client_mode = f"local ({QDRANT_PATH})"
    return _client


def get_embedding_model() -> SentenceTransformer:
    """Lazily load and cache the local CPU SentenceTransformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    return _model


def get_rag_stats() -> Dict[str, Any]:
    """Return live vector store statistics for diagnostics and UI display."""
    try:
        client = get_qdrant_client()
        if client.collection_exists(COLLECTION_NAME):
            info = client.get_collection(COLLECTION_NAME)
            count = getattr(info, "points_count", None) or 0
            return {
                "status": "active",
                "mode": _client_mode,
                "collection": COLLECTION_NAME,
                "chunks_count": count,
                "embedding_model": EMBEDDING_MODEL,
            }
        return {
            "status": "empty",
            "mode": _client_mode,
            "collection": COLLECTION_NAME,
            "chunks_count": 0,
            "embedding_model": EMBEDDING_MODEL,
        }
    except Exception as exc:
        return {
            "status": f"error: {exc}",
            "mode": _client_mode,
            "collection": COLLECTION_NAME,
            "chunks_count": 0,
            "embedding_model": EMBEDDING_MODEL,
        }


def retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Return the *top_k* most relevant document chunks for *query*."""
    if not query or not query.strip():
        return []
    try:
        client = get_qdrant_client()
        if not client.collection_exists(COLLECTION_NAME):
            return []

        model = get_embedding_model()
        query_vec = model.encode([query])[0].tolist()

        hits = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            limit=top_k,
        )

        results: List[Dict[str, Any]] = []
        for hit in hits:
            payload = dict(hit.payload or {})
            payload["score"] = round(float(hit.score), 4)
            payload.setdefault("citation", payload.get("filename", payload.get("source", "Knowledge Base")))
            results.append(payload)
        return results
    except Exception as exc:
        print(f"[WARN] Qdrant retrieval warning: {exc}", flush=True)
        return []


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m rag.retriever <query>")
        sys.exit(1)
    q = " ".join(sys.argv[1:])
    print("RAG Stats:", get_rag_stats())
    for r in retrieve(q):
        print(f"[{r.get('score')}] {r.get('citation')} ({r.get('category', 'general')})")
        print(r.get("text", "")[:300])
        print("---")
