# rag/ingest.py
"""Document ingestion pipeline for local/server Qdrant vector store.

Scans data/ (knowledge_base, priority_reference, raw_scenarios, sample_srs, uploads, etc.)
for .txt, .md, .docx, and .pdf files, splits them with paragraph/sentence-aware chunking,
computes CPU embeddings with BAAI/bge-small-en-v1.5, and upserts them into Qdrant.
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, List

logging.getLogger("pypdf").setLevel(logging.ERROR)

from qdrant_client.http import models as rest

from rag.retriever import (
    COLLECTION_NAME,
    get_embedding_model,
    get_qdrant_client,
)


def _read_file_content(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")
    if ext == ".docx":
        try:
            import docx
            doc = docx.Document(path)
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception as exc:
            print(f"[WARN] Failed to parse DOCX {path}: {exc}")
            return ""
    if ext == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(path)
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(p for p in pages if p.strip())
        except Exception as exc:
            print(f"[WARN] Failed to parse PDF {path}: {exc}")
            return ""
    return ""


def _load_documents(root_dir: Path) -> List[Dict]:
    """Load deduplicated financial knowledge base, policy, and uploaded documents."""
    docs: List[Dict] = []
    supported_extensions = {".txt", ".md", ".docx", ".pdf"}
    seen_filenames = set()

    if not root_dir.exists():
        return docs

    ingest_all = os.getenv("INGEST_ALL_DATA", "false").lower() == "true"
    default_subdirs = ["knowledge_base", "priority_reference", "uploads", "indian_docx"]
    search_roots = [root_dir] if ingest_all else [root_dir / s for s in default_subdirs if (root_dir / s).exists()]
    if not search_roots:
        search_roots = [root_dir]

    for base in search_roots:
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in supported_extensions:
                continue
            if path.name in seen_filenames:
                continue
            seen_filenames.add(path.name)

            text = _read_file_content(path)
            clean_text = re.sub(r"\s+", " ", text).strip()
            if not clean_text:
                continue
            try:
                rel = path.relative_to(root_dir)
                category = rel.parts[0] if len(rel.parts) > 1 else "general"
            except ValueError:
                category = "general"

            docs.append(
                {
                    "text": text.strip(),
                    "metadata": {
                        "source": str(path),
                        "filename": path.name,
                        "category": category,
                    },
                }
            )
    return docs


def _chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
    """Split text into overlapping chunks preferring paragraph/sentence boundaries."""
    if not text or not text.strip():
        return []

    normalized = re.sub(r"\n{3,}", "\n\n", text.strip())
    paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]

    chunks: List[str] = []
    current = ""
    for para in paragraphs:
        if len(para) > chunk_size:
            # Flush current buffer first
            if current:
                chunks.append(current.strip())
                current = ""
            # Split oversized paragraph at sentence boundaries or fixed window
            sentences = re.split(r"(?<=[.!?])\s+", para)
            buf = ""
            for sent in sentences:
                if len(sent) > chunk_size:
                    if buf:
                        chunks.append(buf.strip())
                        buf = ""
                    start = 0
                    while start < len(sent):
                        end = min(start + chunk_size, len(sent))
                        chunks.append(sent[start:end].strip())
                        if end == len(sent):
                            break
                        start += max(1, chunk_size - overlap)
                elif len(buf) + len(sent) + 1 <= chunk_size:
                    buf = f"{buf} {sent}".strip()
                else:
                    if buf:
                        chunks.append(buf.strip())
                    tail = buf[-overlap:] if overlap > 0 and len(buf) > overlap else ""
                    buf = f"{tail} {sent}".strip()
            if buf:
                chunks.append(buf.strip())
        elif len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}".strip() if current else para
        else:
            if current:
                chunks.append(current.strip())
            tail = current[-overlap:] if overlap > 0 and len(current) > overlap else ""
            current = f"{tail}\n\n{para}".strip() if tail else para

    if current:
        chunks.append(current.strip())

    return [c for c in chunks if len(c) >= 40]


def main(data_dir: str = "data") -> int:
    client = get_qdrant_client()
    model = get_embedding_model()

    raw_docs = _load_documents(Path(data_dir))
    if not raw_docs:
        print("[INFO] No documents found to ingest.")
        return 0

    chunked_docs: List[Dict] = []
    for doc in raw_docs:
        doc_chunks = _chunk_text(doc["text"])
        for idx, chunk in enumerate(doc_chunks):
            chunked_docs.append(
                {
                    "id": len(chunked_docs),
                    "text": chunk,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index": idx,
                    },
                }
            )

    if not chunked_docs:
        print("[INFO] No non-empty text chunks generated.")
        return 0

    sample_vector = model.encode([chunked_docs[0]["text"]])[0]
    vector_size = len(sample_vector)

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
    )

    texts = [doc["text"] for doc in chunked_docs]
    vectors = model.encode(texts, batch_size=64, show_progress_bar=True).tolist()

    batch_size = 250
    for i in range(0, len(chunked_docs), batch_size):
        batch = chunked_docs[i : i + batch_size]
        batch_vecs = vectors[i : i + batch_size]
        points = [
            rest.PointStruct(
                id=item["id"],
                vector=vec,
                payload={"text": item["text"], **item["metadata"]},
            )
            for item, vec in zip(batch, batch_vecs)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)

    print(
        f'[INFO] Ingested {len(chunked_docs)} chunks from {len(raw_docs)} files into "{COLLECTION_NAME}".'
    )
    return len(chunked_docs)


def ingest_file(file_path: str, category: str = "uploads") -> int:
    """Incrementally ingest a single uploaded file into the existing Qdrant collection."""
    path = Path(file_path)
    if not path.is_file():
        return 0
    text = _read_file_content(path)
    chunks = _chunk_text(text)
    if not chunks:
        return 0

    client = get_qdrant_client()
    model = get_embedding_model()
    vectors = model.encode(chunks, batch_size=32, show_progress_bar=False).tolist()

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=rest.VectorParams(size=len(vectors[0]), distance=rest.Distance.COSINE),
        )

    existing_count = getattr(client.get_collection(COLLECTION_NAME), "points_count", 0) or 0
    points = [
        rest.PointStruct(
            id=existing_count + idx,
            vector=vec,
            payload={
                "text": chunk,
                "source": str(path),
                "filename": path.name,
                "category": category,
                "chunk_index": idx,
            },
        )
        for idx, (chunk, vec) in enumerate(zip(chunks, vectors))
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
    return len(points)


if __name__ == "__main__":
    main()
