import os
import uuid
from dataclasses import dataclass
from typing import Literal

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

COLLECTION = "hr_documents"
VECTOR_DIM = 768


@dataclass
class RetrievedChunk:
    text: str
    page: int
    chunk_type: Literal["text", "table"]
    source_file: str
    score: float


def get_client() -> QdrantClient:
    return QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
    )


def ensure_collection(client: QdrantClient) -> None:
    names = [c.name for c in client.get_collections().collections]
    if COLLECTION not in names:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )


def drop_collection(client: QdrantClient) -> None:
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass


def upsert_chunks(chunks: list, embeddings: list[list[float]]) -> None:
    client = get_client()
    ensure_collection(client)
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb,
            payload={
                "text": chunk.text,
                "page": chunk.page,
                "chunk_type": chunk.chunk_type,
                "source_file": chunk.source_file,
            },
        )
        for chunk, emb in zip(chunks, embeddings)
    ]
    client.upsert(collection_name=COLLECTION, points=points)


def retrieve(query_embedding: list[float]) -> list[RetrievedChunk]:
    threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.45"))
    top_k = int(os.getenv("TOP_K", "5"))
    client = get_client()
    results = client.search(
        collection_name=COLLECTION,
        query_vector=query_embedding,
        limit=top_k,
        score_threshold=threshold,
    )
    return [
        RetrievedChunk(
            text=r.payload["text"],
            page=r.payload["page"],
            chunk_type=r.payload["chunk_type"],
            source_file=r.payload["source_file"],
            score=r.score,
        )
        for r in results
    ]
