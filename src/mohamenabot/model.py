from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
import numpy as np

from processing.chunking import chunk_article

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "src" / "data" / "processed_corpus.json"


@dataclass(frozen=True)
class RetrievalOutput:
    results: list[dict[str, Any]]
    embedding_drift: float


class Retriever(Protocol):
    @property
    def ready(self) -> bool: ...

    def initialize(self) -> None: ...

    def search(
        self,
        query: str,
        top_k: int = 3,
        include_repealed: bool = False,
    ) -> RetrievalOutput: ...


class LegalRAGModel:
    """Sentence-transformer retriever backed by a Qdrant vector collection."""

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-m3",
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "egyptian_civil_code",
        corpus_path: str | Path = DEFAULT_CORPUS_PATH,
        encoder: Any | None = None,
        client: Any | None = None,
        chunk_size: int = 150,
    ) -> None:
        self.embedding_model = embedding_model
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.corpus_path = Path(corpus_path)
        self.encoder = encoder
        self.client = client
        self.chunk_size = chunk_size
        self._reference_centroid: np.ndarray | None = None
        self._reference_p95_distance = 0.0
        self._ready = False

    @classmethod
    def from_environment(cls) -> LegalRAGModel:
        return cls(
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            collection_name=os.getenv(
                "QDRANT_COLLECTION", "egyptian_civil_code"
            ),
            corpus_path=os.getenv("CORPUS_PATH", str(DEFAULT_CORPUS_PATH)),
            chunk_size=int(os.getenv("CHUNK_SIZE", "150")),
        )

    @property
    def ready(self) -> bool:
        return self._ready

    def initialize(self) -> None:
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be greater than zero")
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(self.embedding_model)
        if self.client is None:
            from qdrant_client import QdrantClient

            self.client = QdrantClient(url=self.qdrant_url, timeout=60)

        if not self.corpus_path.is_file():
            raise FileNotFoundError(f"Corpus file not found: {self.corpus_path}")
        with self.corpus_path.open(encoding="utf-8") as corpus_file:
            articles = json.load(corpus_file)
        if not isinstance(articles, list) or not articles:
            raise ValueError(f"Expected a non-empty JSON array in {self.corpus_path}")

        chunks = [
            chunk
            for article in articles
            if isinstance(article, dict)
            for chunk in chunk_article(article, self.chunk_size)
        ]
        if not chunks:
            raise ValueError(f"No searchable article text found in {self.corpus_path}")

        vectors = np.asarray(
            self.encoder.encode(
                [chunk["text"] for chunk in chunks],
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError("The embedding model returned an invalid vector matrix")
        self._set_drift_reference(vectors)

        if not self.client.collection_exists(self.collection_name):
            from qdrant_client.models import Distance, VectorParams

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=int(vectors.shape[1]),
                    distance=Distance.COSINE,
                ),
            )

        count = self.client.count(
            collection_name=self.collection_name,
            exact=True,
        ).count
        if count == 0:
            self._populate_collection(chunks, vectors)
        self._ready = True

    def _set_drift_reference(self, vectors: np.ndarray) -> None:
        centroid = np.mean(vectors, axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm == 0:
            raise ValueError("Corpus embeddings have a zero-length centroid")
        centroid = centroid / centroid_norm
        distances = 1 - np.clip(vectors @ centroid, -1, 1)
        self._reference_centroid = centroid
        self._reference_p95_distance = max(float(np.percentile(distances, 95)), 1e-6)

    def _populate_collection(
        self,
        chunks: list[dict[str, Any]],
        vectors: np.ndarray,
    ) -> None:
        from qdrant_client.models import PointStruct

        points = []
        for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
            payload = {
                key: value
                for key, value in chunk.items()
                if key not in {"text_ar", "text_en"}
            }
            points.append(
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["chunk_id"])),
                    vector=vector.tolist(),
                    payload=payload,
                )
            )
            if len(points) == 256 or index == len(chunks) - 1:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True,
                )
                points = []

    def search(
        self,
        query: str,
        top_k: int = 3,
        include_repealed: bool = False,
    ) -> RetrievalOutput:
        if not self._ready or self.encoder is None or self.client is None:
            raise RuntimeError("The retrieval model has not been initialized")
        if not query.strip():
            raise ValueError("query must not be empty")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")

        query_vector = np.asarray(
            self.encoder.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0],
            dtype=np.float32,
        )
        if self._reference_centroid is None:
            raise RuntimeError("The embedding drift reference was not initialized")
        query_distance = 1 - float(
            np.clip(query_vector @ self._reference_centroid, -1, 1)
        )
        embedding_drift = query_distance / self._reference_p95_distance

        query_filter = None
        if not include_repealed:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            query_filter = Filter(
                must_not=[
                    FieldCondition(
                        key="is_repealed",
                        match=MatchValue(value=True),
                    )
                ]
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        results = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                {
                    "score": float(point.score),
                    "chunk_id": payload.get("chunk_id", ""),
                    "article_number": payload.get("article_number"),
                    "citation": payload.get("citation", ""),
                    "section": payload.get("section", ""),
                    "topic": payload.get("topic", ""),
                    "is_repealed": bool(payload.get("is_repealed", False)),
                    "text": payload.get("text", ""),
                }
            )
        return RetrievalOutput(
            results=results,
            embedding_drift=float(embedding_drift),
        )


class BentoRemoteRetriever:
    """FastAPI-side client for the BentoML production retrieval service."""

    def __init__(self, service_url: str, startup_timeout_seconds: int = 300) -> None:
        self.service_url = service_url.rstrip("/")
        self.startup_timeout_seconds = startup_timeout_seconds
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def initialize(self) -> None:
        deadline = time.monotonic() + self.startup_timeout_seconds
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                response = httpx.get(
                    f"{self.service_url}/readyz",
                    timeout=3,
                )
                if response.is_success:
                    self._ready = True
                    return
                last_error = RuntimeError(
                    f"BentoML readiness check returned HTTP {response.status_code}"
                )
            except httpx.TransportError as error:
                last_error = error
            time.sleep(2)
        raise RuntimeError(
            f"BentoML service did not become ready at {self.service_url}"
        ) from last_error

    def search(
        self,
        query: str,
        top_k: int = 3,
        include_repealed: bool = False,
    ) -> RetrievalOutput:
        if not self._ready:
            raise RuntimeError("The BentoML retrieval service is not ready")
        response = httpx.post(
            f"{self.service_url}/predict",
            json={
                "query": query,
                "top_k": top_k,
                "include_repealed": include_repealed,
            },
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()
        return RetrievalOutput(
            results=result["results"],
            embedding_drift=float(result["embedding_drift"]),
        )


def retriever_from_environment() -> Retriever:
    service_url = os.getenv("MODEL_BACKEND_URL")
    if service_url:
        return BentoRemoteRetriever(service_url)
    return LegalRAGModel.from_environment()
