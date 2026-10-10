from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping


def chunk_article(
    record: Mapping[str, Any],
    max_words_per_chunk: int = 150,
) -> list[dict[str, Any]]:
    """Split one corpus article into bounded chunks without dropping its metadata."""
    if max_words_per_chunk < 1:
        raise ValueError("max_words_per_chunk must be greater than zero")

    text = record.get("text_ar")
    if not isinstance(text, str):
        raise ValueError("Each article must contain string field 'text_ar'")

    article_number = record.get("article_number")
    if article_number is None:
        raise ValueError("Each article must contain 'article_number'")

    text = text.strip()
    if not text:
        return []

    # Prefer legal sentence/clause boundaries, then split oversized clauses by words.
    segments = [
        segment.strip()
        for segment in re.split(r"(?<=[.!?؟؛])\s+|\n+", text)
        if segment.strip()
    ]
    groups: list[str] = []
    pending: list[str] = []
    pending_words = 0

    for segment in segments:
        words = segment.split()
        if len(words) > max_words_per_chunk:
            if pending:
                groups.append(" ".join(pending))
                pending = []
                pending_words = 0
            groups.extend(
                " ".join(words[index : index + max_words_per_chunk])
                for index in range(0, len(words), max_words_per_chunk)
            )
            continue

        if pending_words + len(words) > max_words_per_chunk:
            groups.append(" ".join(pending))
            pending = []
            pending_words = 0
        pending.append(segment)
        pending_words += len(words)

    if pending:
        groups.append(" ".join(pending))

    metadata = {
        key: value
        for key, value in record.items()
        if key not in {"text_ar", "text_en"}
    }
    return [
        {
            **metadata,
            "chunk_id": f"art_{article_number}_c{index}",
            "text": chunk_text,
            "text_ar": chunk_text,
        }
        for index, chunk_text in enumerate(groups, start=1)
    ]


def _retrieval_recall_at_k(
    query_vectors: Any,
    chunk_vectors: Any,
    query_article_numbers: list[int],
    chunk_article_numbers: list[int],
    k: int = 3,
) -> float:
    import numpy as np

    similarities = np.asarray(query_vectors) @ np.asarray(chunk_vectors).T
    hits = 0
    for row, article_number in zip(similarities, query_article_numbers, strict=True):
        top_indices = np.argsort(row)[-k:]
        if any(chunk_article_numbers[index] == article_number for index in top_indices):
            hits += 1
    return hits / len(query_article_numbers)


def run_mlflow_chunking_experiment(
    json_path: str,
    embedding_model_name: str,
    max_words: int,
    run_count: int = 5,
    scorecard_path: str = "metrics/retrieval_scorecard.json",
) -> dict[str, Any]:
    """Compare chunk sizes, log artifacts and register the best retrieval model."""
    if max_words < 1:
        raise ValueError("max_words must be greater than zero")
    if run_count < 1:
        raise ValueError("run_count must be greater than zero")

    import mlflow
    import numpy as np
    from mlflow import MlflowClient
    from sentence_transformers import SentenceTransformer

    corpus_path = Path(json_path)
    with corpus_path.open(encoding="utf-8") as corpus_file:
        records = json.load(corpus_file)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Expected a non-empty JSON array in {corpus_path}")

    articles = [
        record
        for record in records
        if isinstance(record, dict)
        and isinstance(record.get("text_ar"), str)
        and record["text_ar"].strip()
        and isinstance(record.get("article_number"), int)
    ]
    if not articles:
        raise ValueError(f"No valid articles found in {corpus_path}")

    offsets = range(-(run_count // 2), run_count - run_count // 2)
    chunk_sizes: list[int] = []
    for offset in offsets:
        chunk_size = max(1, max_words + offset * 25)
        while chunk_size in chunk_sizes:
            chunk_size += 1
        chunk_sizes.append(chunk_size)
    chunk_sizes.sort()
    encoder = SentenceTransformer(embedding_model_name)
    evaluation_articles = articles[: min(50, len(articles))]
    query_texts = [record["text_ar"].split()[:20] for record in evaluation_articles]
    queries = [" ".join(words) for words in query_texts]
    query_article_numbers = [record["article_number"] for record in evaluation_articles]

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    mlflow.set_experiment("MohamenaBot_Chunking")
    results: list[dict[str, Any]] = []
    best_run_id = ""
    best_score = -1.0

    for chunk_size in chunk_sizes:
        chunks = [
            chunk
            for article in articles
            for chunk in chunk_article(article, chunk_size)
        ]
        if not chunks:
            raise ValueError("Chunking produced no searchable text")

        texts = [chunk["text"] for chunk in chunks]
        vectors = np.asarray(
            encoder.encode(
                texts,
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )
        query_vectors = np.asarray(
            encoder.encode(
                queries,
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )
        chunk_article_numbers = [int(chunk["article_number"]) for chunk in chunks]
        recall_at_3 = _retrieval_recall_at_k(
            query_vectors,
            vectors,
            query_article_numbers,
            chunk_article_numbers,
        )
        average_words = float(np.mean([len(text.split()) for text in texts]))
        run_name = f"chunking_{embedding_model_name.replace('/', '-')}_w{chunk_size}"

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(
                {
                    "chunk_strategy": "sentence_clause_then_word_limit",
                    "max_words_per_chunk": chunk_size,
                    "embedding_model": embedding_model_name,
                    "total_articles": len(articles),
                    "evaluation_queries": len(queries),
                }
            )
            mlflow.log_metrics(
                {
                    "generated_chunks_count": len(chunks),
                    "embedding_dimensions": int(vectors.shape[1]),
                    "average_chunk_words": average_words,
                    "retrieval_recall_at_3": recall_at_3,
                }
            )

            artifact_dir = Path(tempfile.mkdtemp(prefix="mohamenabot-run-"))
            try:
                chunks_path = artifact_dir / "chunks.json"
                chunks_path.write_text(
                    json.dumps(chunks, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                diagnostics_path = artifact_dir / "retrieval_metrics.json"
                diagnostics_path.write_text(
                    json.dumps(
                        {
                            "retrieval_recall_at_3": recall_at_3,
                            "average_chunk_words": average_words,
                            "evaluation_queries": len(queries),
                            "evaluation_note": (
                                "Queries are the first 20 words of their labeled "
                                "articles; use a human-labeled held-out set before "
                                "claiming production retrieval quality."
                            ),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                mlflow.log_artifacts(str(artifact_dir))
            finally:
                for artifact in artifact_dir.iterdir():
                    artifact.unlink()
                artifact_dir.rmdir()

            results.append(
                {
                    "run_id": run.info.run_id,
                    "max_words_per_chunk": chunk_size,
                    "chunks": len(chunks),
                    "retrieval_recall_at_3": recall_at_3,
                    "average_chunk_words": average_words,
                }
            )
            if recall_at_3 > best_score:
                best_score = recall_at_3
                best_run_id = run.info.run_id

    with mlflow.start_run(run_id=best_run_id):
        with tempfile.TemporaryDirectory(prefix="mohamenabot-encoder-") as model_dir:
            encoder.save(model_dir)

            class SentenceEncoderModel(mlflow.pyfunc.PythonModel):
                def load_context(self, context: Any) -> None:
                    self.encoder = SentenceTransformer(
                        context.artifacts["embedding_model"]
                    )

                def predict(
                    self,
                    context: Any,
                    model_input: Any,
                    params: dict[str, Any] | None = None,
                ) -> list[list[float]]:
                    texts = model_input["text"].astype(str).tolist()
                    return self.encoder.encode(texts).tolist()

            model_info = mlflow.pyfunc.log_model(
                artifact_path="registered_model",
                python_model=SentenceEncoderModel(),
                artifacts={"embedding_model": model_dir},
                registered_model_name="MohamenaBotEmbedder",
            )

    client = MlflowClient()
    if model_info.registered_model_version is None:
        raise RuntimeError("MLflow did not return a registered model version")
    client.transition_model_version_stage(
        name="MohamenaBotEmbedder",
        version=model_info.registered_model_version,
        stage="Production",
        archive_existing_versions=True,
    )

    scorecard = {
        "experiment": "MohamenaBot_Chunking",
        "registered_model": "MohamenaBotEmbedder",
        "production_version": model_info.registered_model_version,
        "best_run_id": best_run_id,
        "selection_metric": "retrieval_recall_at_3",
        "best_retrieval_recall_at_3": best_score,
        "runs": results,
        "limitations": [
            "The evaluation queries are generated from article text, not human-labeled questions.",
            "Review the logged retrieval metrics on a held-out legal-question set before production use.",
        ],
    }
    output_path = Path(scorecard_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(scorecard, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return scorecard


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    corpus_path = project_root / "src" / "data" / "processed_corpus.json"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Processed corpus not found at: {corpus_path}")

    run_mlflow_chunking_experiment(
        json_path=str(corpus_path),
        embedding_model_name="BAAI/bge-m3",
        max_words=150,
    )