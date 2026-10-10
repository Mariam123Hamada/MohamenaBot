import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mohamenabot.model import LegalRAGModel


class FakeEncoder:
    def encode(self, texts, **kwargs):
        return [[1.0, 0.0] for _ in texts]


class FakeClient:
    def collection_exists(self, collection_name: str) -> bool:
        return True

    def count(self, collection_name: str, exact: bool) -> SimpleNamespace:
        return SimpleNamespace(count=1)

    def query_points(self, **kwargs) -> SimpleNamespace:
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    payload={
                        "chunk_id": "art_1_c1",
                        "article_number": 1,
                        "citation": "Egyptian Civil Code, Article 1",
                        "section": "General",
                        "topic": "Law",
                        "is_repealed": False,
                        "text": "نص قانوني.",
                    },
                    score=0.9,
                )
            ]
        )


def test_model_search_returns_citations_and_excludes_repealed_by_default(
    tmp_path: Path,
) -> None:
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        json.dumps(
            [
                {
                    "article_number": 1,
                    "citation": "Egyptian Civil Code, Article 1",
                    "section": "General",
                    "topic": "Law",
                    "is_repealed": False,
                    "text_ar": "نص قانوني.",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    model = LegalRAGModel(
        corpus_path=corpus_path,
        encoder=FakeEncoder(),
        client=FakeClient(),
    )
    model.initialize()

    result = model.search("ما هو القانون؟")

    assert model.ready
    assert result.results[0]["citation"] == "Egyptian Civil Code, Article 1"
    assert result.results[0]["text"] == "نص قانوني."
    assert result.embedding_drift == pytest.approx(0.0)


def test_model_rejects_empty_search_query(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        json.dumps(
            [
                {
                    "article_number": 1,
                    "text_ar": "نص قانوني.",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    model = LegalRAGModel(
        corpus_path=corpus_path,
        encoder=FakeEncoder(),
        client=FakeClient(),
    )
    model.initialize()

    with pytest.raises(ValueError, match="must not be empty"):
        model.search("  ")
