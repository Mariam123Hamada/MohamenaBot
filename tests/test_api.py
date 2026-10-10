from fastapi.testclient import TestClient

from mohamenabot.api import create_app
from mohamenabot.model import RetrievalOutput


class FakeRetriever:
    ready = False

    def initialize(self) -> None:
        self.ready = True

    def search(
        self,
        query: str,
        top_k: int,
        include_repealed: bool,
    ) -> RetrievalOutput:
        return RetrievalOutput(
            results=[
                {
                    "score": 0.91,
                    "chunk_id": "art_1_c1",
                    "article_number": 1,
                    "citation": "Egyptian Civil Code, Article 1",
                    "section": "General Provisions",
                    "topic": "Law",
                    "is_repealed": False,
                    "text": "نص قانوني للاختبار.",
                }
            ][:top_k],
            embedding_drift=0.8,
        )


def test_health_and_predict_return_cited_passages() -> None:
    retriever = FakeRetriever()
    app = create_app(model_factory=lambda: retriever)

    with TestClient(app) as client:
        health = client.get("/health")
        prediction = client.post(
            "/predict",
            json={"query": "ما هو النص القانوني؟", "top_k": 1},
        )

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert prediction.status_code == 200
    assert prediction.json()["results"][0]["citation"].endswith("Article 1")
    assert prediction.json()["results"][0]["text"] == "نص قانوني للاختبار."
    assert "not legal advice" in prediction.json()["disclaimer"]


def test_predict_rejects_empty_query() -> None:
    app = create_app(model_factory=FakeRetriever)

    with TestClient(app) as client:
        response = client.post("/predict", json={"query": " "})

    assert response.status_code == 422
