from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Annotated, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from mohamenabot.model import RetrievalOutput, Retriever, retriever_from_environment

PREDICTION_REQUESTS = Counter(
    "mohamenabot_prediction_requests_total",
    "Number of prediction requests served.",
)
PREDICTION_LATENCY = Histogram(
    "mohamenabot_prediction_latency_seconds",
    "Time spent serving prediction requests.",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)
EMBEDDING_DRIFT = Gauge(
    "mohamenabot_embedding_input_drift",
    "Query distance from the corpus embedding centroid, normalized by corpus p95.",
)
RETRIEVED_RESULTS = Histogram(
    "mohamenabot_retrieved_results",
    "Number of legal provisions returned for a prediction.",
    buckets=(0, 1, 2, 3, 5, 10, 20),
)


class PredictRequest(BaseModel):
    query: Annotated[str, Field(min_length=1, max_length=4000, strip_whitespace=True)]
    top_k: Annotated[int, Field(ge=1, le=20)] = 3
    include_repealed: bool = False


class Passage(BaseModel):
    score: float
    chunk_id: str
    article_number: int | None
    citation: str
    section: str
    topic: str
    is_repealed: bool
    text: str


class PredictResponse(BaseModel):
    query: str
    results: list[Passage]
    embedding_drift: float
    disclaimer: str = (
        "Retrieved legal text for reference only; this is not legal advice."
    )


def create_app(
    model_factory: Callable[[], Retriever] = retriever_from_environment,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        model = model_factory()
        await run_in_threadpool(model.initialize)
        app.state.model = model
        yield

    application = FastAPI(
        title="MohamenaBot API",
        version="0.1.0",
        description="Citation-first retrieval over the Egyptian Civil Code.",
        lifespan=lifespan,
    )

    @application.get("/health")
    async def health(request: Request) -> dict[str, str]:
        model = getattr(request.app.state, "model", None)
        ready = model is not None and model.ready
        if not ready:
            raise HTTPException(status_code=503, detail="Retrieval model is not ready")
        return {"status": "ok", "model": "ready"}

    @application.post("/predict", response_model=PredictResponse)
    async def predict(
        payload: PredictRequest,
        request: Request,
    ) -> PredictResponse:
        model: Retriever | None = getattr(request.app.state, "model", None)
        if model is None or not model.ready:
            raise HTTPException(status_code=503, detail="Retrieval model is not ready")

        started_at = time.perf_counter()
        try:
            output: RetrievalOutput = await run_in_threadpool(
                model.search,
                payload.query,
                payload.top_k,
                payload.include_repealed,
            )
            EMBEDDING_DRIFT.set(output.embedding_drift)
            RETRIEVED_RESULTS.observe(len(output.results))
            return PredictResponse(
                query=payload.query,
                results=output.results,
                embedding_drift=output.embedding_drift,
            )
        finally:
            PREDICTION_REQUESTS.inc()
            PREDICTION_LATENCY.observe(time.perf_counter() - started_at)

    @application.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return application


app = create_app()
