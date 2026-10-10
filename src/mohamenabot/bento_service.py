import os

import bentoml

from mohamenabot.api import Passage, PredictRequest, PredictResponse
from mohamenabot.model import LegalRAGModel


@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 120},
)
class LegalRetrievalService:
    def __init__(self) -> None:
        self.model = LegalRAGModel.from_environment()
        self.model.initialize()

    @bentoml.api
    def predict(self, payload: PredictRequest) -> PredictResponse:
        output = self.model.search(
            payload.query,
            top_k=payload.top_k,
            include_repealed=payload.include_repealed,
        )
        return PredictResponse(
            query=payload.query,
            results=[Passage.model_validate(result) for result in output.results],
            embedding_drift=output.embedding_drift,
        )
