from locust import HttpUser, between, task


class LegalRetrievalUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def retrieve_legal_provisions(self) -> None:
        self.client.post(
            "/predict",
            json={
                "query": "ما هي آثار العقد؟",
                "top_k": 3,
            },
            name="/predict",
        )
