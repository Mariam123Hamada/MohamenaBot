from typing import Any
from pathlib import Path

from mohamenabot.model import LegalRAGModel


class LegalVectorDatabase:
    """Compatibility wrapper for scripts that used the original vector DB API."""

    def __init__(
        self,
        collection_name: str = "egyptian_civil_code",
        embedding_model: str = "BAAI/bge-m3",
    ) -> None:
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.model: LegalRAGModel | None = None

    def create_and_populate_index(self, json_corpus_path: str) -> None:
        self.model = LegalRAGModel(
            embedding_model=self.embedding_model,
            collection_name=self.collection_name,
            corpus_path=json_corpus_path,
        )
        self.model.initialize()

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("Create the vector index before searching")
        return self.model.search(query, top_k=top_k).results


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]
    CORPUS_PATH = BASE_DIR / "src" / "data" / "processed_corpus.json"
    
    db = LegalVectorDatabase()
    db.create_and_populate_index(str(CORPUS_PATH))
    
    # Test query
    sample_query = "ما هي شروط العقد وآثاره؟"
    print(f"\nTesting query: '{sample_query}'")
    hits = db.search(sample_query, top_k=2)
    for hit in hits:
        print(f" - [{hit['citation']}] (Score: {hit['score']:.4f}): {hit['text'][:100]}...")