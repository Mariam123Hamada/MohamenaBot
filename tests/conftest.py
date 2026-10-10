import pytest 
import json 
from pathlib import Path 

@pytest.fixture
def corpus_data():
    json_path = Path(__file__).resolve().parent.parent / "src"/"data" / "processed_corpus.json"
    assert json_path.exists(), f"Corpus JSON not found at {json_path}"
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)
