import json
from pathlib import Path

import pytest

@pytest.fixture
def corpus_data():
    json_path = Path(__file__).resolve().parent.parent / "src"/"data" / "processed_corpus.json"
    if not json_path.exists():
        pytest.skip(
            "Processed corpus is not present; retrieve it with DVC before "
            "running corpus validation."
        )
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

