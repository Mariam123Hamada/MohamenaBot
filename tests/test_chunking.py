import pytest

from processing.chunking import chunk_article


def test_chunk_article_splits_flattened_text_with_word_limit() -> None:
    record = {
        "article_number": 12,
        "citation": "Egyptian Civil Code, Article 12",
        "section": "Contracts",
        "topic": "Obligations",
        "is_repealed": False,
        "text_ar": " ".join(f"word{index}" for index in range(31)),
    }

    chunks = chunk_article(record, max_words_per_chunk=10)

    assert [len(chunk["text"].split()) for chunk in chunks] == [10, 10, 10, 1]
    assert [chunk["chunk_id"] for chunk in chunks] == [
        "art_12_c1",
        "art_12_c2",
        "art_12_c3",
        "art_12_c4",
    ]
    assert all(chunk["citation"] == record["citation"] for chunk in chunks)
    assert " ".join(chunk["text"] for chunk in chunks) == record["text_ar"]


def test_chunk_article_keeps_short_article_as_one_chunk() -> None:
    record = {
        "article_number": 1,
        "book": "Civil Code",
        "text_ar": "تبدأ شخصية الإنسان بتمام ولادته حيا.",
    }

    chunks = chunk_article(record)

    assert len(chunks) == 1
    assert chunks[0]["text"] == record["text_ar"]
    assert chunks[0]["book"] == "Civil Code"


def test_chunk_article_rejects_invalid_limit() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        chunk_article({"article_number": 1, "text_ar": "نص"}, 0)
