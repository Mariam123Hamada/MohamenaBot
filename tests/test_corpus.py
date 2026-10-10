def test_article_numbers_contiguous_no_gaps(corpus_data):
  
    article_nums = [record["article_number"] for record in corpus_data]
    sorted_nums = sorted(article_nums)
    
    # Check uniqueness
    assert len(sorted_nums) == len(set(sorted_nums)), "Duplicate article numbers detected!"
    
    # Check range continuity
    min_art, max_art = min(sorted_nums), max(sorted_nums)
    expected_range = set(range(min_art, max_art + 1))
    missing_articles = expected_range - set(sorted_nums)
    
    assert len(missing_articles) == 0, f"Unexplained article gaps found: {missing_articles}"

def test_non_empty_arabic_text(corpus_data):
  
    for record in corpus_data:
        art_num = record["article_number"]
        assert "text_ar" in record, f"Article {art_num} missing 'text_ar' key."
        assert isinstance(record["text_ar"], str), f"Article {art_num} 'text_ar' is not a string."
        assert len(record["text_ar"].strip()) > 0, f"Article {art_num} has empty 'text_ar'."

def test_sane_article_length(corpus_data):
    
    MAX_SANE_CHARACTERS = 15000  # ~2,500 words
    for record in corpus_data:
        art_num = record["article_number"]
        text_len = len(record["text_ar"])
        assert text_len <= MAX_SANE_CHARACTERS, (
            f"Article {art_num} has length {text_len} chars, exceeding threshold. "
            "Likely a failed split combining multiple articles."
        )

def test_repealed_articles_flagged(corpus_data):
   
    for record in corpus_data:
        art_num = record["article_number"]
        if 54 <= art_num <= 80:
            assert record["is_repealed"] is True, (
                f"Article {art_num} should be flagged as is_repealed=True."
            )