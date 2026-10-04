"""
Unit tests for text_cleaner module.
"""

import pytest
from src.text_cleaner import clean_text, remove_stopwords, tokenize_text


def test_clean_text_basic():
    raw = "Hello World! Check out https://example.com or email test@example.com."
    cleaned = clean_text(raw)
    assert "https" not in cleaned
    assert "test@example.com" not in cleaned
    assert "hello" in cleaned
    assert "world" in cleaned


def test_tech_aliases_preservation():
    raw = "Experienced in C++, C#, .NET, Node.js, and React.js development."
    cleaned = clean_text(raw, remove_stop_words=False)
    assert "c++" in cleaned
    assert "c#" in cleaned
    assert ".net" in cleaned
    assert "node.js" in cleaned
    assert "react.js" in cleaned


def test_remove_stopwords():
    raw = "this is a test string with some useless words"
    result = remove_stopwords(raw)
    assert "this" not in result.split()
    assert "test" in result.split()
