"""
Text Cleaning and Preprocessing Module for Resume Screening System.
Uses regex, NLTK, and spaCy for cleaning unstructured text.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy

# Load spaCy English model safely
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Download NLTK stopwords safely
try:
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))


def clean_text(text: str, remove_stop_words: bool = True, lemmatize: bool = False) -> str:
    """
    Cleans raw resume or job description text.
    - Preserves technology aliases (e.g. c++, c#, .net, react.js, ci/cd)
    - Removes emails, phone numbers, URLs, bullet symbols, extra spaces
    - Optionally removes standard stopwords and lemmatizes tokens.
    """
    if not text or not isinstance(text, str):
        return ""

    # Convert lowercase
    text_lower = text.lower()

    # Remove URLs
    text_clean = re.sub(r'https?://\S+|www\.\S+', ' ', text_lower)
    
    # Remove Email addresses
    text_clean = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' ', text_clean)

    # Remove Phone numbers
    text_clean = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', ' ', text_clean)

    # Replace special tech tokens temporarily to prevent regex destruction
    tech_replacements = {
        'c++': ' cpp ',
        'c#': ' csharp ',
        '.net': ' dotnet ',
        'react.js': ' reactjs ',
        'node.js': ' nodejs ',
        'vue.js': ' vuejs ',
        'next.js': ' nextjs ',
        'd3.js': ' d3js ',
        'ci/cd': ' cicd '
    }

    for tech, placeholder in tech_replacements.items():
        text_clean = text_clean.replace(tech, placeholder)

    # Replace bullet points and non-alphanumeric chars (except whitespace)
    text_clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', text_clean)

    # Normalize whitespace
    text_clean = re.sub(r'\s+', ' ', text_clean).strip()

    if lemmatize:
        doc = nlp(text_clean)
        tokens = [token.lemma_ for token in doc if not (remove_stop_words and token.is_stop)]
        text_clean = " ".join(tokens)
    elif remove_stop_words:
        tokens = text_clean.split()
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
        text_clean = " ".join(tokens)

    # Restore tech token placeholders
    tech_reverse = {
        'cpp': 'c++',
        'csharp': 'c#',
        'dotnet': '.net',
        'reactjs': 'react.js',
        'nodejs': 'node.js',
        'vuejs': 'vue.js',
        'nextjs': 'next.js',
        'd3js': 'd3.js',
        'cicd': 'ci/cd'
    }

    for placeholder, tech in tech_reverse.items():
        text_clean = re.sub(r'\b' + placeholder + r'\b', tech, text_clean)

    return text_clean


def tokenize_text(text: str) -> list[str]:
    """Tokenizes text into a list of word tokens."""
    cleaned = clean_text(text, remove_stop_words=True, lemmatize=False)
    return cleaned.split()


def remove_stopwords(text: str) -> str:
    """Removes common stopwords from string."""
    tokens = text.split()
    filtered = [w for w in tokens if w.lower() not in STOPWORDS]
    return " ".join(filtered)
