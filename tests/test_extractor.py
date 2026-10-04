"""
Unit tests for skill_extractor module.
"""

import pytest
from src.skill_extractor import SkillExtractor


def test_skill_extraction_data_science():
    extractor = SkillExtractor()
    text = "Proficient in Python, Machine Learning, Scikit-learn, PyTorch, and SQL."
    extracted = extractor.extract_skills(text)
    
    unique_skills = extracted["unique_skills"]
    assert "python" in unique_skills
    assert "machine learning" in unique_skills
    assert "scikit-learn" in unique_skills
    assert "pytorch" in unique_skills
    assert "sql" in unique_skills


def test_skill_extraction_web():
    extractor = SkillExtractor()
    text = "Full stack developer skilled in JavaScript, React, Node.js, and MongoDB."
    extracted = extractor.extract_skills(text)
    
    unique_skills = extracted["unique_skills"]
    assert "javascript" in unique_skills
    assert "react" in unique_skills
    assert "node.js" in unique_skills
    assert "mongodb" in unique_skills
