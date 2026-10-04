"""
Unit tests for matcher module.
"""

import pytest
from src.matcher import ResumeMatcher


def test_resume_matching_and_ranking():
    matcher = ResumeMatcher()
    
    jd = "Seeking a Data Scientist proficient in Python, Machine Learning, Scikit-learn, and SQL."
    
    cand1 = {
        "id": "CAND_01",
        "name": "Alex Rivera",
        "text": "Data Scientist with experience in Python, Machine Learning, Scikit-learn, SQL, and Deep Learning."
    }
    cand2 = {
        "id": "CAND_02",
        "name": "David Miller",
        "text": "Human Resources Specialist in recruiting, onboarding, and employee relations."
    }

    ranked = matcher.rank_candidates([cand1, cand2], job_description_text=jd)

    assert len(ranked) == 2
    assert ranked[0]["candidate_id"] == "CAND_01"
    assert ranked[0]["overall_score"] > ranked[1]["overall_score"]
    assert "python" in ranked[0]["matched_skills"]
    assert ranked[0]["fit_tier"] in ["Strong Fit", "Good Fit"]
