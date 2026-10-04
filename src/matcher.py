"""
Candidate Matcher, Scoring, and Ranking Engine.
Computes TF-IDF Cosine Similarity, Skill Overlap, Mandatory Skill Weighting, and Skill Gap Analysis.
"""

from typing import Dict, List, Tuple, Union, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .text_cleaner import clean_text
from .skill_extractor import SkillExtractor


class ResumeMatcher:
    """Matches resumes against a target job description and ranks candidates."""

    def __init__(self, skill_extractor: SkillExtractor = None):
        if skill_extractor is None:
            self.skill_extractor = SkillExtractor()
        else:
            self.skill_extractor = skill_extractor

    def compute_tfidf_similarity(self, resume_text: str, job_description_text: str) -> float:
        """
        Computes TF-IDF Cosine Similarity between cleaned resume text and job description.
        Returns score scaled between 0.0 and 1.0.
        """
        cleaned_resume = clean_text(resume_text, remove_stop_words=True, lemmatize=True)
        cleaned_jd = clean_text(job_description_text, remove_stop_words=True, lemmatize=True)

        if not cleaned_resume or not cleaned_jd:
            return 0.0

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        tfidf_matrix = vectorizer.fit_transform([cleaned_jd, cleaned_resume])
        
        sim_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(np.nan_to_num(sim_score, nan=0.0))

    def evaluate_candidate(
        self,
        candidate_id: str,
        candidate_name: str,
        resume_text: str,
        job_description_text: str,
        mandatory_skills: List[str] = None,
        weights: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Evaluates a single resume against a target job description.
        Returns comprehensive score dictionary.
        """
        if weights is None:
            weights = {
                "tfidf_weight": 0.40,
                "skill_coverage_weight": 0.40,
                "mandatory_weight": 0.20
            }

        if mandatory_skills is None:
            mandatory_skills = []

        mandatory_skills_clean = [s.lower().strip() for s in mandatory_skills if s.strip()]

        # Extract skills
        resume_skills_data = self.skill_extractor.extract_skills(resume_text)
        jd_skills_data = self.skill_extractor.extract_skills(job_description_text)

        resume_skills_set = set(resume_skills_data["unique_skills"])
        jd_skills_set = set(jd_skills_data["unique_skills"])

        # Skill Breakdown
        matched_skills = sorted(list(resume_skills_set.intersection(jd_skills_set)))
        missing_skills = sorted(list(jd_skills_set.difference(resume_skills_set)))
        extra_skills = sorted(list(resume_skills_set.difference(jd_skills_set)))

        # Skill Coverage Ratio
        if len(jd_skills_set) > 0:
            skill_coverage_score = len(matched_skills) / len(jd_skills_set)
        else:
            skill_coverage_score = 0.0

        # Mandatory Skill Score
        if mandatory_skills_clean:
            matched_mandatory = [s for s in mandatory_skills_clean if s in resume_skills_set]
            missing_mandatory = [s for s in mandatory_skills_clean if s not in resume_skills_set]
            mandatory_score = len(matched_mandatory) / len(mandatory_skills_clean)
        else:
            matched_mandatory = []
            missing_mandatory = []
            mandatory_score = 1.0  # Default full score if no mandatory list specified

        # TF-IDF Cosine Similarity
        tfidf_score = self.compute_tfidf_similarity(resume_text, job_description_text)

        # Composite Overall Score (0 - 100%)
        overall_score = (
            (weights["tfidf_weight"] * tfidf_score) +
            (weights["skill_coverage_weight"] * skill_coverage_score) +
            (weights["mandatory_weight"] * mandatory_score)
        ) * 100.0

        overall_score = round(min(100.0, max(0.0, overall_score)), 2)

        # Determine Fit Tier Label
        if overall_score >= 78.0:
            fit_tier = "Strong Fit"
            badge_color = "green"
        elif overall_score >= 62.0:
            fit_tier = "Good Fit"
            badge_color = "blue"
        elif overall_score >= 45.0:
            fit_tier = "Potential Fit"
            badge_color = "orange"
        else:
            fit_tier = "Low Fit"
            badge_color = "red"

        # Generate Natural Language Recruiter Explanation
        explanation = self.generate_recruiter_explanation(
            candidate_name=candidate_name,
            overall_score=overall_score,
            fit_tier=fit_tier,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            missing_mandatory=missing_mandatory,
            tfidf_score=tfidf_score
        )

        return {
            "candidate_id": candidate_id,
            "candidate_name": candidate_name,
            "overall_score": overall_score,
            "fit_tier": fit_tier,
            "badge_color": badge_color,
            "tfidf_similarity_percent": round(tfidf_score * 100, 2),
            "skill_coverage_percent": round(skill_coverage_score * 100, 2),
            "mandatory_score_percent": round(mandatory_score * 100, 2),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "extra_skills": extra_skills,
            "matched_mandatory": matched_mandatory,
            "missing_mandatory": missing_mandatory,
            "total_resume_skills": len(resume_skills_set),
            "total_jd_skills": len(jd_skills_set),
            "explanation": explanation
        }

    def rank_candidates(
        self,
        candidates: List[Dict[str, str]],
        job_description_text: str,
        mandatory_skills: List[str] = None,
        weights: Dict[str, float] = None
    ) -> List[Dict[str, Any]]:
        """
        Ranks a list of candidate dictionaries against a job description.
        candidates: list of dicts with keys 'id', 'name', 'text'
        Returns sorted list of evaluated candidate result dicts (highest score first).
        """
        results = []
        for i, candidate in enumerate(candidates):
            cid = candidate.get("id", f"CAND_{i+1:03d}")
            cname = candidate.get("name", f"Candidate {i+1}")
            ctext = candidate.get("text", "")

            eval_res = self.evaluate_candidate(
                candidate_id=cid,
                candidate_name=cname,
                resume_text=ctext,
                job_description_text=job_description_text,
                mandatory_skills=mandatory_skills,
                weights=weights
            )
            results.append(eval_res)

        # Sort by overall score descending
        ranked_results = sorted(results, key=lambda x: x["overall_score"], reverse=True)

        # Assign rank positions
        for rank, res in enumerate(ranked_results, start=1):
            res["rank"] = rank

        return ranked_results

    def generate_recruiter_explanation(
        self,
        candidate_name: str,
        overall_score: float,
        fit_tier: str,
        matched_skills: List[str],
        missing_skills: List[str],
        missing_mandatory: List[str],
        tfidf_score: float
    ) -> str:
        """Generates executive non-technical explanation for HR personnel."""
        top_matches = ", ".join(matched_skills[:5]) if matched_skills else "None"
        key_gaps = ", ".join(missing_skills[:5]) if missing_skills else "None"

        explanation = f"**Assessment for {candidate_name} ({fit_tier} - {overall_score:.1f}%):**\n"
        
        if fit_tier in ["Strong Fit", "Good Fit"]:
            explanation += f"- **Strengths:** Strong alignment with job requirements. Demonstrates core competencies in {top_matches}.\n"
        else:
            explanation += f"- **Strengths:** Partial skill overlap found in {top_matches}.\n"

        if missing_mandatory:
            explanation += f"- **Critical Gaps:** Missing mandatory required skills: **{', '.join(missing_mandatory)}**.\n"
        elif missing_skills:
            explanation += f"- **Skill Gaps:** Lacks secondary skills such as {key_gaps}.\n"
        else:
            explanation += "- **Skill Gaps:** Candidate meets or exceeds all identified skill requirements.\n"

        if tfidf_score >= 0.5:
            explanation += "- **Context Alignment:** High textual and domain experience match."
        elif tfidf_score >= 0.3:
            explanation += "- **Context Alignment:** Moderate domain background alignment."
        else:
            explanation += "- **Context Alignment:** Limited contextual overlap with job description language."

        return explanation
