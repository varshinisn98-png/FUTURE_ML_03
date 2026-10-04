"""
Skill Extraction & Entity Recognition Engine.
Extracts technical and soft skills from resume and job description text using spaCy and taxonomy matching.
"""

import json
import os
import re
from typing import Dict, List, Set, Tuple, Any
import spacy
from spacy.matcher import PhraseMatcher


class SkillExtractor:
    """Extracts skills and technical keywords from raw text."""

    def __init__(self, skills_db_path: str = None):
        if skills_db_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            skills_db_path = os.path.join(base_dir, "data", "skills_db.json")

        self.skills_db_path = skills_db_path
        self.skills_db = self._load_skills_db()
        self.all_skills_flat = self._flatten_skills_db()
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self._build_phrase_matcher()

    def _load_skills_db(self) -> Dict[str, List[str]]:
        """Loads JSON taxonomy database of skills."""
        if os.path.exists(self.skills_db_path):
            with open(self.skills_db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _flatten_skills_db(self) -> Set[str]:
        """Creates a set of all skill keywords in lowercase."""
        skills_set = set()
        for category, skills in self.skills_db.items():
            for skill in skills:
                skills_set.add(skill.lower().strip())
        return skills_set

    def _build_phrase_matcher(self):
        """Populates spaCy phrase matcher with multi-word skill patterns."""
        patterns = []
        for skill in self.all_skills_flat:
            patterns.append(self.nlp.make_doc(skill))
        self.phrase_matcher.add("SKILL_PATTERNS", patterns)

    def extract_skills(self, text: str) -> Dict[str, Any]:
        """
        Extracts skills from text using spaCy phrase matching and regex word-boundary lookup.
        Returns:
            dict containing:
            - 'unique_skills': sorted list of unique extracted skills
            - 'categorized_skills': skills grouped by domain category
            - 'skill_count': total count of extracted skills
        """
        if not text:
            return {"unique_skills": [], "categorized_skills": {}, "skill_count": 0}

        text_clean = text.lower()
        extracted_skills = set()

        # 1. spaCy PhraseMatcher for exact multi-word & single-word matching
        doc = self.nlp(text_clean)
        matches = self.phrase_matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            extracted_skills.add(span.text.lower().strip())

        # 2. Regex word-boundary fallback for tech skills with symbols (e.g. c++, c#, .net, react.js, ci/cd)
        for skill in self.all_skills_flat:
            if skill in extracted_skills:
                continue
            
            # Escaping special regex chars
            pattern = r'\b' + re.escape(skill) + r'\b'
            # Special check for skills starting/ending with punctuation like c++, c#
            if skill in ['c++', 'c#', '.net', 'react.js', 'node.js', 'vue.js', 'next.js', 'ci/cd']:
                pattern = re.escape(skill)
                if re.search(pattern, text_clean):
                    extracted_skills.add(skill)
            elif re.search(pattern, text_clean):
                extracted_skills.add(skill)

        # Categorize extracted skills
        categorized = {}
        for category, skills_list in self.skills_db.items():
            cat_matched = []
            category_lower_skills = [s.lower() for s in skills_list]
            for skill in extracted_skills:
                if skill in category_lower_skills:
                    cat_matched.append(skill)
            if cat_matched:
                categorized[category] = sorted(list(set(cat_matched)))

        sorted_skills = sorted(list(extracted_skills))

        return {
            "unique_skills": sorted_skills,
            "categorized_skills": categorized,
            "skill_count": len(sorted_skills)
        }

    def add_custom_skills(self, custom_skills: List[str]):
        """Dynamically adds new skills to taxonomy."""
        for skill in custom_skills:
            cleaned_skill = skill.lower().strip()
            if cleaned_skill:
                self.all_skills_flat.add(cleaned_skill)
                # Add to Custom category in db
                if "Custom Skills" not in self.skills_db:
                    self.skills_db["Custom Skills"] = []
                if cleaned_skill not in self.skills_db["Custom Skills"]:
                    self.skills_db["Custom Skills"].append(cleaned_skill)
        self._build_phrase_matcher()
