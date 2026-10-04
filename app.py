"""
Streamlit Web Application: ML-Powered Resume Screening & Ranking System.
"""

import os
import glob
import pandas as pd
import streamlit as st
import plotly.express as px

from src.parser import DocumentParser
from src.skill_extractor import SkillExtractor
from src.matcher import ResumeMatcher
from src.visualizer import (
    create_candidate_comparison_chart,
    create_skill_gap_chart,
    create_multi_metric_radar_chart
)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AI Resume Screening & Ranking System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for HR dashboard
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #2563EB;
        margin-bottom: 10px;
    }
    .badge-strong {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
    .badge-good {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
    .badge-potential {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
    .badge-low {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
    .skill-chip-matched {
        background-color: #2ECC71;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        display: inline-block;
        margin: 2px;
    }
    .skill-chip-missing {
        background-color: #E74C3C;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        display: inline-block;
        margin: 2px;
    }
    .skill-chip-extra {
        background-color: #3498DB;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        display: inline-block;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_system_modules():
    """Cached initialization of NLP components."""
    extractor = SkillExtractor()
    matcher = ResumeMatcher(skill_extractor=extractor)
    return extractor, matcher


skill_extractor, resume_matcher = load_system_modules()

# App Header
st.markdown("<div class='main-title'>🔍 Smart ML Resume Screening & Ranking System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Automated Candidate Shortlisting, Skill Gap Identification, and Match Analytics for Hiring Teams</div>", unsafe_allow_html=True)

# Sidebar Setup
st.sidebar.header("⚙️ Screening Configuration")

# 1. Select or Upload Job Description
jd_option = st.sidebar.selectbox(
    "Target Job Role",
    [
        "Data Scientist / ML Engineer (Sample)",
        "Full Stack Developer (Sample)",
        "DevOps & Cloud Engineer (Sample)",
        "Paste Custom Job Description",
        "Upload Job Description File (.txt/.docx/.pdf)"
    ]
)

base_dir = os.path.dirname(os.path.abspath(__file__))
jd_samples_dir = os.path.join(base_dir, "data", "sample_job_descriptions")

jd_text = ""

if "Sample" in jd_option:
    if "Data Scientist" in jd_option:
        jd_path = os.path.join(jd_samples_dir, "data_scientist.txt")
    elif "Full Stack" in jd_option:
        jd_path = os.path.join(jd_samples_dir, "fullstack_developer.txt")
    else:
        jd_path = os.path.join(jd_samples_dir, "devops_engineer.txt")
    
    if os.path.exists(jd_path):
        with open(jd_path, "r", encoding="utf-8") as f:
            jd_text = f.read()

elif jd_option == "Paste Custom Job Description":
    jd_text = st.sidebar.text_area(
        "Paste Job Description Here",
        height=200,
        placeholder="We are looking for a Senior Developer proficient in Python, React, AWS, and Docker..."
    )

elif jd_option == "Upload Job Description File (.txt/.docx/.pdf)":
    uploaded_jd = st.sidebar.file_uploader("Upload JD File", type=["txt", "pdf", "docx"], key="jd_uploader")
    if uploaded_jd:
        jd_text = DocumentParser.parse_file(uploaded_jd, uploaded_jd.name)

# Display extracted JD skills preview in sidebar
if jd_text:
    extracted_jd_skills = skill_extractor.extract_skills(jd_text)["unique_skills"]
    st.sidebar.markdown(f"**Extracted JD Skills ({len(extracted_jd_skills)}):**")
    st.sidebar.caption(", ".join(extracted_jd_skills) if extracted_jd_skills else "No skills recognized yet.")

    # 2. Mandatory Skill Selection
    mandatory_skills = st.sidebar.multiselect(
        "Filter / Weight Mandatory Skills",
        options=extracted_jd_skills,
        default=extracted_jd_skills[:min(3, len(extracted_jd_skills))]
    )
else:
    mandatory_skills = []

# 3. Model Weightings Adjustment
st.sidebar.markdown("---")
st.sidebar.subheader("⚖️ Scoring Formula Weights")
tfidf_wt = st.sidebar.slider("TF-IDF Text Similarity Weight", 0.0, 1.0, 0.40, 0.05)
skill_wt = st.sidebar.slider("Skill Match Coverage Weight", 0.0, 1.0, 0.40, 0.05)
mand_wt = st.sidebar.slider("Mandatory Skill Bonus Weight", 0.0, 1.0, 0.20, 0.05)

total_wt = tfidf_wt + skill_wt + mand_wt
if total_wt == 0:
    total_wt = 1.0

norm_weights = {
    "tfidf_weight": tfidf_wt / total_wt,
    "skill_coverage_weight": skill_wt / total_wt,
    "mandatory_weight": mand_wt / total_wt
}

# Main Area - Tabs
tab1, tab2, tab3 = st.tabs(["🏆 Candidate Leaderboard", "🔬 Candidate Deep-Dive", "📚 Skill Taxonomy"])

with tab1:
    st.subheader("📁 Ingest Resumes for Screening")
    
    col_input1, col_input2 = st.columns([3, 1])
    
    with col_input1:
        uploaded_resumes = st.file_uploader(
            "Upload Candidate Resumes (.pdf, .docx, .txt)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )
    
    with col_input2:
        st.markdown("<br>", unsafe_allow_html=True)
        load_sample_btn = st.button("🚀 Load 5 Benchmark Resumes", use_container_width=True)

    candidates_list = []

    if load_sample_btn or not uploaded_resumes:
        # Load sample benchmark resumes from data/sample_resumes/
        sample_resumes_dir = os.path.join(base_dir, "data", "sample_resumes")
        sample_files = glob.glob(os.path.join(sample_resumes_dir, "*.txt"))
        for sfile in sample_files:
            fname = os.path.basename(sfile)
            name_part = fname.replace(".txt", "").replace("_", " ").title()
            with open(sfile, "r", encoding="utf-8") as f:
                content = f.read()
            candidates_list.append({
                "id": fname,
                "name": name_part,
                "text": content
            })
        if load_sample_btn:
            st.success("Loaded 5 benchmark resumes from dataset!")

    if uploaded_resumes:
        candidates_list = []
        for ufile in uploaded_resumes:
            try:
                parsed_content = DocumentParser.parse_file(ufile, ufile.name)
                c_name = os.path.splitext(ufile.name)[0].replace("_", " ").title()
                candidates_list.append({
                    "id": ufile.name,
                    "name": c_name,
                    "text": parsed_content
                })
            except Exception as e:
                st.error(f"Could not parse {ufile.name}: {e}")

    if not jd_text:
        st.warning("⚠️ Please select or upload a Job Description in the sidebar to run candidate ranking.")
    elif not candidates_list:
        st.info("ℹ️ Upload resumes above or click 'Load 5 Benchmark Resumes' to view screening results.")
    else:
        # Perform Ranking
        ranked_results = resume_matcher.rank_candidates(
            candidates=candidates_list,
            job_description_text=jd_text,
            mandatory_skills=mandatory_skills,
            weights=norm_weights
        )

        st.markdown("---")
        st.subheader("📊 Executive Screening Summary")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Resumes Screened", len(ranked_results))
        top_cand = ranked_results[0]
        m2.metric("Top Ranked Fit", f"{top_cand['overall_score']}%", delta=top_cand['candidate_name'])
        avg_score = sum(r['overall_score'] for r in ranked_results) / len(ranked_results)
        m3.metric("Cohort Average Score", f"{avg_score:.1f}%")
        m4.metric("Top Skill Coverage", f"{top_cand['skill_coverage_percent']}%")

        # Leaderboard Table
        st.markdown("### 🏆 Candidate Ranking Leaderboard")
        
        table_data = []
        for r in ranked_results:
            table_data.append({
                "Rank": f"#{r['rank']}",
                "Candidate Name": r["candidate_name"],
                "Fit Tier": r["fit_tier"],
                "Overall Fit Score": f"{r['overall_score']}%",
                "TF-IDF Similarity": f"{r['tfidf_similarity_percent']}%",
                "Skill Coverage": f"{r['skill_coverage_percent']}%",
                "Matched Skills": ", ".join(r["matched_skills"][:5]) if r["matched_skills"] else "None",
                "Missing Skills": ", ".join(r["missing_skills"][:5]) if r["missing_skills"] else "None"
            })
        
        df_display = pd.DataFrame(table_data)
        st.dataframe(df_display, use_container_width=True)

        # Visualizations
        col_chart1, col_chart2 = st.columns([1, 1])
        with col_chart1:
            chart_fig = create_candidate_comparison_chart(ranked_results, engine="plotly")
            if chart_fig:
                st.plotly_chart(chart_fig, use_container_width=True)
        
        with col_chart2:
            radar_fig = create_multi_metric_radar_chart(ranked_results)
            if radar_fig:
                st.plotly_chart(radar_fig, use_container_width=True)

        # Download CSV report
        csv_data = pd.DataFrame(ranked_results).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Screening Report (CSV)",
            data=csv_data,
            file_name="candidate_screening_report.csv",
            mime="text/csv"
        )

        # Store results in session state for Deep-Dive tab
        st.session_state["ranked_results"] = ranked_results
        st.session_state["jd_text"] = jd_text


with tab2:
    st.subheader("🔬 Deep-Dive Candidate Assessment")

    if "ranked_results" not in st.session_state or not st.session_state["ranked_results"]:
        st.info("Run candidate screening in Tab 1 first to view candidate deep-dives.")
    else:
        results = st.session_state["ranked_results"]
        candidate_names = [r["candidate_name"] for r in results]

        selected_cand_name = st.selectbox("Select Candidate to Inspect:", candidate_names)
        cand_data = next(r for r in results if r["candidate_name"] == selected_cand_name)

        col_d1, col_d2 = st.columns([1.2, 1])

        with col_d1:
            st.markdown(f"### {cand_data['candidate_name']} — Rank #{cand_data['rank']}")
            st.markdown(f"**Role Fit Tier:** <span class='badge-{cand_data['fit_tier'].lower().replace(' ', '-')}'>{cand_data['fit_tier']}</span>", unsafe_allow_html=True)
            st.markdown(f"**Overall Match Score:** `{cand_data['overall_score']}%`")

            # Explanation Box
            st.markdown("#### 💡 Executive Recruiter Explanation")
            st.info(cand_data["explanation"])

            # Skill Tags
            st.markdown("#### 🛠️ Skill Breakdown")
            
            st.markdown("**Matched Skills (Found in Resume & JD):**")
            if cand_data["matched_skills"]:
                chips_html = "".join([f"<span class='skill-chip-matched'>{s}</span>" for s in cand_data["matched_skills"]])
                st.markdown(chips_html, unsafe_allow_html=True)
            else:
                st.caption("No matching skills identified.")

            st.markdown("<br>**Missing Skills (Required by JD):**", unsafe_allow_html=True)
            if cand_data["missing_skills"]:
                chips_html = "".join([f"<span class='skill-chip-missing'>{s}</span>" for s in cand_data["missing_skills"]])
                st.markdown(chips_html, unsafe_allow_html=True)
            else:
                st.caption("No missing skills!")

            st.markdown("<br>**Extra Candidate Skills:**", unsafe_allow_html=True)
            if cand_data["extra_skills"]:
                chips_html = "".join([f"<span class='skill-chip-extra'>{s}</span>" for s in cand_data["extra_skills"]])
                st.markdown(chips_html, unsafe_allow_html=True)
            else:
                st.caption("No extra skills found.")

        with col_d2:
            pie_fig = create_skill_gap_chart(cand_data)
            if pie_fig:
                st.plotly_chart(pie_fig, use_container_width=True)

            st.markdown("#### 📏 Detailed Score Breakdown")
            st.write(f"- **TF-IDF Context Similarity:** `{cand_data['tfidf_similarity_percent']}%`")
            st.write(f"- **Skill Coverage Ratio:** `{cand_data['skill_coverage_percent']}%`")
            st.write(f"- **Mandatory Skill Match:** `{cand_data['mandatory_score_percent']}%`")
            if cand_data["missing_mandatory"]:
                st.warning(f"⚠️ Missing Mandatory Skills: {', '.join(cand_data['missing_mandatory'])}")


with tab3:
    st.subheader("📚 Active Skill Taxonomy & Custom Rules")
    
    st.markdown("This system uses spaCy Entity Matching combined with a curated skills database.")

    current_db = skill_extractor.skills_db
    
    st.markdown("#### 📂 Skill Categories & Counts")
    for category, skills in current_db.items():
        with st.expander(f"{category} ({len(skills)} skills)"):
            st.write(", ".join(skills))

    st.markdown("---")
    st.markdown("#### ➕ Add Custom Skills to Taxonomy")
    new_skills_input = st.text_input("Enter new skills (comma-separated):", placeholder="golang, snowflake, lang-chain")
    if st.button("Add Skills to Live Taxonomy"):
        if new_skills_input:
            added_list = [s.strip() for s in new_skills_input.split(",") if s.strip()]
            skill_extractor.add_custom_skills(added_list)
            st.success(f"Added {len(added_list)} skills to live taxonomy: {', '.join(added_list)}")
