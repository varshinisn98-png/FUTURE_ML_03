"""
Visualization Module for Resume Screening & Ranking System.
Generates Plotly and Matplotlib charts for candidate evaluation dashboards.
"""

from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


def create_candidate_comparison_chart(ranked_results: List[Dict[str, any]], engine: str = "plotly"):
    """
    Generates a horizontal bar chart comparing ranked candidates by overall score and fit breakdown.
    """
    df = pd.DataFrame(ranked_results)
    if df.empty:
        return None

    if engine == "plotly":
        fig = px.bar(
            df,
            x="overall_score",
            y="candidate_name",
            orientation="h",
            color="fit_tier",
            color_discrete_map={
                "Strong Fit": "#2ecc71",
                "Good Fit": "#3498db",
                "Potential Fit": "#f39c12",
                "Low Fit": "#e74c3c"
            },
            text="overall_score",
            labels={"overall_score": "Overall Fit Score (%)", "candidate_name": "Candidate"},
            title="Candidate Role Fit Leaderboard"
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400, margin=dict(l=20, r=20, t=50, b=20))
        return fig
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(
            data=df,
            x="overall_score",
            y="candidate_name",
            palette="viridis",
            ax=ax
        )
        ax.set_title("Candidate Screening Leaderboard")
        ax.set_xlabel("Overall Match Score (%)")
        ax.set_ylabel("Candidate")
        plt.tight_layout()
        return fig


def create_skill_gap_chart(candidate_eval: Dict[str, any]):
    """
    Generates a breakdown gauge / donut chart of candidate's Matched vs Missing skills.
    """
    matched_cnt = len(candidate_eval.get("matched_skills", []))
    missing_cnt = len(candidate_eval.get("missing_skills", []))
    
    labels = ["Matched Skills", "Missing Skills"]
    values = [matched_cnt, missing_cnt]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.4,
        marker=dict(colors=["#2ecc71", "#e74c3c"])
    )])
    fig.update_layout(
        title=f"Skill Breakdown - {candidate_eval.get('candidate_name', 'Candidate')}",
        annotations=[dict(text=f"{candidate_eval.get('overall_score', 0):.0f}%", x=0.5, y=0.5, font_size=20, showarrow=False)],
        height=350,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def create_multi_metric_radar_chart(ranked_results: List[Dict[str, any]]):
    """
    Generates a Radar / Spider chart comparing top candidates across TF-IDF, Skill Coverage, and Mandatory Score.
    """
    if not ranked_results:
        return None

    top_candidates = ranked_results[:5]  # Limit to top 5 for visual clarity
    categories = ['Context Similarity (TF-IDF)', 'Skill Coverage', 'Mandatory Skills']

    fig = go.Figure()

    for cand in top_candidates:
        r_values = [
            cand.get("tfidf_similarity_percent", 0),
            cand.get("skill_coverage_percent", 0),
            cand.get("mandatory_score_percent", 0)
        ]
        r_values.append(r_values[0])  # Close radar loop

        fig.add_trace(go.Scatterpolar(
            r=r_values,
            theta=categories + [categories[0]],
            fill='toself',
            name=cand.get("candidate_name", "Candidate")
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        showlegend=True,
        title="Top Candidates Multi-Dimensional Profile Comparison",
        height=450,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig
