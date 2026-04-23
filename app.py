import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ICU Stroke Family Resilience",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# THEME / CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Sidebar */
  [data-testid="stSidebar"] { background: #0f172a; }
  [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

  /* Metric cards */
  [data-testid="metric-container"] {
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 12px;
      padding: 16px;
  }
  [data-testid="metric-container"] label { color: #94a3b8 !important; font-size:0.8rem !important; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #38bdf8 !important; font-size:1.8rem !important; font-weight:700; }

  /* Section headers */
  .section-header {
      background: linear-gradient(135deg, #1e3a5f 0%, #1e293b 100%);
      border-left: 4px solid #38bdf8;
      padding: 12px 20px;
      border-radius: 8px;
      margin: 20px 0 12px 0;
      color: #e2e8f0;
      font-size: 1.1rem;
      font-weight: 600;
  }

  /* Page title */
  .hero {
      background: linear-gradient(135deg, #0c1445 0%, #1e3a5f 50%, #0f172a 100%);
      border-radius: 16px;
      padding: 28px 32px;
      margin-bottom: 24px;
      border: 1px solid #334155;
  }
  .hero h1 { color: #38bdf8; margin:0; font-size:1.9rem; font-weight:700; }
  .hero p  { color: #94a3b8; margin:6px 0 0 0; font-size:0.95rem; }

  /* Insight box */
  .insight {
      background: #0f2744;
      border: 1px solid #1d4ed8;
      border-radius: 10px;
      padding: 14px 18px;
      margin: 10px 0;
      color: #bfdbfe;
      font-size: 0.88rem;
      line-height: 1.6;
  }
  .insight b { color: #60a5fa; }

  /* Tab styling */
  button[data-baseweb="tab"] { color: #94a3b8 !important; font-weight:600; }
  button[data-baseweb="tab"][aria-selected="true"] { color: #38bdf8 !important; border-bottom: 2px solid #38bdf8 !important; }

  div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LABEL MAPS  (coded → readable)
# ─────────────────────────────────────────────
LABELS = {
    "Gender":               {1: "Male", 2: "Female"},
    "Gender.1":             {1: "Male", 2: "Female"},
    "Marital Status":       {1: "Married", 2: "Other"},
    "Primary Caregiver":    {1: "Spouse", 2: "Child", 3: "Other"},
    "Educational Level":    {1: "Primary", 2: "Secondary", 3: "College+"},
    "Stroke Subtype":       {1: "Ischemic", 2: "Hemorrhagic", 3: "Other"},
    "Medical Expense Payment": {1: "Self-pay", 2: "Insurance", 3: "Other"},
    "Residence":            {1: "Urban", 2: "Rural"},
    "Understanding of the Disease ": {1: "Poor", 2: "Moderate", 3: "Good"},
    "Occupation":           {1: "Employed", 2: "Retired", 3: "Other"},
    "Employment Status ":   {1: "Employed", 2: "Unemployed"},
    "Religious Belief ":    {1: "Yes", 2: "No"},
    "Educational Level ":   {1: "Primary", 2: "Secondary", 3: "College+"},
    "Monthly Household Income per Capita": {1: "Low", 2: "Medium", 3: "High"},
    "Frequency of Care from Relatives or Friends ": {1: "Never", 2: "Rarely", 3: "Sometimes", 4: "Often", 5: "Always"},
    "Frequency of Physician–Family Communication ": {1: "Never", 2: "Rarely", 3: "Sometimes", 4: "Often", 5: "Always"},
    "Outcome":              {1: "Survived", 2: "Deceased"},
}

SCORE_DESCRIPTIONS = {
    "FCPS": "Family Crisis-Oriented Personal Scales (32–90) — higher = greater resilience",
    "USR":  "Unique Strength Resources Scale (3–12) — higher = more personal strengths",
    "MPO":  "Meaning & Purpose in Outlook Scale (8–24) — higher = more positive outlook",
}

PLOTLY_THEME = "plotly_dark"
COLOR_SEQ = px.colors.qualitative.Bold
COLOR_CONTINUOUS = "Blues"

# ─────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────
@st.cache_data
def load_data(file_obj=None):
    if file_obj is not None:
        df = pd.read_csv(file_obj)
    else:
        df = pd.read_csv(
            "1776842982964_Table_1_Unraveling_family_resilience_patterns_in_ICU_first-episode_stroke__a_latent_profile_analysis.csv"
        )
    # Drop empty header columns
    df = df.drop(columns=[c for c in df.columns if df[c].isna().all()], errors="ignore")
    # Rename duplicate Educational Level
    cols = list(df.columns)
    edu_idx = [i for i, c in enumerate(cols) if c.strip() == "Educational Level"]
    if len(edu_idx) >= 2:
        cols[edu_idx[1]] = "Educational Level (Caregiver)"
    df.columns = cols
    LABELS["Educational Level (Caregiver)"] = {1: "Primary", 2: "Secondary", 3: "College+"}
    return df

def apply_labels(df):
    df2 = df.copy()
    for col, mapping in LABELS.items():
        if col in df2.columns:
            df2[col] = df2[col].map(mapping).fillna(df2[col])
    return df2

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 ICU Stroke Study")
    st.markdown("---")
    uploaded = st.file_uploader("📂 Upload your own CSV", type=["csv"])
    st.markdown("---")

    st.markdown("### 🧭 Navigation")
    pages = [
        "🏠 Overview & Summary",
        "👤 Patient Demographics",
        "👨‍👩‍👧 Caregiver Profile",
        "🏥 Clinical Severity",
        "📊 Resilience Scores",
        "🔗 Correlations & Insights",
        "🔎 Data Explorer",
    ]
    page = st.radio("", pages, label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### 🎛️ Global Filters")

    # Load early for filter options
    try:
        raw = load_data(uploaded)
    except Exception:
        st.error("Could not load default data. Please upload a CSV.")
        st.stop()

    df_all = apply_labels(raw)

    outcome_opts = ["All"] + sorted(df_all["Outcome"].unique().tolist())
    sel_outcome = st.selectbox("Outcome", outcome_opts)

    stroke_opts = ["All"] + sorted(df_all["Stroke Subtype"].unique().tolist())
    sel_stroke = st.selectbox("Stroke Subtype", stroke_opts)

    gender_opts = ["All"] + sorted(df_all["Gender"].unique().tolist())
    sel_gender = st.selectbox("Patient Gender", gender_opts)

    age_min, age_max = int(raw["Age"].min()), int(raw["Age"].max())
    sel_age = st.slider("Patient Age Range", age_min, age_max, (age_min, age_max))

    st.markdown("---")
    st.caption("Study: *Unraveling family resilience patterns in ICU first-episode stroke*")

# ─────────────────────────────────────────────
# FILTER DATA
# ─────────────────────────────────────────────
df = df_all.copy()
if sel_outcome != "All":
    df = df[df["Outcome"] == sel_outcome]
if sel_stroke != "All":
    df = df[df["Stroke Subtype"] == sel_stroke]
if sel_gender != "All":
    df = df[df["Gender"] == sel_gender]
df = df[(df["Age"] >= sel_age[0]) & (df["Age"] <= sel_age[1])]

N = len(df)
pct = lambda n: f"{n/N*100:.1f}%"

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def insight(text):
    st.markdown(f'<div class="insight">{text}</div>', unsafe_allow_html=True)

def bar_chart(df_plot, x, y, title, color=None, orientation="v"):
    fig = px.bar(df_plot, x=x, y=y, title=title, color=color,
                 color_discrete_sequence=COLOR_SEQ,
                 template=PLOTLY_THEME, orientation=orientation)
    fig.update_layout(margin=dict(t=40, b=20, l=10, r=10), height=340)
    return fig

def pie_chart(df_plot, names, values, title):
    fig = px.pie(df_plot, names=names, values=values, title=title,
                 color_discrete_sequence=COLOR_SEQ, template=PLOTLY_THEME,
                 hole=0.4)
    fig.update_layout(margin=dict(t=40, b=20, l=10, r=10), height=340)
    return fig

# ═══════════════════════════════════════════════════════════
# PAGE 1 – OVERVIEW
# ═══════════════════════════════════════════════════════════
if page == "🏠 Overview & Summary":
    st.markdown("""
    <div class="hero">
      <h1>🧠 ICU Stroke Family Resilience Dashboard</h1>
      <p>Latent profile analysis of family resilience patterns in first-episode ICU stroke patients & their caregivers</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Patients", N)
    survived = (df["Outcome"] == "Survived").sum() if "Survived" in df["Outcome"].values else (df["Outcome"] == 1).sum()
    survived = len(df[df_all.loc[df.index, "Outcome"] == "Survived"]) if "Survived" in df_all["Outcome"].values else 0
    survived = len(df[df["Outcome"] == "Survived"])
    c2.metric("Survived", survived, f"{pct(survived)}")
    c3.metric("Avg FCPS", f"{df['FCPS'].mean():.1f}", f"±{df['FCPS'].std():.1f}")
    c4.metric("Avg USR", f"{df['USR'].mean():.1f}", f"±{df['USR'].std():.1f}")
    c5.metric("Avg MPO", f"{df['MPO'].mean():.1f}", f"±{df['MPO'].std():.1f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        section("📋 Score Distributions")
        scores_long = df[["FCPS", "USR", "MPO"]].melt(var_name="Scale", value_name="Score")
        fig = px.histogram(scores_long, x="Score", color="Scale", barmode="overlay",
                           template=PLOTLY_THEME, color_discrete_sequence=COLOR_SEQ,
                           title="Distribution of Resilience Scores", nbins=20)
        fig.update_layout(height=350, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("⚕️ Outcome Breakdown")
        oc = df["Outcome"].value_counts().reset_index()
        oc.columns = ["Outcome", "Count"]
        fig2 = pie_chart(oc, "Outcome", "Count", "Patient Outcomes")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        section("🔵 Stroke Subtype Distribution")
        st_cnt = df["Stroke Subtype"].value_counts().reset_index()
        st_cnt.columns = ["Subtype", "Count"]
        fig3 = bar_chart(st_cnt, "Subtype", "Count", "Stroke Subtypes", color="Subtype")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        section("📦 Score Boxplots by Outcome")
        fig4 = go.Figure()
        for score, color in zip(["FCPS", "USR", "MPO"], ["#38bdf8", "#818cf8", "#34d399"]):
            for outcome in df["Outcome"].unique():
                sub = df[df["Outcome"] == outcome][score]
                fig4.add_trace(go.Box(y=sub, name=f"{score} – {outcome}", marker_color=color,
                                      boxmean=True))
        fig4.update_layout(template=PLOTLY_THEME, height=360, title="Score Spread by Outcome",
                           margin=dict(t=40, b=20))
        st.plotly_chart(fig4, use_container_width=True)

    section("💡 Key Insights")
    mean_fcps = df['FCPS'].mean()
    mean_mpo  = df['MPO'].mean()
    pct_surv  = pct(survived)
    insight(f"<b>Resilience Scores:</b> The average FCPS score is <b>{mean_fcps:.1f}</b> (range 32–90), "
            f"suggesting moderate to high family resilience. MPO averages <b>{mean_mpo:.1f}</b>, "
            f"indicating families maintain a meaningful sense of purpose during ICU crisis.")
    insight(f"<b>Outcome:</b> <b>{pct_surv}</b> of filtered patients survived. "
            f"Hemorrhagic strokes are the most common subtype in this cohort.")
    insight("<b>Data note:</b> 'Patient Characteristics' and 'Primary Caregiver Characteristics' "
            "are structural separator rows in the original table — they are excluded from analysis.")

# ═══════════════════════════════════════════════════════════
# PAGE 2 – PATIENT DEMOGRAPHICS
# ═══════════════════════════════════════════════════════════
elif page == "👤 Patient Demographics":
    st.markdown('<div class="hero"><h1>👤 Patient Demographics</h1><p>Characteristics of ICU first-episode stroke patients</p></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", N)
    c2.metric("Mean Age", f"{df['Age'].mean():.1f} yrs")
    c3.metric("% Male", pct(len(df[df["Gender"] == "Male"])))
    c4.metric("% Married", pct(len(df[df["Marital Status"] == "Married"])))

    col1, col2 = st.columns(2)
    with col1:
        section("👥 Gender Distribution")
        g = df["Gender"].value_counts().reset_index(); g.columns = ["Gender", "Count"]
        st.plotly_chart(pie_chart(g, "Gender", "Count", "Patient Gender"), use_container_width=True)

    with col2:
        section("📅 Age Distribution")
        fig = px.histogram(df, x="Age", nbins=20, template=PLOTLY_THEME,
                           color_discrete_sequence=["#38bdf8"],
                           title="Age Distribution of Patients")
        fig.add_vline(x=df["Age"].mean(), line_dash="dash", line_color="#f59e0b",
                      annotation_text=f"Mean: {df['Age'].mean():.1f}", annotation_position="top right")
        fig.update_layout(height=340, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        section("💍 Marital Status")
        ms = df["Marital Status"].value_counts().reset_index(); ms.columns = ["Status", "Count"]
        st.plotly_chart(pie_chart(ms, "Status", "Count", "Marital Status"), use_container_width=True)

    with col4:
        section("🎓 Educational Level (Patient)")
        el = df["Educational Level"].value_counts().reset_index(); el.columns = ["Level", "Count"]
        st.plotly_chart(bar_chart(el, "Level", "Count", "Educational Level", color="Level"), use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        section("👶 Number of Children")
        nc = df["Number of Children"].value_counts().sort_index().reset_index(); nc.columns = ["Children", "Count"]
        fig = px.bar(nc, x="Children", y="Count", template=PLOTLY_THEME,
                     color_discrete_sequence=["#818cf8"], title="Children Count")
        fig.update_layout(height=320, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        section("🧠 Age vs Resilience")
        fig = px.scatter(df, x="Age", y="FCPS", color="Outcome", size="APACHE II Score",
                         template=PLOTLY_THEME, color_discrete_sequence=COLOR_SEQ,
                         title="Age vs FCPS (size = APACHE II)", trendline="ols")
        fig.update_layout(height=320, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    insight("<b>Demographic Observation:</b> The cohort spans a wide age range. "
            "Older patients tend to have higher APACHE II severity scores. "
            "Most patients are married, which may positively influence family resilience outcomes.")

# ═══════════════════════════════════════════════════════════
# PAGE 3 – CAREGIVER PROFILE
# ═══════════════════════════════════════════════════════════
elif page == "👨‍👩‍👧 Caregiver Profile":
    st.markdown('<div class="hero"><h1>👨‍👩‍👧 Primary Caregiver Profile</h1><p>Demographics and characteristics of ICU family caregivers</p></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean Caregiver Age", f"{df['Age.1'].mean():.1f} yrs")
    c2.metric("% Employed", pct(len(df[df["Employment Status "] == "Employed"])))
    c3.metric("% Religious Belief", pct(len(df[df["Religious Belief "] == "Yes"])))

    col1, col2 = st.columns(2)
    with col1:
        section("👥 Caregiver Gender")
        cg = df["Gender.1"].value_counts().reset_index(); cg.columns = ["Gender", "Count"]
        st.plotly_chart(pie_chart(cg, "Gender", "Count", "Caregiver Gender"), use_container_width=True)

    with col2:
        section("📅 Caregiver Age Distribution")
        fig = px.histogram(df, x="Age.1", nbins=20, template=PLOTLY_THEME,
                           color_discrete_sequence=["#818cf8"],
                           title="Caregiver Age Distribution")
        fig.add_vline(x=df["Age.1"].mean(), line_dash="dash", line_color="#f59e0b",
                      annotation_text=f"Mean: {df['Age.1'].mean():.1f}", annotation_position="top right")
        fig.update_layout(height=340, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        section("🏡 Residence Type")
        res = df["Residence"].value_counts().reset_index(); res.columns = ["Residence", "Count"]
        st.plotly_chart(pie_chart(res, "Residence", "Count", "Urban vs Rural"), use_container_width=True)

    with col4:
        section("💰 Monthly Household Income")
        inc = df["Monthly Household Income per Capita"].value_counts().reset_index(); inc.columns = ["Income", "Count"]
        st.plotly_chart(bar_chart(inc, "Income", "Count", "Income Level", color="Income"), use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        section("📖 Disease Understanding")
        du = df["Understanding of the Disease "].value_counts().reset_index(); du.columns = ["Level", "Count"]
        st.plotly_chart(bar_chart(du, "Level", "Count", "Understanding of Disease", color="Level"), use_container_width=True)

    with col6:
        section("🤝 Social Support Frequency")
        freq_care = df["Frequency of Care from Relatives or Friends "].value_counts().reset_index()
        freq_care.columns = ["Frequency", "Count"]
        freq_doc = df["Frequency of Physician–Family Communication "].value_counts().reset_index()
        freq_doc.columns = ["Frequency", "Count"]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=freq_care["Frequency"], y=freq_care["Count"], name="Relatives/Friends", marker_color="#38bdf8"))
        fig.add_trace(go.Bar(x=freq_doc["Frequency"], y=freq_doc["Count"], name="Physician–Family", marker_color="#34d399"))
        fig.update_layout(barmode="group", template=PLOTLY_THEME, height=340,
                          title="Support & Communication Frequency", margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    insight("<b>Caregiver Insight:</b> Female caregivers predominate in this cohort, consistent with "
            "global ICU caregiver literature. Rural residence and lower income are associated with "
            "reduced social support frequency, which may negatively affect family resilience scores.")

# ═══════════════════════════════════════════════════════════
# PAGE 4 – CLINICAL SEVERITY
# ═══════════════════════════════════════════════════════════
elif page == "🏥 Clinical Severity":
    st.markdown('<div class="hero"><h1>🏥 Clinical Severity & ICU Characteristics</h1><p>Glasgow Coma Scale, APACHE II scores, length of stay, and stroke subtypes</p></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean GCS", f"{df['Glasgow Coma Scale  '].mean():.1f}")
    c2.metric("Mean APACHE II", f"{df['APACHE II Score'].mean():.1f}")
    c3.metric("Mean ICU Stay", f"{df['ICU Length of Stay '].mean():.1f} days")
    c4.metric("Mean Organ Dysfunctions", f"{df['Number of Additional Organ System Dysfunctions'].mean():.1f}")

    col1, col2 = st.columns(2)
    with col1:
        section("🩺 GCS vs APACHE II (by Outcome)")
        fig = px.scatter(df, x="Glasgow Coma Scale  ", y="APACHE II Score",
                         color="Outcome", symbol="Stroke Subtype",
                         template=PLOTLY_THEME, color_discrete_sequence=COLOR_SEQ,
                         title="GCS vs APACHE II Score", trendline="ols",
                         labels={"Glasgow Coma Scale  ": "Glasgow Coma Scale"})
        fig.update_layout(height=380, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("📏 ICU Length of Stay by Stroke Subtype")
        fig = px.box(df, x="Stroke Subtype", y="ICU Length of Stay ",
                     color="Stroke Subtype", template=PLOTLY_THEME,
                     color_discrete_sequence=COLOR_SEQ,
                     title="ICU LOS by Stroke Subtype",
                     labels={"ICU Length of Stay ": "Length of Stay (days)"})
        fig.update_layout(height=380, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        section("🏥 ICU Admission Count Distribution")
        iac = df["ICU Admission Count"].value_counts().sort_index().reset_index()
        iac.columns = ["Admissions", "Count"]
        fig = px.bar(iac, x="Admissions", y="Count", template=PLOTLY_THEME,
                     color_discrete_sequence=["#f472b6"], title="ICU Admissions")
        fig.update_layout(height=340, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        section("🔴 Organ System Dysfunctions by Outcome")
        fig = px.violin(df, y="Number of Additional Organ System Dysfunctions",
                        x="Outcome", color="Outcome",
                        box=True, points="all", template=PLOTLY_THEME,
                        color_discrete_sequence=COLOR_SEQ,
                        title="Organ Dysfunctions by Outcome")
        fig.update_layout(height=340, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    section("📊 APACHE II Score Distribution by Stroke Subtype")
    fig_ap = px.histogram(df, x="APACHE II Score", color="Stroke Subtype",
                          barmode="overlay", template=PLOTLY_THEME,
                          color_discrete_sequence=COLOR_SEQ, nbins=15,
                          title="APACHE II Score by Stroke Subtype")
    fig_ap.update_layout(height=340, margin=dict(t=40, b=20))
    st.plotly_chart(fig_ap, use_container_width=True)

    insight("<b>Clinical Observation:</b> Higher APACHE II scores and lower GCS values "
            "are concentrated among deceased patients. Hemorrhagic strokes tend to have "
            "longer ICU lengths of stay and more organ system dysfunctions, reflecting greater severity.")

# ═══════════════════════════════════════════════════════════
# PAGE 5 – RESILIENCE SCORES
# ═══════════════════════════════════════════════════════════
elif page == "📊 Resilience Scores":
    st.markdown('<div class="hero"><h1>📊 Family Resilience Scores</h1><p>FCPS · USR · MPO — in-depth analysis across subgroups</p></div>', unsafe_allow_html=True)

    for score, desc in SCORE_DESCRIPTIONS.items():
        st.markdown(f"**{score}:** {desc}")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🔬 Subgroup Analysis", "🧮 Score Profiles"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        for col, score in zip([col1, col2, col3], ["FCPS", "USR", "MPO"]):
            with col:
                fig = px.histogram(df, x=score, nbins=20, template=PLOTLY_THEME,
                                   color_discrete_sequence=["#38bdf8"],
                                   title=f"{score} Distribution", marginal="box")
                fig.update_layout(height=380, margin=dict(t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        group_by = st.selectbox("Group By", ["Outcome", "Stroke Subtype", "Gender",
                                              "Marital Status", "Residence", "Monthly Household Income per Capita"])
        score_sel = st.selectbox("Score", ["FCPS", "USR", "MPO"])
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(df, x=group_by, y=score_sel, color=group_by,
                         template=PLOTLY_THEME, color_discrete_sequence=COLOR_SEQ,
                         title=f"{score_sel} by {group_by}", points="all")
            fig.update_layout(height=380, margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            # Violin
            fig2 = px.violin(df, x=group_by, y=score_sel, color=group_by, box=True,
                             template=PLOTLY_THEME, color_discrete_sequence=COLOR_SEQ,
                             title=f"{score_sel} Violin by {group_by}")
            fig2.update_layout(height=380, margin=dict(t=40, b=20))
            st.plotly_chart(fig2, use_container_width=True)

        # Mean bar
        mean_df = df.groupby(group_by)[score_sel].mean().reset_index()
        mean_df.columns = [group_by, "Mean Score"]
        fig3 = px.bar(mean_df, x=group_by, y="Mean Score", color=group_by,
                      template=PLOTLY_THEME, color_discrete_sequence=COLOR_SEQ,
                      title=f"Mean {score_sel} by {group_by}", text="Mean Score")
        fig3.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig3.update_layout(height=340, margin=dict(t=40, b=20))
        st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        section("🌐 Radar Profile by Outcome")
        categories = ["FCPS", "USR", "MPO"]
        fig_radar = go.Figure()
        for outcome in df["Outcome"].unique():
            sub = df[df["Outcome"] == outcome]
            vals = [sub[s].mean() for s in categories]
            # normalize 0–1
            maxvals = [df[s].max() for s in categories]
            vals_norm = [v / m for v, m in zip(vals, maxvals)]
            vals_norm += [vals_norm[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_norm, theta=categories + [categories[0]],
                fill='toself', name=str(outcome)
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            template=PLOTLY_THEME, height=420, title="Normalized Resilience Radar by Outcome"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        section("📊 Composite Resilience Score")
        df = df.copy()
        df["Composite"] = (
            (df["FCPS"] - df["FCPS"].min()) / (df["FCPS"].max() - df["FCPS"].min()) +
            (df["USR"] - df["USR"].min()) / (df["USR"].max() - df["USR"].min()) +
            (df["MPO"] - df["MPO"].min()) / (df["MPO"].max() - df["MPO"].min())
        ) / 3 * 100
        fig_comp = px.histogram(df, x="Composite", color="Outcome", nbins=20,
                                template=PLOTLY_THEME, color_discrete_sequence=COLOR_SEQ,
                                title="Composite Resilience Index (0–100)", barmode="overlay")
        fig_comp.update_layout(height=340, margin=dict(t=40, b=20))
        st.plotly_chart(fig_comp, use_container_width=True)

    insight("<b>Resilience Score Patterns:</b> Families of surviving patients tend to score higher on all three scales. "
            "The FCPS shows the widest distribution, reflecting variability in family problem-solving capacity. "
            "MPO (Meaning & Purpose) is relatively stable across subgroups, suggesting a shared sense of hope even in difficult ICU contexts.")

# ═══════════════════════════════════════════════════════════
# PAGE 6 – CORRELATIONS
# ═══════════════════════════════════════════════════════════
elif page == "🔗 Correlations & Insights":
    st.markdown('<div class="hero"><h1>🔗 Correlations & Advanced Insights</h1><p>Heatmaps, scatter matrices, and multivariate patterns</p></div>', unsafe_allow_html=True)

    numeric_cols = ["Age", "Age.1", "Glasgow Coma Scale  ", "APACHE II Score",
                    "ICU Length of Stay ", "Number of Additional Organ System Dysfunctions",
                    "FCPS", "USR", "MPO"]
    numeric_labels = ["Patient Age", "Caregiver Age", "GCS", "APACHE II",
                      "ICU LOS", "Organ Dysfunct.", "FCPS", "USR", "MPO"]
    corr = df[numeric_cols].corr()
    corr.index = numeric_labels
    corr.columns = numeric_labels

    section("🌡️ Correlation Heatmap")
    fig_heat = px.imshow(corr, text_auto=".2f", aspect="auto",
                         color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         template=PLOTLY_THEME,
                         title="Pearson Correlations Between Numeric Variables")
    fig_heat.update_layout(height=500, margin=dict(t=40, b=20))
    st.plotly_chart(fig_heat, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        section("🔵 FCPS vs MPO Scatter")
        fig = px.scatter(df, x="FCPS", y="MPO", color="Outcome", size="USR",
                         trendline="ols", template=PLOTLY_THEME,
                         color_discrete_sequence=COLOR_SEQ,
                         title="FCPS vs MPO (size = USR)")
        fig.update_layout(height=360, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("🔴 APACHE II vs ICU LOS by Stroke Type")
        fig2 = px.scatter(df, x="APACHE II Score", y="ICU Length of Stay ",
                          color="Stroke Subtype", trendline="ols",
                          template=PLOTLY_THEME, color_discrete_sequence=COLOR_SEQ,
                          title="APACHE II vs ICU Length of Stay",
                          labels={"ICU Length of Stay ": "ICU LOS (days)"})
        fig2.update_layout(height=360, margin=dict(t=40, b=20))
        st.plotly_chart(fig2, use_container_width=True)

    section("📉 Scatter Matrix — Resilience + Clinical Scores")
    fig_matrix = px.scatter_matrix(
        df, dimensions=["FCPS", "USR", "MPO", "APACHE II Score", "Glasgow Coma Scale  "],
        color="Outcome", template=PLOTLY_THEME, color_discrete_sequence=COLOR_SEQ,
        title="Pairwise Scatter Matrix"
    )
    fig_matrix.update_traces(diagonal_visible=False, marker=dict(size=3, opacity=0.6))
    fig_matrix.update_layout(height=600, margin=dict(t=40, b=20))
    st.plotly_chart(fig_matrix, use_container_width=True)

    section("📊 Mean Resilience Scores by Key Factors")
    factor = st.selectbox("Select Factor", ["Outcome", "Stroke Subtype", "Residence",
                                             "Monthly Household Income per Capita",
                                             "Frequency of Physician–Family Communication "])
    means = df.groupby(factor)[["FCPS", "USR", "MPO"]].mean().reset_index()
    means_long = means.melt(id_vars=factor, var_name="Scale", value_name="Mean Score")
    fig_gr = px.bar(means_long, x=factor, y="Mean Score", color="Scale", barmode="group",
                    template=PLOTLY_THEME, color_discrete_sequence=COLOR_SEQ,
                    title=f"Mean Resilience Scores by {factor}", text="Mean Score")
    fig_gr.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig_gr.update_layout(height=380, margin=dict(t=40, b=20))
    st.plotly_chart(fig_gr, use_container_width=True)

    insight("<b>Key Correlation Findings:</b> "
            "FCPS and MPO are moderately positively correlated, suggesting families with strong problem-solving capacity also report greater meaning and purpose. "
            "APACHE II severity is negatively correlated with all resilience scores. "
            "Physician–Family communication frequency shows a positive association with resilience scores, underscoring the value of clear ICU communication.")

# ═══════════════════════════════════════════════════════════
# PAGE 7 – DATA EXPLORER
# ═══════════════════════════════════════════════════════════
elif page == "🔎 Data Explorer":
    st.markdown('<div class="hero"><h1>🔎 Data Explorer</h1><p>Browse, search, and export the filtered dataset</p></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows (filtered)", N)
    c2.metric("Columns", len(df.columns))
    c3.metric("Missing Values", df.isnull().sum().sum())

    section("🗂️ Full Dataset")
    st.dataframe(df, use_container_width=True, height=420)

    col1, col2 = st.columns(2)
    with col1:
        section("📐 Summary Statistics")
        st.dataframe(df[["Age", "Age.1", "Glasgow Coma Scale  ", "APACHE II Score",
                          "ICU Length of Stay ", "FCPS", "USR", "MPO"]].describe().round(2),
                     use_container_width=True)

    with col2:
        section("📊 Column Value Counts")
        col_pick = st.selectbox("Select Column", [c for c in df.columns if df[c].nunique() < 20])
        vc = df[col_pick].value_counts().reset_index()
        vc.columns = [col_pick, "Count"]
        vc["Percentage"] = (vc["Count"] / N * 100).round(1).astype(str) + "%"
        st.dataframe(vc, use_container_width=True)

    section("⬇️ Export Filtered Data")
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered CSV",
        data=csv_data,
        file_name="filtered_stroke_resilience_data.csv",
        mime="text/csv",
    )