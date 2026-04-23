import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ICU Stroke Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ─────────────────────────────────────────────
# DATA LOADER (FULLY FIXED)
# ─────────────────────────────────────────────
@st.cache_data
def load_data(file_obj=None):
    try:
        if file_obj:
            df = pd.read_csv(file_obj)
        else:
            df = pd.read_csv("data.csv")
    except:
        st.error("❌ Failed to load dataset")
        st.stop()

    # Clean column names
    df.columns = df.columns.str.strip()

    # Drop unwanted columns
    df = df.drop(columns=[
        "Patient Characteristics",
        "Primary Caregiver Characteristics"
    ], errors="ignore")

    # Handle duplicate columns safely
    cols_to_drop = []
    for col in df.columns:
        col_data = df.loc[:, col]

        if isinstance(col_data, pd.DataFrame):
            if col_data.isna().all().all():
                cols_to_drop.append(col)
        else:
            if col_data.isna().all():
                cols_to_drop.append(col)

    df = df.drop(columns=cols_to_drop, errors="ignore")

    return df


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 ICU Dashboard")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        st.success("Using uploaded dataset")
    else:
        st.info("Using default dataset (data.csv)")

    pages = ["Overview", "Advanced Analysis", "Data Explorer"]

    page = st.radio("Navigation", pages, label_visibility="collapsed")

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df = load_data(uploaded)

# ─────────────────────────────────────────────
# PAGE 1: OVERVIEW
# ─────────────────────────────────────────────
if page == "Overview":
    st.title("📊 ICU Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(df))

    if "Age" in df.columns:
        col2.metric("Avg Age", f"{df['Age'].mean():.1f}")

    if "Outcome" in df.columns:
        survival_rate = (df["Outcome"] == 1).mean() * 100
        col3.metric("Survival Rate", f"{survival_rate:.1f}%")

    st.markdown("---")

    if "Age" in df.columns:
        fig = px.histogram(df, x="Age", nbins=20, title="Age Distribution")
        st.plotly_chart(fig, use_container_width=True)

    if "Gender" in df.columns:
        fig = px.pie(df, names="Gender", title="Gender Distribution")
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 2: ADVANCED ANALYSIS
# ─────────────────────────────────────────────
elif page == "Advanced Analysis":
    st.title("🚀 Advanced Analysis")

    # 🔹 Survival vs Severity
    st.subheader("🧠 Survival vs Severity")

    col1, col2 = st.columns(2)

    if "APACHE II Score" in df.columns:
        fig = px.box(df, x="Outcome", y="APACHE II Score",
                     title="APACHE II vs Outcome")
        col1.plotly_chart(fig, use_container_width=True)

    if "Glasgow Coma Scale" in df.columns:
        fig = px.box(df, x="Outcome", y="Glasgow Coma Scale",
                     title="GCS vs Outcome")
        col2.plotly_chart(fig, use_container_width=True)

    # 🔹 ICU Stay vs Severity
    st.subheader("🏥 ICU Stay vs Severity")

    if "ICU Length of Stay" in df.columns and "APACHE II Score" in df.columns:
        fig = px.scatter(df,
                         x="APACHE II Score",
                         y="ICU Length of Stay",
                         color="Outcome",
                         trendline="ols")
        st.plotly_chart(fig, use_container_width=True)

    # 🔹 Correlation Heatmap
    st.subheader("🔥 Correlation Heatmap")

    numeric_df = df.select_dtypes(include='number')

    if not numeric_df.empty:
        corr = numeric_df.corr()

        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r"
        )
        st.plotly_chart(fig, use_container_width=True)

    # 🔹 Resilience Scores
    st.subheader("💪 Resilience Scores")

    for score in ["FCPS", "USR", "MPO"]:
        if score in df.columns:
            fig = px.box(df, x="Outcome", y=score,
                         title=f"{score} vs Outcome")
            st.plotly_chart(fig, use_container_width=True)

    # 🔹 Composite Score
    st.subheader("📊 Composite Score")

    if all(col in df.columns for col in ["FCPS", "USR", "MPO"]):
        df["Composite"] = (
            (df["FCPS"] - df["FCPS"].min()) / (df["FCPS"].max() - df["FCPS"].min()) +
            (df["USR"] - df["USR"].min()) / (df["USR"].max() - df["USR"].min()) +
            (df["MPO"] - df["MPO"].min()) / (df["MPO"].max() - df["MPO"].min())
        ) / 3 * 100

        fig = px.histogram(df, x="Composite", color="Outcome")
        st.plotly_chart(fig, use_container_width=True)

    # 🔹 Feature Importance (ML)
    st.subheader("🤖 Feature Importance")

    df_ml = df.select_dtypes(include='number').dropna()

    if "Outcome" in df_ml.columns and len(df_ml) > 5:
        X = df_ml.drop("Outcome", axis=1)
        y = df_ml["Outcome"]

        model = RandomForestClassifier()
        model.fit(X, y)

        importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        fig = px.bar(importance, x="Importance", y="Feature",
                     orientation="h")
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 3: DATA EXPLORER
# ─────────────────────────────────────────────
elif page == "Data Explorer":
    st.title("🔎 Data Explorer")

    st.dataframe(df, use_container_width=True)

    st.subheader("Summary Statistics")
    st.write(df.describe())
