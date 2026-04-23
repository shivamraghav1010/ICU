import streamlit as st
import pandas as pd
import plotly.express as px
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
# 🔥 FUNCTION: MAKE COLUMN NAMES UNIQUE
# ─────────────────────────────────────────────
def make_unique_columns(cols):
    seen = {}
    new_cols = []

    for col in cols:
        col = col.strip()
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)

    return new_cols

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
    except Exception:
        st.error("❌ Failed to load dataset")
        st.stop()

    # ✅ Clean column names
    df.columns = df.columns.str.strip()

    # ✅ 🔥 FORCE UNIQUE COLUMN NAMES (MAIN FIX)
    df.columns = make_unique_columns(df.columns)

    # ✅ Remove unwanted separator columns
    df = df.drop(columns=[
        "Patient Characteristics",
        "Primary Caregiver Characteristics"
    ], errors="ignore")

    # ✅ Drop fully empty columns safely
    df = df.loc[:, ~df.isna().all()]

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

    # Survival vs Severity
    if "APACHE II Score" in df.columns and "Outcome" in df.columns:
        fig = px.box(df, x="Outcome", y="APACHE II Score")
        st.plotly_chart(fig, use_container_width=True)

    # Correlation Heatmap
    numeric_df = df.select_dtypes(include='number')

    if not numeric_df.empty:
        corr = numeric_df.corr()
        fig = px.imshow(corr, text_auto=True)
        st.plotly_chart(fig, use_container_width=True)

    # Resilience Scores
    for score in ["FCPS", "USR", "MPO"]:
        if score in df.columns:
            fig = px.box(df, x="Outcome", y=score)
            st.plotly_chart(fig, use_container_width=True)

    # Feature Importance
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

        fig = px.bar(importance, x="Importance", y="Feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 3: DATA EXPLORER
# ─────────────────────────────────────────────
elif page == "Data Explorer":
    st.title("🔎 Data Explorer")

    st.dataframe(df, use_container_width=True)
    st.write(df.describe())
