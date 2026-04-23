import streamlit as st
import pandas as pd
import plotly.express as px

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ICU Stroke Family Resilience",
    page_icon="🧠",
    layout="wide",
)

# ─────────────────────────────────────────────
# DATA LOADER (FIXED ERROR HERE)
# ─────────────────────────────────────────────
@st.cache_data
def load_data(file_obj=None):
    try:
        if file_obj is not None:
            df = pd.read_csv(file_obj)
        else:
            df = pd.read_csv("data.csv")
    except Exception:
        st.error("❌ Failed to load data. Please upload a valid CSV.")
        st.stop()

    # ✅ Clean column names
    df.columns = df.columns.str.strip()

    # ✅ Remove unwanted separator columns
    df = df.drop(columns=[
        "Patient Characteristics",
        "Primary Caregiver Characteristics"
    ], errors="ignore")

    # ✅ FIX: handle duplicate columns safely
    cols_to_drop = []
    for col in df.columns:
        col_data = df.loc[:, col]

        # If duplicate → returns DataFrame
        if isinstance(col_data, pd.DataFrame):
            if col_data.isna().all().all():
                cols_to_drop.append(col)
        else:
            if col_data.isna().all():
                cols_to_drop.append(col)

    df = df.drop(columns=cols_to_drop, errors="ignore")

    # Rename duplicate Educational Level
    cols = list(df.columns)
    edu_idx = [i for i, c in enumerate(cols) if c == "Educational Level"]
    if len(edu_idx) >= 2:
        cols[edu_idx[1]] = "Educational Level (Caregiver)"
    df.columns = cols

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

    pages = ["Overview", "Demographics", "Explorer"]

    # ✅ FIX: label cannot be empty
    page = st.radio("Navigation", pages, label_visibility="collapsed")


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
raw = load_data(uploaded)
df = raw.copy()

# ─────────────────────────────────────────────
# PAGE 1: OVERVIEW
# ─────────────────────────────────────────────
if page == "Overview":
    st.title("📊 Overview")

    st.metric("Total Records", len(df))

    if "Age" in df.columns:
        st.metric("Average Age", f"{df['Age'].mean():.1f}")

        fig = px.histogram(df, x="Age", nbins=20)
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 2: DEMOGRAPHICS
# ─────────────────────────────────────────────
elif page == "Demographics":
    st.title("👤 Demographics")

    if "Gender" in df.columns:
        fig = px.pie(df, names="Gender")
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 3: DATA EXPLORER
# ─────────────────────────────────────────────
elif page == "Explorer":
    st.title("🔎 Data Explorer")

    st.dataframe(df, use_container_width=True)
