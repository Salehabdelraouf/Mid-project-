
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="FIFA 21 Player Explorer",
    page_icon="⚽",
    layout="wide",
)

DATA_PATH = r"C:\courses\Data Science course\sessions\Mid_Project\data\fifa21 raw data v2.csv"


# ----------------------------------------------------------------------------
# Data loading & cleaning (mirrors the cleaning notebook)
# ----------------------------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    # Drop columns not needed for analysis
    df = df.drop(["ID", "photoUrl", "playerUrl"], axis=1)

    # Clean Club
    df["Club"] = df["Club"].str.strip()
    df["Club"] = df["Club"].replace("No Club", np.nan)

    # Extract Contract_Start / Contract_End, then drop raw Contract
    def contract_start_year(x):
        x = str(x).strip()
        if "~" in x:
            return int(x.split("~")[0].strip())
        return np.nan

    def contract_end_year(x):
        x = str(x).strip()
        if "~" in x:
            return int(x.split("~")[1].strip())
        return np.nan

    df["Contract_Start"] = df["Contract"].apply(contract_start_year).astype("Int64")
    df["Contract_End"] = df["Contract"].apply(contract_end_year).astype("Int64")
    df = df.drop(["Contract", "Loan Date End"], axis=1)

    # Height / Weight: strip units, convert imperial rows, cast to numeric
    def height_to_cm(x):
        x = str(x).strip()
        if x.endswith("cm"):
            return int(x[:-2])
        if "'" in x:
            ft, inch = x.replace('"', "").split("'")
            return int(round(int(ft) * 30.48 + int(inch) * 2.54))
        return np.nan

    def weight_to_kg(x):
        x = str(x).strip()
        if x.endswith("kg"):
            return int(x[:-2])
        if x.endswith("lbs"):
            return int(round(int(x[:-3]) * 0.453592))
        return np.nan

    df["Height"] = df["Height"].apply(height_to_cm)
    df["Weight"] = df["Weight"].apply(weight_to_kg)

    # Rename OVA
    df = df.rename(columns={"↓OVA": "OVA"})

    # Money columns: strip currency/suffix, convert to numeric (in units of money, not K/M)
    def money_to_number(x):
        x = str(x).replace("€", "").strip()
        if x.endswith("M"):
            return float(x[:-1]) * 1_000_000
        if x.endswith("K"):
            return float(x[:-1]) * 1_000
        try:
            return float(x)
        except ValueError:
            return np.nan

    for col in ["Value", "Wage", "Release Clause"]:
        df[col] = df[col].apply(money_to_number)

    # Star ratings
    for col in ["W/F", "SM", "IR"]:
        df[col] = df[col].str.replace("★", "", regex=False).str.strip().astype(int)

    # Joined date
    df["Joined"] = pd.to_datetime(df["Joined"], format="%b %d, %Y", errors="coerce")

    # Hits
    def hits_to_number(x):
        if pd.isna(x):
            return np.nan
        x = str(x).strip()
        if x.endswith("K"):
            return float(x[:-1]) * 1000
        try:
            return float(x)
        except ValueError:
            return np.nan

    df["Hits"] = df["Hits"].apply(hits_to_number)

    # Growth potential
    df["difference_OVA_POT"] = df["POT"] - df["OVA"]

    return df


df = load_data(DATA_PATH)

# ----------------------------------------------------------------------------
# Sidebar filters
# ----------------------------------------------------------------------------
st.sidebar.header("Filters")

nationalities = sorted(df["Nationality"].dropna().unique())
selected_nats = st.sidebar.multiselect("Nationality", nationalities)

clubs = sorted(df["Club"].dropna().unique())
selected_clubs = st.sidebar.multiselect("Club", clubs)

positions = sorted(df["Best Position"].dropna().unique())
selected_positions = st.sidebar.multiselect("Best Position", positions)

age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))

ova_min, ova_max = int(df["OVA"].min()), int(df["OVA"].max())
ova_range = st.sidebar.slider("Overall Rating (OVA) range", ova_min, ova_max, (ova_min, ova_max))

filtered = df[
    df["Age"].between(*age_range) & df["OVA"].between(*ova_range)
]
if selected_nats:
    filtered = filtered[filtered["Nationality"].isin(selected_nats)]
if selected_clubs:
    filtered = filtered[filtered["Club"].isin(selected_clubs)]
if selected_positions:
    filtered = filtered[filtered["Best Position"].isin(selected_positions)]

st.sidebar.caption(f"{len(filtered):,} of {len(df):,} players match the current filters")

# ----------------------------------------------------------------------------
# Header & KPIs
# ----------------------------------------------------------------------------
st.title("⚽ FIFA 21 Player Explorer")
st.caption("Explore, compare and rank FIFA 21 players — filter from the sidebar.")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Players", f"{len(filtered):,}")
k2.metric("Avg Age", f"{filtered['Age'].mean():.1f}")
k3.metric("Avg OVA", f"{filtered['OVA'].mean():.1f}")
k4.metric("Avg Value", f"€{filtered['Value'].mean():,.0f}")
k5.metric("Nationalities", f"{filtered['Nationality'].nunique():,}")

st.divider()

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_overview, tab_players, tab_clubs_nat, tab_money, tab_correlation, tab_data = st.tabs(
    ["Overview", "Top Players", "Clubs & Nationalities", "Value & Wages", "Correlations", "Raw Data"]
)

# --- Overview ---------------------------------------------------------------
with tab_overview:
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(filtered, x="Age", nbins=30, title="Age Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(filtered, x="OVA", nbins=30, title="Overall Rating Distribution")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        foot_counts = filtered["Preferred Foot"].value_counts().reset_index()
        foot_counts.columns = ["Preferred Foot", "Players"]
        fig = px.bar(foot_counts, x="Preferred Foot", y="Players", title="Left vs Right Foot")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        avg_ova_foot = (
            filtered.groupby("Preferred Foot")["OVA"].mean().reset_index().round(2)
        )
        fig = px.bar(avg_ova_foot, x="Preferred Foot", y="OVA", title="Average OVA by Preferred Foot")
        st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        filtered[filtered["OVA"] < filtered["POT"]],
        x="OVA",
        y="POT",
        hover_name="Name",
        opacity=0.3,
        title="Players with Growth Potential (POT > OVA)",
    )
    fig.add_shape(type="line", x0=40, y0=40, x1=100, y1=100, line=dict(color="red", dash="dash"))
    st.plotly_chart(fig, use_container_width=True)

# --- Top Players --------------------------------------------------------
with tab_players:
    n = st.slider("How many players to show", 5, 25, 10, key="n_players")

    col1, col2 = st.columns(2)

    with col1:
        top_rated = filtered.sort_values("OVA", ascending=False)[
            ["Name", "OVA", "Age", "Club", "Best Position"]
        ].head(n)
        fig = px.bar(top_rated, x="Name", y="OVA", color="Best Position", title=f"Top {n} Highest-Rated Players")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(top_rated, use_container_width=True, hide_index=True)

    with col2:
        top_potential = filtered.sort_values("POT", ascending=False)[
            ["Name", "POT", "Age", "Club", "Best Position"]
        ].head(n)
        fig = px.bar(top_potential, x="Name", y="POT", color="Best Position", title=f"Top {n} Highest-Potential Players")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(top_potential, use_container_width=True, hide_index=True)

    st.subheader("Biggest Growth Gap (POT − OVA)")
    top_growth = filtered.sort_values("difference_OVA_POT", ascending=False)[
        ["Name", "Age", "OVA", "POT", "difference_OVA_POT", "Club"]
    ].head(n)
    fig = px.bar(
        top_growth.sort_values("difference_OVA_POT"),
        x="difference_OVA_POT",
        y="Name",
        orientation="h",
        color="difference_OVA_POT",
        color_continuous_scale="Greens",
        title=f"Top {n} Players by Growth Potential",
        labels={"difference_OVA_POT": "POT − OVA"},
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Clubs & Nationalities ------------------------------------------------
with tab_clubs_nat:
    n = st.slider("How many to show", 5, 25, 10, key="n_clubs_nat")

    col1, col2 = st.columns(2)

    with col1:
        top_nat = filtered["Nationality"].value_counts().head(n).reset_index()
        top_nat.columns = ["Nationality", "Players"]
        fig = px.bar(top_nat, x="Nationality", y="Players", title=f"Top {n} Nationalities by Player Count")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        top_clubs = (
            filtered[filtered["Club"].notna()]["Club"].value_counts().head(n).reset_index()
        )
        top_clubs.columns = ["Club", "Players"]
        fig = px.bar(top_clubs, x="Players", y="Club", orientation="h", title=f"Top {n} Clubs by Player Count")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        avg_ova_nat = (
            filtered.groupby("Nationality")["OVA"].mean().sort_values(ascending=False).head(n).reset_index().round(2)
        )
        fig = px.bar(avg_ova_nat, x="Nationality", y="OVA", title=f"Top {n} Nationalities by Average OVA")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        avg_ova_club = (
            filtered[filtered["Club"].notna()]
            .groupby("Club")["OVA"]
            .mean()
            .sort_values(ascending=False)
            .head(n)
            .reset_index()
            .round(2)
        )
        fig = px.bar(avg_ova_club, x="Club", y="OVA", title=f"Top {n} Clubs by Average OVA")
        st.plotly_chart(fig, use_container_width=True)

# --- Value & Wages ----------------------------------------------------------
with tab_money:
    n = st.slider("How many to show", 5, 25, 10, key="n_money")

    col1, col2 = st.columns(2)

    with col1:
        top_value = (
            filtered.sort_values("Value", ascending=False)[["Name", "Value", "OVA"]].head(n)
        )
        fig = px.bar(top_value, x="Value", y="Name", orientation="h", title=f"Top {n} Most Valuable Players")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        top_wage = filtered.sort_values("Wage", ascending=False)[["Name", "Wage", "OVA"]].head(n)
        fig = px.bar(top_wage, x="Wage", y="Name", orientation="h", title=f"Top {n} Highest-Paid Players")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        top_club_wage = (
            filtered[filtered["Club"].notna()]
            .groupby("Club")["Wage"]
            .sum()
            .sort_values(ascending=False)
            .head(n)
            .reset_index()
        )
        fig = px.bar(top_club_wage, x="Wage", y="Club", orientation="h", title=f"Top {n} Clubs by Total Wage Bill")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.scatter(
            filtered,
            x="OVA",
            y="Value",
            hover_name="Name",
            opacity=0.3,
            title="Overall Rating vs Market Value",
            log_y=True,
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Correlations -----------------------------------------------------------
with tab_correlation:
    core_stats = [c for c in ["OVA", "POT", "PAC", "SHO", "PAS", "DRI", "DEF", "PHY"] if c in filtered.columns]
    corr = filtered[core_stats].corr().round(2)
    fig = px.imshow(corr, text_auto=True, title="Correlation Between Core Stats")
    st.plotly_chart(fig, use_container_width=True)

    correlation_ova = (
        filtered.corr(numeric_only=True)["OVA"].sort_values(ascending=False).round(2)
    )
    correlation_ova = correlation_ova.drop("OVA", errors="ignore").head(15)
    fig = px.bar(
        x=correlation_ova.values,
        y=correlation_ova.index,
        orientation="h",
        title="Attributes Most Strongly Correlated with OVA",
        labels={"x": "Correlation", "y": "Attribute"},
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Raw data -----------------------------------------------------------
with tab_data:
    st.caption("Filtered dataset, cleaned the same way as the analysis notebook.")
    st.dataframe(filtered, use_container_width=True)
    st.download_button(
        "Download filtered data as CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="fifa21_filtered.csv",
        mime="text/csv",
    )
