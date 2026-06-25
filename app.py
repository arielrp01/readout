"""
Pharma Pipeline Intelligence
----------------------------
Traces the drug development pipeline from clinical trial to FDA approval.
Data sources: ClinicalTrials.gov API v2 + openFDA Drugs API 
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date
import time

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Pharma Pipeline Intelligence",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# THEME / STYLE
# ─────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Palette ─────────────────────────────── */
  :root {
    --ink:       #1a2332;
    --paper:     #f7f9fb;
    --slate:     #334e5e;
    --accent:    #0d9488;       /* clinical teal */
    --success:   #16a34a;
    --warning:   #d97706;
    --danger:    #dc2626;
    --border:    #d1dae2;
    --muted:     #64748b;
    --card-bg:   #ffffff;
  }

  /* ── Typography ──────────────────────────── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--ink);
    background: var(--paper);
  }

  /* ── Header ──────────────────────────────── */
  .app-header {
    border-bottom: 2px solid var(--slate);
    padding-bottom: 0.75rem;
    margin-bottom: 1.5rem;
  }
  .app-header h1 {
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin: 0;
    color: var(--slate);
  }
  .app-header p {
    color: var(--muted);
    font-size: 0.85rem;
    margin: 0.25rem 0 0;
  }

  /* ── Metric cards ────────────────────────── */
  .metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
  .metric-card {
    flex: 1; min-width: 140px;
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.25rem;
  }
  .metric-card .label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
  }
  .metric-card .value {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--slate);
    line-height: 1.1;
    margin: 0.2rem 0;
  }
  .metric-card .delta {
    font-size: 0.75rem;
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
  }

  /* ── Section headers ─────────────────────── */
  .section-label {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
  }

  /* ── Trial table ─────────────────────────── */
  .status-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 9999px;
    font-size: 0.7rem;
    font-weight: 600;
  }
  .status-recruiting    { background: #dcfce7; color: #15803d; }
  .status-completed     { background: #dbeafe; color: #1d4ed8; }
  .status-terminated    { background: #fee2e2; color: #b91c1c; }
  .status-active        { background: #fef9c3; color: #92400e; }
  .status-other         { background: #f3f4f6; color: #374151; }

  /* ── Sidebar ─────────────────────────────── */
  section[data-testid="stSidebar"] {
    background: #f0f4f7;
    border-right: 1px solid #d1dae2;
  }
  section[data-testid="stSidebar"] * {
    color: #334e5e !important;
  }
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stSlider label,
  section[data-testid="stSidebar"] .stMultiSelect label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b !important;
  }
  section[data-testid="stSidebar"] .stButton > button {
    background: #14b8a6 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
  }
  section[data-testid="stSidebar"] .stButton > button:hover {
    background: #0d9488 !important;
  }

  /* ── Plotly chart background ─────────────── */
  .js-plotly-plot .plotly .bg { fill: transparent !important; }

  /* ── Insight callout ─────────────────────── */
  .insight-box {
    background: #f0fdfa;
    border-left: 3px solid var(--accent);
    border-radius: 0 6px 6px 0;
    padding: 0.75rem 1rem;
    margin: 1rem 0;
    font-size: 0.82rem;
    line-height: 1.6;
  }
  .insight-box strong { color: var(--accent); }

  /* Hide Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
CT_BASE = "https://clinicaltrials.gov/api/v2/studies"
FDA_BASE = "https://api.fda.gov/drug/drugsfda.json"

THERAPEUTIC_AREAS = {
    "Oncology":         "cancer OR oncology OR tumor OR carcinoma OR lymphoma OR leukemia",
    "Cardiovascular":   "cardiovascular OR heart failure OR hypertension OR coronary",
    "Immunology":       "autoimmune OR rheumatoid arthritis OR lupus OR Crohn OR psoriasis",
    "Neurology":        "alzheimer OR parkinson OR multiple sclerosis OR epilepsy OR dementia",
    "Rare Disease":     "rare disease OR orphan drug OR genetic disorder",
    "Infectious Disease":"HIV OR hepatitis OR tuberculosis OR COVID OR influenza",
    "Metabolic":        "diabetes OR obesity OR NASH OR fatty liver",
}

PHASE_ORDER  = ["EARLY_PHASE1", "PHASE1", "PHASE1_PHASE2", "PHASE2", "PHASE2_PHASE3", "PHASE3", "PHASE4"]
PHASE_LABELS = {"EARLY_PHASE1": "Early Phase 1", "PHASE1": "Phase 1", "PHASE1_PHASE2": "Phase 1/2",
                "PHASE2": "Phase 2", "PHASE2_PHASE3": "Phase 2/3", "PHASE3": "Phase 3", "PHASE4": "Phase 4"}

STATUS_COLORS = {
    # Positive / completed states — teal family
    "COMPLETED":                "#0d9488",  # teal
    "ACTIVE_NOT_RECRUITING":    "#65a30d",  # soft green (still running, post-enrollment)

    # In-progress / enrolling states — warm family
    "RECRUITING":               "#ea7c30",  # desaturated orange (actively enrolling)
    "NOT_YET_RECRUITING":       "#f59e0b",  # amber
    "ENROLLING_BY_INVITATION":  "#fbbf24",  # light amber

    # Stopped states — red family
    "TERMINATED":               "#b91c1c",  # muted red
    "WITHDRAWN":                "#c2410c",  # orange-red
    "SUSPENDED":                "#7c3aed",  # purple (rare, distinct safety signal)

    # Neutral
    "UNKNOWN":                  "#94a3b8",  # slate gray
}


# ─────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_trials(query: str, max_results: int = 1000) -> pd.DataFrame:
    """Pull clinical trials from ClinicalTrials.gov API v2."""
    fields = [
        "NCTId", "BriefTitle", "Phase", "OverallStatus", "StartDate",
        "PrimaryCompletionDate", "CompletionDate", "LeadSponsorName",
        "LeadSponsorClass", "Condition", "InterventionType",
        "InterventionName", "EnrollmentCount", "StudyType", "HasResults"
    ]
    records, token = [], None
    page_size = min(100, max_results)

    while len(records) < max_results:
        params = {
            "query.term": query,
            "pageSize":   page_size,
            "format":     "json",
            "fields":     "|".join(fields),
        }
        if token:
            params["pageToken"] = token

        try:
            r = requests.get(CT_BASE, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            st.error(f"ClinicalTrials.gov API error: {e}")
            break

        studies = data.get("studies", [])
        if not studies:
            break

        # Filter to interventional studies in Python
        studies = [s for s in studies
                   if s.get("protocolSection", {})
                     .get("designModule", {})
                     .get("studyType") == "INTERVENTIONAL"]

        for s in studies:
            proto  = s.get("protocolSection", {})
            id_mod = proto.get("identificationModule", {})
            stat   = proto.get("statusModule", {})
            design = proto.get("designModule", {})
            sponsor= proto.get("sponsorCollaboratorsModule", {})
            cond   = proto.get("conditionsModule", {})
            interv = proto.get("armsInterventionsModule", {})
            enroll = design.get("enrollmentInfo", {})

            phases = design.get("phases", [])
            phase  = phases[0] if phases else None

            intv_types = [i.get("type","") for i in interv.get("interventions", [])]
            intv_names = [i.get("name","") for i in interv.get("interventions", [])]

            records.append({
                "nct_id":         id_mod.get("nctId"),
                "title":          id_mod.get("briefTitle"),
                "phase":          phase,
                "status":         stat.get("overallStatus"),
                "start_date":     stat.get("startDateStruct", {}).get("date"),
                "primary_completion": stat.get("primaryCompletionDateStruct", {}).get("date"),
                "completion_date":stat.get("completionDateStruct", {}).get("date"),
                "sponsor":        sponsor.get("leadSponsor", {}).get("name"),
                "sponsor_class":  sponsor.get("leadSponsor", {}).get("class"),
                "conditions":     ", ".join(cond.get("conditions", [])),
                "intervention_types": ", ".join(set(intv_types)),
                "intervention_names": "; ".join(intv_names[:3]),
                "enrollment":     enroll.get("count"),
                "has_results":    s.get("hasResults", False),
            })

        token = data.get("nextPageToken")
        if not token:
            break
        time.sleep(0.1)  # polite rate limiting

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Parse dates
    for col in ["start_date", "primary_completion", "completion_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["enrollment"] = pd.to_numeric(df["enrollment"], errors="coerce")
    df["year_started"] = df["start_date"].dt.year.astype("Int64")  # nullable int, handles NaT
    df["phase_label"] = df["phase"].map(PHASE_LABELS).fillna(df["phase"].fillna("Unknown"))

    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fda_approvals(sponsor_hint: str = "", limit: int = 1000) -> pd.DataFrame:
    """Pull recent drug approvals from openFDA Drugs@FDA endpoint."""
    try:
        params = {
            "limit": limit,
        }

        r = requests.get(FDA_BASE, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        st.warning(f"openFDA API unavailable: {e}")
        return pd.DataFrame()

    rows = []
    for result in data.get("results", []):
        sponsor = result.get("sponsor_name", "").title()
        app_no  = result.get("application_number", "")

        # Filter to NDA/BLA applications only (skip ANDAs/generics)
        if not (app_no.startswith("NDA") or app_no.startswith("BLA")):
            continue
        app_type = "NDA" if app_no.startswith("NDA") else "BLA"

        for prod in result.get("products", []):
            for sub in result.get("submissions", []):
                # Only original approvals — not supplements or tentative approvals
                if sub.get("submission_type") == "ORIG" and sub.get("submission_status") == "AP":
                    rows.append({
                        "application_number": app_no,
                        "sponsor":    sponsor,
                        "brand_name": prod.get("brand_name", ""),
                        "generic_name": prod.get("active_ingredients", [{}])[0].get("name", "") if prod.get("active_ingredients") else "",
                        "dosage_form": prod.get("dosage_form", ""),
                        "route":       prod.get("route", ""),
                        "approval_date": pd.to_datetime(
                            sub.get("submission_status_date"), format="%Y%m%d", errors="coerce"
                        ),
                        "submission_type": app_type,
                    })

    df = pd.DataFrame(rows).drop_duplicates(subset=["application_number", "brand_name"])
    if not df.empty:
        df["approval_year"] = df["approval_date"].dt.year
    return df


def compute_pipeline_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate trial duration in months per study."""
    df = df.copy()
    df["duration_months"] = (
        (df["completion_date"] - df["start_date"]).dt.days / 30.44
    ).round(1)
    return df[df["duration_months"].between(1, 300)]  # sanity filter


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    area = st.selectbox(
        "Therapeutic Area",
        list(THERAPEUTIC_AREAS.keys()),
        index=0
    )

    st.markdown("**Filter by Phase**")
    all_phases = ["EARLY_PHASE1", "PHASE1", "PHASE1_PHASE2", "PHASE2", "PHASE2_PHASE3", "PHASE3", "PHASE4"]
    sel_phases = st.multiselect(
        "Phases",
        options=all_phases,
        default=all_phases,
        format_func=lambda x: PHASE_LABELS.get(x, x),
        label_visibility="collapsed",
    )

    st.markdown("**Year Range**")
    year_min, year_max = st.slider(
        "Start year",
        min_value=2005,
        max_value=2025,
        value=(2015, 2025),
        label_visibility="collapsed",
    )

    st.markdown("**Sponsor Type**")
    sponsor_types = st.multiselect(
        "Sponsor class",
        ["INDUSTRY", "NIH", "NETWORK", "OTHER_GOV", "INDIV", "UNKNOWN"],
        default=["INDUSTRY"],
        label_visibility="collapsed",
    )

    max_results = st.select_slider(
        "Max trials to load",
        options=[200, 500, 1000],
        value=500,
    )

    st.markdown("---")
    load_btn = st.button("Run Analysis", use_container_width=True)
    if st.button("Clear Cache", use_container_width=True, type="secondary"):
        st.cache_data.clear()
        st.rerun()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class='app-header'>
  <h1>Pharma Pipeline Intelligence</h1>
  <p>Trial activity to FDA approval · ClinicalTrials.gov &amp; openFDA Drugs · Live Data - Updates daily</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "df_trials" not in st.session_state:
    st.session_state.df_trials = None
    st.session_state.df_fda    = None
    st.session_state.area      = None


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
if load_btn or st.session_state.df_trials is None:
    query = THERAPEUTIC_AREAS[area]
    with st.spinner(f"Fetching up to {max_results} {area} trials…"):
        df_raw = fetch_trials(query, max_results=max_results)

    with st.spinner("Fetching FDA approval records…"):
        df_fda = fetch_fda_approvals(limit=1000)

    st.session_state.df_trials = df_raw
    st.session_state.df_fda    = df_fda
    st.session_state.area      = area

df_raw = st.session_state.df_trials
df_fda = st.session_state.df_fda

if df_raw is None or df_raw.empty:
    st.warning("No trial data loaded. Click **▶ Run Analysis** to start.")
    st.stop()

# ── Apply sidebar filters ──────────────────────
df = df_raw.copy()
df = df[df["phase"].isin(sel_phases)]            # empty selection → empty df, correctly
df = df[df["sponsor_class"].isin(sponsor_types)] # same
df = df[df["year_started"].between(year_min, year_max) | df["year_started"].isna()]

if df.empty:
    st.warning("No trials match current filters. Try widening your phase, sponsor, or year selection.")
    st.stop()


# ─────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────
total_trials   = len(df)
recruiting     = (df["status"] == "RECRUITING").sum()
completed      = (df["status"] == "COMPLETED").sum()
with_results   = df["has_results"].sum()
top_sponsor    = df["sponsor"].value_counts().idxmax() if not df["sponsor"].isna().all() else "—"
median_enroll  = int(df["enrollment"].median()) if df["enrollment"].notna().any() else "—"

st.markdown(f"""
<div class='metric-row'>
  <div class='metric-card'>
    <div class='label'>Trials Loaded</div>
    <div class='value'>{total_trials:,}</div>
    <div class='delta'>{st.session_state.area}</div>
  </div>
  <div class='metric-card'>
    <div class='label'>Recruiting Now</div>
    <div class='value'>{recruiting:,}</div>
    <div class='delta'>{recruiting/total_trials*100:.0f}% of filtered</div>
  </div>
  <div class='metric-card'>
    <div class='label'>Completed</div>
    <div class='value'>{completed:,}</div>
    <div class='delta'>{with_results} have posted results</div>
  </div>
  <div class='metric-card'>
    <div class='label'>Median Enrollment</div>
    <div class='value'>{median_enroll}</div>
    <div class='delta'>patients per trial</div>
  </div>
  <div class='metric-card'>
    <div class='label'>Top Sponsor</div>
    <div class='value' style='font-size:1rem; padding-top:0.3rem;'>{top_sponsor[:28] if isinstance(top_sponsor, str) else top_sponsor}</div>
    <div class='delta'>{df[df.sponsor==top_sponsor].shape[0]} trials</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ROW 1 — Phase funnel + Status breakdown
# ─────────────────────────────────────────────
st.markdown("<div class='section-label'>Pipeline Structure</div>", unsafe_allow_html=True)
col1, col2 = st.columns([3, 2])

with col1:
    phase_counts = (
        df.groupby("phase_label")
          .size()
          .reindex([PHASE_LABELS[p] for p in PHASE_ORDER if PHASE_LABELS[p] in df["phase_label"].values])
          .dropna()
          .reset_index()
    )
    phase_counts.columns = ["Phase", "Trials"]

    phase_order_labels = [PHASE_LABELS[p] for p in PHASE_ORDER]
    phase_counts["Phase"] = pd.Categorical(phase_counts["Phase"], categories=phase_order_labels, ordered=True)
    phase_counts = phase_counts.sort_values("Phase")

    fig_phase = px.bar(
        phase_counts, x="Trials", y="Phase", orientation="h",
        color="Trials",
        color_continuous_scale=["#99f6e4", "#0d9488"],
        labels={"Trials": "Number of Trials"},
        height=280,
    )
    fig_phase.update_traces(
        hovertemplate="<b>%{y}</b><br>%{x} trials<extra></extra>"
    )
    fig_phase.update_layout(
        margin=dict(l=0, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        xaxis=dict(gridcolor="#e5e7eb"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        font=dict(family="Inter", size=11),
    )
    st.plotly_chart(fig_phase, use_container_width=True)

with col2:
    status_counts = df["status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]
    status_counts["Color"] = status_counts["Status"].map(STATUS_COLORS).fillna("#9ca3af")
    status_counts["Label"] = status_counts["Status"].str.replace("_", " ").str.title()

    fig_status = px.pie(
        status_counts, values="Count", names="Label",
        color="Status",
        color_discrete_map={r["Status"]: r["Color"] for _, r in status_counts.iterrows()},
        hole=0.55,
        height=280,
    )
    fig_status.update_traces(
        textposition="inside",
        textinfo="percent",
        showlegend=True,
        hovertemplate="<b>%{label}</b><br>%{value} trials (%{percent})<extra></extra>",
    )
    fig_status.update_layout(
        margin=dict(l=0, r=0, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", size=11),
        legend=dict(
            orientation="v",
            x=1.02,
            y=0.5,
            font=dict(size=11),
    ),
)
    st.plotly_chart(fig_status, use_container_width=True)


# ─────────────────────────────────────────────
# ROW 2 — Trial starts over time + Sponsor landscape
# ─────────────────────────────────────────────
st.markdown("<div class='section-label'>Activity Trends &amp; Sponsor Landscape</div>", unsafe_allow_html=True)
col3, col4 = st.columns(2)

with col3:
    yearly = (
        df.dropna(subset=["year_started"])
          .assign(year_started=lambda d: d["year_started"].astype(int))
          .groupby(["year_started", "phase_label"])
          .size()
          .reset_index(name="count")
    )
    # Convert to plain Python int (Plotly doesn't handle pandas Int64 cleanly)
    yearly["year_started"] = yearly["year_started"].astype("int64")

    if yearly.empty:
        st.info("No trial start dates available for the current filter selection.")
    else:
        fig_time = px.area(
            yearly, x="year_started", y="count", color="phase_label",
            color_discrete_sequence=["#ccfbf1", "#5eead4", "#14b8a6", "#0d9488", "#0f766e", "#134e4a", "#1a2e2c"],
            labels={"year_started": "Year", "count": "New Trials", "phase_label": "Phase"},
            height=280,
        )
        fig_time.update_traces(
            hovertemplate="<b>%{fullData.name}</b><br>%{x}: %{y} new trials<extra></extra>"
        )
        fig_time.update_layout(
            margin=dict(l=0, r=0, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.2, font=dict(size=9)),
            xaxis=dict(gridcolor="#e5e7eb", tickformat="d", dtick=1, type="linear"),
            yaxis=dict(gridcolor="#e5e7eb"),
            font=dict(family="Inter", size=11),
        )
        st.plotly_chart(fig_time, use_container_width=True)

with col4:
    top_sponsors = df["sponsor"].value_counts().head(12).reset_index()
    top_sponsors.columns = ["Sponsor", "Trials"]
    top_sponsors["Short"] = top_sponsors["Sponsor"].str.slice(0, 28)

    top_sponsors_sorted = top_sponsors.sort_values("Trials")
    fig_sponsors = px.bar(
        top_sponsors_sorted, x="Trials", y="Short",
        orientation="h",
        color="Trials",
        color_continuous_scale=["#99f6e4", "#0d9488"],
        height=280,
        custom_data=["Sponsor"],
    )
    fig_sponsors.update_traces(
        hovertemplate="<b>%{customdata[0]}</b><br>%{x} trials<extra></extra>"
    )
    fig_sponsors.update_layout(
        margin=dict(l=0, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        xaxis=dict(gridcolor="#e5e7eb"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", title=None),
        font=dict(family="Inter", size=10),
    )
    st.plotly_chart(fig_sponsors, use_container_width=True)


# ─────────────────────────────────────────────
# ROW 3 — Trial duration analysis (Phase 2 → 3)
# ─────────────────────────────────────────────
st.markdown("<div class='section-label'>Time-in-Phase Analysis</div>", unsafe_allow_html=True)

df_dur = compute_pipeline_duration(df[df["phase"].isin(["PHASE2", "PHASE3"])])
if not df_dur.empty:
    fig_dur = px.box(
        df_dur, x="phase_label", y="duration_months",
        color="phase_label",
        color_discrete_map={"Phase 2": "#14b8a6", "Phase 3": "#0d9488", "Phase 2/3": "#5eead4"},
        points="outliers",
        labels={"duration_months": "Duration (months)", "phase_label": "Phase"},
        height=260,
        category_orders={"phase_label": ["Phase 2", "Phase 2/3", "Phase 3"]},
    )
    fig_dur.update_traces(
        hovertemplate="<b>%{x}</b><br>Duration: %{y} months<extra></extra>"
    )
    fig_dur.update_layout(
        margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor="#e5e7eb"),
        font=dict(family="Inter", size=11),
    )
    col5, col6 = st.columns([2, 1])
    with col5:
        st.plotly_chart(fig_dur, use_container_width=True)
    with col6:
        import math
        med_ph2 = df_dur[df_dur["phase"] == "PHASE2"]["duration_months"].median()
        med_ph3 = df_dur[df_dur["phase"] == "PHASE3"]["duration_months"].median()

        ph2_str = f"{med_ph2:.0f} months" if not math.isnan(med_ph2) else "—"
        ph3_str = f"{med_ph3:.0f} months" if not math.isnan(med_ph3) else "—"

        if not math.isnan(med_ph2) and not math.isnan(med_ph3) and med_ph2 > 0:
            ratio_line = (
                f"Phase 3 trials in {st.session_state.area} run roughly "
                f"<strong>{med_ph3/med_ph2:.1f}×</strong> as long as Phase 2. "
                f"This shapes how sponsors model development timelines and resource allocation."
            )
        else:
            ratio_line = (
                "Not enough completed trials with full date records in this area "
                "to compute a Phase 2 vs Phase 3 duration ratio."
            )

        st.markdown(f"""
        <div class='insight-box'>
          <strong>Duration benchmarks</strong><br>
          Median Phase 2: <strong>{ph2_str}</strong><br>
          Median Phase 3: <strong>{ph3_str}</strong><br><br>
          {ratio_line}
        </div>
        """, unsafe_allow_html=True)
else:
    st.info(
        "No completed Phase 2 or Phase 3 trials with full start and completion dates "
        "in the current filter selection. Try widening the year range or selecting "
        "a broader therapeutic area to see duration benchmarks."
    )


# ─────────────────────────────────────────────
# ROW 4 — FDA Approvals (if data available)
# ─────────────────────────────────────────────
if df_fda is not None and not df_fda.empty:
    st.markdown("<div class='section-label'>FDA Approval Activity (NDA/BLA)</div>", unsafe_allow_html=True)

    col7, col8 = st.columns(2)
    with col7:
        appr_by_year = df_fda.groupby("approval_year").size().reset_index(name="Approvals")
        appr_by_year = appr_by_year[appr_by_year["approval_year"].between(2010, 2025)]
        fig_appr = px.bar(
            appr_by_year, x="approval_year", y="Approvals",
            color="Approvals",
            color_continuous_scale=["#99f6e4", "#0d9488"],
            labels={"approval_year": "Year"},
            height=320,
        )
        fig_appr.update_traces(
            hovertemplate="<b>%{x}</b><br>%{y} approvals<extra></extra>"
        )
        fig_appr.update_layout(
            margin=dict(l=0, r=0, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="#e5e7eb", dtick=2),
            yaxis=dict(gridcolor="#e5e7eb"),
            font=dict(family="Inter", size=11),
            title=dict(text="NDA/BLA Approvals per Year", font=dict(size=11), x=0),
        )
        st.plotly_chart(fig_appr, use_container_width=True)

    with col8:
        top_fda_sponsors = df_fda["sponsor"].value_counts().head(10).reset_index()
        top_fda_sponsors.columns = ["Sponsor", "Approvals"]
        top_fda_sponsors["Short"] = top_fda_sponsors["Sponsor"].str.slice(0, 25)
        top_fda_sorted = top_fda_sponsors.sort_values("Approvals")
        fig_fda_s = px.bar(
            top_fda_sorted, x="Approvals", y="Short",
            orientation="h",
            color="Approvals",
            color_continuous_scale=["#99f6e4", "#0d9488"],
            height=320,
            custom_data=["Sponsor"],
        )
        fig_fda_s.update_traces(
            hovertemplate="<b>%{customdata[0]}</b><br>%{x} approvals<extra></extra>"
        )
        fig_fda_s.update_layout(
            margin=dict(l=0, r=20, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="#e5e7eb"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", title=None, tickmode="linear", automargin=True),
            font=dict(family="Inter", size=10),
            title=dict(text="Top NDA/BLA Sponsors (2010–2025)", font=dict(size=11), x=0),
        )
        st.plotly_chart(fig_fda_s, use_container_width=True)


# ─────────────────────────────────────────────
# ROW 5 — Searchable trial table
# ─────────────────────────────────────────────
st.markdown("<div class='section-label'>Trial Explorer</div>", unsafe_allow_html=True)

search_term = st.text_input(
    "Search by sponsor, intervention, or condition",
    placeholder="e.g. Pfizer, pembrolizumab, non-small cell lung, etc.",
)

df_table = df.copy()
if search_term:
    mask = (
        df_table["sponsor"].str.contains(search_term, case=False, na=False) |
        df_table["conditions"].str.contains(search_term, case=False, na=False) |
        df_table["intervention_names"].str.contains(search_term, case=False, na=False)
    )
    df_table = df_table[mask]

if search_term and df_table.empty:
    st.info(
        f'No trials match "{search_term}". '
        "Try a sponsor name, generic drug name, or condition. "
        "Brand name drugs are often not indexed."
    )
else:
    display_cols = {
        "nct_id":              "NCT ID",
        "title":               "Title",
        "phase_label":         "Phase",
        "status":              "Status",
        "sponsor":             "Sponsor",
        "enrollment":          "Enrollment",
        "start_date":          "Start",
        "completion_date":     "Est. Completion",
        "has_results":         "Results?",
    }

    df_display = df_table[list(display_cols.keys())].rename(columns=display_cols).copy()
    df_display["Start"] = df_display["Start"].dt.strftime("%Y-%m").fillna("—")
    df_display["Est. Completion"] = df_display["Est. Completion"].dt.strftime("%Y-%m").fillna("—")
    df_display["Enrollment"] = df_display["Enrollment"].fillna(0).astype(int).replace(0, "—")
    df_display["Results?"] = df_display["Results?"].map({True: "✓", False: "—"})
    df_display["NCT ID"] = df_display["NCT ID"].apply(
        lambda x: f"https://clinicaltrials.gov/study/{x}" if pd.notna(x) else None
    )

    st.dataframe(
        df_display.head(200),
        use_container_width=True,
        hide_index=True,
        height=340,
        column_config={
            "NCT ID": st.column_config.LinkColumn(
                "NCT ID",
                display_text="https://clinicaltrials.gov/study/(NCT[0-9]+)",
            ),
            "Title":  st.column_config.TextColumn("Title", width="large"),
        }
    )
    st.caption(f"Showing {min(200, len(df_table)):,} of {len(df_table):,} filtered trials · Click NCT ID to open on ClinicalTrials.gov")


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='font-size:0.7rem; color:#9ca3af; text-align:center;'>"
    "Data sourced from <a href='https://clinicaltrials.gov' style='color:#6b7280;'>ClinicalTrials.gov</a> "
    "and <a href='https://open.fda.gov' style='color:#6b7280;'>openFDA</a> "
    "· Not for clinical decision-making. · Built with Streamlit + Plotly"
    "</div>",
    unsafe_allow_html=True
)
