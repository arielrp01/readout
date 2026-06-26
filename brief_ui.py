"""
brief_ui.py
-----------
Drop-in Streamlit UI component for the LLM analyst brief.

INTEGRATION INSTRUCTIONS FOR app.py
=====================================

1. At the top of app.py, add these imports:
   ----------------------------------------
   from generate_brief import generate_analyst_brief
   import streamlit as st  # already present

2. Add these keys to your Streamlit secrets:
   ------------------------------------------
   In the Streamlit Cloud dashboard → App settings → Secrets, add:
   
       ANTHROPIC_API_KEY = "sk-ant-..."
       BRIEF_ENABLED = true
   
   For local dev, create .streamlit/secrets.toml:
   
       ANTHROPIC_API_KEY = "sk-ant-..."
       BRIEF_ENABLED = true
   
   To disable the feature without a code change, flip BRIEF_ENABLED to false
   and redeploy. The button will be replaced with a maintenance message.

3. After your existing charts section, paste the block below:
   -----------------------------------------------------------
   (Search for where you render your final chart/table and add
    the brief section immediately after.)

PASTE THIS BLOCK INTO app.py AFTER YOUR CHARTS:
================================================
"""

# ─── COPY FROM HERE ───────────────────────────────────────────────────────────

import streamlit as st

def render_brief_section(df_filtered, filters: dict):
    """
    Renders the Generate Brief button and collapsible/tabbed brief output.
    
    Args:
        df_filtered: The same filtered DataFrame used to render your charts
        filters: Dict of active filter values, e.g.:
                 {
                     "therapeutic_area": selected_ta,
                     "phases": selected_phases,
                     "year_range": (start_year, end_year),
                     "sponsor_type": selected_sponsor_type,
                 }
    """

    # ── Constants ──────────────────────────────────────────────────────────────
    SESSION_LIMIT = 3          # max brief generations per user session
    DATASET_ROW_CAP = 500      # rows passed to context builder; quality stable above this

    st.divider()

    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.25rem;">
            <span style="font-size:1.1rem; font-weight:600; color:#f8fafc;">🤖 AI Analyst Brief</span>
            <span style="font-size:0.7rem; background:#1e3a5f; color:#60a5fa;
                         padding:2px 8px; border-radius:10px; font-weight:500;">
                Powered by Claude
            </span>
        </div>
        <p style="color:#94a3b8; font-size:0.85rem; margin-top:0; margin-bottom:1rem;">
            Generate a plain-language competitive intelligence brief from the filtered dataset.
            Synthesis takes 5–10 seconds.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ── Kill switch — flip BRIEF_ENABLED = false in Streamlit secrets to disable ──
    if not st.secrets.get("BRIEF_ENABLED", True):
        st.info("Brief generation is temporarily unavailable. Check back soon.")
        return

    # ── No data guard ──────────────────────────────────────────────────────────
    if df_filtered is None or len(df_filtered) == 0:
        st.info("Apply filters above to enable brief generation.")
        return

    # ── Session-based rate limit ───────────────────────────────────────────────
    if "brief_count" not in st.session_state:
        st.session_state.brief_count = 0

    remaining = SESSION_LIMIT - st.session_state.brief_count

    if remaining <= 0:
        st.warning(
            f"You've used all {SESSION_LIMIT} brief generations for this session. "
            "Refresh the page to start a new session."
        )
        return

    # Show remaining uses (only once one has been used, to avoid cluttering a fresh load)
    if st.session_state.brief_count > 0:
        st.caption(f"{remaining} of {SESSION_LIMIT} brief generations remaining this session.")

    if st.button(
        "⚡ Generate Brief",
        type="primary",
        help="Calls the Claude API to synthesize a competitive intelligence brief from the current filtered view.",
        use_container_width=False,
    ):
        # ── Dataset size cap — sample large views before the API call ──────────
        df_for_brief = (
            df_filtered.sample(n=DATASET_ROW_CAP, random_state=42)
            if len(df_filtered) > DATASET_ROW_CAP
            else df_filtered
        )

        with st.spinner("Synthesizing analyst brief…"):
            api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
            brief = generate_analyst_brief(df_for_brief, filters, api_key=api_key)

        # Increment counter only on a completed call (success or API error, not guard returns)
        st.session_state.brief_count += 1

        if brief.get("error"):
            st.error(brief["error"])
            return

        # ── Tabbed output ──────────────────────────────────────────────────
        tab_summary, tab_competitive, tab_signals, tab_takeaway = st.tabs([
            "📊 Pipeline Summary",
            "🏢 Competitive Landscape",
            "🔔 Key Signals",
            "📝 Takeaway",
        ])

        with tab_summary:
            st.markdown(
                f"""
                <div style="background:#0f172a; border:1px solid #1e293b; border-radius:8px;
                             padding:1.25rem 1.5rem; color:#e2e8f0; line-height:1.7;
                             font-size:0.93rem;">
                {brief['pipeline_summary']}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with tab_competitive:
            st.markdown(
                f"""
                <div style="background:#0f172a; border:1px solid #1e293b; border-radius:8px;
                             padding:1.25rem 1.5rem; color:#e2e8f0; line-height:1.7;
                             font-size:0.93rem;">
                {brief['competitive_landscape']}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with tab_signals:
            st.markdown(
                f"""
                <div style="background:#0f172a; border:1px solid #1e293b; border-radius:8px;
                             padding:1.25rem 1.5rem; color:#e2e8f0; line-height:1.7;
                             font-size:0.93rem;">
                {brief['key_signals']}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with tab_takeaway:
            st.markdown(
                f"""
                <div style="background:#0c1929; border-left:3px solid #3b82f6;
                             border-radius:0 8px 8px 0; padding:1.25rem 1.5rem;
                             color:#e2e8f0; line-height:1.75; font-size:0.95rem;">
                {brief['takeaway']}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Subtle disclosure
        st.caption(
            "Generated by Claude (claude-sonnet-4-6) from ClinicalTrials.gov + openFDA data. "
            "Not for clinical decision-making."
        )

# ─── END COPY ──────────────────────────────────────────────────────────────────


# ─── EXAMPLE of how to call this in app.py ────────────────────────────────────
#
# # ... (your existing chart rendering code) ...
#
# # After the last chart:
# from brief_ui import render_brief_section
#
# active_filters = {
#     "therapeutic_area": therapeutic_area,   # your sidebar variable
#     "phases": phases,                        # your sidebar variable
#     "year_range": (start_year, end_year),    # your sidebar variable
#     "sponsor_type": sponsor_type,            # your sidebar variable
# }
# render_brief_section(df_filtered, active_filters)
#
# ──────────────────────────────────────────────────────────────────────────────
