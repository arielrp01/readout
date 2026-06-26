"""
generate_brief.py
-----------------
Takes a filtered DataFrame of clinical trial / FDA data and calls the
Claude API to produce a structured analyst brief.

Usage (called from app.py):
    from generate_brief import generate_analyst_brief
    brief = generate_analyst_brief(df_filtered, filters)
"""

import anthropic
import json
import pandas as pd
from typing import Optional


def _build_context_summary(df: pd.DataFrame, filters: dict) -> str:
    """Condense the filtered dataset into a structured text block for the prompt."""

    total_trials = len(df)
    ta = filters.get("therapeutic_area", "All")
    phase_sel = filters.get("phases", [])
    year_range = filters.get("year_range", ("N/A", "N/A"))
    sponsor_type = filters.get("sponsor_type", "All")

    # Phase distribution
    if "Phase" in df.columns:
        phase_counts = (
            df["Phase"]
            .value_counts()
            .head(6)
            .to_dict()
        )
        phase_str = ", ".join(f"{k}: {v}" for k, v in phase_counts.items())
    else:
        phase_str = "Not available"

    # Top sponsors
    sponsor_col = next(
        (c for c in ["Sponsor", "LeadSponsorName", "sponsor"] if c in df.columns), None
    )
    if sponsor_col:
        top_sponsors = (
            df[sponsor_col]
            .value_counts()
            .head(8)
            .to_dict()
        )
        sponsor_str = ", ".join(f"{k} ({v})" for k, v in top_sponsors.items())
    else:
        sponsor_str = "Not available"

    # Status distribution
    status_col = next(
        (c for c in ["OverallStatus", "Status", "status"] if c in df.columns), None
    )
    if status_col:
        status_counts = df[status_col].value_counts().head(5).to_dict()
        status_str = ", ".join(f"{k}: {v}" for k, v in status_counts.items())
    else:
        status_str = "Not available"

    # Year trend (trial starts per year if available)
    date_col = next(
        (c for c in ["StartDate", "start_date", "StartYear"] if c in df.columns), None
    )
    if date_col:
        try:
            df["_year"] = pd.to_datetime(df[date_col], errors="coerce").dt.year
            year_trend = df["_year"].value_counts().sort_index().tail(8).to_dict()
            trend_str = ", ".join(f"{int(k)}: {v}" for k, v in year_trend.items() if pd.notna(k))
        except Exception:
            trend_str = "Not available"
    else:
        trend_str = "Not available"

    # FDA approvals if present
    fda_col = next(
        (c for c in ["fda_approvals", "ApprovalDate", "approval_count"] if c in df.columns), None
    )
    fda_note = ""
    if fda_col:
        fda_count = df[fda_col].notna().sum()
        fda_note = f"\nFDA Approvals (NDA) in dataset: {fda_count}"

    context = f"""
FILTER CONTEXT
--------------
Therapeutic Area: {ta}
Phase Selection: {', '.join(phase_sel) if phase_sel else 'All phases'}
Year Range: {year_range[0]}–{year_range[1]}
Sponsor Type: {sponsor_type}

DATASET SUMMARY
---------------
Total Interventional Trials: {total_trials}
Phase Distribution: {phase_str}
Trial Status Breakdown: {status_str}
Top Sponsors by Trial Count: {sponsor_str}
Trial Start Trend (by year): {trend_str}{fda_note}
""".strip()

    return context


def generate_analyst_brief(
    df: pd.DataFrame,
    filters: dict,
    api_key: Optional[str] = None,
) -> dict:
    """
    Call the Claude API with a condensed dataset summary and return a
    structured brief as a dict with four sections.

    Returns:
        {
            "pipeline_summary": str,
            "competitive_landscape": str,
            "key_signals": str,
            "takeaway": str,
            "error": str | None
        }
    """
    if df is None or len(df) == 0:
        return {
            "pipeline_summary": "",
            "competitive_landscape": "",
            "key_signals": "",
            "takeaway": "",
            "error": "No data available for the current filters. Adjust your selections and try again.",
        }

    context_block = _build_context_summary(df, filters)

    system_prompt = """You are a senior pharmaceutical industry analyst specializing in pipeline intelligence and competitive strategy. You write analyst briefs for business development, clinical operations, and strategy teams at biopharma companies.

Your briefs are precise, evidence-grounded, and free of filler. You do not hedge excessively. You flag what is significant and explain why. You write for an audience that already understands trial phases, regulatory pathways, and competitive dynamics — you don't define basics.

Respond ONLY with a JSON object. No preamble, no markdown fencing. The JSON must have exactly these four keys:
- pipeline_summary
- competitive_landscape  
- key_signals
- takeaway

Each value is a plain string (2–4 sentences each, except takeaway which is 1 concise paragraph a BD director would actually read)."""

    user_prompt = f"""Generate an analyst brief based on the following filtered ClinicalTrials.gov + openFDA dataset summary:

{context_block}

Write as if briefing a VP of Business Development before a competitive review meeting. Be specific about what the numbers show — not what they "may suggest." If the data reveals concentration, gaps, or acceleration, say so directly."""

    try:
        client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            system=system_prompt,
        )

        raw = message.content[0].text.strip()

        # Strip markdown fences if model included them despite instructions
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()

        result = json.loads(raw)

        # Validate all four keys present
        required = {"pipeline_summary", "competitive_landscape", "key_signals", "takeaway"}
        missing = required - set(result.keys())
        if missing:
            raise ValueError(f"Response missing keys: {missing}")

        result["error"] = None
        return result

    except json.JSONDecodeError as e:
        return {
            "pipeline_summary": "",
            "competitive_landscape": "",
            "key_signals": "",
            "takeaway": "",
            "error": f"Brief generation failed (parse error). Try again or adjust filters. Detail: {e}",
        }
    except anthropic.APIError as e:
        return {
            "pipeline_summary": "",
            "competitive_landscape": "",
            "key_signals": "",
            "takeaway": "",
            "error": f"API error: {e}",
        }
    except Exception as e:
        return {
            "pipeline_summary": "",
            "competitive_landscape": "",
            "key_signals": "",
            "takeaway": "",
            "error": f"Unexpected error: {e}",
        }
