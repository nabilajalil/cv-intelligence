
"""CV Intelligence Streamlit dashboard.

Phase 7.3:
- Original Agentic RAG search
- Protected multimodal search
"""

import os
from pathlib import Path

import requests
import streamlit as st


API_BASE_URL = os.getenv(
    "CV_INTELLIGENCE_API_URL",
    "http://127.0.0.1:8001",
).rstrip("/")

SEARCH_URL = f"{API_BASE_URL}/search"
PROTECTED_SEARCH_URL = (
    f"{API_BASE_URL}/multimodal/protected-search"
)


st.set_page_config(
    page_title="CV Intelligence",
    page_icon="📄",
    layout="wide",
)

st.title("CV Intelligence")
st.caption(
    "Evidence-grounded recruitment search "
    "with Agentic RAG and protected multimodal retrieval"
)

st.info(
    "CV evidence shows what a document states. "
    "It does not independently establish a candidate's "
    "qualifications or proficiency."
)


def call_api(url, payload, timeout=120):
    """Send a request to the FastAPI backend."""
    response = requests.post(
        url,
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def render_agentic_search():
    """Preserve the original Agentic RAG interface."""

    st.subheader("Agentic RAG Candidate Search")

    query = st.text_area(
        "Describe the candidates you're looking for",
        value=(
            "Find backend developers experienced "
            "with Python, Docker, and AWS"
        ),
        height=100,
        key="agentic_query",
    )

    limit = st.slider(
        "Maximum candidates",
        min_value=1,
        max_value=20,
        value=5,
        key="agentic_limit",
    )

    if st.button(
        "Search candidates",
        type="primary",
        key="agentic_search_button",
    ):
        if not query.strip():
            st.warning("Please enter a recruiter request.")
        else:
            with st.spinner(
                "Searching and evaluating CVs..."
            ):
                try:
                    st.session_state["search_results"] = (
                        call_api(
                            SEARCH_URL,
                            {
                                "query": query,
                                "limit": limit,
                            },
                        )
                    )
                except requests.RequestException as exc:
                    st.error(
                        f"API request failed: {exc}"
                    )

    if "search_results" not in st.session_state:
        return

    data = st.session_state["search_results"]
    candidates = data.get("candidates", [])
    required_skills = data.get("required_skills", [])

    st.divider()
    st.subheader("Candidate search results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Candidates found", len(candidates))

    with col2:
        st.metric(
            "Required skills",
            len(required_skills),
        )

    st.write(
        "**Extracted skills:**",
        ", ".join(required_skills) or "None identified",
    )

    for rank, candidate in enumerate(
        candidates,
        start=1,
    ):
        name = candidate.get(
            "candidate_name",
            "Unknown candidate",
        )
        candidate_id = candidate.get(
            "candidate_id",
            "",
        )
        filename = candidate.get("filename", "")

        matched = candidate.get(
            "matched_skills",
            [],
        )
        not_found = candidate.get(
            "skills_not_found_in_evidence",
            [],
        )

        total = len(required_skills)
        matched_count = len(matched)
        coverage = (
            matched_count / total
            if total
            else 0
        )

        with st.container(border=True):
            title_col, score_col = st.columns(
                [4, 1]
            )

            with title_col:
                st.subheader(
                    f"#{rank} — {name}"
                )
                st.caption(
                    f"Candidate ID: {candidate_id} | "
                    f"Source: {filename}"
                )

            with score_col:
                st.metric(
                    "Skill matches",
                    f"{matched_count}/{total}",
                )

            st.write(
                "**Required skill coverage**"
            )
            st.progress(coverage)

            if total:
                st.caption(
                    f"{coverage:.0%} of explicitly "
                    "required skills were mentioned "
                    "in the indexed CV."
                )
            else:
                st.caption(
                    "No explicit required skills "
                    "were extracted."
                )

            st.write(
                "**Skills mentioned in CV**"
            )

            if matched:
                for skill in matched:
                    st.write(f"✓ {skill}")
            else:
                st.write(
                    "No required skill mentions found."
                )

            if not_found:
                st.write(
                    "**Skills without indexed evidence**"
                )
                for skill in not_found:
                    st.write(f"• {skill}")

            st.write("**Recruiter summary**")
            st.write(
                candidate.get(
                    "summary",
                    "No summary available.",
                )
            )

            metadata = candidate.get("response_metadata")

            if metadata:
                with st.expander("AI response diagnostics"):
                    st.write(
                        "**Response mode:**",
                        metadata.get("response_mode"),
                    )
                    st.write(
                        "**Prompt version:**",
                        metadata.get("prompt_version"),
                    )
                    st.write(
                        "**Presentation style:**",
                        metadata.get("selected_style"),
                    )

                    latency = metadata.get("latency_seconds")
                    if latency is not None:
                        st.metric(
                            "Response latency",
                            f"{latency:.3f} s",
                        )

                    usage = metadata.get("usage_metadata", {})
                    if usage:
                        st.json(usage)

                    if metadata.get("fallback_reason"):
                        st.info(
                            "Deterministic fallback: "
                            + metadata["fallback_reason"]
                        )

                    budget = metadata.get("token_budget", {})
                    if budget:
                        st.write("**Estimated token budget**")
                        st.json(budget)

            with st.expander(
                "View supporting CV evidence"
            ):
                evidence = candidate.get(
                    "skill_evidence",
                    {},
                )

                if not evidence:
                    st.write(
                        "No supporting skill "
                        "evidence available."
                    )

                for skill, chunks in evidence.items():
                    st.markdown(f"**{skill}**")

                    for chunk in chunks:
                        st.caption(
                            "Section: "
                            f"{chunk.get('section', 'unknown')} "
                            "| Page: "
                            f"{chunk.get('page', 'unknown')}"
                        )
                        st.write(
                            chunk.get("text", "")
                        )
                        st.divider()


def render_protected_search():
    """Display fingerprint-protected multimodal results."""

    st.subheader("Protected Multimodal Search")

    st.caption(
        "Searches the controlled synthetic "
        "multimodal benchmark. Evidence is "
        "accepted only after backend verification "
        "and source-document integrity checks."
    )

    query = st.text_input(
        "Candidate search query",
        value="Python Docker",
        key="protected_query",
    )

    skills_text = st.text_input(
        "Required skills (comma-separated)",
        value="Python, Docker",
        key="protected_skills",
    )

    required_skills = [
        skill.strip()
        for skill in skills_text.split(",")
        if skill.strip()
    ]

    if st.button(
        "Search verified evidence",
        type="primary",
        key="protected_search_button",
    ):
        if not query.strip():
            st.warning(
                "Please enter a search query."
            )
        else:
            with st.spinner(
                "Checking candidate evidence "
                "and PDF integrity..."
            ):
                try:
                    result = call_api(
                        PROTECTED_SEARCH_URL,
                        {
                            "query": query.strip(),
                            "required_skills": (
                                required_skills
                            ),
                        },
                        timeout=30,
                    )

                    st.session_state[
                        "protected_results"
                    ] = result

                except requests.RequestException:
                    st.session_state.pop(
                        "protected_results",
                        None,
                    )
                    st.error(
                        "Protected search is unavailable. "
                        "Check that FastAPI is running "
                        "and the verification files exist."
                    )

    if "protected_results" not in st.session_state:
        return

    result = st.session_state[
        "protected_results"
    ]

    candidates = result.get(
        "candidates",
        [],
    )

    st.divider()
    st.subheader("Verified candidate results")

    st.metric(
        "Eligible candidates",
        result.get(
            "candidate_count",
            len(candidates),
        ),
    )

    st.caption(
        "Required skills: "
        + (
            ", ".join(
                result.get("required_skills", [])
            )
            or "None specified"
        )
    )

    if not candidates:
        st.warning(
            "No candidates have sufficient "
            "verified evidence for this query."
        )
        return

    for candidate in candidates:
        candidate_id = candidate.get(
            "candidate_id",
            "Unknown",
        )

        evidence = candidate.get(
            "accepted_evidence",
            [],
        )

        withheld_count = candidate.get(
            "withheld_evidence_count",
            0,
        )

        with st.container(border=True):
            st.subheader(
                f"Candidate {candidate_id}"
            )

            st.success(
                "Eligible based on verified "
                "CV mentions."
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Accepted evidence",
                    len(evidence),
                )

            with col2:
                st.metric(
                    "Withheld evidence",
                    withheld_count,
                )

            st.write("**Verified CV evidence**")

            for item in evidence:
                st.markdown(
                    f"**{item.get('text', '')}**"
                )

                source_name = Path(
                    item.get(
                        "source_file",
                        "",
                    )
                ).name

                st.caption(
                    "Field: "
                    f"{item.get('field', 'unknown')} "
                    "| Source: "
                    f"{source_name} "
                    "| Page: "
                    f"{item.get('page_number', '?')}"
                )

                st.caption(
                    "Decision: "
                    f"{item.get('decision', 'unknown')} "
                    "| Integrity: "
                    f"{item.get('integrity_status', 'unknown')}"
                )

                st.divider()

    with st.expander(
        "View backend-grounded answer"
    ):
        st.text(
            result.get(
                "answer",
                "No grounded answer returned.",
            )
        )


agentic_tab, protected_tab = st.tabs(
    [
        "Agentic RAG Search",
        "Protected Multimodal Search",
    ]
)

with agentic_tab:
    render_agentic_search()

with protected_tab:
    render_protected_search()
