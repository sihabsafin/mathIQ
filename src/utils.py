"""
MathIQ Utilities — API key validation + Streamlit HTML rendering
"""

import os
import streamlit as st


TOOL_ICONS = {
    "Calculator":         "∑",
    "ReasoningStructurer": "🧠",
    "PythonSolver":       "🐍",
    "WikipediaLookup":    "📖",
    "unknown":            "⚙️",
}


def validate_api_keys() -> None:
    """
    Check that required API keys are present.
    Reads from Streamlit secrets first, then environment variables.
    Raises ValueError with a clear message if missing.
    """
    # Streamlit Cloud secrets → inject into env
    try:
        secrets = st.secrets
        for key in ["GROQ_API_KEY", "LANGCHAIN_API_KEY", "LANGSMITH_API_KEY", "LANGCHAIN_PROJECT"]:
            if key in secrets and secrets[key]:
                os.environ[key] = secrets[key]
    except Exception:
        pass  # Not on Streamlit Cloud or no secrets file

    if not os.getenv("GROQ_API_KEY"):
        raise ValueError(
            "GROQ_API_KEY is missing.\n\n"
            "• Streamlit Cloud: add it in App Settings → Secrets\n"
            "• Local: add it to .env or secrets.toml\n"
            "See README.md for full setup instructions."
        )


def format_response_html(
    data: dict,
    show_trace: bool = True,
    show_tools: bool = True,
) -> None:
    """
    Render a structured MathIQ response into Streamlit using HTML components.
    """
    if not data:
        st.warning("No response data to display.")
        return

    # Error state
    if "error" in data and data["error"]:
        st.markdown(
            f'<div class="error-box">❌ {data["error"]}</div>',
            unsafe_allow_html=True,
        )
        return

    # ── Tool badges ──────────────────────────────────────────────────────────
    if show_tools and data.get("tools_used"):
        badges_html = "".join(
            f'<span class="badge">{TOOL_ICONS.get(t, "⚙️")} {t}</span>'
            for t in data["tools_used"]
        )
        st.markdown(
            f'<div style="margin-bottom:12px">{badges_html}</div>',
            unsafe_allow_html=True,
        )

    # ── Problem Understanding ────────────────────────────────────────────────
    if data.get("understanding"):
        st.markdown(
            f"""
            <div style="background:#1a2035;border:1px solid #1e2d47;border-radius:8px;padding:12px 16px;margin-bottom:12px">
                <div style="font-size:10px;font-weight:700;color:#64748b;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:6px">
                    Problem Understanding
                </div>
                <div style="font-size:13px;color:#e2e8f0;line-height:1.7">{data["understanding"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Reasoning Trace ──────────────────────────────────────────────────────
    if show_trace and data.get("steps"):
        with st.expander("📐 Reasoning Trace", expanded=True):
            for i, step in enumerate(data["steps"], 1):
                st.markdown(
                    f"""
                    <div class="step-box">
                        <div style="font-size:10px;font-weight:700;color:#6366f1;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px">
                            Step {i}
                        </div>
                        <div style="white-space:pre-wrap;line-height:1.7">{step}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ── Calculation Breakdown ────────────────────────────────────────────────
    if data.get("calculation"):
        with st.expander("🔢 Calculation Breakdown", expanded=False):
            st.markdown(
                f'<div class="step-box" style="white-space:pre-wrap">{data["calculation"]}</div>',
                unsafe_allow_html=True,
            )

    # ── Final Answer ─────────────────────────────────────────────────────────
    answer_text = data.get("answer", "No answer generated.")
    st.markdown(
        f"""
        <div class="answer-box">
            <div style="font-size:10px;font-weight:700;color:#22c55e;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:8px">
                ✓ Final Answer
            </div>
            <div style="font-size:16px;color:#e2e8f0;font-weight:500;line-height:1.7">{answer_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Verification ─────────────────────────────────────────────────────────
    if data.get("verification"):
        st.markdown(
            f'<div class="verify-box">✓ Verified: {data["verification"]}</div>',
            unsafe_allow_html=True,
        )

    # ── Tool Trace (detailed) ─────────────────────────────────────────────────
    if show_tools and data.get("tool_trace"):
        with st.expander("🔧 Tool Execution Log", expanded=False):
            for entry in data["tool_trace"]:
                icon = TOOL_ICONS.get(entry["tool"], "⚙️")
                st.markdown(
                    f"""
                    <div style="margin-bottom:10px;padding:10px 14px;background:#111827;border:1px solid #1e2d47;border-radius:8px">
                        <div style="font-size:11px;font-weight:700;color:#818cf8;margin-bottom:4px">{icon} {entry["tool"]}</div>
                        <div style="font-size:11px;color:#64748b;font-family:monospace">Input: {entry["input"]}</div>
                        <div style="font-size:11px;color:#94a3b8;font-family:monospace;margin-top:4px">Output: {entry["output"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ── Mode badge ────────────────────────────────────────────────────────────
    mode = data.get("mode", "")
    if mode:
        st.markdown(
            f'<div style="font-size:10px;color:#64748b;margin-top:8px;font-family:monospace">mode: {mode} · model: gemma2-9b-it · via groq</div>',
            unsafe_allow_html=True,
        )
