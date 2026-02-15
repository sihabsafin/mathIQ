"""
MathIQ — Advanced AI Math Reasoning Platform
Built with Groq + Gemma 2 + LangChain + LangSmith + Streamlit
"""

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# ── Page config (MUST be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="MathIQ · AI Math Reasoning Engine",
    page_icon="∑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject global CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;700&display=swap');

/* Root palette */
:root {
    --bg:        #0a0f1e;
    --surface:   #111827;
    --alt:       #1a2035;
    --border:    #1e2d47;
    --accent:    #6366f1;
    --accentD:   #4f46e5;
    --glow:      rgba(99,102,241,0.15);
    --text:      #e2e8f0;
    --muted:     #64748b;
    --dim:       #94a3b8;
    --success:   #22c55e;
    --error:     #ef4444;
    --highlight: #818cf8;
}

/* Global resets */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Inputs */
input, textarea {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
input:focus, textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--glow) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accentD), var(--accent)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    filter: brightness(1.1) !important;
    box-shadow: 0 0 20px var(--glow) !important;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    margin-bottom: 12px !important;
}

/* Chat input */
[data-testid="stChatInput"] {
    background: var(--surface) !important;
    border-top: 1px solid var(--border) !important;
}
[data-testid="stChatInput"] textarea {
    background: var(--bg) !important;
    color: var(--text) !important;
}

/* Selectbox / radio */
[data-testid="stSelectbox"] > div, [data-testid="stRadio"] > div {
    background: var(--alt) !important;
    border-radius: 8px !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: var(--alt) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: var(--alt) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 12px !important;
}

/* Code blocks */
code, pre {
    background: var(--bg) !important;
    color: var(--highlight) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* Success / error banners */
.success-box {
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.3);
    border-left: 3px solid #22c55e;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
}
.error-box {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.3);
    border-left: 3px solid #ef4444;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
}
.step-box {
    background: rgba(99,102,241,0.05);
    border: 1px solid #1e2d47;
    border-left: 3px solid #6366f1;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    color: #94a3b8;
}
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border: 1px solid var(--accent);
    background: var(--glow);
    color: var(--highlight);
    margin-right: 6px;
}
.answer-box {
    background: #111827;
    border: 1px solid rgba(34,197,94,0.4);
    border-top: 2px solid #22c55e;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 12px 0;
}
.verify-box {
    background: rgba(34,197,94,0.05);
    border: 1px solid rgba(34,197,94,0.2);
    border-radius: 8px;
    padding: 8px 14px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #22c55e;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Lazy imports (only after page config) ───────────────────────────────────
from src.agent import build_agent, run_agent
from src.utils import validate_api_keys, format_response_html

# ── Session state defaults ───────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = "Exam Mode"
if "show_trace" not in st.session_state:
    st.session_state.show_trace = True
if "show_tools" not in st.session_state:
    st.session_state.show_tools = True
if "agent" not in st.session_state:
    st.session_state.agent = None

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 16px">
        <div style="font-size:22px;font-weight:700;color:#fff;letter-spacing:-0.02em">
            Math<span style="color:#6366f1">IQ</span>
        </div>
        <div style="font-size:10px;color:#64748b;letter-spacing:0.12em;text-transform:uppercase;margin-top:2px">
            AI Reasoning Engine
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Model status badge — no API key shown in UI
    st.markdown("""
    <div style="background:rgba(99,102,241,0.1);border:1px solid #6366f1;border-radius:8px;padding:10px 14px;margin-bottom:16px;display:flex;align-items:center;gap:10px">
        <div style="width:8px;height:8px;border-radius:50%;background:#6366f1;animation:pulse 2s infinite"></div>
        <div>
            <div style="font-size:12px;font-weight:600;color:#818cf8">Gemma 2 9B · Groq</div>
            <div style="font-size:10px;color:#64748b">temperature: 0.2 · via LangChain</div>
        </div>
    </div>
    <style>@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}</style>
    """, unsafe_allow_html=True)

    # LangSmith status
    ls_active = bool(os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY"))
    ls_color = "#22c55e" if ls_active else "#ef4444"
    ls_label = "LangSmith Tracing Active" if ls_active else "LangSmith Not Configured"
    st.markdown(f"""
    <div style="font-size:11px;color:{ls_color};margin-bottom:16px;display:flex;align-items:center;gap:6px">
        <span>●</span> {ls_label}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Mode selector
    st.markdown('<div style="font-size:10px;font-weight:700;color:#64748b;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:8px">Reasoning Mode</div>', unsafe_allow_html=True)
    mode = st.selectbox(
        "Mode",
        ["Solve Only", "Explain Like I'm 12", "Exam Mode", "Proof Mode"],
        index=2,
        label_visibility="collapsed",
    )
    st.session_state.mode = mode

    MODE_DESCS = {
        "Solve Only":          "⚡ Fast, direct answer with minimal steps",
        "Explain Like I'm 12": "🎓 Beginner-friendly, plain English walkthrough",
        "Exam Mode":           "📋 Full step-by-step, no shortcuts, exam-ready",
        "Proof Mode":          "∀  Formal mathematical proof and logic",
    }
    st.markdown(f'<div style="font-size:11px;color:#64748b;margin-top:4px;margin-bottom:16px">{MODE_DESCS[mode]}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Display toggles
    st.markdown('<div style="font-size:10px;font-weight:700;color:#64748b;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:8px">Display Options</div>', unsafe_allow_html=True)
    st.session_state.show_trace = st.toggle("Show Reasoning Trace", value=st.session_state.show_trace)
    st.session_state.show_tools = st.toggle("Show Tool Usage", value=st.session_state.show_tools)

    st.markdown("---")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Footer
    st.markdown("""
    <div style="font-size:10px;color:#64748b;margin-top:20px;line-height:1.8">
        <div>LangChain · LangSmith · Groq API</div>
        <div style="color:#6366f1">mathiq.platform · MVP v1.0</div>
    </div>
    """, unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;align-items:center;gap:12px;padding:0 0 16px;border-bottom:1px solid #1e2d47;margin-bottom:24px">
    <div style="width:44px;height:44px;border-radius:12px;background:linear-gradient(135deg,#4f46e5,#a855f7);display:flex;align-items:center;justify-content:center;font-size:22px;box-shadow:0 0 24px rgba(99,102,241,0.3)">∑</div>
    <div>
        <div style="display:flex;align-items:center;gap:8px">
            <span style="font-size:20px;font-weight:700;color:#fff">Math<span style="color:#6366f1">IQ</span></span>
            <span class="badge">Gemma 2 · Groq</span>
            <span style="display:inline-block;padding:2px 10px;border-radius:20px;font-size:10px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;border:1px solid rgba(34,197,94,0.4);background:rgba(34,197,94,0.08);color:#22c55e">{mode}</span>
        </div>
        <div style="font-size:11px;color:#64748b;margin-top:2px">AI Math Reasoning Engine — Structured · Traceable · Verifiable</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Empty state / welcome ─────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center;padding:32px 0 24px">
        <p style="font-size:13px;color:#64748b;line-height:1.6;max-width:560px;margin:0 auto">
            Powered by <strong style="color:#818cf8">Gemma 2 via Groq</strong> ·
            LangChain Agent Orchestration · LangSmith Tracing<br>
            Ask any math problem below — algebra, calculus, probability, word problems.
        </p>
    </div>
    """, unsafe_allow_html=True)

    sample_problems = [
        "A train leaves Chicago at 60 mph. Another leaves Denver (1000 miles away) at 80 mph toward Chicago. Where do they meet?",
        "Find the derivative of f(x) = 3x³ - 5x² + 2x - 7",
        "Solve the system: 2x + 3y = 12 and x - y = 1",
        "A bag has 4 red and 6 blue balls. What's the probability of drawing 2 red balls without replacement?",
        "Integrate ∫(x² + 2x + 1)dx from 0 to 3",
    ]

    st.markdown('<div style="font-size:10px;font-weight:700;color:#64748b;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:10px">Try a sample problem</div>', unsafe_allow_html=True)
    cols = st.columns(1)
    for i, prob in enumerate(sample_problems):
        if st.button(f"→  {prob}", key=f"sample_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": prob})
            st.rerun()

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    for col, icon, label, desc in [
        (c1, "⚡", "Fast Inference",      "Groq-powered responses"),
        (c2, "🔍", "Traceable Reasoning", "LangSmith step logs"),
        (c3, "✅", "Verified Output",     "Automatic validation"),
        (c4, "🧩", "Tool Orchestration",  "Calculator + Solver"),
    ]:
        with col:
            st.markdown(f"""
            <div style="background:#111827;border:1px solid #1e2d47;border-radius:10px;padding:14px;text-align:center">
                <div style="font-size:22px">{icon}</div>
                <div style="font-size:12px;font-weight:600;color:#e2e8f0;margin-top:6px">{label}</div>
                <div style="font-size:11px;color:#64748b;margin-top:4px">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "∑"):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            # Render structured AI response
            data = msg.get("data", {})
            format_response_html(
                data,
                show_trace=st.session_state.show_trace,
                show_tools=st.session_state.show_tools,
            )

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Enter a math problem… (Enter to send)"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="∑"):
        with st.spinner("Reasoning through your problem…"):
            try:
                # Validate keys exist
                validate_api_keys()

                # Build agent if not cached
                if st.session_state.agent is None:
                    st.session_state.agent = build_agent(mode=st.session_state.mode)

                result = run_agent(
                    agent=st.session_state.agent,
                    problem=prompt,
                    mode=st.session_state.mode,
                )
                format_response_html(
                    result,
                    show_trace=st.session_state.show_trace,
                    show_tools=st.session_state.show_tools,
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result.get("answer", ""),
                    "data": result,
                })

            except ValueError as e:
                st.markdown(f'<div class="error-box">⚠️ {str(e)}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f'<div class="error-box">❌ Agent error: {str(e)}<br><small>Check your API keys in Streamlit secrets or .env file.</small></div>', unsafe_allow_html=True)
