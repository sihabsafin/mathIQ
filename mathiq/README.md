# MathIQ — Advanced AI Math Reasoning Platform

> Powered by **Groq + Gemma 2 + LangChain + LangSmith + Streamlit**

A production-grade AI math tutor with structured reasoning, tool orchestration, and full LangSmith tracing. Solves word problems, algebra, calculus, probability, and more — step by step.

---

## 🚀 Quick Start (3 paths)

### Path 1 — Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/mathiq.git
cd mathiq

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your API keys
cp .env.example .env
# Open .env and fill in GROQ_API_KEY (required)

# 5. Run
streamlit run app.py
```

### Path 2 — Deploy to Streamlit Cloud (free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo → `app.py` → **Deploy**
4. Click **App Settings → Secrets** and paste:

```toml
GROQ_API_KEY = "gsk_your_key_here"

# Optional — enables LangSmith tracing
LANGCHAIN_API_KEY    = "ls__your_key_here"
LANGCHAIN_TRACING_V2 = "true"
LANGCHAIN_PROJECT    = "mathiq-platform"
```

5. Click **Save** → app restarts → **live! ✅**

### Path 3 — Docker

```bash
docker build -t mathiq .
docker run -p 8501:8501 \
  -e GROQ_API_KEY=gsk_your_key \
  mathiq
```

---

## 🔑 API Keys

| Key | Required | Where to get |
|-----|----------|-------------|
| `GROQ_API_KEY` | ✅ Yes | [console.groq.com](https://console.groq.com) — free tier available |
| `LANGCHAIN_API_KEY` | Optional | [smith.langchain.com](https://smith.langchain.com) — enables tracing |

> **Security**: Never hardcode keys in code. Use `.env` locally or Streamlit Secrets on Cloud. The `.gitignore` already excludes `secrets.toml` and `.env`.

---

## 🧠 Architecture

```
User Input
    ↓
Streamlit Chat UI (app.py)
    ↓
Intent + Mode Detection
    ↓
LangChain ReAct Agent (src/agent.py)
    ├── Calculator Tool      — safe math evaluation
    ├── ReasoningStructurer  — step decomposition
    ├── PythonSolver         — sympy / numpy
    └── WikipediaLookup      — theorem background
    ↓
Groq API → Gemma 2 9B (temp: 0.2)
    ↓
LangSmith Tracing (automatic)
    ↓
Structured Response Parser (src/utils.py)
    ↓
Chat UI — Problem Understanding + Steps + Answer + Verification
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **4 Reasoning Modes** | Solve Only · ELI-12 · Exam Mode · Proof Mode |
| **Structured Responses** | Understanding → Steps → Calculation → Answer → Verification |
| **Tool Orchestration** | Calculator, Python REPL, Wikipedia, Reasoning Structurer |
| **LangSmith Tracing** | Full agent step logs, tool calls, timing, error capture |
| **Dark Mode UI** | Professional `#0a0f1e` dark theme, IBM Plex Sans typography |
| **Reasoning Trace Toggle** | Show/hide step-by-step expandable trace |
| **Tool Usage Badges** | Visual indicator of which tools were invoked |

---

## 📁 Project Structure

```
mathiq/
├── app.py                    # Main Streamlit app
├── src/
│   ├── __init__.py
│   ├── agent.py              # LangChain agent + Groq + tools
│   └── utils.py              # Key validation + HTML rendering
├── .streamlit/
│   ├── config.toml           # Dark theme config
│   └── secrets.toml          # ← YOUR KEYS GO HERE (git-ignored)
├── requirements.txt
├── .env.example              # Local dev template
├── .gitignore                # Protects secrets
└── README.md
```

---

## 🛠️ Tech Stack

- **LLM**: Gemma 2 9B via Groq (ultra-fast inference)
- **Agent**: LangChain ReAct agent
- **Tracing**: LangSmith
- **Frontend**: Streamlit
- **Math Tools**: Python REPL, SymPy, Calculator
- **Hosting**: Streamlit Cloud (free)

---

## 🔧 Troubleshooting

**"GROQ_API_KEY not found"**
→ Check your `.env` file (local) or Streamlit Secrets (cloud). Ensure no trailing spaces.

**"Agent error: model not found"**
→ Groq free tier supports `gemma2-9b-it`. Check [console.groq.com](https://console.groq.com) for available models.

**LangSmith not showing traces**
→ Verify `LANGCHAIN_API_KEY` is set and `LANGCHAIN_TRACING_V2=true`. Check [smith.langchain.com](https://smith.langchain.com).

---

## 📄 License

MIT — free to use, modify, and deploy.
