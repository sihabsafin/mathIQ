"""
MathIQ Agent — LangGraph + Groq (Gemma 2) + LangSmith tracing
Compatible with langchain>=1.0, langgraph>=1.0, langchain-groq>=1.0
"""

import os
import re
from typing import Any

# ── LangChain core (modern imports) ──────────────────────────────────────────
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

# ── LangGraph prebuilt agent ──────────────────────────────────────────────────
from langgraph.prebuilt import create_react_agent

# ── MODE SYSTEM PROMPTS ───────────────────────────────────────────────────────
MODE_INSTRUCTIONS = {
    "Solve Only": (
        "Solve the math problem directly and concisely. "
        "Show only the essential steps needed to reach the answer. Be brief but accurate."
    ),
    "Explain Like I'm 12": (
        "Explain the math problem as if teaching a 12-year-old. "
        "Use simple language, real-world analogies, and avoid jargon. "
        "Make every step crystal clear and friendly."
    ),
    "Exam Mode": (
        "Solve this as a full exam solution. Show ALL steps explicitly. "
        "Define variables. Set up equations. Show every calculation. "
        "Verify the final answer. Be thorough and leave nothing implicit."
    ),
    "Proof Mode": (
        "Construct a formal mathematical proof or rigorous logical argument. "
        "State axioms, definitions, and theorems used. "
        "Each step must follow logically from the previous."
    ),
}

SYSTEM_TEMPLATE = """\
You are MathIQ, an elite AI math reasoning assistant powered by Gemma 2.
Your task: solve math problems with precision, clarity, and step-by-step reasoning.

Mode: {mode_instruction}

ALWAYS structure your final response EXACTLY like this:

**PROBLEM UNDERSTANDING**
[Restate what is being asked and identify key variables]

**STEP-BY-STEP SOLUTION**
1. [First step with reasoning]
2. [Second step with reasoning]
3. [Continue for all steps...]

**CALCULATION BREAKDOWN**
[Show all arithmetic/algebraic steps explicitly]

**FINAL ANSWER**
[State the answer clearly, with units if applicable]

**VERIFICATION**
[Check the answer by substituting back or using an alternative method]

Rules:
- Never fabricate formulas. If unsure, say so clearly.
- Always verify your arithmetic.
- Be precise and avoid overconfident tone.
- Use clear mathematical notation.
"""


# ── TOOLS (modern @tool decorator) ───────────────────────────────────────────

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.
    Input should be a valid math expression like '2 * (3 + 4) / 7' or 'sqrt(144)'.
    Supports: +, -, *, /, **, sqrt, sin, cos, tan, log, exp, pi, e, abs, round.
    """
    import math
    safe_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    safe_names["abs"] = abs
    safe_names["round"] = round
    clean = expression.strip()
    try:
        result = eval(clean, {"__builtins__": {}}, safe_names)
        return f"Result: {result}"
    except Exception as exc:
        return f"Calculation error: {exc}. Expression: '{expression}'"


@tool
def reasoning_structurer(problem: str) -> str:
    """
    Break down a complex math word problem into a structured reasoning scaffold.
    Use this FIRST for word problems before attempting calculations.
    Input: the full problem statement.
    """
    return (
        f"Structured breakdown for: {problem}\n\n"
        "IDENTIFY:\n"
        "  - Known quantities: [list what is given]\n"
        "  - Unknown quantities: [list what to find]\n"
        "  - Relationships: [how quantities are connected]\n\n"
        "APPROACH:\n"
        "  - Method: [equation, substitution, formula, etc.]\n"
        "  - Steps: [outline the solution path]\n\n"
        "Proceed with calculations using the calculator tool."
    )


@tool
def python_solver(code: str) -> str:
    """
    Execute Python code for complex math computations.
    Use for: solving equations with sympy, numerical methods, matrix operations.
    Input: valid Python code. Example: 'import sympy as sp; x = sp.Symbol(\"x\"); print(sp.solve(x**2 - 4, x))'
    """
    import io
    import contextlib
    safe_globals = {"__builtins__": __builtins__}
    stdout_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, safe_globals)
        output = stdout_capture.getvalue()
        return output if output else "Code executed successfully (no output)"
    except Exception as exc:
        return f"Python execution error: {exc}"


@tool
def wikipedia_lookup(query: str) -> str:
    """
    Look up mathematical theorems, definitions, or historical context.
    Use ONLY when you need background on a specific math theorem or concept.
    Input: a concept name like 'Pythagorean theorem' or 'quadratic formula'.
    """
    try:
        import wikipediaapi
        wiki = wikipediaapi.Wikipedia(language="en", user_agent="MathIQ-Platform/1.0")
        page = wiki.page(query)
        if page.exists():
            return page.summary[:600] + "..."
        return f"No Wikipedia article found for '{query}'."
    except ImportError:
        return "Wikipedia lookup unavailable. Proceeding with built-in knowledge."
    except Exception as exc:
        return f"Wikipedia error: {exc}"


# ── AGENT BUILDER ─────────────────────────────────────────────────────────────

def build_agent(mode: str = "Exam Mode"):
    """Build the LangGraph ReAct agent with Groq + LangSmith."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Add it to Streamlit Secrets (App Settings → Secrets) or your .env file."
        )

    # LangSmith tracing
    langsmith_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
    if langsmith_key:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_PROJECT", "mathiq-platform"))
        if "LANGCHAIN_API_KEY" not in os.environ:
            os.environ["LANGCHAIN_API_KEY"] = langsmith_key

    llm = ChatGroq(
        model="gemma2-9b-it",
        temperature=0.2,
        max_tokens=2048,
        api_key=groq_api_key,
    )

    tools = [calculator, reasoning_structurer, python_solver, wikipedia_lookup]
    mode_instruction = MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS["Exam Mode"])
    system_prompt = SYSTEM_TEMPLATE.format(mode_instruction=mode_instruction)

    agent = create_react_agent(model=llm, tools=tools, prompt=system_prompt)
    return agent


# ── RUN AGENT & PARSE OUTPUT ──────────────────────────────────────────────────

def run_agent(agent, problem: str, mode: str = "Exam Mode") -> dict[str, Any]:
    """Run the agent on a math problem and return a structured result dict."""
    try:
        result = agent.invoke({"messages": [HumanMessage(content=problem)]})
        messages = result.get("messages", [])

        # Get final AI text response
        raw_output = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                raw_output = msg.content
                break

        # Extract tool usage
        tools_used = []
        tool_trace = []
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.get("name", "unknown")
                    if name not in tools_used:
                        tools_used.append(name)
            if hasattr(msg, "name") and msg.name and hasattr(msg, "content"):
                tool_trace.append({
                    "tool": msg.name,
                    "input": "",
                    "output": str(msg.content)[:300],
                })

        parsed = _parse_structured_output(raw_output)
        parsed["tools_used"] = tools_used
        parsed["tool_trace"] = tool_trace
        parsed["raw"] = raw_output
        parsed["mode"] = mode
        return parsed

    except Exception as exc:
        return {
            "error": str(exc),
            "understanding": "Error during reasoning.",
            "steps": [],
            "calculation": "",
            "answer": f"Agent error: {exc}",
            "verification": "",
            "tools_used": [],
            "tool_trace": [],
            "raw": str(exc),
            "mode": mode,
        }


def _parse_structured_output(text: str) -> dict[str, Any]:
    """Extract structured sections from the agent's final output."""
    sections: dict[str, Any] = {
        "understanding": "",
        "steps": [],
        "calculation": "",
        "answer": "",
        "verification": "",
    }
    patterns = {
        "understanding": r"\*\*PROBLEM UNDERSTANDING\*\*\s*(.*?)(?=\*\*STEP-BY-STEP|\Z)",
        "steps_raw":     r"\*\*STEP-BY-STEP SOLUTION\*\*\s*(.*?)(?=\*\*CALCULATION|\Z)",
        "calculation":   r"\*\*CALCULATION BREAKDOWN\*\*\s*(.*?)(?=\*\*FINAL ANSWER|\Z)",
        "answer":        r"\*\*FINAL ANSWER\*\*\s*(.*?)(?=\*\*VERIFICATION|\Z)",
        "verification":  r"\*\*VERIFICATION\*\*\s*(.*?)(?=\*\*|\Z)",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            if key == "steps_raw":
                raw_steps = re.split(r"\n(?=\d+\.)", val)
                sections["steps"] = [s.strip() for s in raw_steps if s.strip()]
            else:
                sections[key] = val

    # Fallback: use full output if parsing fails
    if not sections["answer"] and not sections["understanding"]:
        sections["answer"] = text.strip()

    return sections
