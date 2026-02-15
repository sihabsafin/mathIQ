"""
MathIQ Agent — LangChain + Groq (Gemma 2) + LangSmith tracing
"""

import os
import re
from typing import Any

# LangChain
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_experimental.tools import PythonREPLTool

# LangSmith
from langsmith import Client as LangSmithClient

# ── MODE SYSTEM PROMPTS ───────────────────────────────────────────────────────
MODE_INSTRUCTIONS = {
    "Solve Only": (
        "Solve the math problem directly and concisely. "
        "Show only the essential steps needed to reach the answer. "
        "Be brief but accurate."
    ),
    "Explain Like I'm 12": (
        "Explain the math problem as if you're teaching a 12-year-old. "
        "Use simple language, real-world analogies, and avoid jargon. "
        "Make every step crystal clear and friendly."
    ),
    "Exam Mode": (
        "Solve this as if writing a full exam solution. "
        "Show ALL steps explicitly. Define variables. Set up equations. "
        "Show every calculation. Verify the final answer. Be thorough."
    ),
    "Proof Mode": (
        "Construct a formal mathematical proof or rigorous logical argument. "
        "State axioms, definitions, and theorems used. "
        "Each step must follow logically from the previous."
    ),
}

# ── AGENT PROMPT ──────────────────────────────────────────────────────────────
REACT_TEMPLATE = """\
You are MathIQ, an elite AI math reasoning assistant powered by Gemma 2.
Your task: solve math problems with precision, clarity, and step-by-step reasoning.

Mode instruction: {mode_instruction}

You have access to these tools:
{tools}

Tool names: {tool_names}

STRICT RESPONSE FORMAT — always structure your FINAL answer as:

**PROBLEM UNDERSTANDING**
[Restate what is being asked and identify key variables]

**STEP-BY-STEP SOLUTION**
[Numbered steps with clear reasoning]

**CALCULATION BREAKDOWN**
[Show all arithmetic/algebraic steps explicitly]

**FINAL ANSWER**
[State the answer clearly, with units if applicable]

**VERIFICATION**
[Check the answer by substituting back or alternative method]

Rules:
- Never fabricate formulas. If unsure, say so.
- Always verify your arithmetic.
- Use LaTeX notation for math expressions: $expression$
- Be precise and avoid overconfident tone.

Begin!

Question: {input}
{agent_scratchpad}
"""

REACT_PROMPT = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tools", "tool_names", "mode_instruction"],
    template=REACT_TEMPLATE,
)


# ── CUSTOM TOOLS ──────────────────────────────────────────────────────────────

def calculator_tool_fn(expression: str) -> str:
    """Safe math expression evaluator."""
    import math
    allowed = {
        k: v for k, v in math.__dict__.items() if not k.startswith("__")
    }
    allowed["__builtins__"] = {}
    try:
        # Sanitize: only allow math characters
        clean = re.sub(r"[^0-9+\-*/().,e\sEmsqrtlogincoshtanp]", "", expression)
        result = eval(clean, {"__builtins__": {}}, allowed)  # noqa: S307
        return f"Result: {result}"
    except Exception as exc:
        return f"Calculation error: {exc}. Expression: {expression}"


def reasoning_tool_fn(problem: str) -> str:
    """Structured reasoning decomposer."""
    return (
        f"Reasoning about: {problem}\n\n"
        "Step 1: Identify what is known and what is unknown.\n"
        "Step 2: Choose the appropriate mathematical method.\n"
        "Step 3: Set up equations or logical structure.\n"
        "Step 4: Solve systematically.\n"
        "Step 5: Verify the result.\n\n"
        "Proceed with the chosen method."
    )


def build_tools() -> list[Tool]:
    """Build and return the MathIQ tool suite."""
    tools = []

    # 1. Calculator
    tools.append(Tool(
        name="Calculator",
        func=calculator_tool_fn,
        description=(
            "Evaluate mathematical expressions. Input: a valid math expression "
            "like '2 * (3 + 4) / 7' or 'sqrt(144)'. Returns numeric result."
        ),
    ))

    # 2. Reasoning Structurer
    tools.append(Tool(
        name="ReasoningStructurer",
        func=reasoning_tool_fn,
        description=(
            "Use this to break down complex word problems into structured reasoning steps. "
            "Input: the problem statement. Returns a structured reasoning scaffold."
        ),
    ))

    # 3. Python REPL (for complex calculations)
    try:
        python_repl = PythonREPLTool()
        python_repl.name = "PythonSolver"
        python_repl.description = (
            "Run Python code for complex math: solving equations, symbolic math with sympy, "
            "numerical computation. Input: valid Python code. Use for non-trivial calculations."
        )
        tools.append(python_repl)
    except Exception:
        pass

    # 4. Wikipedia (optional knowledge lookup)
    try:
        wiki = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=800)
        tools.append(Tool(
            name="WikipediaLookup",
            func=wiki.run,
            description=(
                "Look up mathematical theorems, definitions, or historical context. "
                "Use sparingly — only when theorem background is needed. "
                "Input: a math concept or theorem name."
            ),
        ))
    except Exception:
        pass

    return tools


# ── AGENT BUILDER ─────────────────────────────────────────────────────────────

def build_agent(mode: str = "Exam Mode") -> AgentExecutor:
    """Build the LangChain ReAct agent with Groq + LangSmith."""

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Add it to Streamlit secrets or your .env file."
        )

    # LangSmith tracing (automatic via env vars)
    langsmith_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
    if langsmith_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "mathiq-platform")
        if "LANGCHAIN_API_KEY" not in os.environ and langsmith_key:
            os.environ["LANGCHAIN_API_KEY"] = langsmith_key

    # LLM — Groq + Gemma 2
    llm = ChatGroq(
        model="gemma2-9b-it",
        temperature=0.2,
        max_tokens=2048,
        groq_api_key=groq_api_key,
    )

    tools = build_tools()
    mode_instruction = MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS["Exam Mode"])

    prompt = REACT_PROMPT.partial(mode_instruction=mode_instruction)

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=8,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


# ── RUN AGENT & PARSE OUTPUT ──────────────────────────────────────────────────

def run_agent(agent: AgentExecutor, problem: str, mode: str = "Exam Mode") -> dict[str, Any]:
    """
    Run the agent on a math problem and return a structured result dict.
    """
    try:
        result = agent.invoke({"input": problem})
        raw_output: str = result.get("output", "")
        steps = result.get("intermediate_steps", [])

        # Parse tools used
        tools_used = []
        tool_trace = []
        for action, observation in steps:
            tool_name = getattr(action, "tool", "unknown")
            tool_input = getattr(action, "tool_input", "")
            if tool_name not in tools_used:
                tools_used.append(tool_name)
            tool_trace.append({
                "tool": tool_name,
                "input": str(tool_input)[:200],
                "output": str(observation)[:300],
            })

        # Parse structured sections from raw output
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
            "answer": f"Error: {exc}",
            "verification": "",
            "tools_used": [],
            "tool_trace": [],
            "raw": str(exc),
            "mode": mode,
        }


def _parse_structured_output(text: str) -> dict[str, Any]:
    """Extract structured sections from the agent's final output."""
    sections = {
        "understanding": "",
        "steps": [],
        "calculation": "",
        "answer": "",
        "verification": "",
    }

    patterns = {
        "understanding": r"\*\*PROBLEM UNDERSTANDING\*\*\s*(.*?)(?=\*\*STEP|$)",
        "steps_raw":     r"\*\*STEP-BY-STEP SOLUTION\*\*\s*(.*?)(?=\*\*CALCULATION|$)",
        "calculation":   r"\*\*CALCULATION BREAKDOWN\*\*\s*(.*?)(?=\*\*FINAL|$)",
        "answer":        r"\*\*FINAL ANSWER\*\*\s*(.*?)(?=\*\*VERIFICATION|$)",
        "verification":  r"\*\*VERIFICATION\*\*\s*(.*?)(?=\*\*|$)",
    }

    for key, pattern in patterns.items():
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            if key == "steps_raw":
                # Split numbered steps
                raw_steps = re.split(r"\n(?=\d+\.|\*Step)", val)
                sections["steps"] = [s.strip() for s in raw_steps if s.strip()]
            else:
                sections[key] = val

    # Fallback if parsing fails
    if not sections["answer"]:
        sections["answer"] = text.strip()

    return sections
