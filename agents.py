"""
SMIT Teaching Assistant - Agents
================================

Four cooperating agents, each a focused call to Claude (Anthropic API):

1. CodeReviewAgent      -> finds syntax/logic/structure issues
2. TutorAgent           -> explains the issues in simple Roman Urdu/English
3. AssignmentRubricAgent-> scores the submission against the retrieved rubric
4. FeedbackAgent        -> compiles everything into a final report + corrected code

All agents are RAG-aware: they receive context retrieved from the ChromaDB
knowledge base (rubrics, common mistakes, class notes) as part of their prompt.

Requires: ANTHROPIC_API_KEY environment variable.
"""

import os
import json
import anthropic

# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------
# Two ways to power the agents:
#
# 1. OpenCode Zen (used if OPENCODE_ZEN_API_KEY is set)
#    - Claude models ("claude-*") -> Anthropic-compatible /v1/messages
#      at base_url "https://opencode.ai/zen" (paid, requires billing set up
#      at https://opencode.ai/auth)
#    - All other models -> OpenAI-compatible /v1/chat/completions
#      at base_url "https://opencode.ai/zen/v1" (includes several FREE
#      models, e.g. "big-pickle", "deepseek-v4-flash-free")
#    - Default model is a FREE model so this works without any billing setup.
#      Once you've confirmed billing on your Zen account, set
#      ZEN_MODEL=claude-sonnet-4-6 (or another Claude model) for better quality.
#
# 2. Direct Anthropic API (used if ANTHROPIC_API_KEY is set instead)
#    - model "claude-sonnet-4-6"
#
# Override the model with ZEN_MODEL / ANTHROPIC_MODEL env vars if desired.
# See https://opencode.ai/docs/zen/ for the full model list.

ZEN_ANTHROPIC_BASE_URL = "https://opencode.ai/zen"
ZEN_OPENAI_BASE_URL = "https://opencode.ai/zen/v1"
ZEN_MODEL_DEFAULT = "big-pickle"  # free, OpenAI-compatible -- works without billing
ANTHROPIC_MODEL_DEFAULT = "claude-sonnet-4-6"

_client = None
_model = None
_mode = None  # "anthropic" or "openai_compatible"


def get_client():
    global _client, _model, _mode
    if _client is not None:
        return _client, _model, _mode

    zen_key = os.environ.get("OPENCODE_ZEN_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if zen_key:
        _model = os.environ.get("ZEN_MODEL", ZEN_MODEL_DEFAULT)
        if _model.startswith("claude"):
            _client = anthropic.Anthropic(api_key=zen_key, base_url=ZEN_ANTHROPIC_BASE_URL)
            _mode = "anthropic"
        else:
            import openai
            _client = openai.OpenAI(api_key=zen_key, base_url=ZEN_OPENAI_BASE_URL)
            _mode = "openai_compatible"
    elif anthropic_key:
        _client = anthropic.Anthropic(api_key=anthropic_key)
        _model = os.environ.get("ANTHROPIC_MODEL", ANTHROPIC_MODEL_DEFAULT)
        _mode = "anthropic"
    else:
        raise RuntimeError(
            "No API key found. Set ONE of:\n"
            "  export OPENCODE_ZEN_API_KEY=...   (https://opencode.ai/auth)\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...\n"
        )
    return _client, _model, _mode


def _call_claude(system: str, user: str, max_tokens: int = 1500) -> str:
    client, model, mode = get_client()

    if mode == "anthropic":
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if block.type == "text")

    # mode == "openai_compatible"
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content


def _format_context(context_chunks: list[dict]) -> str:
    if not context_chunks:
        return "(no extra context retrieved)"
    lines = []
    for c in context_chunks:
        lines.append(f"[{c['metadata']['type']}] {c['text']}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Agent 1: Code Review Agent
# ---------------------------------------------------------------------------
class CodeReviewAgent:
    """Checks syntax, logic, file naming, folder structure, best practices."""

    SYSTEM_PROMPT = (
        "You are a strict but fair Code Review Agent for a programming bootcamp (SMIT). "
        "You review beginner student code (JavaScript/HTML/CSS or similar). "
        "You check: syntax errors, logic errors, naming conventions, file/folder "
        "structure issues, use of best practices (let/const, ===, modular functions), "
        "and code quality (indentation, comments, dead code). "
        "Respond ONLY with a JSON object, no markdown fences, no extra text, with this shape:\n"
        '{\n'
        '  "syntax_issues": ["..."],\n'
        '  "logic_issues": ["..."],\n'
        '  "structure_issues": ["..."],\n'
        '  "best_practice_issues": ["..."],\n'
        '  "positives": ["..."]\n'
        "}\n"
        "If a category has no issues, return an empty list for it."
    )

    def review(self, filename: str, code: str, context_chunks: list[dict]) -> dict:
        context = _format_context(context_chunks)
        user_prompt = (
            f"Filename: {filename}\n\n"
            f"--- STUDENT CODE START ---\n{code}\n--- STUDENT CODE END ---\n\n"
            f"Relevant class material (from RAG knowledge base):\n{context}\n\n"
            "Review this submission and return the JSON object as instructed."
        )
        raw = _call_claude(self.SYSTEM_PROMPT, user_prompt)
        return _safe_json(raw, fallback_keys=[
            "syntax_issues", "logic_issues", "structure_issues",
            "best_practice_issues", "positives"
        ])


# ---------------------------------------------------------------------------
# Agent 2: Tutor Agent
# ---------------------------------------------------------------------------
class TutorAgent:
    """Explains the issues found in simple, beginner-friendly Roman Urdu/English."""

    SYSTEM_PROMPT = (
        "You are a friendly Tutor Agent for SMIT students who are beginners in programming. "
        "Many students are more comfortable with a mix of Roman Urdu and English (Urdlish). "
        "Given a list of code issues, explain EACH issue in 1-3 short sentences using "
        "simple Roman Urdu mixed with English technical terms (the way a friendly senior "
        "explains to a junior). Be encouraging, not harsh. Use the provided class notes / "
        "common-mistake explanations as reference where relevant, but write in your own words.\n\n"
        "Respond ONLY with a JSON object, no markdown fences, with this shape:\n"
        '{\n'
        '  "explanations": [\n'
        '     {"issue": "<short copy of the issue>", "explanation_roman_urdu_english": "..."}\n'
        "  ]\n"
        "}"
    )

    def explain(self, issues: dict, context_chunks: list[dict]) -> dict:
        context = _format_context(context_chunks)
        all_issues = (
            (issues.get("syntax_issues") or [])
            + (issues.get("logic_issues") or [])
            + (issues.get("structure_issues") or [])
            + (issues.get("best_practice_issues") or [])
        )
        if not all_issues:
            return {"explanations": []}

        user_prompt = (
            f"Issues found in the student's code:\n"
            + "\n".join(f"- {i}" for i in all_issues)
            + f"\n\nReference material (common mistakes / class notes from RAG):\n{context}\n\n"
            "Explain each issue as instructed (JSON only)."
        )
        raw = _call_claude(self.SYSTEM_PROMPT, user_prompt)
        return _safe_json(raw, fallback_keys=["explanations"])


# ---------------------------------------------------------------------------
# Agent 3: Assignment Rubric Agent
# ---------------------------------------------------------------------------
class AssignmentRubricAgent:
    """Marks the submission according to the retrieved class rubric."""

    SYSTEM_PROMPT = (
        "You are the Assignment Rubric Agent for SMIT. You assign marks to a student's "
        "code submission STRICTLY based on the rubric criteria provided to you (retrieved "
        "from the class's rubric database via RAG). For each criterion, decide how many of "
        "the available marks the student earned, and give a one-line justification. "
        "Be fair: give partial credit for partially-correct work.\n\n"
        "Respond ONLY with a JSON object, no markdown fences, with this shape:\n"
        '{\n'
        '  "rubric_used": "<assignment name from rubric>",\n'
        '  "criteria_scores": [\n'
        '     {"criterion": "...", "max_marks": 0, "awarded_marks": 0, "justification": "..."}\n'
        "  ],\n"
        '  "total_awarded": 0,\n'
        '  "total_possible": 0,\n'
        '  "pass_fail": "pass" | "fail"\n'
        "}"
    )

    def score(self, code: str, issues: dict, context_chunks: list[dict]) -> dict:
        rubric_chunks = [c for c in context_chunks if c["metadata"]["type"] == "rubric"]
        context = _format_context(rubric_chunks if rubric_chunks else context_chunks)
        user_prompt = (
            f"--- STUDENT CODE ---\n{code}\n--- END CODE ---\n\n"
            f"Issues found by the Code Review Agent (for reference):\n{json.dumps(issues, indent=2)}\n\n"
            f"Rubric retrieved via RAG:\n{context}\n\n"
            "Score this submission against the rubric criteria (JSON only)."
        )
        raw = _call_claude(self.SYSTEM_PROMPT, user_prompt, max_tokens=2000)
        return _safe_json(raw, fallback_keys=[
            "rubric_used", "criteria_scores", "total_awarded", "total_possible", "pass_fail"
        ])


# ---------------------------------------------------------------------------
# Agent 4: Feedback Agent
# ---------------------------------------------------------------------------
class FeedbackAgent:
    """Generates the final personalized feedback report, including corrected code."""

    SYSTEM_PROMPT = (
        "You are the Feedback Agent for SMIT. You receive: the original student code, "
        "the Code Review Agent's findings, the Tutor Agent's beginner-friendly explanations, "
        "and the Rubric Agent's score. Your job is to produce a final, encouraging, "
        "personalized feedback report for the student, AND a corrected version of their code "
        "that fixes the issues while preserving their original approach/style as much as possible.\n\n"
        "Respond ONLY with a JSON object, no markdown fences, with this shape:\n"
        '{\n'
        '  "summary": "<2-4 sentence overall summary, encouraging tone>",\n'
        '  "score_line": "<e.g. 72 / 100 - Pass>",\n'
        '  "top_priority_fixes": ["..."],\n'
        '  "corrected_code": "<full corrected code as a single string>",\n'
        '  "what_changed": ["short bullet list of what was changed and why"]\n'
        "}"
    )

    def compile(self, code: str, review: dict, explanations: dict, rubric_result: dict) -> dict:
        user_prompt = (
            f"--- ORIGINAL STUDENT CODE ---\n{code}\n--- END ---\n\n"
            f"Code Review findings:\n{json.dumps(review, indent=2)}\n\n"
            f"Tutor explanations:\n{json.dumps(explanations, indent=2)}\n\n"
            f"Rubric scoring:\n{json.dumps(rubric_result, indent=2)}\n\n"
            "Produce the final feedback report and corrected code as instructed (JSON only)."
        )
        raw = _call_claude(self.SYSTEM_PROMPT, user_prompt, max_tokens=3000)
        return _safe_json(raw, fallback_keys=[
            "summary", "score_line", "top_priority_fixes", "corrected_code", "what_changed"
        ])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_json(raw: str, fallback_keys: list[str]) -> dict:
    """Strip markdown fences if present and parse JSON; return a safe dict on failure."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"_raw_response": raw, "_parse_error": True, **{k: None for k in fallback_keys}}
