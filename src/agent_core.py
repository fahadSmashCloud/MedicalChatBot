"""ReAct Agentic AI — orchestrator using Groq function calling.

ARCHITECTURE OVERVIEW
─────────────────────
                         ┌─────────────────────┐
    User Task  ──────►   │   Orchestrator LLM   │  (Groq: llama-3.3-70b)
                         │  (Reason + Decide)   │
                         └──────────┬──────────┘
                                    │ tool_calls (JSON)
                                    ▼
                         ┌─────────────────────┐
                         │   Tool Dispatcher    │
                         │  TOOLS_BY_NAME[name] │
                         │  .func(args) → str   │
                         └──────────┬──────────┘
                                    │ observation (str)
                                    ▼
                         ┌─────────────────────┐
                         │   Messages buffer    │  ← injected back as "tool" role
                         │  [system, user,      │
                         │   assistant, tool,…] │
                         └──────────┬──────────┘
                                    │  loop until Final Answer or max_iterations
                                    ▼
                         ┌─────────────────────┐
                         │    Final Answer      │  streamed to caller as AgentStep
                         └─────────────────────┘

ReAct Pattern (Yao et al., 2022 — arxiv.org/abs/2210.03629)
────────────────────────────────────────────────────────────
  REASON: LLM generates text (Thought) explaining its reasoning.
  ACT:    LLM selects a tool and arguments (Action).
  OBSERVE:Tool result is injected back into the conversation (Observation).
  Repeat until the LLM produces a final answer without a tool call.

Groq Function Calling Protocol
───────────────────────────────
  1. Pass ``tools`` list (JSON Schema) + ``tool_choice="auto"`` to the API.
  2. Model response includes ``message.tool_calls`` when it wants to call a tool.
  3. Caller executes the tool and appends a ``{"role": "tool", ...}`` message.
  4. Model sees the observation and decides next action or gives a Final Answer.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Generator

from groq import Groq

from src.agent_tools import TOOLS, TOOLS_BY_NAME

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "llama-3.3-70b-versatile"
MAX_ITERATIONS = 10


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class AgentStep:
    """
    One atomic step in the agent's reasoning trace.

    step_type values:
      "thought"     — LLM's textual reasoning (visible to user)
      "action"      — Tool call being dispatched
      "observation" — Raw result from the tool
      "final"       — Agent's concluded answer
      "error"       — Unrecoverable error (Groq API failure, etc.)
    """
    step_type: str
    content: str
    tool_name: str | None = None
    tool_args: dict | None = None
    iteration: int = 0


@dataclass
class AgentResult:
    """Collected output after the agent loop finishes."""
    steps: list[AgentStep] = field(default_factory=list)
    final_answer: str = ""
    total_iterations: int = 0
    error: str | None = None


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a powerful AI agent with access to real-world tools.
Your mission: solve the user's task step by step with precision and transparency.

## Strategy
1. **Think first** — Briefly explain your reasoning before choosing a tool.
2. **Act precisely** — Call one tool at a time with well-formed arguments.
3. **Observe carefully** — Read the tool result fully before your next step.
4. **Synthesise** — When you have enough information, give a clear Final Answer.

## Rules
- Never repeat the exact same tool call with identical arguments.
- If a tool returns an error, try a different approach or different arguments.
- Prefer search_web for current events; prefer calculate for math; prefer
  query_medical_knowledge for medical topics (if the knowledge base exists).
- Be concise in tool arguments; be thorough in the Final Answer.
- When asked for calculations, always use the calculate tool (do not guess).
- Always cite sources (URLs) when using search_web or scrape_url results.
"""


# ── Tool definition builder ───────────────────────────────────────────────────

def _groq_tool_definitions(enabled_tools: list[str] | None = None) -> list[dict]:
    """
    Converts ToolSpec objects into the Groq/OpenAI function-calling schema.

    Args:
        enabled_tools: Optional whitelist of tool names. None = all tools.
    """
    specs = TOOLS if enabled_tools is None else [
        t for t in TOOLS if t.name in enabled_tools
    ]
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in specs
    ]


# ── Agent loop ────────────────────────────────────────────────────────────────

def run_agent(
    task: str,
    model: str = DEFAULT_MODEL,
    max_iterations: int = MAX_ITERATIONS,
    enabled_tools: list[str] | None = None,
    extra_system: str = "",
) -> Generator[AgentStep, None, AgentResult]:
    """
    Run the ReAct agent loop. This is a Python generator — it **yields**
    AgentStep objects as they are produced so callers can display them
    incrementally (Streamlit UI, SSE stream, etc.).

    Example usage
    ─────────────
    >>> steps = []
    >>> for step in run_agent("What is 2^32?"):
    ...     print(step.step_type, step.content[:60])
    ...     steps.append(step)

    Args:
        task            : The user's task or question.
        model           : Groq model ID (must support function calling).
        max_iterations  : Hard cap on reasoning iterations (prevents runaway loops).
        enabled_tools   : Whitelist of tool names. None = all tools enabled.
        extra_system    : Optional extra instructions appended to the system prompt.

    Yields:
        AgentStep objects as the agent reasons, acts, and observes.

    Returns (via StopIteration.value):
        AgentResult with all steps, final answer, and iteration count.
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        err = AgentStep(step_type="error", content="GROQ_API_KEY is not set.")
        yield err
        return AgentResult(error=err.content)

    client = Groq(api_key=api_key)
    tools_def = _groq_tool_definitions(enabled_tools)

    system_content = _SYSTEM_PROMPT
    if extra_system:
        system_content += f"\n\n{extra_system}"

    # Conversation buffer — grows as the agent reasons
    messages: list[dict] = [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": task},
    ]

    result = AgentResult()
    seen_calls: set[str] = set()   # de-duplicate identical tool calls

    for iteration in range(1, max_iterations + 1):
        result.total_iterations = iteration

        # ── Call the LLM ──────────────────────────────────────────────────────
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools_def,
                tool_choice="auto",
                temperature=0.2,
                max_tokens=1024,
            )
        except Exception as exc:
            exc_str = str(exc)
            # Groq returns HTTP 400 with code "output_parse_failed" when the
            # model generates reasoning text that its function-calling parser
            # mistakes for a malformed tool call.  Fix: retry the same messages
            # with tool_choice="none" to force a plain-text final answer.
            if "output_parse_failed" in exc_str or (
                hasattr(exc, "status_code") and getattr(exc, "status_code", 0) == 400
                and "failed_generation" in exc_str
            ):
                parse_warn = AgentStep(
                    step_type="thought",
                    content=(
                        "[Auto-recovery] Groq output parser failed on previous response "
                        "— retrying without tool calls to produce a final answer."
                    ),
                    iteration=iteration,
                )
                result.steps.append(parse_warn)
                yield parse_warn
                try:
                    fallback = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tool_choice="none",   # force plain text — no tool calls
                        temperature=0.3,
                        max_tokens=1024,
                    )
                    answer = fallback.choices[0].message.content or "(No response generated.)"
                    final = AgentStep(
                        step_type="final",
                        content=answer,
                        iteration=iteration,
                    )
                    result.steps.append(final)
                    result.final_answer = answer
                    yield final
                    return result
                except Exception as fallback_exc:
                    exc_str = f"Groq API error (iteration {iteration}): {fallback_exc}"
            err_step = AgentStep(
                step_type="error",
                content=f"Groq API error (iteration {iteration}): {exc_str}",
                iteration=iteration,
            )
            result.steps.append(err_step)
            result.error = err_step.content
            yield err_step
            return result

        msg = response.choices[0].message

        # ── Thought (optional text before tool call) ──────────────────────────
        if msg.content:
            thought = AgentStep(
                step_type="thought",
                content=msg.content,
                iteration=iteration,
            )
            result.steps.append(thought)
            yield thought

        # ── No tool calls → Final Answer ──────────────────────────────────────
        if not msg.tool_calls:
            answer = msg.content or "(No response generated.)"
            final = AgentStep(
                step_type="final",
                content=answer,
                iteration=iteration,
            )
            result.steps.append(final)
            result.final_answer = answer
            yield final
            return result

        # ── Append assistant message (with tool_calls) to buffer ──────────────
        messages.append({
            "role": "assistant",
            "content": msg.content,    # may be None — Groq accepts None here
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments,  # raw JSON string
                    },
                }
                for tc in msg.tool_calls
            ],
        })

        # ── Execute each tool call ────────────────────────────────────────────
        for tc in msg.tool_calls:
            tool_name = tc.function.name

            try:
                tool_args: dict = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            # De-duplicate: same tool + same args = skip
            call_sig = f"{tool_name}::{json.dumps(tool_args, sort_keys=True)}"
            if call_sig in seen_calls:
                observation = "Skipped: identical call already made. Try different arguments."
            else:
                seen_calls.add(call_sig)

                # Yield action step before executing (so UI updates immediately)
                action_step = AgentStep(
                    step_type="action",
                    content=(
                        f"Calling **{tool_name}** with:\n"
                        f"```json\n{json.dumps(tool_args, indent=2, ensure_ascii=False)}\n```"
                    ),
                    tool_name=tool_name,
                    tool_args=tool_args,
                    iteration=iteration,
                )
                result.steps.append(action_step)
                yield action_step

                # Execute the tool
                tool_spec = TOOLS_BY_NAME.get(tool_name)
                if tool_spec:
                    try:
                        observation = tool_spec.func(tool_args)
                    except Exception as exc:
                        observation = f"Tool raised an exception: {exc}"
                else:
                    observation = f"Unknown tool '{tool_name}'. Available: {list(TOOLS_BY_NAME)}"

            # Yield observation step
            obs_step = AgentStep(
                step_type="observation",
                content=observation,
                tool_name=tool_name,
                iteration=iteration,
            )
            result.steps.append(obs_step)
            yield obs_step

            # Inject tool result back into the conversation
            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      observation,
            })

    # ── Max iterations reached ────────────────────────────────────────────────
    summary = "\n".join(
        s.content for s in result.steps if s.step_type == "observation"
    )
    timeout_content = (
        f"Maximum iterations ({max_iterations}) reached. "
        f"Here is a summary of what was found:\n\n{summary or '(no observations)'}"
    )
    timeout_step = AgentStep(
        step_type="final",
        content=timeout_content,
        iteration=max_iterations,
    )
    result.steps.append(timeout_step)
    result.final_answer = timeout_content
    yield timeout_step
    return result
