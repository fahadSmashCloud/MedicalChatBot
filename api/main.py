"""FastAPI REST API — Agentic AI as an OpenAPI service.

HOW TO RUN
──────────
    # From the project root:
    uvicorn api.main:app --reload --port 8000

    # Then open the interactive docs:
    http://localhost:8000/docs          ← Swagger UI
    http://localhost:8000/redoc         ← ReDoc
    http://localhost:8000/openapi.json  ← raw OpenAPI 3.0 schema

ENDPOINTS
─────────
    GET  /                  → redirect to /docs
    GET  /health            → service + tool availability
    GET  /agent/tools       → list all tools with their JSON schemas
    POST /agent/run         → blocking agent run (returns full trace)
    POST /agent/run/stream  → streaming agent run (NDJSON, one step per line)

AUTHENTICATION
──────────────
    Set GROQ_API_KEY as an environment variable (or in a .env file).
    The API itself has no authentication — add a reverse proxy (nginx + JWT)
    for production deployments.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterator

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

# ── Path setup — make src/ importable when running from project root ──────────
sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv(find_dotenv())

from src.agent_core import (
    DEFAULT_MODEL,
    MAX_ITERATIONS,
    AgentStep,
    run_agent,
)
from src.agent_tools import TOOLS

# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI app
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="AI Workbench — Agentic AI",
    description="""
## Overview
A **ReAct-pattern** AI agent powered by [Groq](https://groq.com) LLMs with
real-world tool use — web search, math calculation, web scraping, and medical
knowledge retrieval.

## How it works
1. Send a **task** string to `/agent/run`.
2. The agent **reasons** (Thought), **acts** (Tool call), **observes** (Tool result).
3. It loops until it has a complete answer (max 10 iterations by default).
4. The full reasoning trace is returned alongside the final answer.

## Tools available
| Tool | Description |
|------|-------------|
| `search_web` | DuckDuckGo web search (no API key required) |
| `calculate` | Safe math expression evaluator |
| `scrape_url` | Web page text extractor |
| `query_medical_knowledge` | FAISS RAG search over ingested PDFs |

## Streaming
Use `/agent/run/stream` to receive each step as a JSON line (NDJSON)
as soon as it is produced — ideal for real-time UIs or SSE proxies.

## Setup
Set `GROQ_API_KEY` as an environment variable before starting the server.
""",
    version="1.0.0",
    contact={
        "name":  "Fahad",
        "email": "ijazfahad683@gmail.com",
    },
    license_info={"name": "MIT"},
    openapi_tags=[
        {"name": "health",  "description": "Service availability and configuration"},
        {"name": "tools",   "description": "Inspect available agent tools"},
        {"name": "agent",   "description": "Run the ReAct agent on a task"},
    ],
)


# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic models (request / response)
# ═══════════════════════════════════════════════════════════════════════════════

class RunRequest(BaseModel):
    """Request body for POST /agent/run and /agent/run/stream."""

    task: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The task or question for the agent to solve.",
        examples=["What is the square root of 1764 times 3.14?"],
    )
    model: str = Field(
        DEFAULT_MODEL,
        description=(
            "Groq model ID. Must support function calling. "
            "Recommended: llama-3.3-70b-versatile (default), "
            "llama-3.1-8b-instant (faster, less capable)."
        ),
    )
    max_iterations: int = Field(
        MAX_ITERATIONS,
        ge=1,
        le=20,
        description="Maximum reasoning iterations before the agent is forced to stop.",
    )
    enabled_tools: list[str] | None = Field(
        None,
        description=(
            "Optional list of tool names to enable. "
            "If null, all tools are enabled. "
            "Example: [\"search_web\", \"calculate\"]"
        ),
    )

    @field_validator("enabled_tools")
    @classmethod
    def validate_tools(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        valid_names = {t.name for t in TOOLS}
        bad = [n for n in v if n not in valid_names]
        if bad:
            raise ValueError(
                f"Unknown tool(s): {bad}. Valid names: {sorted(valid_names)}"
            )
        return v


class ToolParameterDoc(BaseModel):
    """A single tool's public documentation."""
    name: str
    description: str
    parameters: dict


class AgentStepOut(BaseModel):
    """One step in the agent's reasoning trace."""
    step_type: str = Field(description="thought | action | observation | final | error")
    content: str
    tool_name: str | None = None
    tool_args: dict | None = None
    iteration: int


class AgentRunResponse(BaseModel):
    """Full response for blocking /agent/run."""
    final_answer: str
    total_iterations: int
    steps: list[AgentStepOut]
    error: str | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _step_to_out(step: AgentStep) -> AgentStepOut:
    return AgentStepOut(
        step_type=step.step_type,
        content=step.content,
        tool_name=step.tool_name,
        tool_args=step.tool_args,
        iteration=step.iteration,
    )


def _check_groq_key() -> None:
    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GROQ_API_KEY is not configured on this server.",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect root to the interactive Swagger docs."""
    return RedirectResponse("/docs")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    tags=["health"],
    summary="Health check",
    description="Returns the service status and configuration summary.",
)
def health() -> dict:
    return {
        "status":            "ok",
        "groq_api_key_set":  bool(os.environ.get("GROQ_API_KEY")),
        "tools_available":   len(TOOLS),
        "default_model":     DEFAULT_MODEL,
        "max_iterations":    MAX_ITERATIONS,
    }


# ── Tools ─────────────────────────────────────────────────────────────────────

@app.get(
    "/agent/tools",
    response_model=list[ToolParameterDoc],
    tags=["tools"],
    summary="List all agent tools",
    description=(
        "Returns every registered tool with its name, description, "
        "and JSON Schema parameter definition."
    ),
)
def list_tools() -> list[ToolParameterDoc]:
    return [
        ToolParameterDoc(
            name=t.name,
            description=t.description,
            parameters=t.parameters,
        )
        for t in TOOLS
    ]


# ── Blocking agent run ────────────────────────────────────────────────────────

@app.post(
    "/agent/run",
    response_model=AgentRunResponse,
    tags=["agent"],
    summary="Run agent (blocking)",
    description=(
        "Run the ReAct agent on a task. **Blocks until the agent finishes** "
        "and returns the complete reasoning trace + final answer. "
        "For real-time streaming, use `/agent/run/stream` instead."
    ),
)
def run_agent_blocking(req: RunRequest) -> AgentRunResponse:
    _check_groq_key()

    steps: list[AgentStep] = []
    final_answer = ""
    error = None

    try:
        gen = run_agent(
            task=req.task,
            model=req.model,
            max_iterations=req.max_iterations,
            enabled_tools=req.enabled_tools,
        )
        for step in gen:
            steps.append(step)
            if step.step_type == "final":
                final_answer = step.content
            if step.step_type == "error":
                error = step.content
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    tool_calls = [s for s in steps if s.step_type == "action"]

    return AgentRunResponse(
        final_answer=final_answer,
        total_iterations=len(tool_calls),
        steps=[_step_to_out(s) for s in steps],
        error=error,
    )


# ── Streaming agent run ───────────────────────────────────────────────────────

@app.post(
    "/agent/run/stream",
    tags=["agent"],
    summary="Run agent (streaming NDJSON)",
    description=(
        "Run the ReAct agent and **stream each step** as a newline-delimited "
        "JSON (NDJSON) line. Suitable for SSE proxies, WebSocket bridges, or "
        "any client that can read a streaming HTTP response.\n\n"
        "Each line is a JSON object matching the `AgentStepOut` schema.\n\n"
        "Example client (Python):\n"
        "```python\n"
        "import requests, json\n"
        "resp = requests.post(\n"
        "    'http://localhost:8000/agent/run/stream',\n"
        "    json={'task': 'What is pi to 10 decimal places?'},\n"
        "    stream=True\n"
        ")\n"
        "for line in resp.iter_lines():\n"
        "    step = json.loads(line)\n"
        "    print(step['step_type'], step['content'][:60])\n"
        "```"
    ),
    response_class=StreamingResponse,
)
def run_agent_stream(req: RunRequest) -> StreamingResponse:
    _check_groq_key()

    def _generate() -> Iterator[str]:
        try:
            for step in run_agent(
                task=req.task,
                model=req.model,
                max_iterations=req.max_iterations,
                enabled_tools=req.enabled_tools,
            ):
                payload = {
                    "step_type": step.step_type,
                    "content":   step.content,
                    "tool_name": step.tool_name,
                    "tool_args": step.tool_args,
                    "iteration": step.iteration,
                }
                yield json.dumps(payload, ensure_ascii=False) + "\n"
        except Exception as exc:
            yield json.dumps({"step_type": "error", "content": str(exc)}) + "\n"

    return StreamingResponse(_generate(), media_type="application/x-ndjson")
