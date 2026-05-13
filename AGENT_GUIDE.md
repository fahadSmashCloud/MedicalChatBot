# Agentic AI — Complete Technical Guide

A deep-dive into the architecture, patterns, and code behind the **Agentic AI**
module in this project. This guide is written so you can learn the concepts
from first principles and then map them directly to the production code.

---

## Table of Contents

1. [What is Agentic AI?](#1-what-is-agentic-ai)
2. [Architecture Overview](#2-architecture-overview)
3. [The ReAct Pattern](#3-the-react-pattern)
4. [Groq Function Calling — Deep Dive](#4-groq-function-calling--deep-dive)
5. [Code Walkthrough — File by File](#5-code-walkthrough--file-by-file)
6. [Tool System — Adding Your Own Tools](#6-tool-system--adding-your-own-tools)
7. [OpenAPI REST Layer](#7-openapi-rest-layer)
8. [Running the Stack](#8-running-the-stack)
9. [Advanced Topics](#9-advanced-topics)
10. [Learning Resources](#10-learning-resources)

---

## 1. What is Agentic AI?

### Simple chat vs. Agentic AI

| | Simple Chat (ChatBot) | Agentic AI |
|---|---|---|
| **Pattern** | Single LLM call | Multi-step reasoning loop |
| **Memory** | Conversation history | Conversation + Tool observations |
| **External world** | No access | Web, APIs, databases, code exec |
| **Planning** | None | Decides which tool to use next |
| **Predictability** | One response | Variable number of steps |
| **Best for** | Q&A, summarisation | Research, multi-step problem solving |

### The core insight

A plain LLM only knows what it was trained on (knowledge cutoff) and what
you tell it in the prompt. An **agent** can:

- **search the web** for live information
- **run calculations** precisely (no hallucination)
- **read web pages** for full article content
- **query a knowledge base** your own documents

The LLM becomes a *reasoning engine* that orchestrates external tools rather
than trying to recall everything from its weights.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACES                                │
│                                                                          │
│   ┌──────────────────────┐         ┌──────────────────────────────────┐ │
│   │  Streamlit Web UI    │         │   FastAPI REST API               │ │
│   │  (medibot.py)        │         │   (api/main.py)                  │ │
│   │                      │         │                                  │ │
│   │  Task text area  ────┼─────────┼──► POST /agent/run              │ │
│   │  Run button          │         │    POST /agent/run/stream        │ │
│   │  Step cards (UI) ◄───┼─────────┼──  GET  /agent/tools            │ │
│   └──────────────────────┘         └──────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────────────────┘
                             │ task (str)
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       AGENT CORE  (src/agent_core.py)                   │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │               ReAct Loop  (max 10 iterations)                     │  │
│  │                                                                   │  │
│  │    iteration 1                                                    │  │
│  │    ┌──────────┐   tool_calls JSON   ┌───────────────────────┐    │  │
│  │    │  REASON  │────────────────────►│  Tool Dispatcher      │    │  │
│  │    │  (Groq   │                     │  TOOLS_BY_NAME[name]  │    │  │
│  │    │   LLM)   │◄────────────────────│  .func(args) → str    │    │  │
│  │    └──────────┘   observation str   └───────────────────────┘    │  │
│  │         │                                                         │  │
│  │         │ yield AgentStep(thought/action/observation)             │  │
│  │         ▼                                                         │  │
│  │    iteration 2  →  iteration 3  →  …  →  Final Answer            │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  Groq model: llama-3.3-70b-versatile  (function calling enabled)        │
└──────────────────────────────────────────┬──────────────────────────────┘
                                           │ calls
                                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     TOOL REGISTRY  (src/agent_tools.py)                  │
│                                                                          │
│  ┌──────────────┐  ┌───────────────┐  ┌────────────┐  ┌─────────────┐  │
│  │  search_web  │  │   calculate   │  │ scrape_url │  │query_medical│  │
│  │              │  │               │  │            │  │_knowledge   │  │
│  │ DuckDuckGo   │  │ Python eval() │  │requests +  │  │ FAISS RAG   │  │
│  │ (no key)     │  │ + math module │  │ BeautifulS │  │ vectorstore │  │
│  └──────────────┘  └───────────────┘  └────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data flow for one complete agent run

```
1.  User types:   "What is 12! divided by the number of planets?"

2.  Agent THINKS: "I need to calculate 12 factorial and know the planet count."

3.  Agent ACTS:   calculate(expression="factorial(12)")
                  → "Result: 479001600"

4.  Agent THINKS: "Now I need the planet count. 8 planets in the solar system."

5.  Agent ACTS:   calculate(expression="479001600 / 8")
                  → "Result: 59875200.0"

6.  Agent ANSWERS: "12! = 479,001,600. Divided by 8 planets = 59,875,200."
```

---

## 3. The ReAct Pattern

### Origin

ReAct ("Reasoning + Acting") was introduced in the paper:

> **ReAct: Synergizing Reasoning and Acting in Language Models**
> Yao et al., 2022 — https://arxiv.org/abs/2210.03629

The key insight: interleaving **reasoning traces** (thoughts) with
**actions** (tool calls) produces far better results than pure reasoning
(chain-of-thought) or pure acting (tool-only) alone.

### The loop in pseudocode

```python
messages = [system_prompt, user_task]

for iteration in range(MAX_ITERATIONS):
    response = llm.chat(messages, tools=tool_definitions)

    if response.has_text_content:
        # REASON step — LLM explains its thinking
        yield AgentStep(type="thought", content=response.text)

    if not response.tool_calls:
        # No tool call = LLM is done reasoning → Final Answer
        yield AgentStep(type="final", content=response.text)
        break

    for tool_call in response.tool_calls:
        # ACT step — execute the chosen tool
        yield AgentStep(type="action", tool=tool_call.name)
        result = execute_tool(tool_call.name, tool_call.args)

        # OBSERVE step — inject result back
        yield AgentStep(type="observation", content=result)
        messages.append(tool_result_message(result))
```

### Why it works

```
Round 1:
  Thought → "I need live weather data for Karachi"
  Action  → search_web("Karachi weather today")
  Observe → "35°C, sunny, humidity 65%..."
       ↓
Round 2:
  Thought → "I have the data. I can answer now."
  Final   → "Karachi is 35°C and sunny today."
```

The LLM **sees its own previous observations** in the context window, so it
can chain multi-step reasoning naturally — just like how you think when
solving a problem: gather info → think → gather more → conclude.

---

## 4. Groq Function Calling — Deep Dive

### What is function calling?

When you call `client.chat.completions.create(..., tools=[...])`, you are
telling the LLM: "Here are tools you can call. If you want to use one,
respond with a structured JSON payload instead of plain text."

The LLM doesn't actually run any code — it just *describes* the call in
its response. Your code reads that description and runs the real function.

### Step-by-step protocol

**Step 1 — Define tools (JSON Schema)**

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for recent information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query":     {"type": "string", "description": "Search query"},
                    "n_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    }
]
```

**Step 2 — Call the LLM with tools**

```python
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=messages,
    tools=tools,
    tool_choice="auto",   # LLM decides whether to call a tool
    temperature=0.2,      # low temp → more deterministic tool selection
)
```

**Step 3 — Read the response**

```python
msg = response.choices[0].message

if msg.tool_calls:
    for tc in msg.tool_calls:
        name = tc.function.name               # "search_web"
        args = json.loads(tc.function.arguments)  # {"query": "Python news"}
        # → run your real search function here
```

**Step 4 — Inject tool result back**

```python
# The assistant message MUST be appended first (with tool_calls)
messages.append({
    "role": "assistant",
    "content": msg.content,    # may be None
    "tool_calls": [...]
})

# Then one "tool" role message per tool call
messages.append({
    "role":         "tool",
    "tool_call_id": tc.id,    # must match the id from step 3
    "content":      result,   # string — what the tool returned
})
```

**Step 5 — Call the LLM again** (it now sees the tool result)

The LLM will either:
- Call another tool (→ loop again)
- Return a `message.content` with no `tool_calls` (→ Final Answer)

### Message buffer evolution

```
[system]               ← stays fixed
[user: "task"]         ← stays fixed
                       ← after round 1:
[assistant: thought + tool_calls=[search_web(...)]]
[tool: result of search_web]
                       ← after round 2:
[assistant: "Based on the search results, here is my answer: ..."]
```

---

## 5. Code Walkthrough — File by File

### `src/agent_tools.py`

```
Purpose : Defines ToolSpec dataclass + all tool implementations.
Pattern : Registry pattern — TOOLS list + TOOLS_BY_NAME dict for O(1) lookup.
Adding  : Write _tool_X(args: dict) → str, append a ToolSpec to TOOLS.
```

Key data structure:

```python
@dataclass
class ToolSpec:
    name:        str               # tool identifier (used by LLM)
    description: str               # capability description (fed to LLM)
    parameters:  dict              # JSON Schema → passed to Groq tools API
    func:        Callable[[dict], str]  # actual implementation
```

**Calculator safety model:**

```python
_MATH_NS = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
_MATH_NS.update({"abs": abs, "round": round, "pow": pow})

result = eval(expression, {"__builtins__": {}}, _MATH_NS)
#                          ↑                    ↑
#                          No builtins          Only math functions
```

Setting `__builtins__` to `{}` prevents access to `import`, `open`, `os`,
`__class__`, etc. Combined with the forbidden-keyword regex, this is safe
for math evaluation in a controlled environment.

**Web scraper pipeline:**

```
requests.get(url)
    → BeautifulSoup(html, "lxml")
    → remove <script>, <style>, <nav>, <footer>
    → .get_text(separator="\n", strip=True)
    → collapse 3+ newlines
    → truncate to max_chars
```

---

### `src/agent_core.py`

```
Purpose : ReAct orchestrator — the main agent loop.
Pattern : Generator function that yields AgentStep objects.
Why gen : Callers can display each step incrementally (UI, SSE stream, etc.)
```

**Generator pattern:**

```python
def run_agent(task, model, max_iterations, ...) -> Generator[AgentStep, None, AgentResult]:
    # ...
    for iteration in range(max_iterations):
        response = groq_call(messages, tools)
        if response.tool_calls:
            yield AgentStep(type="action", ...)    # ← caller sees this now
            result = execute_tool(...)
            yield AgentStep(type="observation", ...) # ← and this
        else:
            yield AgentStep(type="final", ...)
            return AgentResult(...)                # ← return value of generator
```

Callers iterate with `for step in run_agent(...)`.

**De-duplication:**

```python
seen_calls: set[str] = set()
call_sig = f"{tool_name}::{json.dumps(tool_args, sort_keys=True)}"
if call_sig in seen_calls:
    observation = "Skipped: identical call already made."
```

This prevents infinite loops where the LLM repeatedly calls the same tool.

---

### `api/main.py`

```
Purpose : FastAPI REST API wrapping the agent.
Pattern : Thin layer — validates input, runs agent, serialises output.
Docs    : Auto-generated at /docs (Swagger UI) and /redoc (ReDoc).
```

**Three endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service status + config check |
| `/agent/tools` | GET | List all tools with JSON schemas |
| `/agent/run` | POST | Blocking run — waits for completion |
| `/agent/run/stream` | POST | Streaming NDJSON — one step per line |

**Streaming response:**

```python
def _generate() -> Iterator[str]:
    for step in run_agent(task, model, max_iterations):
        yield json.dumps({...step fields...}) + "\n"  # NDJSON

return StreamingResponse(_generate(), media_type="application/x-ndjson")
```

NDJSON (Newline-Delimited JSON) is perfect for streaming structured data:
each line is a valid JSON object, parseable independently.

---

## 6. Tool System — Adding Your Own Tools

Adding a tool takes exactly 3 steps:

### Step 1 — Write the implementation function

```python
# In src/agent_tools.py

def _tool_get_weather(args: dict) -> str:
    """
    Fetches current weather for a city using the Open-Meteo API (free, no key).
    """
    city = str(args.get("city", "")).strip()
    if not city:
        return "Error: 'city' argument is required."

    # Geocode city → lat/lon (using Open-Meteo geocoding)
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=5,
        ).json()
        result = geo["results"][0]
        lat, lon = result["latitude"], result["longitude"]

        # Fetch weather
        weather = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": True},
            timeout=5,
        ).json()
        cw = weather["current_weather"]
        return (
            f"Weather in {city}: {cw['temperature']}°C, "
            f"windspeed {cw['windspeed']} km/h, "
            f"weathercode {cw['weathercode']}"
        )
    except Exception as e:
        return f"Weather lookup failed: {e}"
```

### Step 2 — Register a ToolSpec

```python
TOOLS.append(
    ToolSpec(
        name="get_weather",
        description=(
            "Get the current weather for any city in the world. "
            "Returns temperature, windspeed, and weather condition."
        ),
        parameters={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'Karachi' or 'London'",
                },
            },
            "required": ["city"],
        },
        func=_tool_get_weather,
    )
)
```

### Step 3 — Done

Restart the app. The new tool:
- Appears in the sidebar tool checkboxes in Streamlit
- Appears in `GET /agent/tools` in the REST API
- Is available to the LLM for function calling

No other changes needed — the registry is fully automatic.

---

## 7. OpenAPI REST Layer

### Starting the API server

```bash
# From the project root:
uvicorn api.main:app --reload --port 8000

# With a specific host (for LAN access):
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Interactive documentation

| URL | Description |
|-----|-------------|
| `http://localhost:8000/docs` | Swagger UI — try endpoints interactively |
| `http://localhost:8000/redoc` | ReDoc — clean reference documentation |
| `http://localhost:8000/openapi.json` | Raw OpenAPI 3.0 schema (JSON) |

### API examples

**Health check:**

```bash
curl http://localhost:8000/health
# {
#   "status": "ok",
#   "groq_api_key_set": true,
#   "tools_available": 4,
#   "default_model": "llama-3.3-70b-versatile"
# }
```

**List tools:**

```bash
curl http://localhost:8000/agent/tools
# [{"name": "search_web", "description": "...", "parameters": {...}}, ...]
```

**Run agent (blocking):**

```bash
curl -X POST http://localhost:8000/agent/run \
  -H "Content-Type: application/json" \
  -d '{
    "task": "What is the square root of 1764?",
    "max_iterations": 5
  }'
# {
#   "final_answer": "The square root of 1764 is 42.",
#   "total_iterations": 1,
#   "steps": [
#     {"step_type": "action", "tool_name": "calculate", ...},
#     {"step_type": "observation", "content": "Result: 42.0", ...},
#     {"step_type": "final", "content": "The square root of 1764 is 42.", ...}
#   ]
# }
```

**Run agent (streaming NDJSON):**

```bash
curl -X POST http://localhost:8000/agent/run/stream \
  -H "Content-Type: application/json" \
  -d '{"task": "Search for Python 3.13 new features"}' \
  --no-buffer
# {"step_type": "thought", "content": "I will search the web...", "iteration": 1}
# {"step_type": "action", "tool_name": "search_web", "tool_args": {...}, "iteration": 1}
# {"step_type": "observation", "content": "[1] Python 3.13...", "iteration": 1}
# {"step_type": "final", "content": "Python 3.13 introduces...", "iteration": 2}
```

**Python client (streaming):**

```python
import requests
import json

resp = requests.post(
    "http://localhost:8000/agent/run/stream",
    json={"task": "What is the GDP of Pakistan in 2024?"},
    stream=True,
)
for line in resp.iter_lines():
    if line:
        step = json.loads(line)
        icon = {"thought": "💭", "action": "🔧", "observation": "👁", "final": "✅"}.get(
            step["step_type"], "•"
        )
        print(f"{icon} [{step['step_type']}] {step['content'][:100]}")
```

**JavaScript / fetch (streaming):**

```javascript
const response = await fetch("http://localhost:8000/agent/run/stream", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ task: "Search for AI news today" }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  const lines = decoder.decode(value).split("\n").filter(Boolean);
  for (const line of lines) {
    const step = JSON.parse(line);
    console.log(step.step_type, step.content.slice(0, 80));
  }
}
```

---

## 8. Running the Stack

### Option A — Streamlit only (default)

```bash
# Install all dependencies
pip install -r requirements.txt

# Set your Groq API key
echo "GROQ_API_KEY=gsk_..." >> .env

# Run the Streamlit app
streamlit run medibot.py --server.fileWatcherType none

# Open: http://localhost:8501
# → Click "🤖 Agentic AI" in the sidebar
```

### Option B — Streamlit + FastAPI (full stack)

Run both services in separate terminals:

```bash
# Terminal 1 — Streamlit
streamlit run medibot.py --server.fileWatcherType none

# Terminal 2 — FastAPI REST API
uvicorn api.main:app --reload --port 8000
```

Both services share the same Python environment and source code.
The FastAPI server auto-reloads when you edit `api/main.py` or `src/agent_*.py`.

### Option C — Docker (both services)

Add this service to `docker-compose.yml`:

```yaml
services:
  app:
    build: .
    ports:
      - "8501:8501"
    env_file: .env

  api:
    build: .
    command: uvicorn api.main:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    env_file: .env
```

---

## 9. Advanced Topics

### Memory and state across agent runs

The current implementation is **stateless** — each `run_agent()` call starts
fresh. For persistent memory, you can:

1. **Summarise previous runs** and inject them as `extra_system` context:

```python
summary = f"Previous session: User researched {topic}. Key findings: {findings}."
for step in run_agent(new_task, extra_system=summary):
    ...
```

2. **Store facts in a database** and retrieve them as a tool (similar to
   `query_medical_knowledge` but for agent-specific memory).

### Multi-agent orchestration

For complex tasks, you can route to specialised sub-agents:

```
Orchestrator Agent
    ├── Research Agent     (has: search_web, scrape_url)
    ├── Math Agent         (has: calculate only)
    └── Medical Agent      (has: query_medical_knowledge only)
```

Implement this by giving the orchestrator a `delegate_to_agent` tool:

```python
def _tool_delegate(args: dict) -> str:
    sub_agent = args["agent"]      # "research" | "math" | "medical"
    sub_task  = args["task"]
    enabled   = AGENT_TOOL_MAP[sub_agent]
    steps = list(run_agent(sub_task, enabled_tools=enabled))
    final = next((s for s in reversed(steps) if s.step_type == "final"), None)
    return final.content if final else "Sub-agent found no answer."
```

### Structured output + validation

Instead of asking the agent to return plain text, you can request structured
JSON output with Pydantic validation:

```python
from pydantic import BaseModel

class ResearchReport(BaseModel):
    summary: str
    key_facts: list[str]
    sources: list[str]
    confidence: float

# Ask the agent to output JSON matching this schema
task = f"Research X. Respond ONLY with valid JSON matching: {ResearchReport.schema_json()}"
```

### Rate limiting and cost control

The current code has no rate limiting. For production:

```python
import time

_last_call_time: float = 0

def _rate_limited_call(*args, **kwargs):
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < 1.0:   # minimum 1 second between calls
        time.sleep(1.0 - elapsed)
    _last_call_time = time.time()
    return client.chat.completions.create(*args, **kwargs)
```

### Async agent for FastAPI

For true async streaming (non-blocking I/O), use the async Groq client:

```python
from groq import AsyncGroq

async def run_agent_async(task: str):
    client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
    async with client as c:
        response = await c.chat.completions.create(...)
        # ...
        yield AgentStep(...)
```

This allows FastAPI to serve multiple concurrent agent requests without blocking.

---

## 10. Learning Resources

### Papers to read

| Paper | Why it matters |
|-------|----------------|
| [ReAct (2022)](https://arxiv.org/abs/2210.03629) | Foundation of this agent's reasoning pattern |
| [Toolformer (2023)](https://arxiv.org/abs/2302.04761) | Teaching LLMs to use tools via self-supervised learning |
| [HuggingGPT (2023)](https://arxiv.org/abs/2303.04671) | Using LLMs to orchestrate other AI models |
| [AutoGPT (2023)](https://arxiv.org/abs/2306.02224) | Long-horizon autonomous agent design |
| [Chain-of-Thought (2022)](https://arxiv.org/abs/2201.11903) | Basis for structured reasoning in LLMs |

### Groq / OpenAI documentation

- [Groq function calling](https://console.groq.com/docs/tool-use) — tool use protocol
- [Groq models](https://console.groq.com/docs/models) — model capabilities table
- [OpenAI function calling](https://platform.openai.com/docs/guides/function-calling) — same format

### Libraries used in this project

| Library | Purpose | Docs |
|---------|---------|------|
| `groq` | Groq Python SDK | https://github.com/groq/groq-python |
| `fastapi` | REST API framework | https://fastapi.tiangolo.com |
| `pydantic` | Data validation | https://docs.pydantic.dev |
| `duckduckgo-search` | Web search (no key) | https://github.com/deedy5/duckduckgo_search |
| `beautifulsoup4` | HTML parsing | https://www.crummy.com/software/BeautifulSoup |
| `uvicorn` | ASGI server | https://www.uvicorn.org |

### Courses

- **DeepLearning.AI — Building Agentic AI Systems** (free short courses)
  → https://www.deeplearning.ai/short-courses/
- **LangGraph** — state machine based agent framework
  → https://langchain-ai.github.io/langgraph/
- **Semantic Kernel** (Microsoft) — production agent framework
  → https://learn.microsoft.com/en-us/semantic-kernel/

---

## Appendix — Tool JSON Schema Reference

JSON Schema is the standard used to describe tool parameters.
The Groq/OpenAI function calling API follows it exactly.

```json
{
  "type": "object",
  "properties": {
    "name":         { "type": "string" },
    "count":        { "type": "integer", "minimum": 1, "maximum": 100 },
    "temperature":  { "type": "number", "default": 0.7 },
    "enabled":      { "type": "boolean" },
    "mode":         { "type": "string", "enum": ["fast", "careful", "balanced"] },
    "tags":         { "type": "array",  "items": { "type": "string" } }
  },
  "required": ["name"]
}
```

Types: `string`, `integer`, `number`, `boolean`, `array`, `object`

Optional fields: `description`, `default`, `minimum`, `maximum`, `enum`, `items`

Always include `"description"` for every property — the LLM uses descriptions
to understand what each argument means and how to fill it correctly.

---

*This guide covers the full implementation as of the current project version.
For questions or contributions, open an issue or PR on the repository.*
