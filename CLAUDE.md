# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**AI Workbench** — a single Streamlit app (`medibot.py`, ~2100 lines) exposing nine "domains" (tabs) over Groq LLMs: Medical RAG, Live PSX Stocks, Jobs Finder, Career Roadmap, Resume Analyzer, Code Assistant, Interview Guider, Agentic AI (ReAct), and Quant Agent. A separate FastAPI service (`api/main.py`) re-exposes the ReAct agent as a REST/OpenAPI endpoint.

Each domain's logic lives in its own `src/<domain>.py` module; `medibot.py` is the UI shell + router that imports them.

## Commands

```bash
# Run the app (port 8501)
streamlit run medibot.py
# Windows + Python 3.13 watchdog semaphore errors → disable file watcher:
streamlit run medibot.py --server.fileWatcherType none

# Build / rebuild the Medical RAG FAISS index from PDFs in Data/
python app.py
# (PDFs can also be uploaded live via the sidebar on the Medical tab — see ingest_uploaded_pdfs)

# Run the Agentic AI REST API (port 8000; docs at /docs)
uvicorn api.main:app --reload --port 8000

install:  pip install -r requirements.txt   # `-e .` installs this repo as a package via setup.py
```

There is no test suite, linter, or CI config. `test1.py` and `research/` are ad-hoc scratch files, not a test runner.

## Required configuration

`.env` in project root (loaded via `load_dotenv(find_dotenv())`):
- `GROQ_API_KEY` — **required**; every LLM call fails without it.
- `ADZUNA_APP_ID` / `ADZUNA_APP_KEY`, `RAPIDAPI_KEY` — optional, enable extra Jobs Finder sources.

## Architecture

### LLM access — two distinct call paths
The codebase uses Groq through **two separate clients**; don't conflate them:
1. **LangChain `ChatGroq`** (`src/helper.py:build_llm`) — only the Medical RAG chain, which needs retriever + prompt-template composition.
2. **Raw `groq.Groq` SDK** (`src/stock_chat.py`) — everything else. `stream_chat()` yields token deltas for chat UIs; `call_groq_json(temperature=0.1)` for structured/JSON outputs (resume, interview eval). Both wrap calls in exponential-backoff retry (1s, 2s) on 429/503, skipping retry on 401/403.

Model IDs live in two registries: `AVAILABLE_MODELS` (`helper.py`) and `STOCK_LLM_MODELS` (`stock_chat.py`). Default agent model is `llama-3.3-70b-versatile` (`agent_core.py:DEFAULT_MODEL`).

### Prompts are centralized
**All** system prompts and suggested-question lists live in `src/prompt.py` (~400 lines), imported by name into `medibot.py`. Add/edit prompt text there, not inline in the UI.

### Medical RAG pipeline (`src/helper.py` + `app.py`)
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace). FAISS index at `vectorstore/db_faiss/`.
- `build_chain()` returns a **(retrieve_runnable, answer_chain) tuple** — retrieval is a separate step so the UI can show sources (`format_docs_with_sources`).
- `load_vectorstore()` uses `allow_dangerous_deserialization=True` (trusted local index).
- Note: `app.py` is a standalone, older duplicate of the ingest logic in `helper.py` (`helper.ingest_uploaded_pdfs` is the live-upload path; both must stay consistent on chunk_size=500/overlap=50 and the embedding model).

### ReAct Agent (`src/agent_core.py` + `src/agent_tools.py`)
- `run_agent()` is a **generator** yielding `AgentStep` objects (`thought`/`action`/`observation`/`final`/`error`); both the Streamlit UI and FastAPI consume the same generator. Loops up to `MAX_ITERATIONS=10`.
- Uses Groq native function-calling (`tool_choice="auto"`), not text parsing.
- **Adding a tool:** write `_tool_<name>(args: dict) -> str` in `agent_tools.py`, append a `ToolSpec` to the `TOOLS` list. `agent_core`, `api/main.py`, and the UI pick it up automatically via `TOOLS` / `TOOLS_BY_NAME`. Tools: `search_web`, `calculate`, `scrape_url`, `query_medical_knowledge`.

### Quant Agent (`src/quant_agent.py`)
Synchronous pipeline of dataclass "agents": Market → TA (RSI/MACD/SMA/EMA) → News → Signal (BUY/SELL/HOLD). Keyless data from Yahoo Finance + CoinGecko. Asset list is hard-coded in `ASSET_UNIVERSE`. No DB, no async.

### Auth & domain gating (`src/auth.py`)
- SQLite at `data/users.db`, bcrypt-hashed passwords, two roles: `superadmin` and `user`. DB + a seeded superadmin (`SUPERADMIN_EMAIL` / `SUPERADMIN_DEFAULT_PASSWORD` constants) are created on first run.
- `medibot.py` blocks all content behind an auth gate (`render_auth_gate()` + `st.stop()`).
- **Domain visibility is role-based and defined in `medibot.py`:** `ADMIN_DOMAINS` (superadmin: all 9 + Admin Panel) vs `USER_DOMAINS` (regular users: Medical, Career Roadmap, Quant Agent only). When adding a domain, update both lists plus the `render_<domain>()` dispatch.

### UI conventions in `medibot.py`
- Each tab is a `render_<domain>()` function; the sidebar `st.radio` over `_domain_list` selects which one runs.
- State is held in `st.session_state` (auth user, per-domain chat histories, selected model).
- Large inline CSS block near the top defines `.main-header`, `.disclaimer*`, role badges, etc.

## Reference docs in repo
- `README.md` — user-facing feature/module overview and full model table.
- `AGENT_GUIDE.md` — deep dive on the ReAct agent + FastAPI layer.
- `QUANT_AGENT_GUIDE.md` — Quant Agent pipeline internals.
- `DEPLOYMENT.md` — Docker / cloud deployment.
