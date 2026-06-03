# AI Workbench — Claude Code Plugin Demo Guide

A 3-domain demo: **Quant Agent**, **ReAct Agent**, **Medical RAG** — invoked by natural language inside Claude Code via the `ai-workbench` plugin.

Audience: engineers. Total run time: ~8 min.

---

## 0. Pre-demo checklist (do this BEFORE the audience is watching)

```powershell
# From the repo root: c:\Users\DELL\Desktop\Fahad Data\MedicalChatBot

# 1. Virtualenv active + deps installed
.venv\Scripts\activate
pip install -r requirements.txt          # (already done; just confirm no errors)

# 2. GROQ_API_KEY present (Medical + ReAct need it; Quant does NOT)
Get-Content .env                          # confirm GROQ_API_KEY=gsk_...

# 3. FAISS index exists (Medical RAG). If missing, build it:
Test-Path vectorstore\db_faiss            # True = ready; False = run: python app.py

# 4. Smoke-test all three bridges (no Claude needed):
python .claude\scripts\quant_run.py BTC                 # must print a SIGNAL
python .claude\scripts\agent_run.py "what is 2+2"       # must print steps + final
python .claude\scripts\medical_query.py "test"          # must print answer + sources
```

**Optional servers** (only if you want to show the FastAPI path live):
```powershell
# Terminal A — REST API on :8000
uvicorn api.main:app --reload --port 8000
# Terminal B — Streamlit UI on :8501 (visual backdrop)
streamlit run medibot.py --server.fileWatcherType none
```
Open http://localhost:8000/docs and http://localhost:8501 in tabs as eye-candy. The Claude Code demo itself does **not** require either server — bridges import `src/` directly.

**Confirm the plugin loaded:** in Claude Code, type `/` and check `/quant`, `/agent`, `/medical` appear; skills `quant-agent`, `react-agent`, `medical-rag`, `workbench-orchestrator` are listed.

---

## Demo 1 — Quant Agent (open with this; zero dependencies, always works)

**Type into Claude Code:**
```
What's the quant signal on TSLA right now?
```

**What happens:** auto-triggers the `quant-agent` skill → runs `quant_run.py TSLA` → live Yahoo Finance data.

**Wow moment:** real-time price + a transparent BUY/SELL/HOLD with RSI/MACD/trend and a confidence %. No API key, no mocks — live market data through plain natural language.

**Expected shape:**
```
SIGNAL: HOLD (confidence 50%)
Price 422.71  24h +7.72% | RSI 53.9 | MACD neutral | trend sideways
Rationale: ... not financial advice.
```

**Slash-command variant (faster):** `/quant NVDA`

---

## Demo 2 — ReAct Agent (show the reasoning trace)

**Type into Claude Code:**
```
Use the ReAct agent to research the latest Groq LPU announcements and summarize the top 3 points.
```

**What happens:** auto-triggers `react-agent` → `agent_run.py` → reason → `search_web` → observe → loop → final answer.

**Wow moment:** the visible **🤔 thought → 🔧 action(tool) → 👁️ observation → ✅ final** trace. Engineers see the agent *decide* to search, not just answer. Point out it's the same `run_agent()` generator that powers the FastAPI `/agent/run/stream` endpoint.

**Math variant (deterministic, fast, no network flake):**
```
Ask the ReAct agent: what is the square root of 1764 times 3.14?
```
Expect it to call the `calculate` tool and return ~131.88.

**Slash variant:** `/agent research recent Groq model releases`

---

## Demo 3 — Medical RAG (cited answers from private PDFs)

**Type into Claude Code:**
```
Ask MediBot what my documents say about <topic in your ingested PDFs> — with sources.
```

**What happens:** auto-triggers `medical-rag` → `medical_query.py` → FAISS retrieval + Groq → answer + citations.

**Wow moment:** the answer ends with `[1] <file>, p.<page>` citations — grounded in *their* PDFs, not the model's memory. Contrast with a raw LLM that would hallucinate.

**Slash variant:** `/medical what does the handbook say about hypertension`

---

## Demo 4 — Orchestrator + Specialist (the closer: multi-domain chain)

**Type into Claude Code:**
```
Analyze TSLA for me: get the quant signal, then use the ReAct agent to find recent TSLA news, then summarize both into one report.
```

**What happens:** `workbench-orchestrator` routes and chains; for heavy chains it hands off to the `workbench-specialist` subagent (its own context window) which runs all three bridges and returns one synthesized report.

**Wow moment:** one English sentence fans out across three capabilities and comes back as a single brief — signal → news → synthesis. This is the "plugin = team-wide capability" payoff.

---

## Fallbacks (if something breaks mid-demo)

| Symptom | Fast fix / pivot |
|---|---|
| Quant: network/Yahoo timeout | Re-run with a different asset: `/quant BTC` then `ETH`. One usually responds instantly. |
| ReAct: web search flaky/slow | Pivot to the math prompt (`sqrt(1764)*3.14`) — uses `calculate`, no network. |
| Medical: "FAISS index missing" | `python app.py` to build it (needs PDFs in `Data/`), or skip to Demo 4 quant-only. |
| Any: "GROQ_API_KEY not set" | `Get-Content .env`; re-activate venv so `.env` loads. Quant still works without it. |
| Skill didn't auto-trigger | Use the slash command instead: `/quant`, `/agent`, `/medical`. |
| cp1252 / UnicodeEncodeError | Already patched (UTF-8 stdout). If seen, re-pull latest scripts. |

---

## Closing — show distribution

Pitch: *"This whole workbench is now a Claude Code plugin any teammate installs once."*

```
.claude/
  .claude-plugin/plugin.json     # name: ai-workbench, v0.1.0
  skills/        medical-rag, react-agent, quant-agent, workbench-orchestrator
  agents/        workbench-specialist (own context window)
  commands/      /medical  /agent  /quant
  scripts/       bridges into src/
```

**Closing prompts to type:**
```
List the ai-workbench plugin capabilities and how a teammate would invoke each.
```
```
Show me what the workbench-specialist subagent can do.
```

To distribute: commit `.claude/` to the repo — teammates get the plugin on clone. (Next step: publish to a plugin marketplace for one-command install.)
