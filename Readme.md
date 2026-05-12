# MediBot + PSX-Sense + JobScout + RoadmapCoach

A four-in-one Streamlit workbench for a senior engineer — built on open-source LLMs (hosted free on Groq).

- **Tab 1: Bring-your-own-books RAG chatbot** — upload any PDFs (medical, finance, legal, technical) → chunked, embedded, indexed in FAISS, queried with citations.
- **Tab 2: Live PSX stock analyst** — chat grounded in real-time Pakistan Stock Exchange data scraped from `dps.psx.com.pk`.
- **Tab 3: Premium Jobs Finder** — aggregate high-paying global roles from RemoteOK, Remotive, Arbeitnow + optional Adzuna and JSearch (legal proxy for LinkedIn / Indeed / Glassdoor). Filter by salary, search, remote-only, then ask an LLM to rank fit and spot skill gaps from your pasted profile.
- **Tab 4: Top-1% Engineer Roadmap** — checklist across 10 skill tracks (Data Engineering / Snowflake, Data Analysis, Python, Oracle, Odoo, System Design, Cloud, AI/ML, DevOps, Full-Stack Craft) with progress persisted to disk and an LLM coach that picks your next milestones.

## Features

### 📚 Books / RAG tab
- **Open-source models on Groq** — GPT-OSS 120B (strongest reasoning), Llama 3.3 70B, Qwen 3 32B, Llama 3.1 8B (fastest).
- **Streaming responses** with source citations (file + page number).
- **Two modes** — Strict (context-only) or Assisted (supplements with general knowledge, clearly marked).
- **Conversational by default** — greetings ("hi", "thanks") get natural replies, not "I don't have information on that".
- **Runtime PDF upload** — chunked, embedded, merged into the FAISS index without restart.
- **Adjustable** top-k retrieval (1–8), temperature (0.0–1.0), conversation memory (0–8 turns).
- **Clear / export** chat as Markdown.

### 📈 Stocks tab
- **Live PSX market-watch** scraped from `dps.psx.com.pk` (cached 60s, ~487 symbols).
- **Top gainers / losers / volume leaders** panels with sortable tables.
- **Personal watchlist** — pick from common PSX tickers or type your own; watchlist quotes get injected into the LLM context.
- **Chat grounded in live data** — the LLM never invents prices, refuses to predict future moves, and always cites the live snapshot.
- **Market-status indicator** (Open / Closed in PKT).

### 💼 Jobs Finder tab
- **Multi-source aggregator** — fans out to RemoteOK, Remotive, Arbeitnow in parallel (no API key required). Optional Adzuna (10 countries with salary data) and JSearch (LinkedIn / Indeed / Glassdoor / ZipRecruiter listings) when their keys are present.
- **Filters** — keyword search, minimum USD-equivalent salary, remote-only toggle, source picker.
- **De-duplicated, salary-ranked table** — postings sorted by midpoint USD desc. Click any row to apply.
- **Profile-aware fit analysis** — paste your resume/profile in the sidebar; the LLM ranks the snapshot by fit, surfaces recurring skills in top-paying postings, and drafts cover-letter angles.
- **Why no direct LinkedIn scraping** — their ToS prohibits it and they IP-ban scrapers within minutes. JSearch is the legal route to the same listings.

### 🎯 Career Roadmap tab
- **10 default skill tracks** — Data Engineering (Snowflake / dbt / Airflow / Spark / Kafka), Data Analysis & BI, Python Mastery, Oracle & Databases, Odoo & ERP, System Design, Cloud (AWS / GCP / Azure), AI / ML Engineering, DevOps & Platform, Full-Stack Craft.
- **Milestone-level checkboxes** with notes — each milestone tagged `core / advanced / expert`.
- **Progress persisted** to `data/roadmap.json` — survives restarts. Reset-to-defaults button in the sidebar.
- **LLM mentor (RoadmapCoach)** — sees your full progress and recommends next milestones, sequencing, resources, and effort estimates based on what you've already checked.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=...

# Optional — enable extra job-board sources:
ADZUNA_APP_ID=...
ADZUNA_APP_KEY=...
RAPIDAPI_KEY=...
```

- Groq key (required): https://console.groq.com/keys
- Adzuna keys (optional, free ~250 calls/month, salary data): https://developer.adzuna.com/
- RapidAPI key (optional, free 200 calls/month, JSearch covers LinkedIn / Indeed / Glassdoor): https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch

## Run

```bash
streamlit run medibot.py
```

Switch between **Medical (RAG)**, **Stocks (Live PSX)**, **Jobs Finder**, and **Career Roadmap** in the sidebar.

## Build the RAG index from disk

Drop PDFs into `Data/`, then run:

```bash
python app.py
```

This populates `vectorstore/db_faiss/`. You can also upload PDFs at runtime from the Streamlit sidebar (Books mode).

## CLI version (RAG only)

```bash
python connect_memory_withllm.py
```

## Project layout

```
medibot.py                  Streamlit UI (Books + Stocks + Jobs + Roadmap)
app.py                      Build the FAISS index from Data/
connect_memory_withllm.py   CLI Q&A loop
src/
  helper.py                 FAISS, Groq, RAG chain, PDF ingestion
  prompt.py                 System prompts + suggested questions + PSX tickers
  stocks.py                 PSX scraper + LLM context formatter
  stock_chat.py             Groq streaming chat (reused by Jobs + Roadmap too)
  jobs.py                   Multi-source job aggregator (RemoteOK, Remotive,
                            Arbeitnow, Adzuna, JSearch)
  roadmap.py                Skill-track checklist model + persistence
Data/                       PDFs to be indexed
data/roadmap.json           Persisted roadmap progress (auto-seeded)
vectorstore/db_faiss/       Persisted FAISS index
```

## Models

All models are open-source weight; inference is hosted free on Groq's API.

| Model | Best for |
|-------|----------|
| `openai/gpt-oss-120b`     | Strongest reasoning (default) — OpenAI's open-weight Apache-2.0 release |
| `llama-3.3-70b-versatile` | General-purpose, balanced |
| `qwen/qwen3-32b`          | Reasoning, smaller |
| `llama-3.1-8b-instant`    | Fastest replies |

> **Note:** Groq decommissions models periodically. If you see `model_decommissioned` errors, check the live list at https://console.groq.com/docs/models and update `AVAILABLE_MODELS` in `src/helper.py` and `STOCK_LLM_MODELS` in `src/stock_chat.py`.

## Disclaimers

- **Books / RAG:** for educational and reference use. For medical, legal, or financial decisions, consult a qualified professional.
- **Stocks:** PSX-Sense is a data-analysis tool — *not* investment advice. It explains past and present numbers; it does not predict future prices. Markets are volatile; past performance does not predict future returns. Consult a SECP-licensed advisor before trading.
