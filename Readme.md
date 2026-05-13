# AI Workbench

A production-grade, multi-module AI assistant built with Streamlit, Groq LLMs, PyTorch, and sentence-transformers. Seven specialised tabs, one unified interface.

---

## Modules

| Tab | Description | Key Tech |
|-----|-------------|----------|
| **Medical (RAG)** | Chat with your own PDF library — retrieval-augmented, cited answers | LangChain, FAISS, HuggingFace Embeddings |
| **PSX Stocks** | Live Pakistan Stock Exchange analyst — streaming chat grounded in real-time data | Groq streaming, BeautifulSoup |
| **Jobs Finder** | Aggregates RemoteOK, Remotive, Arbeitnow, Adzuna, JSearch; LLM fit analysis | REST APIs, Groq |
| **Career Roadmap** | Personalised Top-1% engineer checklist with progress tracking and coaching chat | JSON persistence, Groq |
| **Resume Analyzer** | PDF → ATS score → STAR bullet rewrites → cover letter → interview prep | PyPDFLoader, Groq JSON, LangChain |
| **Code Assistant** | 8 language presets, 8 task shortcuts, CodeSensei streaming chat | Groq streaming |
| **Interview Guider** | Hard questions by topic & difficulty, dual scoring (PyTorch + LLM), coaching chat | sentence-transformers, PyTorch, Groq |

---

## ML / AI Stack

### sentence-transformers + PyTorch (Interview Guider)

The Interview Guider is the only module that uses PyTorch directly for numeric scoring:

```python
# src/interview_guide.py

import torch
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def compute_semantic_score(user_answer: str, ideal_answer: str) -> float:
    # Step 1: Encode both strings → (2, 384) float32 torch.Tensor
    embeddings: torch.Tensor = model.encode(
        [user_answer, ideal_answer],
        convert_to_tensor=True,
        normalize_embeddings=True,   # L2-normalise for stable cosine
    )

    # Step 2: Dot product of L2-normalised vectors == cosine similarity
    # torch.dot is explicit and avoids 2-D matrix overhead for 1-D vectors
    cos_sim: float = torch.dot(embeddings[0], embeddings[1]).item()

    # Step 3: Clamp to [0, 1], scale to [0, 100]
    return round(max(0.0, min(1.0, cos_sim)) * 100, 1)
```

**Why two scoring signals?**

| Signal | Source | What it measures |
|--------|--------|-----------------|
| LLM Score (0–10) | Groq LLM | Logical correctness, depth, trade-offs, communication |
| Semantic Match (0–100%) | PyTorch cosine | Vocabulary and concept overlap with ideal answer |

LLMs can be generous or inconsistent scorers. The embedding-based score is deterministic and purely semantic — combining both gives a richer, more calibrated picture.

### FAISS Vector Store (Medical RAG)

```python
# src/helper.py
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Same all-MiniLM-L6-v2 model reused — no extra download needed
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(chunks, embeddings)
```

### Groq LLM (all modules)

Two call patterns depending on the use case:

```python
# Streaming — for conversational responses (src/stock_chat.py)
def stream_chat(...) -> Iterator[str]:
    stream = client.chat.completions.create(..., stream=True)
    for chunk in stream:
        if delta := chunk.choices[0].delta.content:
            yield delta

# Non-streaming + JSON — for structured outputs (resume analysis, interview eval)
def call_groq_json(...) -> str:
    response = client.chat.completions.create(
        ..., temperature=0.1, stream=False
    )
    return response.choices[0].message.content
```

Both patterns include exponential backoff retry (1 s, 2 s) for transient 429/503 errors, skipping retry on auth errors (401/403).

---

## Quick Start

### Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com/keys)

### Installation

```bash
git clone <repo-url>
cd MedicalChatBot

# Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
# Required
GROQ_API_KEY=gsk_your_key_here

# Optional — enables Adzuna job source
ADZUNA_APP_ID=your_adzuna_app_id
ADZUNA_APP_KEY=your_adzuna_app_key

# Optional — enables JSearch (LinkedIn/Indeed/Glassdoor proxy via RapidAPI)
RAPIDAPI_KEY=your_rapidapi_key
```

### Run the app

```bash
# Standard
streamlit run medibot.py

# Windows Python 3.13 — use this if you get watchdog semaphore errors
streamlit run medibot.py --server.fileWatcherType none
```

Open `http://localhost:8501` in your browser.

### Build the Medical RAG index (optional)

```bash
# Place PDFs in data/ then:
python app.py
# OR upload PDFs directly via the sidebar → Medical tab
```

---

## Project Structure

```
MedicalChatBot/
├── medibot.py               # Streamlit app — 7 tabs, sidebar, routing (~1100 lines)
├── app.py                   # One-shot FAISS index builder
├── requirements.txt         # All dependencies with ML library annotations
├── .env                     # API keys (git-ignored)
├── README.md                # This file
├── DEPLOYMENT.md            # Docker, cloud deployment guide
│
├── src/
│   ├── helper.py            # LangChain RAG pipeline, FAISS helpers, LLM builder
│   ├── prompt.py            # All system prompts and suggested questions
│   ├── stock_chat.py        # Groq streaming + JSON call with retry + backoff
│   ├── stocks.py            # PSX live data scraper
│   ├── jobs.py              # Multi-source job aggregator (5 sources)
│   ├── roadmap.py           # Engineer roadmap data and JSON persistence
│   ├── resume_analyzer.py   # PDF extraction, ATS scoring, markdown rendering
│   ├── code_assistant.py    # Language presets and task shortcut templates
│   └── interview_guide.py   # PyTorch semantic scoring, question/eval logic
│
├── data/                    # Source PDFs for Medical RAG
├── vectorstore/db_faiss/    # FAISS index (auto-created after ingestion)
└── data/roadmap.json        # User roadmap progress (auto-created)
```

---

## Interview Guider — Deep Dive

### Topics (8 areas, 6 subtopics each)

| Topic | Subtopics |
|-------|-----------|
| Data Structures & Algorithms | Arrays & Strings, Trees & Graphs, Dynamic Programming, Sorting & Searching, Hash Maps, Recursion |
| System Design | Scalability, DB Design & Sharding, Caching, Message Queues, Microservices, CAP Theorem |
| Python Internals | GIL & Concurrency, Async/Await, Decorators & Metaclasses, Generators, Type Hints, Memory |
| SQL & Databases | Query Optimisation, Indexing, ACID, Window Functions, Partitioning, NoSQL vs SQL |
| Machine Learning & AI | ML Fundamentals, Neural Networks, LLMs & Prompts, RAG & Vector DBs, Model Eval, MLOps |
| Behavioral (STAR) | Leadership, Conflict, Project Failures, Cross-team, Ownership, Ambiguity |
| API Design | REST, Auth, Rate Limiting, Versioning, GraphQL vs REST vs gRPC, OpenAPI |
| DevOps & Cloud | Docker, Kubernetes, CI/CD, Observability, Incident Response, Terraform |

### Difficulty Levels

| Level | Expectation |
|-------|-------------|
| Junior (0–2 yrs) | Fundamentals, common patterns, correctness |
| Mid-level (3–5 yrs) | Trade-offs, edge cases, design decisions |
| Senior (5–8 yrs) | Architecture, failure at scale, mentoring trade-offs |
| Staff/Principal (8+ yrs) | Ambiguous open-ended problems, org-level impact, technical vision |

### Scoring Rubric

Each answer is evaluated on two independent axes:

**LLM rubric (Groq, temp=0.1):**

| Dimension | What it scores |
|-----------|----------------|
| Conceptual Accuracy | Correct coverage of expected concepts |
| Completeness | All expected concepts present |
| Depth & Nuance | Trade-offs, edge cases, real-world wisdom |
| Communication | Clear structure, terminology, conciseness |
| Overall / Hire Signal | 9–10 = Strong hire, 7–8 = Hire, 5–6 = Maybe, <5 = No hire |

**Semantic Match (PyTorch, deterministic):**
- Cosine similarity between your answer embedding and ideal answer embedding
- Range: 0–100%
- Uses `all-MiniLM-L6-v2` — same model as the FAISS vector store

---

## Environment Variables

| Variable | Required | Source |
|----------|----------|--------|
| `GROQ_API_KEY` | Yes | [console.groq.com/keys](https://console.groq.com/keys) |
| `ADZUNA_APP_ID` | No | [developer.adzuna.com](https://developer.adzuna.com) |
| `ADZUNA_APP_KEY` | No | [developer.adzuna.com](https://developer.adzuna.com) |
| `RAPIDAPI_KEY` | No | [rapidapi.com](https://rapidapi.com) — JSearch API |

---

## Available LLM Models

| Display Name | Model ID | Best For |
|-------------|----------|----------|
| GPT-OSS 120B (strongest) | `openai/gpt-oss-120b` | Complex reasoning, system design |
| Llama 3.3 70B (balanced) | `llama-3.3-70b-versatile` | General purpose (default) |
| Qwen 3 32B (reasoning) | `qwen/qwen3-32b` | Structured JSON, analysis |
| Llama 3.1 8B (fastest) | `llama-3.1-8b-instant` | Quick Q&A, low latency |

---

## Sidebar Health Status

| Indicator | Meaning |
|-----------|---------|
| `🟢 Groq API` | GROQ_API_KEY is configured |
| `🟢/🟡 FAISS index` | Vector store exists / missing |
| `🟢/⚪ JSearch` | RAPIDAPI_KEY configured / not set |
| `🟢/⚪ Adzuna` | ADZUNA_APP_ID configured / not set |

---

## License

MIT
