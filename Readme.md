# MediBot — Medical Reference Chatbot

A retrieval-augmented chatbot that answers questions from medical reference PDFs. Built with Streamlit, LangChain, FAISS, and Groq.

## Features

- **Modern Streamlit UI** with sidebar controls and a clean chat layout.
- **Multiple Groq models** — switch between Llama 3.3 70B, Llama 3.1 8B, Gemma 2 9B, or Llama 3 70B at runtime.
- **Streaming responses** — answers appear token-by-token via `st.write_stream`.
- **Source citations** — every answer comes with an expandable panel showing the retrieved PDF chunks with file name and page number.
- **Multi-turn memory** — configurable rolling conversation history (0–8 turns).
- **Two answering modes:**
  - *Strict context-only* — refuses to answer questions outside the indexed material.
  - *Context + general knowledge* — supplements the context, marking general knowledge as such.
- **PDF upload** — drop new PDFs in the sidebar and they're chunked, embedded, and merged into the existing FAISS index.
- **Adjustable retrieval** — top-k slider (1–8).
- **Adjustable temperature** — 0.0 to 1.0.
- **Suggested starter questions** for first-time users.
- **Clear & export chat** — download the full conversation as Markdown.
- **Medical disclaimer banner** at the top of every page.
- **Index stats** — shows total number of indexed chunks in the sidebar.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_key_here
```

Get a free key at https://console.groq.com/keys.

## Build the index

Drop your PDFs into `Data/` then run:

```bash
python app.py
```

This populates `vectorstore/db_faiss/`. (You can also add PDFs at runtime via the sidebar uploader.)

## Run

```bash
streamlit run medibot.py
```

Opens at http://localhost:8501.

## CLI version

For a terminal Q&A loop without the UI:

```bash
python connect_memory_withllm.py
```

## Project layout

```
medibot.py                  Streamlit UI
app.py                      Build the FAISS index from Data/
connect_memory_withllm.py   CLI Q&A loop
src/
  helper.py                 Vectorstore, LLM, chain, ingestion helpers
  prompt.py                 System prompts + suggested questions
Data/                       PDFs to be indexed
vectorstore/db_faiss/       Persisted FAISS index
```

## Disclaimer

MediBot is for educational and reference use only. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for any medical concern.
