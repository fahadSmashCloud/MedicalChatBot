"""Agent tool implementations.

Each tool is described by a ToolSpec dataclass that carries:
  - name        : identifier used by the LLM in function-calling
  - description : natural-language capability description (fed to the LLM)
  - parameters  : JSON Schema object (OpenAI/Groq function-calling format)
  - func        : callable that accepts the parsed args dict and returns a str

ADDING A NEW TOOL
-----------------
1. Write a ``_tool_<name>`` function that accepts ``args: dict`` and returns ``str``.
2. Append a ``ToolSpec`` entry to the ``TOOLS`` list at the bottom of this file.
3. That's it — the agent core and FastAPI layer pick it up automatically.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable

import requests
from bs4 import BeautifulSoup

try:
    from duckduckgo_search import DDGS as _DDGS
    _DDGS_OK = True
except ImportError:
    _DDGS_OK = False


# ── ToolSpec ──────────────────────────────────────────────────────────────────

@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict          # JSON Schema — passed verbatim to Groq tools API
    func: Callable[[dict], str]


# ═══════════════════════════════════════════════════════════════════════════════
# Tool implementations
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1. Web search via DuckDuckGo ──────────────────────────────────────────────

def _tool_search_web(args: dict) -> str:
    """
    Uses the duckduckgo-search library (no API key needed).
    Returns up to n_results results, each with title, URL, and snippet.
    """
    query = str(args.get("query", "")).strip()
    n_results = max(1, min(int(args.get("n_results", 5)), 10))

    if not query:
        return "Error: 'query' argument is required."
    if not _DDGS_OK:
        return "Error: duckduckgo-search package not installed. Run: pip install duckduckgo-search"

    try:
        with _DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=n_results))
        if not hits:
            return "No search results found for that query."
        parts = [
            f"[{i}] {h.get('title', '(no title)')}\n"
            f"URL: {h.get('href', '')}\n"
            f"{h.get('body', '')}"
            for i, h in enumerate(hits, 1)
        ]
        return "\n\n".join(parts)
    except Exception as exc:
        return f"Search error: {exc}"


# ── 2. Safe math calculator ───────────────────────────────────────────────────

_MATH_NS: dict = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
_MATH_NS.update({"abs": abs, "round": round, "pow": pow, "int": int, "float": float})

_FORBIDDEN_CALC = re.compile(
    r"\b(import|exec|eval|open|__|\bos\b|\bsys\b|\bsubprocess\b|\bfile\b)\b",
    re.IGNORECASE,
)


def _tool_calculate(args: dict) -> str:
    """
    Evaluates math expressions safely using Python's eval() with a restricted
    namespace containing only the math module and basic builtins (abs, round, pow).

    No file system, network, or import access is possible because:
      - __builtins__ is set to {} (empty)
      - The namespace only exposes math functions
      - A regex blocklist rejects obviously dangerous tokens

    Supported: +, -, *, /, //, %, **, sqrt, log, sin, cos, tan, pi, e, floor,
               ceil, factorial, gcd, hypot, degrees, radians, and all math.*
    """
    expr = str(args.get("expression", "")).strip()
    if not expr:
        return "Error: 'expression' argument is required."
    if _FORBIDDEN_CALC.search(expr):
        return "Error: forbidden keyword in expression."
    try:
        result = eval(expr, {"__builtins__": {}}, _MATH_NS)  # noqa: S307
        return f"Result: {result}"
    except ZeroDivisionError:
        return "Error: division by zero."
    except Exception as exc:
        return f"Calculation error: {exc}"


# ── 3. Web page scraper ───────────────────────────────────────────────────────

_SCRAPE_STRIP_TAGS = ["script", "style", "nav", "footer", "header", "aside", "form"]
_NEWLINE_RE = re.compile(r"\n{3,}")


def _tool_scrape_url(args: dict) -> str:
    """
    Fetches a URL, strips navigation/scripts/styles, and returns clean text.
    Truncated at max_chars (default 3000) to stay within LLM context limits.
    """
    url = str(args.get("url", "")).strip()
    max_chars = max(200, min(int(args.get("max_chars", 3000)), 10_000))

    if not url:
        return "Error: 'url' argument is required."
    if not url.startswith(("http://", "https://")):
        return "Error: URL must start with http:// or https://"

    try:
        resp = requests.get(
            url,
            timeout=12,
            headers={"User-Agent": "Mozilla/5.0 (compatible; AIWorkbench/1.0)"},
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        for tag in soup(_SCRAPE_STRIP_TAGS):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        text = _NEWLINE_RE.sub("\n\n", text)
        truncated = text[:max_chars]
        if len(text) > max_chars:
            truncated += f"\n\n[... truncated at {max_chars} chars ...]"
        return truncated
    except requests.HTTPError as exc:
        return f"HTTP error {exc.response.status_code}: {exc}"
    except Exception as exc:
        return f"Scrape error: {exc}"


# ── 4. Medical knowledge base (FAISS RAG) ────────────────────────────────────

def _tool_query_medical(args: dict) -> str:
    """
    Performs a semantic similarity search against the FAISS vector store that
    was built from ingested PDF documents in the Medical (RAG) tab.

    Requires:
      - At least one PDF ingested via the Medical tab (vectorstore/db_faiss/)
      - GROQ_API_KEY set (for the embedding model via HuggingFace)

    Returns the top-k most relevant chunks from the knowledge base.
    """
    query = str(args.get("query", "")).strip()
    k = max(1, min(int(args.get("k", 3)), 6))

    if not query:
        return "Error: 'query' argument is required."

    try:
        from src.helper import load_vectorstore   # lazy import — heavy model load
        db = load_vectorstore()
        docs = db.similarity_search(query, k=k)
        if not docs:
            return "No relevant information found in the medical knowledge base."
        parts = [
            f"[Source {i}]\n{doc.page_content}"
            for i, doc in enumerate(docs, 1)
        ]
        return "\n\n".join(parts)
    except FileNotFoundError:
        return (
            "Medical knowledge base not found. "
            "Please ingest PDF documents via the Medical (RAG) tab first."
        )
    except Exception as exc:
        return f"Medical knowledge base error: {exc}"


# ═══════════════════════════════════════════════════════════════════════════════
# Tool Registry — the agent core and API layer iterate over this list
# ═══════════════════════════════════════════════════════════════════════════════

TOOLS: list[ToolSpec] = [
    ToolSpec(
        name="search_web",
        description=(
            "Search the web for current information, news, facts, or any topic "
            "that requires up-to-date knowledge. Returns titles, URLs, and snippets."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string",
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of results to return (1–10, default 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
        func=_tool_search_web,
    ),
    ToolSpec(
        name="calculate",
        description=(
            "Safely evaluate mathematical expressions. Supports arithmetic, "
            "math.sqrt, math.log, math.sin, math.cos, math.pi, math.e, "
            "math.factorial, math.floor, math.ceil, and all standard math functions. "
            "Example: 'sqrt(144) + 2**8' or 'factorial(10) / 1000'"
        ),
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Python math expression to evaluate",
                },
            },
            "required": ["expression"],
        },
        func=_tool_calculate,
    ),
    ToolSpec(
        name="scrape_url",
        description=(
            "Fetch and extract the text content from any web page URL. "
            "Strips navigation, ads, scripts, and styles. "
            "Use this to read articles, documentation, or any web content."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full URL (must start with http:// or https://)",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return (200–10000, default 3000)",
                    "default": 3000,
                },
            },
            "required": ["url"],
        },
        func=_tool_scrape_url,
    ),
    ToolSpec(
        name="query_medical_knowledge",
        description=(
            "Search the local medical knowledge base (built from ingested PDFs) "
            "for relevant information on a medical topic, drug, symptom, or condition. "
            "Only useful after PDFs have been ingested in the Medical tab."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Medical question or topic to search for",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of document chunks to retrieve (1–6, default 3)",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
        func=_tool_query_medical,
    ),
]

# Fast name-based lookup used by the agent core dispatcher
TOOLS_BY_NAME: dict[str, ToolSpec] = {t.name: t for t in TOOLS}
