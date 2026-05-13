"""Code Assistant — language presets, task shortcuts, and context helpers."""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Language presets
# Each preset tunes the system-prompt focus for that tech stack.
# ---------------------------------------------------------------------------
LANGUAGE_PRESETS: dict[str, dict[str, str]] = {
    "Python": {
        "icon": "🐍",
        "focus": (
            "Python 3.12+, type hints + mypy strict, async/await + asyncio, "
            "FastAPI + Pydantic v2, pytest + fixtures, uv / pyproject.toml packaging, "
            "performance profiling (py-spy, scalene), and Pythonic idioms."
        ),
    },
    "SQL / Oracle": {
        "icon": "🗄️",
        "focus": (
            "Oracle PL/SQL (procedures, packages, triggers), PostgreSQL, SQL Server — "
            "window functions, CTEs, recursive queries, execution plans (EXPLAIN/AWR), "
            "index design, partitioning strategies, and query optimisation."
        ),
    },
    "Odoo ORM": {
        "icon": "🟣",
        "focus": (
            "Odoo 17 ORM (models, fields, constraints, SQL views), custom module structure, "
            "XML views & QWeb reports, inheritance patterns (_inherit/_inherits), "
            "computed/related fields, security rules (ir.rule), Odoo.sh CI, and Studio."
        ),
    },
    "React / TypeScript": {
        "icon": "⚛️",
        "focus": (
            "React 19, Next.js 15 App Router (RSC, Server Actions, Suspense), "
            "TypeScript strict mode (generics, conditional types, discriminated unions), "
            "shadcn/ui, Zustand, React Query v5, and performance patterns (memo, useDeferredValue)."
        ),
    },
    "Docker / K8s": {
        "icon": "🐳",
        "focus": (
            "Dockerfile multi-stage builds, layer caching, docker-compose networking, "
            "Kubernetes (pods, services, ingress, HPA, resource limits), Helm charts, "
            "kubectl debugging, and GitHub Actions CI/CD pipelines."
        ),
    },
    "Data Engineering": {
        "icon": "🏗️",
        "focus": (
            "PySpark (DataFrames, UDFs, optimisation), dbt (models, tests, macros, semantic layer), "
            "Airflow DAGs (operators, sensors, XCom), Snowflake SQL + Snowpark, "
            "Delta Lake / Iceberg, Kafka consumers, and streaming with Flink."
        ),
    },
    "AI / ML": {
        "icon": "🤖",
        "focus": (
            "LLM API usage (Groq, OpenAI, Anthropic), prompt engineering, RAG pipelines "
            "(chunking, FAISS, pgvector), LangChain / LangGraph, fine-tuning (LoRA/QLoRA), "
            "eval frameworks (Ragas, lm-eval), and production deployment (vLLM, TGI)."
        ),
    },
    "General": {
        "icon": "💻",
        "focus": (
            "Any language or framework. Be as specific as possible in your question — "
            "include the language, framework version, and exact error if debugging."
        ),
    },
}

# ---------------------------------------------------------------------------
# Task shortcut templates — pre-fill the chat input
# ---------------------------------------------------------------------------
TASK_SHORTCUTS: list[tuple[str, str]] = [
    (
        "🔍 Review code",
        "Review this code for bugs, performance issues, security problems, and best practices."
        " Give me: Critical > Warning > Suggestion > Praise.\n\n```\n# paste your code here\n```",
    ),
    (
        "📖 Explain code",
        "Explain what this code does, step by step. Highlight any non-obvious parts.\n\n"
        "```\n# paste your code here\n```",
    ),
    (
        "🐛 Debug error",
        "Help me debug this error. Explain WHY it happens and show the fix.\n\n"
        "**Error:**\n```\n# paste error / traceback here\n```\n\n"
        "**Code:**\n```\n# paste relevant code here\n```",
    ),
    (
        "✍️ Write function",
        "Write a well-typed, production-ready function that:\n\n"
        "- Does: [describe what it should do]\n"
        "- Input: [describe parameters]\n"
        "- Output: [describe return value]\n"
        "- Edge cases to handle: [list any]",
    ),
    (
        "⚡ Optimize query",
        "Optimise this SQL query for performance. Show the original vs optimised version "
        "and explain the changes (index usage, join order, etc.).\n\n"
        "```sql\n-- paste your query here\n```",
    ),
    (
        "🧪 Write tests",
        "Write comprehensive pytest unit tests for this function. "
        "Include: happy path, edge cases, and error cases. Use fixtures where appropriate.\n\n"
        "```python\n# paste your function here\n```",
    ),
    (
        "🔐 Security review",
        "Do a security review of this code. Check for: injection, auth issues, "
        "insecure dependencies, secrets in code, and OWASP Top 10 vulnerabilities.\n\n"
        "```\n# paste your code here\n```",
    ),
    (
        "📐 Design pattern",
        "Suggest the best design pattern for this situation and show a concrete implementation:\n\n"
        "[Describe the problem / requirement here]",
    ),
]

# ---------------------------------------------------------------------------
# Suggested starter questions for the chat
# ---------------------------------------------------------------------------
CODE_SUGGESTED_QUESTIONS: list[str] = [
    "Review this code for production-readiness.",
    "What's the most Pythonic way to handle this pattern?",
    "Optimise this Snowflake query — it's scanning too much.",
    "How should I structure this Odoo module for maintainability?",
]
