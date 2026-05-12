"""Personal Top-1% engineer roadmap.

Skill tracks are persisted to data/roadmap.json so progress survives
between sessions.  On first launch, the file is seeded with the default
roadmap (DEFAULT_TRACKS below).

Schema:
    {
      "updated": "2026-05-12T12:30:00",
      "tracks": [
        {
          "id":          "data_engineering",
          "name":        "Data Engineering",
          "description": "...",
          "milestones": [
            {"id": "de-1", "title": "...", "level": "core|advanced|expert",
             "done": false, "notes": ""},
            ...
          ]
        },
        ...
      ]
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

ROADMAP_PATH = Path("data/roadmap.json")


@dataclass
class Milestone:
    id: str
    title: str
    level: str = "core"     # core | advanced | expert
    done: bool = False
    notes: str = ""


@dataclass
class Track:
    id: str
    name: str
    description: str
    milestones: list[Milestone] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.milestones)

    @property
    def done_count(self) -> int:
        return sum(1 for m in self.milestones if m.done)

    @property
    def progress(self) -> float:
        return self.done_count / self.total if self.total else 0.0


# ---------------------------------------------------------------------------
# DEFAULT TRACKS — seeded on first run, editable in-app afterwards.
# ---------------------------------------------------------------------------
DEFAULT_TRACKS: list[Track] = [
    Track(
        id="data_engineering",
        name="Data Engineering",
        description="Pipelines, warehouses, big-data — the modern stack.",
        milestones=[
            Milestone("de-1",  "SQL: window functions, CTEs, query plans", "core"),
            Milestone("de-2",  "Snowflake architecture + SnowSQL + Snowpark", "core"),
            Milestone("de-3",  "dbt: models, tests, exposures, semantic layer", "core"),
            Milestone("de-4",  "Airflow / Prefect / Dagster orchestration", "core"),
            Milestone("de-5",  "Apache Spark (PySpark, Spark SQL, tuning)", "advanced"),
            Milestone("de-6",  "Kafka: topics, partitions, consumer groups", "advanced"),
            Milestone("de-7",  "Delta Lake / Iceberg lakehouse format", "advanced"),
            Milestone("de-8",  "Streaming: Flink or Spark Structured Streaming", "expert"),
            Milestone("de-9",  "Data modelling: Kimball, Inmon, Data Vault", "advanced"),
            Milestone("de-10", "Cost optimisation on Snowflake/BigQuery", "expert"),
        ],
    ),
    Track(
        id="data_analysis",
        name="Data Analysis & BI",
        description="From raw data to dashboards stakeholders actually use.",
        milestones=[
            Milestone("da-1", "SQL advanced (analytic functions, performance)", "core"),
            Milestone("da-2", "Pandas + Polars proficiency", "core"),
            Milestone("da-3", "Statistics: hypothesis testing, A/B tests", "core"),
            Milestone("da-4", "Tableau or Power BI dashboards", "core"),
            Milestone("da-5", "Storytelling with data (Cole Knaflic)", "advanced"),
            Milestone("da-6", "dbt + Looker / Metabase semantic layer", "advanced"),
            Milestone("da-7", "Causal inference basics (DAGs, IV, DiD)", "expert"),
        ],
    ),
    Track(
        id="python_mastery",
        name="Python Mastery",
        description="Beyond syntax — performance, async, packaging, internals.",
        milestones=[
            Milestone("py-1", "Type hints + mypy strict", "core"),
            Milestone("py-2", "AsyncIO + httpx + concurrent.futures", "core"),
            Milestone("py-3", "Testing: pytest fixtures, mocking, coverage", "core"),
            Milestone("py-4", "Packaging: pyproject.toml, hatchling, uv", "core"),
            Milestone("py-5", "Profiling: py-spy, scalene, memory_profiler", "advanced"),
            Milestone("py-6", "C-extensions / Cython / mypyc / Rust via PyO3", "expert"),
            Milestone("py-7", "CPython internals (GIL, GC, bytecode)", "expert"),
            Milestone("py-8", "FastAPI / Pydantic v2 production patterns", "advanced"),
        ],
    ),
    Track(
        id="oracle_db",
        name="Oracle & Databases",
        description="Deep RDBMS + Oracle ecosystem.",
        milestones=[
            Milestone("or-1", "PL/SQL: procedures, packages, triggers", "core"),
            Milestone("or-2", "Indexes, partitions, query plans (Oracle)", "core"),
            Milestone("or-3", "Oracle APEX low-code apps", "core"),
            Milestone("or-4", "RAC, Data Guard, RMAN backups", "advanced"),
            Milestone("or-5", "Performance tuning (AWR, ASH reports)", "advanced"),
            Milestone("or-6", "Oracle Cloud Infrastructure (OCI) basics", "advanced"),
            Milestone("or-7", "PostgreSQL & SQL Server comparative depth", "expert"),
        ],
    ),
    Track(
        id="odoo_erp",
        name="Odoo & ERP",
        description="ERP customisation and full-stack module development.",
        milestones=[
            Milestone("od-1", "Odoo architecture + ORM + XML views", "core"),
            Milestone("od-2", "Build a custom Odoo module from scratch", "core"),
            Milestone("od-3", "QWeb reports + website templates", "core"),
            Milestone("od-4", "Inheritance: extending core models cleanly", "advanced"),
            Milestone("od-5", "Odoo.sh deployment + CI", "advanced"),
            Milestone("od-6", "Odoo Studio / no-code for stakeholders", "advanced"),
            Milestone("od-7", "Performance: SQL views, computed fields", "expert"),
        ],
    ),
    Track(
        id="system_design",
        name="System Design",
        description="Designing services that handle real scale.",
        milestones=[
            Milestone("sd-1", "CAP theorem, consistency models, quorum", "core"),
            Milestone("sd-2", "Caching strategies (write-through, cache-aside)", "core"),
            Milestone("sd-3", "Queues vs streams (SQS, Kafka, Pub/Sub)", "core"),
            Milestone("sd-4", "Designing for idempotency + exactly-once", "advanced"),
            Milestone("sd-5", "Sharding, partitioning, hot keys", "advanced"),
            Milestone("sd-6", "Rate limiting + circuit breakers", "advanced"),
            Milestone("sd-7", "Multi-region active/active design", "expert"),
            Milestone("sd-8", "Read 'Designing Data-Intensive Applications'", "core"),
        ],
    ),
    Track(
        id="cloud",
        name="Cloud (AWS / GCP / Azure)",
        description="Pick one cloud and go deep — generalise later.",
        milestones=[
            Milestone("cl-1", "IAM done right (least privilege, roles)", "core"),
            Milestone("cl-2", "Networking: VPC, subnets, peering, NAT", "core"),
            Milestone("cl-3", "Compute + serverless (EC2/EKS/Lambda equivs)", "core"),
            Milestone("cl-4", "Managed data (RDS, DynamoDB, S3, Redshift)", "core"),
            Milestone("cl-5", "Infra as code: Terraform OR Pulumi", "core"),
            Milestone("cl-6", "Observability: CloudWatch / OpenTelemetry", "advanced"),
            Milestone("cl-7", "Cost engineering (FinOps fundamentals)", "advanced"),
            Milestone("cl-8", "Cloud-native cert (AWS SA Pro / GCP PCA)", "expert"),
        ],
    ),
    Track(
        id="ai_ml",
        name="AI / ML Engineering",
        description="LLM apps, RAG, fine-tuning, MLOps — where the market is moving.",
        milestones=[
            Milestone("ai-1", "LLM API fluency (OpenAI, Anthropic, Groq)", "core"),
            Milestone("ai-2", "Prompt engineering + structured outputs", "core"),
            Milestone("ai-3", "RAG: chunking, embeddings, vector DBs", "core"),
            Milestone("ai-4", "Agentic workflows (function calling, tools)", "advanced"),
            Milestone("ai-5", "Fine-tuning: LoRA / QLoRA on open weights", "advanced"),
            Milestone("ai-6", "Eval pipelines (Ragas, lm-eval, hallucination)", "advanced"),
            Milestone("ai-7", "Production: streaming, caching, guardrails", "advanced"),
            Milestone("ai-8", "ML basics: scikit-learn, XGBoost, fundamentals", "core"),
            Milestone("ai-9", "Deploy a small model to prod (vLLM/TGI)", "expert"),
        ],
    ),
    Track(
        id="devops",
        name="DevOps & Platform",
        description="Containers, CI/CD, infra automation — own your deploys.",
        milestones=[
            Milestone("do-1", "Docker: multi-stage builds, layer caching", "core"),
            Milestone("do-2", "Kubernetes: pods, services, ingress, helm", "core"),
            Milestone("do-3", "CI/CD with GitHub Actions or GitLab CI", "core"),
            Milestone("do-4", "Terraform modules and remote state", "core"),
            Milestone("do-5", "Prometheus + Grafana monitoring", "advanced"),
            Milestone("do-6", "Service mesh (Istio / Linkerd)", "expert"),
            Milestone("do-7", "Blue/green + canary deploy strategies", "advanced"),
        ],
    ),
    Track(
        id="fullstack_craft",
        name="Full-Stack Craft",
        description="The senior-engineer fundamentals that age well.",
        milestones=[
            Milestone("fs-1", "Clean Architecture / Hexagonal", "core"),
            Milestone("fs-2", "DDD basics: bounded contexts, aggregates", "core"),
            Milestone("fs-3", "TDD discipline + property-based testing", "core"),
            Milestone("fs-4", "OWASP top 10 + auth patterns (OAuth2, JWT)", "core"),
            Milestone("fs-5", "React/Next.js modern patterns (RSC, suspense)", "core"),
            Milestone("fs-6", "TypeScript advanced (generics, conditionals)", "advanced"),
            Milestone("fs-7", "Mentor a junior + lead a tech-spec review", "expert"),
            Milestone("fs-8", "Write & publish 12+ technical blog posts", "advanced"),
        ],
    ),
]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def load_roadmap() -> list[Track]:
    if not ROADMAP_PATH.exists():
        save_roadmap(DEFAULT_TRACKS)
        return DEFAULT_TRACKS

    try:
        data = json.loads(ROADMAP_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        save_roadmap(DEFAULT_TRACKS)
        return DEFAULT_TRACKS

    tracks: list[Track] = []
    for t in data.get("tracks", []):
        ms = [Milestone(**m) for m in t.get("milestones", [])]
        tracks.append(Track(
            id=t["id"], name=t["name"], description=t.get("description", ""),
            milestones=ms,
        ))
    return tracks or DEFAULT_TRACKS


def save_roadmap(tracks: list[Track]) -> None:
    ROADMAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated": datetime.now().isoformat(timespec="seconds"),
        "tracks": [
            {
                "id": t.id, "name": t.name, "description": t.description,
                "milestones": [asdict(m) for m in t.milestones],
            }
            for t in tracks
        ],
    }
    ROADMAP_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def overall_progress(tracks: list[Track]) -> tuple[int, int]:
    total = sum(t.total for t in tracks)
    done = sum(t.done_count for t in tracks)
    return done, total


def find_milestone(tracks: list[Track], mid: str) -> tuple[Track, Milestone] | None:
    for t in tracks:
        for m in t.milestones:
            if m.id == mid:
                return t, m
    return None


# ---------------------------------------------------------------------------
# LLM context
# ---------------------------------------------------------------------------
def format_for_llm(tracks: list[Track]) -> str:
    lines = []
    for t in tracks:
        lines.append(f"## {t.name} — {t.done_count}/{t.total} ({t.progress * 100:.0f}%)")
        for m in t.milestones:
            mark = "[x]" if m.done else "[ ]"
            lines.append(f"  {mark} ({m.level}) {m.title}")
            if m.notes:
                lines.append(f"      note: {m.notes}")
        lines.append("")
    return "\n".join(lines)
