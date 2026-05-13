"""Advanced Interview Guider — topic-based, difficulty-aware, AI-evaluated.

Architecture & ML stack:
  - Question generation: Groq LLM (structured JSON via call_groq_json)
  - Answer evaluation:   Groq LLM (qualitative rubric scores + ideal answer)
  - Semantic scoring:    sentence-transformers + PyTorch cosine similarity
      This gives an OBJECTIVE, embedding-based score independent of the LLM.
      Uses the same all-MiniLM-L6-v2 model that powers the FAISS vector store,
      so no extra download is needed after first run.

PyTorch is used directly (not via LangChain) to:
  1. Encode answers into 384-dim embedding tensors.
  2. Compute dot-product cosine similarity on L2-normalised vectors.
  3. Return a float32 scalar clamped to [0, 100].

Why PyTorch for this instead of just using the LLM?
  LLMs can be generous or inconsistent scorers — the embedding-based score is
  deterministic and purely semantic, making it useful as a calibration signal.
  Combining both gives a richer picture: LLM catches logical correctness,
  sentence-transformers catches vocabulary/concept overlap.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from collections import Counter

import torch
from sentence_transformers import SentenceTransformer

_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    """Lazy-load the embedding model (cached in module scope after first call)."""
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
    return _embed_model


# ---------------------------------------------------------------------------
# Topic catalogue
# ---------------------------------------------------------------------------
INTERVIEW_TOPICS: dict[str, dict] = {
    "Data Structures & Algorithms": {
        "icon": "🧮",
        "description": "Arrays, trees, graphs, dynamic programming, sorting, complexity analysis",
        "subtopics": [
            "Arrays & Strings",
            "Trees & Graphs",
            "Dynamic Programming",
            "Sorting & Searching",
            "Hash Maps & Sets",
            "Recursion & Backtracking",
        ],
    },
    "System Design": {
        "icon": "🏗️",
        "description": "Scalable distributed systems, databases, caching, messaging, CAP theorem",
        "subtopics": [
            "Scalability & Load Balancing",
            "Database Design & Sharding",
            "Caching Strategies",
            "Message Queues & Event Streaming",
            "Microservices & API Gateway",
            "Consistency & Availability Trade-offs",
        ],
    },
    "Python Internals": {
        "icon": "🐍",
        "description": "GIL, async/await, decorators, generators, memory model, type system",
        "subtopics": [
            "GIL & Concurrency",
            "Async/Await & asyncio",
            "Decorators & Metaclasses",
            "Generators & Iterators",
            "Type Hints & mypy",
            "Memory Management & gc",
        ],
    },
    "SQL & Databases": {
        "icon": "🗄️",
        "description": "Query optimisation, indexing, transactions, window functions, execution plans",
        "subtopics": [
            "Query Optimisation & EXPLAIN",
            "Indexing Strategies",
            "Transactions & ACID",
            "Window Functions & CTEs",
            "Partitioning & Sharding",
            "NoSQL vs Relational Trade-offs",
        ],
    },
    "Machine Learning & AI": {
        "icon": "🤖",
        "description": "ML fundamentals, LLMs, RAG pipelines, model evaluation, production ML",
        "subtopics": [
            "ML Fundamentals & Bias/Variance",
            "Neural Networks & Backpropagation",
            "LLMs & Prompt Engineering",
            "RAG & Vector Databases",
            "Model Evaluation & A/B Testing",
            "MLOps & Production Deployment",
        ],
    },
    "Behavioral (STAR)": {
        "icon": "🎭",
        "description": "Leadership, conflict, ownership, failure, cross-functional collaboration",
        "subtopics": [
            "Leadership & Influence Without Authority",
            "Conflict & Disagreement",
            "Project Failures & Lessons",
            "Cross-team Collaboration",
            "Ownership & Technical Decisions",
            "Ambiguity & Rapid Change",
        ],
    },
    "API Design & Architecture": {
        "icon": "🔌",
        "description": "REST, GraphQL, gRPC, auth, rate limiting, versioning, OpenAPI",
        "subtopics": [
            "REST Best Practices",
            "Authentication & Authorization",
            "Rate Limiting & Throttling",
            "API Versioning",
            "GraphQL vs REST vs gRPC",
            "OpenAPI & Contract-First Design",
        ],
    },
    "DevOps & Cloud": {
        "icon": "☁️",
        "description": "Docker, Kubernetes, CI/CD, observability, incident response",
        "subtopics": [
            "Docker & Multi-stage Builds",
            "Kubernetes & HPA",
            "CI/CD Pipelines",
            "Observability & SLIs/SLOs",
            "Incident Response & Post-mortems",
            "Infrastructure as Code (Terraform)",
        ],
    },
}

DIFFICULTY_LEVELS: dict[str, str] = {
    "Junior (0–2 yrs)":          "junior",
    "Mid-level (3–5 yrs)":       "mid",
    "Senior (5–8 yrs)":          "senior",
    "Staff / Principal (8+ yrs)": "staff",
}

_DIFFICULTY_GUIDANCE: dict[str, str] = {
    "junior": (
        "Focus on fundamentals, common patterns, and correctness. "
        "Expect candidates to know the textbook answer and apply it to a straightforward scenario."
    ),
    "mid": (
        "Focus on trade-offs, edge cases, and design decisions. "
        "Expect candidates to justify choices and anticipate failure modes."
    ),
    "senior": (
        "Focus on architecture decisions, failure at scale, cross-cutting concerns, "
        "and mentoring trade-offs. Expect nuanced, opinionated answers."
    ),
    "staff": (
        "Focus on org-level impact, ambiguous open-ended problems, technical vision, "
        "and long-term maintainability. Expect candidates to define the problem space itself."
    ),
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class InterviewQuestion:
    id: int
    question: str
    category: str
    difficulty: str
    expected_concepts: list[str]
    follow_ups: list[str] = field(default_factory=list)


@dataclass
class AnswerEvaluation:
    question_id: int
    user_answer: str
    ideal_answer: str
    semantic_score: float       # 0–100, PyTorch cosine similarity
    conceptual_score: int       # 0–10, LLM-graded
    completeness_score: int     # 0–10, LLM-graded
    depth_score: int            # 0–10, LLM-graded
    communication_score: int    # 0–10, LLM-graded
    overall_score: int          # 0–10, LLM-graded
    strengths: list[str]
    gaps: list[str]
    improvements: list[str]
    follow_up_suggested: str


# ---------------------------------------------------------------------------
# PyTorch semantic scoring
# ---------------------------------------------------------------------------
def compute_semantic_score(user_answer: str, ideal_answer: str) -> float:
    """Cosine similarity between user answer and ideal answer using PyTorch.

    Steps:
      1. Encode both strings with all-MiniLM-L6-v2 → 384-dim float32 tensors.
      2. L2-normalise each tensor (normalize_embeddings=True in encode).
      3. Dot product of normalised vectors == cosine similarity.
      4. Clamp to [0, 1], scale to [0, 100].

    Why torch.dot instead of util.pytorch_cos_sim?
      For two 1-D vectors, torch.dot is explicit and avoids the 2-D matrix
      overhead of pytorch_cos_sim. Both produce identical results.
    """
    if not user_answer.strip() or not ideal_answer.strip():
        return 0.0

    model = _get_embed_model()

    # Returns a (2, 384) torch.Tensor on CPU
    embeddings: torch.Tensor = model.encode(
        [user_answer, ideal_answer],
        convert_to_tensor=True,
        normalize_embeddings=True,   # L2-normalise so dot-product == cosine
    )

    # torch.dot on two 1-D tensors — explicit and efficient
    cos_sim: float = torch.dot(embeddings[0], embeddings[1]).item()

    return round(max(0.0, min(1.0, cos_sim)) * 100, 1)


# ---------------------------------------------------------------------------
# JSON schemas for structured LLM calls
# ---------------------------------------------------------------------------
_QUESTION_SCHEMA = """{
  "questions": [
    {
      "id": <int starting at 1>,
      "question": "<the full interview question>",
      "category": "<subtopic from the list>",
      "expected_concepts": ["<key concept 1>", "<key concept 2>", ...],
      "follow_ups": ["<follow-up probe 1>", "<follow-up probe 2>"]
    }
  ]
}"""

_EVAL_SCHEMA = """{
  "ideal_answer": "<comprehensive ideal answer in 4-7 sentences>",
  "conceptual_score": <int 0-10>,
  "completeness_score": <int 0-10>,
  "depth_score": <int 0-10>,
  "communication_score": <int 0-10>,
  "overall_score": <int 0-10>,
  "strengths": ["<specific strength in the candidate's answer>", ...],
  "gaps": ["<specific missing or wrong concept>", ...],
  "improvements": ["<concrete rewrite or addition the candidate should make>", ...],
  "follow_up_suggested": "<the single best follow-up question to probe depth>"
}"""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
def build_question_prompt(
    topic: str,
    subtopics: list[str],
    difficulty: str,
    n_questions: int,
    focus_areas: str = "",
) -> str:
    guidance = _DIFFICULTY_GUIDANCE.get(difficulty, "")
    focus_block = f"\nFocus especially on: {focus_areas.strip()}" if focus_areas.strip() else ""
    return (
        f"Generate exactly {n_questions} {difficulty}-level interview questions for topic: **{topic}**.\n"
        f"Subtopics to draw from: {', '.join(subtopics)}.\n"
        f"{focus_block}\n\n"
        f"Difficulty guidance: {guidance}\n\n"
        f"RULES (follow all):\n"
        f"- Questions must be nuanced and hard — NOT textbook definitions like 'what is X?'\n"
        f"- Frame questions as: 'how would you...', 'walk me through...', 'design a...', 'debug this...'\n"
        f"- Each question must list 3-5 expected_concepts a strong answer MUST cover.\n"
        f"- Each question must have exactly 2 follow_up probes the interviewer asks to go deeper.\n"
        f"- Vary the subtopics — do not repeat the same subtopic twice.\n"
        f"- No trivial questions. Every question should make a mid-level candidate sweat.\n\n"
        f"Respond ONLY with valid JSON (no markdown fences, no extra text):\n{_QUESTION_SCHEMA}"
    )


def build_eval_prompt(question: InterviewQuestion, user_answer: str) -> str:
    return (
        f"Evaluate this technical interview answer.\n\n"
        f"Question: {question.question}\n"
        f"Expected concepts (MUST all appear in a great answer): {', '.join(question.expected_concepts)}\n"
        f"Difficulty: {question.difficulty}\n\n"
        f"Candidate's answer:\n<ANSWER>\n{user_answer[:2500]}\n</ANSWER>\n\n"
        f"Scoring rubric (be STRICT — a 10 must satisfy a Google/Amazon interviewer):\n"
        f"  conceptual_score:   Did they cover the expected concepts accurately? Wrong concepts = deduct heavily.\n"
        f"  completeness_score: Did they cover ALL expected_concepts? Each missing = -1 to -2 pts.\n"
        f"  depth_score:        Did they discuss trade-offs, edge cases, failure modes? Surface answers = ≤5.\n"
        f"  communication_score: Clear structure, correct terminology, concise? Rambling = deduct.\n"
        f"  overall_score:      Holistic hire signal. 9-10=strong hire, 7-8=hire, 5-6=maybe, <5=no hire.\n\n"
        f"In gaps, name the SPECIFIC missing concept, not vague feedback like 'needs more depth'.\n"
        f"In improvements, give CONCRETE rewrites or additions — show the candidate what to say.\n\n"
        f"Respond ONLY with valid JSON (no fences, no extra text):\n{_EVAL_SCHEMA}"
    )


def build_hint_text(question: InterviewQuestion) -> str:
    """Generate a hint from the question's expected_concepts (no LLM call needed)."""
    concepts = question.expected_concepts
    if not concepts:
        return "Think about the core trade-offs involved and consider failure modes."
    lines = ["**Hint — key concepts a strong answer should cover:**"]
    for c in concepts:
        lines.append(f"- {c}")
    if question.follow_ups:
        lines.append(f"\n*The interviewer might also ask: '{question.follow_ups[0]}'*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON parsers
# ---------------------------------------------------------------------------
def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        inner = parts[1]
        if inner.startswith("json"):
            inner = inner[4:]
        raw = inner.strip()
    return raw


def parse_questions(raw: str, difficulty: str = "") -> list[InterviewQuestion]:
    data = json.loads(_strip_fences(raw))
    questions = []
    for q in data.get("questions", []):
        questions.append(InterviewQuestion(
            id=q.get("id", 0),
            question=q.get("question", ""),
            category=q.get("category", ""),
            difficulty=difficulty,
            expected_concepts=q.get("expected_concepts", []),
            follow_ups=q.get("follow_ups", []),
        ))
    return questions


def parse_evaluation(
    raw: str,
    question_id: int,
    user_answer: str,
) -> AnswerEvaluation:
    d = json.loads(_strip_fences(raw))
    return AnswerEvaluation(
        question_id=question_id,
        user_answer=user_answer,
        ideal_answer=d.get("ideal_answer", ""),
        semantic_score=0.0,          # filled in by caller after torch scoring
        conceptual_score=d.get("conceptual_score", 0),
        completeness_score=d.get("completeness_score", 0),
        depth_score=d.get("depth_score", 0),
        communication_score=d.get("communication_score", 0),
        overall_score=d.get("overall_score", 0),
        strengths=d.get("strengths", []),
        gaps=d.get("gaps", []),
        improvements=d.get("improvements", []),
        follow_up_suggested=d.get("follow_up_suggested", ""),
    )


# ---------------------------------------------------------------------------
# Markdown renderers
# ---------------------------------------------------------------------------
def format_evaluation_md(ev: AnswerEvaluation) -> str:
    sem_color = "🟢" if ev.semantic_score >= 70 else "🟡" if ev.semantic_score >= 45 else "🔴"
    llm_color = "🟢" if ev.overall_score >= 7 else "🟡" if ev.overall_score >= 5 else "🔴"

    lines = [
        f"### {llm_color} LLM Score: **{ev.overall_score}/10** &nbsp;·&nbsp; "
        f"{sem_color} Semantic Match: **{ev.semantic_score:.0f}%**",
        "",
        "| Dimension | Score |",
        "|-----------|-------|",
        f"| Conceptual Accuracy | {ev.conceptual_score}/10 |",
        f"| Completeness | {ev.completeness_score}/10 |",
        f"| Depth & Nuance | {ev.depth_score}/10 |",
        f"| Communication | {ev.communication_score}/10 |",
        f"| Semantic Match *(PyTorch cosine)* | {ev.semantic_score:.0f}% |",
    ]

    if ev.strengths:
        lines += ["", "**✅ What you got right:**"]
        lines += [f"- {s}" for s in ev.strengths]

    if ev.gaps:
        lines += ["", "**⚠️ Missing / incorrect concepts:**"]
        lines += [f"- {g}" for g in ev.gaps]

    if ev.improvements:
        lines += ["", "**💡 Concrete improvements:**"]
        lines += [f"- {imp}" for imp in ev.improvements]

    if ev.ideal_answer:
        lines += ["", "**📖 Ideal answer:**", "", f"> {ev.ideal_answer}"]

    if ev.follow_up_suggested:
        lines += [
            "",
            f"**🔍 Interviewer follow-up:** _{ev.follow_up_suggested}_",
        ]

    return "\n".join(lines)


def session_summary_md(
    questions: list[InterviewQuestion],
    evaluations: dict[int, AnswerEvaluation],
) -> str:
    """Overall session performance summary with study recommendations."""
    answered = list(evaluations.values())
    if not answered:
        return "No answers submitted yet."

    avg_overall = sum(e.overall_score for e in answered) / len(answered)
    avg_semantic = sum(e.semantic_score for e in answered) / len(answered)

    color = "🟢" if avg_overall >= 7 else "🟡" if avg_overall >= 5 else "🔴"

    all_gaps: list[str] = []
    for ev in answered:
        all_gaps.extend(ev.gaps)

    lines = [
        f"## {color} Session Complete — {len(answered)}/{len(questions)} Questions Answered",
        "",
        f"| Metric | Score |",
        f"|--------|-------|",
        f"| Average LLM Score | **{avg_overall:.1f} / 10** |",
        f"| Average Semantic Match | **{avg_semantic:.0f}%** |",
        f"| Questions Attempted | **{len(answered)} / {len(questions)}** |",
        "",
    ]

    if all_gaps:
        gap_counts = Counter(all_gaps)
        top_gaps = gap_counts.most_common(6)
        lines += ["### 📚 Study Priorities (most common gaps)"]
        for gap, count in top_gaps:
            suffix = f" *(appeared {count}×)*" if count > 1 else ""
            lines.append(f"- {gap}{suffix}")
        lines.append("")

    per_q = []
    for i, q in enumerate(questions):
        ev = evaluations.get(q.id)
        if ev:
            per_q.append(
                f"**Q{i+1}** _{q.question[:60]}…_ → "
                f"LLM {ev.overall_score}/10 · Semantic {ev.semantic_score:.0f}%"
            )
        else:
            per_q.append(f"**Q{i+1}** _{q.question[:60]}…_ → *Skipped*")

    lines += ["### Per-Question Breakdown"] + per_q + [""]

    if avg_overall < 5:
        lines.append(
            "**Next step:** Revisit fundamentals — try the Junior difficulty first, "
            "then ramp up once concepts are solid."
        )
    elif avg_overall < 7:
        lines.append(
            "**Next step:** Strong foundation. Drill trade-off and failure-mode questions — "
            "that's what separates mid from senior."
        )
    else:
        lines.append(
            "**Next step:** Hire-level. Practice Staff/Principal level for ambiguous, "
            "open-ended system design and org-impact discussions."
        )

    return "\n".join(lines)
