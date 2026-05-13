"""Resume / CV analyzer powered by Groq.

Flow:
  1. Upload a PDF resume → extract plain text via PyPDFLoader.
  2. Send to Groq with a strict JSON schema prompt.
  3. Parse the JSON response → render as structured markdown.
  4. Optional: paste a job description to get missing-keyword gap analysis.
  5. Optional follow-up chat grounded in the analysis.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------
def extract_resume_text(uploaded_file) -> str:
    """Load a Streamlit-uploaded PDF file and return its plain text.

    Raises ValueError if no extractable text is found.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        pages = PyPDFLoader(tmp_path).load()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not pages:
        raise ValueError("Could not load any pages from the uploaded PDF.")

    text = "\n".join(p.page_content for p in pages).strip()
    if len(text) < 100:
        raise ValueError(
            "PDF appears to be image-based or encrypted — no extractable text found. "
            "Try a text-based PDF or paste your resume as plain text."
        )
    return text


# ---------------------------------------------------------------------------
# Analysis prompt
# ---------------------------------------------------------------------------
_SCHEMA = """{
  "ats_score": <int 0-100>,
  "overall_score": <int 0-10>,
  "summary": "<2-sentence overall assessment>",
  "strengths": ["<strength>", ...],
  "weaknesses": ["<weakness>", ...],
  "skills_detected": ["<skill>", ...],
  "missing_keywords": ["<keyword missing vs JD>", ...],
  "improvement_suggestions": [
    {"section": "<Resume section>", "suggestion": "<specific rewrite or action>"}
  ],
  "impact_bullets": ["<rewritten STAR-format bullet>", ...]
}"""


def build_analysis_prompt(resume_text: str, job_description: str = "") -> str:
    jd_block = (
        f"\n\nJob Description to compare against:\n<JD>\n{job_description[:3000]}\n</JD>"
        if job_description.strip()
        else ""
    )
    return (
        f"Analyse the resume below and respond ONLY with valid JSON matching this schema "
        f"(no markdown fences, no extra text):\n{_SCHEMA}\n\n"
        f"Rules:\n"
        f"- missing_keywords: only populate when a JD is given; otherwise return [].\n"
        f"- impact_bullets: pick up to 5 weak resume bullets and rewrite them in STAR format "
        f"  (Situation → Task → Action → Result). Include quantified outcomes where inferable.\n"
        f"- Be specific — name real technologies, metrics, and companies from the resume.\n"
        f"- ats_score: assess keyword density, section headers, consistent dates, readability.\n"
        f"- overall_score: holistic quality considering seniority signals, achievements, clarity.\n\n"
        f"Resume:\n<RESUME>\n{resume_text[:6000]}\n</RESUME>{jd_block}"
    )


# ---------------------------------------------------------------------------
# JSON parsing (robust — strips markdown fences if the model wraps anyway)
# ---------------------------------------------------------------------------
def parse_analysis(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        # parts[1] is the code block content (may start with "json\n")
        inner = parts[1]
        if inner.startswith("json"):
            inner = inner[4:]
        raw = inner.strip()
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------
def format_analysis_as_markdown(a: dict) -> str:
    ats = a.get("ats_score", 0)
    overall = a.get("overall_score", 0)
    color = "🟢" if ats >= 75 else "🟡" if ats >= 50 else "🔴"

    lines: list[str] = [
        f"## {color} ATS Score: **{ats}/100** &nbsp;·&nbsp; Overall: **{overall}/10**"
    ]

    if s := a.get("summary"):
        lines += ["", s]

    if items := a.get("strengths"):
        lines += ["", "### ✅ Strengths"]
        lines += [f"- {s}" for s in items]

    if items := a.get("weaknesses"):
        lines += ["", "### ⚠️ Weaknesses"]
        lines += [f"- {w}" for w in items]

    if items := a.get("skills_detected"):
        lines += ["", "### 🛠️ Skills Detected"]
        lines.append("  ".join(f"`{s}`" for s in items))

    if items := a.get("missing_keywords"):
        lines += ["", "### 🚨 Missing Keywords (vs Job Description)"]
        lines += [f"- `{k}`" for k in items]

    if items := a.get("improvement_suggestions"):
        lines += ["", "### 💡 Improvement Suggestions"]
        for sg in items:
            lines.append(f"**{sg.get('section', '')}** — {sg.get('suggestion', '')}")

    if items := a.get("impact_bullets"):
        lines += ["", "### 🚀 High-Impact Rewrites (STAR format)"]
        lines += [f"- {b}" for b in items]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick-action prompt builders
# ---------------------------------------------------------------------------
def cover_letter_prompt(resume_text: str, role: str, company: str) -> str:
    return (
        f"Write a concise, compelling cover letter (3 short paragraphs) for the following:\n"
        f"Role: {role}\nCompany: {company}\n\n"
        f"Base it on this resume:\n{resume_text[:4000]}\n\n"
        f"Tone: confident, senior-level. No fluff. End with a call to action."
    )


def interview_prep_prompt(resume_text: str, role: str) -> str:
    return (
        f"Generate 8 likely interview questions for a senior engineer applying for: {role}\n"
        f"Mix: 3 behavioral (STAR), 3 technical (specific to their stack), 2 situational.\n"
        f"For each question, add a 1-sentence coaching tip.\n\n"
        f"Resume context:\n{resume_text[:3000]}"
    )
