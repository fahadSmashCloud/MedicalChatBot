"""Job aggregator across multiple free / freemium APIs.

Sources (NO key required):
  - RemoteOK    https://remoteok.com/api
  - Remotive    https://remotive.com/api/remote-jobs
  - Arbeitnow   https://arbeitnow.com/api/job-board-api

Optional sources (set env var to enable):
  - Adzuna      ADZUNA_APP_ID + ADZUNA_APP_KEY   (free ~250 calls/month, 20 countries
                with salary data). https://developer.adzuna.com/
  - JSearch     RAPIDAPI_KEY                     (free 200 calls/month — proxies
                LinkedIn / Indeed / Glassdoor / ZipRecruiter listings legally).
                https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch

Why not scrape LinkedIn/Indeed directly?  They ban scraper IPs aggressively
and their ToS prohibits it.  JSearch is the legal route to those same
listings.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

import requests

USER_AGENT = "MediBot-JobsFinder/1.0 (personal use)"
REQUEST_TIMEOUT = 15


@dataclass
class Job:
    title: str
    company: str
    location: str
    remote: bool
    url: str
    source: str
    description: str = ""
    salary_min: float | None = None
    salary_max: float | None = None
    salary_currency: str = ""
    posted: str = ""               # ISO date if known
    tags: list[str] = field(default_factory=list)

    @property
    def salary_str(self) -> str:
        if self.salary_min is None and self.salary_max is None:
            return "—"
        cur = self.salary_currency or "USD"
        if self.salary_min and self.salary_max:
            return f"{cur} {self.salary_min:,.0f} – {self.salary_max:,.0f}"
        v = self.salary_min or self.salary_max
        return f"{cur} {v:,.0f}+"

    @property
    def salary_midpoint_usd(self) -> float:
        """Rough USD midpoint for sorting. Returns 0 when unknown."""
        if self.salary_min is None and self.salary_max is None:
            return 0.0
        lo = self.salary_min or self.salary_max or 0
        hi = self.salary_max or self.salary_min or 0
        mid = (lo + hi) / 2
        return mid * _fx_to_usd(self.salary_currency)


_FX = {
    "USD": 1.0, "EUR": 1.08, "GBP": 1.27, "CAD": 0.73, "AUD": 0.66,
    "PKR": 0.0036, "INR": 0.012, "AED": 0.27, "SAR": 0.27, "SGD": 0.74,
}


def _fx_to_usd(currency: str) -> float:
    return _FX.get((currency or "USD").upper(), 1.0)


# ---------------------------------------------------------------------------
# RemoteOK
# ---------------------------------------------------------------------------
def _fetch_remoteok(query: str) -> list[Job]:
    r = requests.get(
        "https://remoteok.com/api",
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    # First element is metadata
    rows = [d for d in data if isinstance(d, dict) and d.get("position")]
    q = query.lower().strip() if query else ""
    out: list[Job] = []
    for d in rows:
        text = " ".join(str(d.get(k, "")) for k in ("position", "company", "description", "tags")).lower()
        if q and q not in text:
            continue
        out.append(Job(
            title=d.get("position", "").strip(),
            company=d.get("company", "").strip(),
            location=d.get("location") or "Worldwide (Remote)",
            remote=True,
            url=d.get("url") or d.get("apply_url", ""),
            source="RemoteOK",
            description=_clean_html(d.get("description", "")),
            salary_min=_to_float(d.get("salary_min")),
            salary_max=_to_float(d.get("salary_max")),
            salary_currency="USD",
            posted=d.get("date", "")[:10],
            tags=list(d.get("tags") or []),
        ))
    return out


# ---------------------------------------------------------------------------
# Remotive
# ---------------------------------------------------------------------------
def _fetch_remotive(query: str) -> list[Job]:
    params = {"search": query} if query else {}
    r = requests.get(
        "https://remotive.com/api/remote-jobs",
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    out: list[Job] = []
    for d in r.json().get("jobs", []):
        salary_lo, salary_hi, cur = _parse_salary_range(d.get("salary", ""))
        out.append(Job(
            title=d.get("title", "").strip(),
            company=d.get("company_name", "").strip(),
            location=d.get("candidate_required_location", "Worldwide (Remote)"),
            remote=True,
            url=d.get("url", ""),
            source="Remotive",
            description=_clean_html(d.get("description", ""))[:1500],
            salary_min=salary_lo,
            salary_max=salary_hi,
            salary_currency=cur or "USD",
            posted=(d.get("publication_date") or "")[:10],
            tags=list(d.get("tags") or []),
        ))
    return out


# ---------------------------------------------------------------------------
# Arbeitnow
# ---------------------------------------------------------------------------
def _fetch_arbeitnow(query: str) -> list[Job]:
    r = requests.get(
        "https://www.arbeitnow.com/api/job-board-api",
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    q = query.lower().strip() if query else ""
    out: list[Job] = []
    for d in r.json().get("data", []):
        text = " ".join(str(d.get(k, "")) for k in ("title", "company_name", "description", "tags")).lower()
        if q and q not in text:
            continue
        out.append(Job(
            title=d.get("title", "").strip(),
            company=d.get("company_name", "").strip(),
            location=d.get("location") or "Europe",
            remote=bool(d.get("remote")),
            url=d.get("url", ""),
            source="Arbeitnow",
            description=_clean_html(d.get("description", ""))[:1500],
            salary_currency="EUR",
            posted=(d.get("created_at") and datetime.utcfromtimestamp(d["created_at"]).date().isoformat()) or "",
            tags=list(d.get("tags") or []),
        ))
    return out


# ---------------------------------------------------------------------------
# Adzuna (optional, with salary data + global coverage)
# ---------------------------------------------------------------------------
ADZUNA_COUNTRIES = ["us", "gb", "de", "ca", "au", "fr", "in", "sg", "nl", "pl"]


def _fetch_adzuna(query: str, country: str = "us") -> list[Job]:
    app_id = os.environ.get("ADZUNA_APP_ID")
    app_key = os.environ.get("ADZUNA_APP_KEY")
    if not (app_id and app_key):
        return []
    r = requests.get(
        f"https://api.adzuna.com/v1/api/jobs/{country}/search/1",
        params={
            "app_id": app_id,
            "app_key": app_key,
            "results_per_page": 30,
            "what": query or "software engineer",
            "sort_by": "salary",
            "content-type": "application/json",
        },
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    out: list[Job] = []
    for d in r.json().get("results", []):
        out.append(Job(
            title=d.get("title", "").strip(),
            company=(d.get("company") or {}).get("display_name", ""),
            location=(d.get("location") or {}).get("display_name", country.upper()),
            remote="remote" in (d.get("title", "") + " " + d.get("description", "")).lower(),
            url=d.get("redirect_url", ""),
            source=f"Adzuna ({country.upper()})",
            description=(d.get("description") or "")[:1500],
            salary_min=_to_float(d.get("salary_min")),
            salary_max=_to_float(d.get("salary_max")),
            salary_currency=_adzuna_currency(country),
            posted=(d.get("created") or "")[:10],
            tags=[(d.get("category") or {}).get("label", "")],
        ))
    return out


def _adzuna_currency(country: str) -> str:
    return {
        "us": "USD", "gb": "GBP", "de": "EUR", "ca": "CAD", "au": "AUD",
        "fr": "EUR", "in": "INR", "sg": "SGD", "nl": "EUR", "pl": "PLN",
    }.get(country, "USD")


# ---------------------------------------------------------------------------
# JSearch via RapidAPI (optional, covers LinkedIn / Indeed / Glassdoor)
# ---------------------------------------------------------------------------
def _fetch_jsearch(query: str) -> list[Job]:
    key = os.environ.get("RAPIDAPI_KEY")
    if not key:
        return []
    r = requests.get(
        "https://jsearch.p.rapidapi.com/search",
        params={"query": query or "senior full stack engineer", "page": "1", "num_pages": "1"},
        headers={
            "X-RapidAPI-Key": key,
            "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
        },
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    out: list[Job] = []
    for d in r.json().get("data", []):
        out.append(Job(
            title=d.get("job_title", "").strip(),
            company=d.get("employer_name", "").strip(),
            location=", ".join(filter(None, [d.get("job_city"), d.get("job_country")])) or "Unknown",
            remote=bool(d.get("job_is_remote")),
            url=d.get("job_apply_link", ""),
            source=f"JSearch ({d.get('job_publisher', 'web')})",
            description=(d.get("job_description") or "")[:1500],
            salary_min=_to_float(d.get("job_min_salary")),
            salary_max=_to_float(d.get("job_max_salary")),
            salary_currency=d.get("job_salary_currency") or "USD",
            posted=(d.get("job_posted_at_datetime_utc") or "")[:10],
            tags=[],
        ))
    return out


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------
def fetch_jobs(
    query: str = "",
    min_salary_usd: float = 0,
    remote_only: bool = False,
    sources: Iterable[str] | None = None,
) -> tuple[list[Job], dict[str, str]]:
    """Fetch jobs in parallel from all enabled sources.

    Returns (jobs, errors) where `errors` maps source -> error message for
    sources that failed.  The job list is de-duped, salary-filtered, and
    sorted by USD midpoint descending (unknown salaries last).
    """
    runners = {
        "RemoteOK":  lambda: _fetch_remoteok(query),
        "Remotive":  lambda: _fetch_remotive(query),
        "Arbeitnow": lambda: _fetch_arbeitnow(query),
        "Adzuna":    lambda: _fetch_adzuna(query),
        "JSearch":   lambda: _fetch_jsearch(query),
    }
    if sources:
        runners = {k: v for k, v in runners.items() if k in sources}

    jobs: list[Job] = []
    errors: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=len(runners)) as pool:
        futures = {pool.submit(fn): name for name, fn in runners.items()}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                jobs.extend(fut.result())
            except Exception as e:
                errors[name] = str(e)[:200]

    # de-dupe by (title|company) lowercase
    seen: set[tuple[str, str]] = set()
    deduped: list[Job] = []
    for j in jobs:
        key = (j.title.lower(), j.company.lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(j)

    if remote_only:
        deduped = [j for j in deduped if j.remote]

    if min_salary_usd > 0:
        # Include jobs with unknown salary but flagged senior/lead/staff
        senior_kw = re.compile(r"\b(senior|sr\.?|lead|staff|principal|architect)\b", re.I)
        deduped = [
            j for j in deduped
            if j.salary_midpoint_usd >= min_salary_usd
            or (j.salary_midpoint_usd == 0 and senior_kw.search(j.title))
        ]

    deduped.sort(key=lambda j: j.salary_midpoint_usd, reverse=True)
    return deduped, errors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_float(v) -> float | None:
    try:
        f = float(v)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


_HTML_TAGS = re.compile(r"<[^>]+>")


def _clean_html(text: str) -> str:
    if not text:
        return ""
    return _HTML_TAGS.sub(" ", text).replace("&nbsp;", " ").strip()


_SALARY_RX = re.compile(
    r"(?P<cur>USD|EUR|GBP|\$|€|£)?\s*(?P<lo>[\d,]{4,})\s*[-–to]+\s*(?P<cur2>USD|EUR|GBP|\$|€|£)?\s*(?P<hi>[\d,]{4,})",
    re.I,
)
_CUR_MAP = {"$": "USD", "€": "EUR", "£": "GBP"}


def _parse_salary_range(s: str) -> tuple[float | None, float | None, str]:
    if not s:
        return None, None, ""
    m = _SALARY_RX.search(s)
    if not m:
        return None, None, ""
    lo = _to_float(m.group("lo").replace(",", ""))
    hi = _to_float(m.group("hi").replace(",", ""))
    cur_raw = (m.group("cur") or m.group("cur2") or "").upper()
    cur = _CUR_MAP.get(cur_raw, cur_raw or "")
    return lo, hi, cur


# ---------------------------------------------------------------------------
# LLM context builder
# ---------------------------------------------------------------------------
def format_jobs_for_llm(jobs: list[Job], limit: int = 15) -> str:
    if not jobs:
        return "No jobs matched the current filters."
    lines = []
    for i, j in enumerate(jobs[:limit], start=1):
        lines.append(
            f"[{i}] {j.title} — {j.company}\n"
            f"    Location: {j.location} {'(Remote)' if j.remote else ''}\n"
            f"    Salary:   {j.salary_str}\n"
            f"    Source:   {j.source}\n"
            f"    Tags:     {', '.join(j.tags[:8]) if j.tags else '—'}\n"
            f"    URL:      {j.url}"
        )
    return "\n\n".join(lines)
