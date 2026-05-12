"""PSX (Pakistan Stock Exchange) live data layer.

Data source: public market-watch page at https://dps.psx.com.pk/market-watch.
The page returns an HTML table of every listed symbol with price, change, and
volume. We scrape it and cache the parsed snapshot for a short window so that
Streamlit reruns don't hammer the endpoint.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta

import requests
from bs4 import BeautifulSoup

# ---------- config ----------

MARKET_WATCH_URL = "https://dps.psx.com.pk/market-watch"
CACHE_TTL_SECONDS = 60
HTTP_TIMEOUT = 15

# PSX market hours, Asia/Karachi: Mon-Fri 09:30 - 15:30
PKT_OFFSET = timedelta(hours=5)
MARKET_OPEN = (9, 30)
MARKET_CLOSE = (15, 30)


# ---------- data types ----------

@dataclass
class Quote:
    symbol: str
    indices: str
    price: float
    open: float
    high: float
    low: float
    prev_close: float
    change: float
    change_pct: float
    volume: int

    def as_dict(self) -> dict:
        return asdict(self)


# ---------- scraper ----------

_cache: dict = {"timestamp": 0.0, "quotes": []}


def _http_get(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml",
    }
    r = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.text


def _to_float(s: str) -> float:
    s = (s or "").replace(",", "").replace("%", "").strip()
    if not s or s in {"-", "—", "N/A"}:
        return 0.0
    try:
        return float(s)
    except ValueError:
        m = re.match(r"\((.+)\)", s)
        return -float(m.group(1)) if m else 0.0


def _to_int(s: str) -> int:
    return int(_to_float(s))


def fetch_market_watch(force: bool = False) -> list[Quote]:
    """Scrape the PSX market-watch page; cached for CACHE_TTL_SECONDS."""
    now = time.time()
    if not force and (now - _cache["timestamp"] < CACHE_TTL_SECONDS) and _cache["quotes"]:
        return _cache["quotes"]

    html = _http_get(MARKET_WATCH_URL)
    soup = BeautifulSoup(html, "lxml")

    quotes: list[Quote] = []
    table = soup.find("table")
    if not table:
        raise RuntimeError("PSX market-watch page did not contain a table — endpoint may have changed.")

    headers = [th.get_text(strip=True).upper() for th in table.find_all("th")]
    def col(name: str) -> int | None:
        for i, h in enumerate(headers):
            if name in h:
                return i
        return None

    idx = {
        "symbol":  col("SYMBOL"),
        "indices": col("LISTED IN"),
        "ldcp":    col("LDCP"),
        "open":    col("OPEN"),
        "high":    col("HIGH"),
        "low":     col("LOW"),
        "current": col("CURRENT"),
        "change":  col("CHANGE"),
        "pct":     next((i for i, h in enumerate(headers) if "%" in h or "CHANGE (%)" in h), None),
        "volume":  col("VOLUME"),
    }

    for row in table.find_all("tr")[1:]:
        cells = [td.get_text(strip=True) for td in row.find_all("td")]
        if len(cells) < 5 or idx["symbol"] is None:
            continue
        sym = cells[idx["symbol"]].upper()
        if not sym or not sym.isalnum():
            continue

        ldcp    = _to_float(cells[idx["ldcp"]])    if idx["ldcp"]    is not None else 0.0
        open_   = _to_float(cells[idx["open"]])    if idx["open"]    is not None else 0.0
        high    = _to_float(cells[idx["high"]])    if idx["high"]    is not None else 0.0
        low     = _to_float(cells[idx["low"]])     if idx["low"]     is not None else 0.0
        current = _to_float(cells[idx["current"]]) if idx["current"] is not None else 0.0
        change  = _to_float(cells[idx["change"]])  if idx["change"]  is not None else 0.0
        pct     = _to_float(cells[idx["pct"]])     if idx["pct"]     is not None else 0.0
        volume  = _to_int(cells[idx["volume"]])    if idx["volume"]  is not None else 0
        indices = cells[idx["indices"]]            if idx["indices"] is not None else ""

        # The "current" column is empty after market close — fall back to LDCP.
        price = current or ldcp

        quotes.append(Quote(
            symbol=sym,
            indices=indices,
            price=price,
            open=open_,
            high=high,
            low=low,
            prev_close=ldcp,
            change=change,
            change_pct=pct,
            volume=volume,
        ))

    _cache["timestamp"] = now
    _cache["quotes"] = quotes
    return quotes


def get_quote(symbol: str) -> Quote | None:
    symbol = symbol.upper().strip()
    for q in fetch_market_watch():
        if q.symbol == symbol:
            return q
    return None


def top_movers(n: int = 10) -> dict[str, list[Quote]]:
    quotes = [q for q in fetch_market_watch() if q.volume > 0]
    gainers = sorted(quotes, key=lambda q: q.change_pct, reverse=True)[:n]
    losers  = sorted(quotes, key=lambda q: q.change_pct)[:n]
    by_vol  = sorted(quotes, key=lambda q: q.volume, reverse=True)[:n]
    return {"gainers": gainers, "losers": losers, "by_volume": by_vol}


# ---------- market hours ----------

def market_is_open(now: datetime | None = None) -> bool:
    now = now or datetime.now(timezone.utc)
    pkt = now + PKT_OFFSET
    if pkt.weekday() >= 5:
        return False
    h, m = pkt.hour, pkt.minute
    after_open  = (h, m) >= MARKET_OPEN
    before_close = (h, m) <= MARKET_CLOSE
    return after_open and before_close


# ---------- LLM context formatting ----------

def format_for_llm(quotes: list[Quote], symbols: list[str] | None = None, top_n: int = 10) -> str:
    """Render a compact, token-friendly snapshot of the market for the LLM."""
    lines: list[str] = []
    now_pkt = datetime.now() + PKT_OFFSET
    lines.append(f"PSX snapshot — {now_pkt:%Y-%m-%d %H:%M PKT} "
                 f"(market {'OPEN' if market_is_open() else 'CLOSED'})")
    lines.append(f"Total symbols in snapshot: {len(quotes)}")
    lines.append("")

    if symbols:
        wanted = {s.upper() for s in symbols}
        focus = [q for q in quotes if q.symbol in wanted]
        if focus:
            lines.append("### Watchlist")
            lines.append(_table(focus))
            lines.append("")

    movers = top_movers(top_n)
    lines.append(f"### Top {top_n} gainers")
    lines.append(_table(movers["gainers"]))
    lines.append("")
    lines.append(f"### Top {top_n} losers")
    lines.append(_table(movers["losers"]))
    lines.append("")
    lines.append(f"### Top {top_n} by volume")
    lines.append(_table(movers["by_volume"]))
    return "\n".join(lines)


def _table(quotes: list[Quote]) -> str:
    if not quotes:
        return "(none)"
    rows = ["| Symbol | Indices | Price | Chg | Chg% | Volume |",
            "|--------|---------|-------|-----|------|--------|"]
    for q in quotes:
        rows.append(
            f"| {q.symbol} | {q.indices[:32]} | {q.price:,.2f} | "
            f"{q.change:+.2f} | {q.change_pct:+.2f}% | {q.volume:,} |"
        )
    return "\n".join(rows)
