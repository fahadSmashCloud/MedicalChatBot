"""PSX (Pakistan Stock Exchange) data + watchlist + alert engine.

Data source: public market-watch page at https://dps.psx.com.pk/market-watch.
The page returns an HTML table of every listed symbol with price, change, and
volume. We scrape it and cache the parsed snapshot for a short window so that
multiple consumers (Streamlit UI + alert worker) don't hammer the endpoint.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup

# ---------- config ----------

MARKET_WATCH_URL = "https://dps.psx.com.pk/market-watch"
SYMBOL_URL = "https://dps.psx.com.pk/symbol/{symbol}"
TIMESERIES_URL = "https://dps.psx.com.pk/timeseries/int/{symbol}"

CACHE_TTL_SECONDS = 60  # don't re-scrape more than once a minute
HTTP_TIMEOUT = 15

WATCHLIST_PATH = Path("data/watchlist.json")
ALERT_COOLDOWN_MINUTES = 60  # don't re-fire the same alert within this window

# PSX market hours, Asia/Karachi: Mon-Fri 09:30 - 15:30
PKT_OFFSET = timedelta(hours=5)
MARKET_OPEN = (9, 30)
MARKET_CLOSE = (15, 30)

# Default daily-briefing slots (PKT). Three sends: open, midday, close.
DEFAULT_BRIEFING_TIMES = ["09:30", "12:30", "15:30"]
BRIEFING_FIRE_WINDOW_MINUTES = 30  # fire if poll happens within this many mins after slot time


# ---------- data types ----------

@dataclass
class Quote:
    symbol: str
    indices: str           # KSE100, KMI30, ALLSHR — which PSX indices the stock is in
    price: float           # current / last traded price
    open: float
    high: float
    low: float
    prev_close: float
    change: float          # absolute change vs prev_close
    change_pct: float
    volume: int

    @property
    def sector(self) -> str:  # alias for backwards compat in templates
        return self.indices

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class AlertRule:
    symbol: str
    rule: str              # rise_pct | drop_pct | price_above | price_below
    threshold: float
    enabled: bool = True

    def describe(self) -> str:
        s = self.symbol
        t = self.threshold
        return {
            "rise_pct":    f"{s} rises >= {t:.2f}% today",
            "drop_pct":    f"{s} drops >= {t:.2f}% today",
            "price_above": f"{s} price >= Rs {t:.2f}",
            "price_below": f"{s} price <= Rs {t:.2f}",
        }.get(self.rule, f"{s} {self.rule} {t}")


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
        # negative values sometimes wrapped in parentheses
        m = re.match(r"\((.+)\)", s)
        return -float(m.group(1)) if m else 0.0


def _to_int(s: str) -> int:
    return int(_to_float(s))


def fetch_market_watch(force: bool = False) -> list[Quote]:
    """Scrape the PSX market-watch page and return a list of Quote objects.

    Cached for CACHE_TTL_SECONDS to avoid hammering PSX on every Streamlit rerun.
    """
    now = time.time()
    if not force and (now - _cache["timestamp"] < CACHE_TTL_SECONDS) and _cache["quotes"]:
        return _cache["quotes"]

    html = _http_get(MARKET_WATCH_URL)
    soup = BeautifulSoup(html, "lxml")

    quotes: list[Quote] = []
    # The page renders one big <table>. Header columns include: SYMBOL, SECTOR,
    # LISTED IN, LDCP, OPEN, HIGH, LOW, CURRENT, CHANGE, CHANGE (%), VOLUME.
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


def by_sector(sector_keyword: str) -> list[Quote]:
    k = sector_keyword.lower()
    return [q for q in fetch_market_watch() if k in q.sector.lower()]


# ---------- market hours ----------

def market_is_open(now: datetime | None = None) -> bool:
    now = now or datetime.now(timezone.utc)
    pkt = now + PKT_OFFSET
    if pkt.weekday() >= 5:  # Sat/Sun
        return False
    h, m = pkt.hour, pkt.minute
    after_open  = (h, m) >= MARKET_OPEN
    before_close = (h, m) <= MARKET_CLOSE
    return after_open and before_close


# ---------- watchlist persistence ----------

@dataclass
class BriefingConfig:
    enabled: bool = False
    times: list[str] = field(default_factory=lambda: list(DEFAULT_BRIEFING_TIMES))  # ["HH:MM", ...] in PKT
    last_sent: dict[str, str] = field(default_factory=dict)  # "YYYY-MM-DD|HH:MM" -> ISO timestamp


@dataclass
class Watchlist:
    """Saved alert config.

    `recipients` is the list of WhatsApp phones (E.164, e.g. "+923001234567") that
    every alert/briefing will be fanned out to. WhatsApp Cloud API does not support
    sending into groups, so the fan-out replaces the "group" idea with a per-number
    broadcast.
    """
    recipients: list[str] = field(default_factory=list)
    rules: list[AlertRule] = field(default_factory=list)
    last_alerts: dict[str, str] = field(default_factory=dict)  # rule_key -> ISO timestamp
    briefings: BriefingConfig = field(default_factory=BriefingConfig)

    # Backwards-compat: old code referenced `.phone` (single). Keep it as the
    # first recipient so any straggler call sites keep working.
    @property
    def phone(self) -> str:
        return self.recipients[0] if self.recipients else ""

    @phone.setter
    def phone(self, value: str) -> None:
        value = (value or "").strip()
        if not value:
            return
        if self.recipients:
            self.recipients[0] = value
        else:
            self.recipients = [value]

    @classmethod
    def load(cls, path: Path = WATCHLIST_PATH) -> "Watchlist":
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return cls()

        # Migration from earlier `phone: str` schema -> `recipients: list[str]`.
        recipients = list(data.get("recipients") or [])
        legacy_phone = (data.get("phone") or "").strip()
        if legacy_phone and legacy_phone not in recipients:
            recipients.insert(0, legacy_phone)

        briefings_data = data.get("briefings") or {}
        return cls(
            recipients=recipients,
            rules=[AlertRule(**r) for r in data.get("rules", [])],
            last_alerts=data.get("last_alerts", {}),
            briefings=BriefingConfig(
                enabled=briefings_data.get("enabled", False),
                times=briefings_data.get("times", list(DEFAULT_BRIEFING_TIMES)),
                last_sent=briefings_data.get("last_sent", {}),
            ),
        )

    def save(self, path: Path = WATCHLIST_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "recipients": self.recipients,
            "rules": [asdict(r) for r in self.rules],
            "last_alerts": self.last_alerts,
            "briefings": asdict(self.briefings),
        }, indent=2), encoding="utf-8")

    def add_recipient(self, phone: str) -> bool:
        phone = (phone or "").strip()
        if not phone or phone in self.recipients:
            return False
        self.recipients.append(phone)
        return True

    def remove_recipient(self, phone: str) -> bool:
        if phone in self.recipients:
            self.recipients.remove(phone)
            return True
        return False

    def symbols(self) -> list[str]:
        return sorted({r.symbol for r in self.rules})


# ---------- alert evaluation ----------

def _rule_key(rule: AlertRule) -> str:
    return f"{rule.symbol}|{rule.rule}|{rule.threshold}"


def _on_cooldown(rule: AlertRule, watchlist: Watchlist) -> bool:
    last = watchlist.last_alerts.get(_rule_key(rule))
    if not last:
        return False
    try:
        ts = datetime.fromisoformat(last)
    except ValueError:
        return False
    return datetime.now(timezone.utc) - ts < timedelta(minutes=ALERT_COOLDOWN_MINUTES)


def _rule_matches(rule: AlertRule, q: Quote) -> bool:
    t = rule.threshold
    match rule.rule:
        case "rise_pct":    return q.change_pct >= t
        case "drop_pct":    return q.change_pct <= -t
        case "price_above": return q.price >= t
        case "price_below": return 0 < q.price <= t
        case _:             return False


def evaluate_alerts(watchlist: Watchlist, quotes: Iterable[Quote] | None = None) -> list[dict]:
    """Return a list of triggered alerts (after honouring cooldowns).

    Each entry is a dict ready to be turned into a WhatsApp message.
    NOTE: caller is responsible for persisting last_alerts after dispatch.
    """
    quote_map = {q.symbol: q for q in (quotes or fetch_market_watch())}
    triggered: list[dict] = []

    for rule in watchlist.rules:
        if not rule.enabled:
            continue
        q = quote_map.get(rule.symbol.upper())
        if not q:
            continue
        if not _rule_matches(rule, q):
            continue
        if _on_cooldown(rule, watchlist):
            continue

        triggered.append({
            "rule": rule,
            "quote": q,
            "rule_key": _rule_key(rule),
            "message": _format_alert_message(rule, q),
        })

    return triggered


def _format_alert_message(rule: AlertRule, q: Quote) -> str:
    arrow = "📈" if q.change_pct >= 0 else "📉"
    return (
        f"{arrow} PSX Alert — {q.symbol}\n"
        f"Rule: {rule.describe()}\n"
        f"Price: Rs {q.price:,.2f} ({q.change:+.2f}, {q.change_pct:+.2f}%)\n"
        f"Open/High/Low: {q.open:.2f} / {q.high:.2f} / {q.low:.2f}\n"
        f"Volume: {q.volume:,}\n"
        f"Indices: {q.indices}\n"
        f"({datetime.now() + PKT_OFFSET:%Y-%m-%d %H:%M PKT})\n"
        f"\nNot investment advice."
    )


def mark_fired(watchlist: Watchlist, rule_key: str) -> None:
    watchlist.last_alerts[rule_key] = datetime.now(timezone.utc).isoformat()


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


# ---------- daily briefings ----------

def _pkt_now(now: datetime | None = None) -> datetime:
    return (now or datetime.now(timezone.utc)) + PKT_OFFSET


def _briefing_key(slot: str, day: datetime | None = None) -> str:
    d = (day or _pkt_now()).strftime("%Y-%m-%d")
    return f"{d}|{slot}"


def _slot_label(slot: str) -> str:
    """Friendly label for a briefing time slot."""
    h, _, m = slot.partition(":")
    try:
        hh = int(h)
    except ValueError:
        return slot
    if hh < 11:
        return f"Morning ({slot} PKT)"
    if hh < 14:
        return f"Midday ({slot} PKT)"
    return f"Closing ({slot} PKT)"


def due_briefings(watchlist: Watchlist, now: datetime | None = None) -> list[str]:
    """Return the briefing slots (HH:MM) that should fire on this poll.

    A slot is due when:
      - briefings are enabled
      - current PKT time is between slot_time and slot_time + window
      - it hasn't already been sent for today
    """
    if not watchlist.briefings.enabled:
        return []
    now_pkt = _pkt_now(now)
    if now_pkt.weekday() >= 5:  # Sat/Sun — skip
        return []

    due: list[str] = []
    for slot in watchlist.briefings.times:
        try:
            hh, mm = (int(x) for x in slot.split(":"))
        except ValueError:
            continue
        slot_dt = now_pkt.replace(hour=hh, minute=mm, second=0, microsecond=0)
        delta = (now_pkt - slot_dt).total_seconds() / 60.0  # minutes after slot
        if 0 <= delta <= BRIEFING_FIRE_WINDOW_MINUTES:
            if _briefing_key(slot, now_pkt) not in watchlist.briefings.last_sent:
                due.append(slot)
    return due


def mark_briefing_sent(watchlist: Watchlist, slot: str, now: datetime | None = None) -> None:
    watchlist.briefings.last_sent[_briefing_key(slot, _pkt_now(now))] = \
        datetime.now(timezone.utc).isoformat()


def generate_briefing(
    quotes: list[Quote],
    watchlist: Watchlist,
    slot: str = "",
    top_n: int = 5,
) -> str:
    """Build a WhatsApp-friendly daily briefing string."""
    now_pkt = _pkt_now()
    label = _slot_label(slot) if slot else f"{now_pkt:%H:%M PKT}"
    lines = [
        f"📊 PSX Briefing — {label}",
        f"{now_pkt:%Y-%m-%d} | Market: {'OPEN' if market_is_open() else 'CLOSED'}",
        f"Symbols tracked: {len(quotes)}",
        "",
    ]

    movers = top_movers(top_n)

    lines.append(f"📈 Top {top_n} Gainers:")
    if movers["gainers"]:
        for q in movers["gainers"]:
            lines.append(f"  • {q.symbol:<8} {q.change_pct:+6.2f}%  (Rs {q.price:,.2f})")
    else:
        lines.append("  (none)")
    lines.append("")

    lines.append(f"📉 Top {top_n} Losers:")
    if movers["losers"]:
        for q in movers["losers"]:
            lines.append(f"  • {q.symbol:<8} {q.change_pct:+6.2f}%  (Rs {q.price:,.2f})")
    else:
        lines.append("  (none)")
    lines.append("")

    lines.append(f"📊 Top {top_n} by Volume:")
    if movers["by_volume"]:
        for q in movers["by_volume"]:
            lines.append(f"  • {q.symbol:<8} vol {q.volume:>14,}  ({q.change_pct:+.2f}%)")
    lines.append("")

    sym_set = set(watchlist.symbols())
    if sym_set:
        wl_quotes = [q for q in quotes if q.symbol in sym_set]
        lines.append("⭐ Your Watchlist:")
        if wl_quotes:
            for q in wl_quotes:
                lines.append(f"  • {q.symbol:<8} Rs {q.price:>8,.2f}  ({q.change_pct:+.2f}%)")
        missing = sym_set - {q.symbol for q in wl_quotes}
        if missing:
            lines.append(f"  (not in snapshot: {', '.join(sorted(missing))})")
        lines.append("")

    lines.append("Not investment advice.")
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
