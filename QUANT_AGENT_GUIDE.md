# 📈 AI Quant Agent — Complete Technical Documentation

> A lightweight, multi-agent financial-intelligence module that fetches market data,
> computes technical indicators, scores news sentiment, and emits an explainable
> **BUY / SELL / HOLD** signal for stocks and crypto — fully integrated into the
> existing **AI Workbench** Streamlit application.

---

## Table of Contents
1. [Module Goal & Scope](#1-module-goal--scope)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Data Sources](#3-data-sources)
4. [Agent-by-Agent Deep Dive](#4-agent-by-agent-deep-dive)
5. [End-to-End Workflow](#5-end-to-end-workflow)
6. [Data Types Reference](#6-data-types-reference)
7. [Streamlit Integration](#7-streamlit-integration)
8. [Access Control](#8-access-control)
9. [How to Run & Test](#9-how-to-run--test)
10. [Extending the Module](#10-extending-the-module)
11. [Limitations & Disclaimers](#11-limitations--disclaimers)
12. [File Inventory](#12-file-inventory)

---

## 1. Module Goal & Scope

### What it does
Given an asset symbol (e.g. `BTC`, `AAPL`), the module:

1. Fetches **live price + 3 months of OHLCV** history.
2. Computes **technical indicators** (RSI, MACD, SMA, EMA).
3. Pulls **recent news headlines** and scores them for sentiment.
4. Aggregates everything into a **rule-based trading signal** with
   confidence + human-readable rationale.
5. Renders the result in a Streamlit dashboard with metrics, charts,
   headlines, and JSON output.

### What it deliberately is NOT
| ✅ In scope                                  | ❌ Out of scope                          |
| -------------------------------------------- | --------------------------------------- |
| Inference + simple decision rules            | Full backtesting engine                 |
| Free / keyless public data sources           | Paid market-data feeds                  |
| Single-asset analysis on demand              | Portfolio optimization                  |
| Lightweight Streamlit plug-in                | Microservices / async / message bus    |
| Transparent, explainable signal logic        | Black-box ML or deep-RL trading models  |

### Design principles
- **Lightweight** — no new heavy dependencies; reuses `requests`, `pandas`, `duckduckgo-search`.
- **Explainable** — the SignalAgent emits a `rationale: list[str]` so the user understands *why* a signal fires.
- **Plug-in friendly** — one new file ([src/quant_agent.py](src/quant_agent.py)) + a thin integration in [medibot.py](medibot.py). No changes to existing modules.
- **Graceful degradation** — Yahoo failure falls back to CoinGecko for crypto; news-fetch failure returns an empty/neutral sentiment instead of crashing the pipeline.

---

## 2. High-Level Architecture

```text
                         ┌──────────────────────────────┐
   User picks asset ───▶ │  Streamlit sidebar           │
   "Run Analysis"        │  (medibot.py → render_quant) │
                         └──────────────┬───────────────┘
                                        │ QuantAgentModule().run(asset)
                                        ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │                        QuantAgentModule                            │
   │   (orchestrator — chains the four agents sequentially)             │
   └──────────┬─────────────┬────────────────┬───────────────┬──────────┘
              │             │                │               │
              ▼             ▼                ▼               ▼
       ┌────────────┐  ┌──────────┐   ┌────────────┐  ┌────────────┐
       │MarketAgent │  │  TAAgent │   │ NewsAgent  │  │SignalAgent │
       │            │  │          │   │            │  │            │
       │Yahoo + CG  │  │RSI MACD  │   │DuckDuckGo  │  │Rule engine │
       │  OHLCV     │  │SMA EMA   │   │ + lexicon  │  │BUY/SELL/HOLD│
       └─────┬──────┘  └────┬─────┘   └─────┬──────┘  └─────┬──────┘
             │              │               │               │
             ▼              ▼               ▼               ▼
        MarketSnapshot  TechnicalAnalysis  NewsSentiment   TradeSignal
                       │                  │
                       └──── combined ────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Streamlit render │
                       │  • signal card   │
                       │  • metrics+chart │
                       │  • headlines     │
                       │  • rationale     │
                       │  • raw JSON      │
                       └──────────────────┘
```

### Pipeline contract
Each agent has a single public method with a deterministic input/output contract:

| Agent         | Input                                | Output                |
| ------------- | ------------------------------------ | --------------------- |
| `MarketAgent` | `asset: str`                         | `MarketSnapshot`      |
| `TAAgent`     | `MarketSnapshot`                     | `TechnicalAnalysis`   |
| `NewsAgent`   | `asset: str`                         | `NewsSentiment`       |
| `SignalAgent` | `MarketSnapshot, TechnicalAnalysis, NewsSentiment` | `TradeSignal` |

The orchestrator (`QuantAgentModule.run()`) simply pipes the outputs through.

---

## 3. Data Sources

All data sources are **public, keyless, and free** — no API tokens required.

### 3.1 Yahoo Finance (primary)
- **Endpoint**: `https://query1.finance.yahoo.com/v8/finance/chart/{symbol}`
- **Params**: `range=3mo`, `interval=1d`
- **Returns**: OHLCV bars + meta block with `regularMarketPrice`, `chartPreviousClose`, `currency`.
- **Symbols**: stocks use raw tickers (`AAPL`); crypto uses suffixed pairs (`BTC-USD`).
- **User-Agent** is set to avoid 403 responses from Yahoo's edge layer.

### 3.2 CoinGecko (fallback for crypto)
- **Endpoint**: `https://api.coingecko.com/api/v3/coins/{id}/market_chart`
- **Params**: `vs_currency=usd`, `days=90`, `interval=daily`
- **Returns**: daily closing prices + total volumes (no true OHLC; open/high/low are synthesised from close).
- Triggered automatically when Yahoo raises an exception for a crypto asset.

### 3.3 DuckDuckGo News (for sentiment)
- Library: `duckduckgo-search` (`DDGS().news(query, max_results=8)`)
- Query templates:
  - Crypto → `"{asset} cryptocurrency news"`
  - Stock  → `"{asset} stock news"`
- Returns title, URL, source, date.
- Headline failures (rate-limit, network) degrade gracefully to `neutral` sentiment with one error headline marker.

### 3.4 Asset Universe
Defined in [src/quant_agent.py](src/quant_agent.py):

```python
ASSET_UNIVERSE = {
    "BTC":  {"kind": "crypto", "yahoo": "BTC-USD", "coingecko": "bitcoin"},
    "ETH":  {"kind": "crypto", "yahoo": "ETH-USD", "coingecko": "ethereum"},
    "SOL":  {"kind": "crypto", "yahoo": "SOL-USD", "coingecko": "solana"},
    "AAPL": {"kind": "stock",  "yahoo": "AAPL",    "coingecko": ""},
    "TSLA": {"kind": "stock",  "yahoo": "TSLA",    "coingecko": ""},
    "MSFT": {"kind": "stock",  "yahoo": "MSFT",    "coingecko": ""},
    "NVDA": {"kind": "stock",  "yahoo": "NVDA",    "coingecko": ""},
    "GOOGL":{"kind": "stock",  "yahoo": "GOOGL",   "coingecko": ""},
}
```

To add an asset, just append an entry — the rest of the pipeline is generic.

---

## 4. Agent-by-Agent Deep Dive

### 4.1 `MarketAgent` — Market Data Fetcher
**Responsibility**: deliver a `MarketSnapshot` for the chosen asset.

**Algorithm**
1. Look up `ASSET_UNIVERSE[asset]` (raises `ValueError` if unknown).
2. Try Yahoo Finance with `range=3mo`, `interval=1d` (≈ 60–90 daily bars).
3. On `requests` exception **and** asset is crypto with a CoinGecko ID → fall back to CoinGecko.
4. Build a pandas DataFrame `[open, high, low, close, volume]` indexed by datetime.
5. Compute `change_pct_24h = (price − prev_close) / prev_close × 100`.

**Why 3 months of daily bars?**
- The TAAgent needs **≥30 points** (it asserts this).
- 3 months ≈ 65 trading days → plenty of headroom for SMA(50) and MACD(26+9).
- Keeps the payload tiny — sub-second fetches.

### 4.2 `TAAgent` — Technical Indicator Engine
Pure-pandas implementation. Indicators:

| Indicator   | Formula                                                         | Library call                                |
| ----------- | --------------------------------------------------------------- | ------------------------------------------- |
| **RSI(14)** | `100 − 100 / (1 + avg_gain/avg_loss)` (Wilder smoothing via EWM) | `delta.clip().ewm(alpha=1/14)`              |
| **SMA(20)** | Simple moving average                                            | `close.rolling(20).mean()`                  |
| **SMA(50)** | Simple moving average (window clamped to data length)            | `close.rolling(min(50, len)).mean()`        |
| **EMA(12)** | Exponential moving average                                       | `close.ewm(span=12, adjust=False).mean()`   |
| **EMA(26)** | Exponential moving average                                       | `close.ewm(span=26, adjust=False).mean()`   |
| **MACD**    | `EMA(12) − EMA(26)`                                              | derived                                     |
| **Signal**  | `EMA(9)` of MACD line                                            | derived                                     |

**Derived state**
- `macd_state`:
  - `bullish` if `macd > signal AND macd > 0`
  - `bearish` if `macd < signal AND macd < 0`
  - else `neutral`
- `trend` (vs SMA20):
  - `up`        if `close > SMA20 × 1.01`
  - `down`      if `close < SMA20 × 0.99`
  - else `sideways`

The ±1 % buffer avoids flip-flopping when price is hovering right on the moving average.

### 4.3 `NewsAgent` — Headline Sentiment Scorer
Two-step pipeline:

**Step 1 — Fetch**
Uses `duckduckgo-search` to grab up to 8 recent headlines for the asset.

**Step 2 — Score**
A custom **finance-tuned lexicon** lives at the top of [src/quant_agent.py](src/quant_agent.py):

- `_POS_WORDS` — *surge, rally, beat, upgrade, breakthrough, …* (≈30 terms)
- `_NEG_WORDS` — *plunge, miss, downgrade, lawsuit, hack, selloff, …* (≈30 terms)

For each headline:

```python
score = (pos − neg) / (pos + neg)        ∈ [−1, +1]
```

Headlines are averaged. The final label maps:
- `score ≥ +0.15` → `positive`
- `score ≤ −0.15` → `negative`
- else            → `neutral`

**Why a lexicon, not an LLM?**
- Zero token cost, no API key, instant.
- Domain-tuned vocabulary (finance lingo > generic sentiment).
- Fully deterministic and auditable.

If you ever want LLM-grade sentiment, the `NewsAgent` is the natural place to swap in a Groq call — the rest of the pipeline doesn't care.

### 4.4 `SignalAgent` — Decision Engine
A transparent **rule-based scoring system** — *not* a black-box model.

It accumulates a `buy_score` and a `sell_score`, then decides:

| Sub-signal              | Rule                                  | Score   |
| ----------------------- | ------------------------------------- | ------- |
| RSI < 30 (oversold)     | strong buy                            | buy +2  |
| RSI > 70 (overbought)   | strong sell                           | sell +2 |
| MACD bullish            | trend confirmation                    | buy +1  |
| MACD bearish            | trend confirmation                    | sell +1 |
| Price > SMA20 (up)      | momentum confirmation                 | buy +1  |
| Price < SMA20 (down)    | momentum confirmation                 | sell +1 |
| 24h change > +3 %       | breakout momentum                     | buy +1  |
| 24h change < −3 %       | selloff momentum                      | sell +1 |
| News sentiment positive | macro tailwind                        | buy +1  |
| News sentiment negative | macro headwind                        | sell +1 |

**Final decision**
```text
if buy  > sell and buy  ≥ 2   →  BUY
if sell > buy  and sell ≥ 2   →  SELL
else                          →  HOLD
```

**Confidence**
```text
confidence = round(winner_score / (buy + sell) × 100)   # capped at 95
```

A `HOLD` defaults to 50 (60 if no signals fired at all — "calm market, no edge").

**Rationale**
Every rule that examines a value appends a one-line explanation to `rationale: list[str]`, surfaced to the user in the dashboard.

---

## 5. End-to-End Workflow

What happens when the user clicks **🚀 Run Analysis**:

```text
1. Streamlit captures st.session_state.quant_asset (e.g. "BTC")
        │
2. medibot.py — sidebar handler clears prior results and calls
        │   QuantAgentModule().run(asset)
        ▼
3. MarketAgent.fetch(asset)
   ├─ Yahoo Finance GET → JSON → pandas DataFrame
   └─ (fallback) CoinGecko GET → DataFrame
                  │
                  ▼  MarketSnapshot
4. TAAgent.analyze(snap)
   └─ close-price series → RSI / SMA / EMA / MACD
                  │
                  ▼  TechnicalAnalysis
5. NewsAgent.get_sentiment(asset)
   ├─ DDGS().news(query)  → headlines
   └─ lexicon scoring     → avg sentiment score + label
                  │
                  ▼  NewsSentiment
6. SignalAgent.generate(snap, ta, news)
   └─ rule scoring → BUY / SELL / HOLD + confidence + rationale
                  │
                  ▼  TradeSignal
7. Orchestrator returns dict {market, technical, news, signal}
8. Streamlit re-renders dashboard from st.session_state.quant_result
```

Latency budget on a typical run: **~1.5–3 s** total (Yahoo ~0.4 s, DDG news ~1–2 s, indicators ~0.05 s, rendering ~0.2 s).

---

## 6. Data Types Reference

All dataclasses live in [src/quant_agent.py](src/quant_agent.py).

```python
@dataclass
class MarketSnapshot:
    asset: str
    kind: str                 # "stock" | "crypto"
    price: float
    change_pct_24h: float
    volume: float
    currency: str             # "USD" etc.
    ohlcv: pd.DataFrame       # open/high/low/close/volume by date
    source: str               # "Yahoo Finance" | "CoinGecko"
    fetched_at: str           # ISO 8601 UTC

@dataclass
class TechnicalAnalysis:
    asset: str
    rsi: float                # 0..100
    sma_20: float
    sma_50: float
    ema_12: float
    ema_26: float
    macd: float               # ema_12 - ema_26
    macd_signal: float        # 9-EMA of macd
    macd_state: str           # "bullish" | "bearish" | "neutral"
    trend: str                # "up" | "down" | "sideways"

@dataclass
class NewsSentiment:
    asset: str
    sentiment: str            # "positive" | "negative" | "neutral"
    score: float              # -1..+1
    headlines: list[dict]     # [{title, url, source, score}, ...]

@dataclass
class TradeSignal:
    asset: str
    signal: str               # "BUY" | "SELL" | "HOLD"
    confidence: int           # 0..95
    rationale: list[str]
```

---

## 7. Streamlit Integration

Lives entirely in [medibot.py](medibot.py). Three touch points:

### 7.1 Domain registration
```python
DOMAINS = [..., "🤖 Agentic AI", "📈 Quant Agent"]
```

### 7.2 Session state (in `init_state()`)
```python
"quant_asset":  "BTC",
"quant_result": None,
"quant_error":  "",
```

### 7.3 Sidebar branch
Shows the asset selector, **Run Analysis** primary button, **Clear results** button, and a one-line caption explaining the pipeline.

### 7.4 `render_quant()` dashboard
Renders, in order:
1. **Final Signal card** — large color-coded BUY (green) / SELL (red) / HOLD (amber) tile with confidence %.
2. **Market summary** — price (+ 24h Δ), volume, kind, source + a `st.line_chart` of close prices.
3. **Technical analysis** — RSI (with oversold/overbought tag), MACD value (with state), SMA 20/50, trend.
4. **News sentiment** — label + score + clickable headline list with per-headline 🟢/🔴/⚪ markers.
5. **Why this signal?** — bullet list from `signal.rationale`.
6. **Raw JSON expander** — every field of every dataclass for power users / API consumers.

### 7.5 Route registration
```python
_ROUTES["📈 Quant Agent"] = render_quant
_msg_key_map["📈 Quant Agent"] = None   # no chat buffer for this domain
```

---

## 8. Access Control

Defined at the top of the sidebar section in [medibot.py](medibot.py):

```python
ADMIN_DOMAINS = DOMAINS + ["🛡️ Admin Panel"]     # super admin sees everything
USER_DOMAINS  = [
    "Medical (RAG)",
    "Career Roadmap",
    "📈 Quant Agent",
]
```

And applied at render time:

```python
_domain_list = ADMIN_DOMAINS if _is_superadmin else USER_DOMAINS
```

- **Super Admin** → all 9 domains + Admin Panel.
- **Regular User** → Medical (RAG), Career Roadmap, 📈 Quant Agent only.

If a regular user's previously-selected domain isn't in their allowed list, they're transparently bounced to **Medical (RAG)** on the next render.

---

## 9. How to Run & Test

### 9.1 Prerequisites
Already in [requirements.txt](requirements.txt):
- `requests`, `pandas`, `beautifulsoup4`, `lxml`
- `duckduckgo-search`
- `streamlit`

No new dependencies were added for this module.

### 9.2 Launch the app
```bash
streamlit run medibot.py
```

### 9.3 Use the module
1. Sign in (super admin or a regular user).
2. Pick **📈 Quant Agent** in the sidebar.
3. Choose an asset (BTC / ETH / SOL / AAPL / TSLA / MSFT / NVDA / GOOGL).
4. Click **🚀 Run Analysis**.
5. Inspect the dashboard.

### 9.4 Quick programmatic test
```python
from src.quant_agent import QuantAgentModule

result = QuantAgentModule().run("BTC")
print(result["signal"].signal, result["signal"].confidence, "%")
for r in result["signal"].rationale:
    print(" -", r)
```

---

## 10. Extending the Module

### 10.1 Add a new asset
Append one entry to `ASSET_UNIVERSE` in [src/quant_agent.py](src/quant_agent.py):

```python
"DOGE": {"kind": "crypto", "yahoo": "DOGE-USD", "coingecko": "dogecoin"},
```

### 10.2 Add a new indicator
Extend `TAAgent.analyze()` and add a field to `TechnicalAnalysis`.

Example — Bollinger Bands:
```python
std20 = close.rolling(20).std().iloc[-1]
upper_band = sma_20 + 2 * std20
lower_band = sma_20 - 2 * std20
```
Optionally wire it into `SignalAgent` as another scored sub-signal.

### 10.3 Upgrade to LLM sentiment
Swap `NewsAgent._score_text()` for a Groq call:

```python
from src.stock_chat import call_groq_json
def _score_text(self, text: str) -> float:
    payload = call_groq_json(
        system_prompt="Score this finance headline from -1 (bearish) to +1 (bullish). Return JSON {\"score\": float}.",
        user_prompt=text,
        model="llama-3.3-70b-versatile",
    )
    return float(payload.get("score", 0.0))
```

### 10.4 Expose as a REST endpoint
Mirror the pattern in [api/main.py](api/main.py):

```python
@app.post("/quant/run")
def run_quant(req: QuantRequest):
    return QuantAgentModule().run(req.asset)
```

You'll need a Pydantic response model that mirrors the four dataclasses (or just call `dataclasses.asdict` minus the DataFrame).

---

## 11. Limitations & Disclaimers

### Educational only
Signals are **rule-based heuristics**, not predictions. Do not trade real money based on them without independent validation.

### Data quality
- Yahoo Finance is unofficial — rate-limit headers can change without notice.
- CoinGecko's free tier synthesises OHLC from daily closes (no intraday bars).
- DuckDuckGo news ranking varies day-to-day; the agent doesn't deduplicate.

### Sentiment lexicon
~60 hand-curated finance terms. Good enough for headline tone, but blind to sarcasm, negation, and ticker mismatches ("Apple announces…" vs "Apple Records to sue…").

### No persistence
Results live in `st.session_state` only — cleared on logout or app restart. No DB writes, no caching.

### No backtesting / no risk model
The signal has zero awareness of position size, stop-loss, drawdown, or correlation. It's a *snapshot*, not a strategy.

---

## 12. File Inventory

| File                                       | Purpose                                                     | Status   |
| ------------------------------------------ | ----------------------------------------------------------- | -------- |
| [src/quant_agent.py](src/quant_agent.py)   | Four agents + orchestrator (≈340 LOC)                       | new      |
| [medibot.py](medibot.py)                   | Sidebar branch, `render_quant`, route, session state, ACL   | modified |
| [QUANT_AGENT_GUIDE.md](QUANT_AGENT_GUIDE.md) | This document                                              | new      |

No changes to: any other `src/*.py`, `requirements.txt`, `api/main.py`, or auth schema.

---

## Appendix — Why each design choice?

| Decision                            | Rationale                                                                |
| ----------------------------------- | ------------------------------------------------------------------------ |
| Sequential agents (not LangGraph)   | Spec asks for *lightweight*; 4 deterministic steps don't need a graph.   |
| Dataclasses (not Pydantic)          | Internal types; no validation needed; cheaper imports.                   |
| Pure-pandas TA (not TA-Lib)         | TA-Lib needs a C compiler on Windows; pandas EWM is identical accuracy. |
| Lexicon sentiment (not transformer) | Zero deps, zero latency, fully auditable, good enough for headlines.    |
| Rule-based signal (not ML)          | Explainable to end users; deterministic; no training data needed.       |
| No backtesting                      | Scope rule from the spec: "inference + simple logic" only.              |
| `_msg_key_map[...] = None`          | This domain has a dashboard, not a chat — disables the shared chat bar. |

---

**End of guide.**
