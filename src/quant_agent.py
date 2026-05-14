"""AI Stock + Crypto Quant Agent — lightweight agent pipeline.

ARCHITECTURE
────────────
    Market Data → Feature Builder → AI Agents → Signal Engine → Streamlit UI

    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
    │ MarketAgent  │──▶│   TAAgent    │──▶│  NewsAgent   │──▶│ SignalAgent  │
    │ (price/OHLC) │   │ (RSI/MACD/   │   │ (sentiment   │   │ (BUY/SELL/   │
    │              │   │  SMA/EMA)    │   │  scoring)    │   │  HOLD)       │
    └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
           │                                                          │
           ▼                                                          ▼
    Yahoo Finance / CoinGecko                              QuantAgentModule.run()

Designed to plug into the existing Streamlit app — no microservices, no DB,
no async. Pure Python + requests + pandas. Keyless data sources.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd
import requests

# ── Asset universe ────────────────────────────────────────────────────────────
# Each entry: (display, kind, yahoo_symbol, coingecko_id)
ASSET_UNIVERSE: dict[str, dict[str, str]] = {
    "BTC":  {"kind": "crypto", "yahoo": "BTC-USD", "coingecko": "bitcoin"},
    "ETH":  {"kind": "crypto", "yahoo": "ETH-USD", "coingecko": "ethereum"},
    "SOL":  {"kind": "crypto", "yahoo": "SOL-USD", "coingecko": "solana"},
    "AAPL": {"kind": "stock",  "yahoo": "AAPL",    "coingecko": ""},
    "TSLA": {"kind": "stock",  "yahoo": "TSLA",    "coingecko": ""},
    "MSFT": {"kind": "stock",  "yahoo": "MSFT",    "coingecko": ""},
    "NVDA": {"kind": "stock",  "yahoo": "NVDA",    "coingecko": ""},
    "GOOGL":{"kind": "stock",  "yahoo": "GOOGL",   "coingecko": ""},
}

_YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
_COINGECKO_URL   = "https://api.coingecko.com/api/v3/coins/{id}/market_chart"

_UA = {"User-Agent": "Mozilla/5.0 (compatible; QuantAgent/1.0)"}


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class MarketSnapshot:
    asset: str
    kind: str
    price: float
    change_pct_24h: float
    volume: float
    currency: str
    ohlcv: pd.DataFrame                # columns: open/high/low/close/volume
    source: str
    fetched_at: str


@dataclass
class TechnicalAnalysis:
    asset: str
    rsi: float                          # 0..100
    sma_20: float
    sma_50: float
    ema_12: float
    ema_26: float
    macd: float                         # ema_12 - ema_26
    macd_signal: float                  # 9-EMA of macd
    macd_state: str                     # bullish | bearish | neutral
    trend: str                          # up | down | sideways


@dataclass
class NewsSentiment:
    asset: str
    sentiment: str                      # positive | negative | neutral
    score: float                        # -1..+1
    headlines: list[dict] = field(default_factory=list)  # {title, url, score}


@dataclass
class TradeSignal:
    asset: str
    signal: str                         # BUY | SELL | HOLD
    confidence: int                     # 0..100
    rationale: list[str]


# ─────────────────────────────────────────────────────────────────────────────
#  1. MarketAgent — fetches live price + OHLCV
# ─────────────────────────────────────────────────────────────────────────────

class MarketAgent:
    """Fetches OHLCV from Yahoo Finance (stocks + crypto). Falls back to
    CoinGecko if Yahoo is unreachable for crypto symbols."""

    def fetch(self, asset: str, *, range_: str = "3mo", interval: str = "1d") -> MarketSnapshot:
        meta = ASSET_UNIVERSE.get(asset.upper())
        if not meta:
            raise ValueError(f"Unknown asset '{asset}'. Choose from {list(ASSET_UNIVERSE)}")

        try:
            return self._from_yahoo(asset.upper(), meta, range_, interval)
        except Exception as primary_exc:
            if meta["kind"] == "crypto" and meta["coingecko"]:
                try:
                    return self._from_coingecko(asset.upper(), meta)
                except Exception as fallback_exc:
                    raise RuntimeError(
                        f"Yahoo failed ({primary_exc}); CoinGecko failed ({fallback_exc})"
                    ) from fallback_exc
            raise

    # ── Yahoo Finance ─────────────────────────────────────────────────────────
    def _from_yahoo(self, asset: str, meta: dict, range_: str, interval: str) -> MarketSnapshot:
        url = _YAHOO_CHART_URL.format(symbol=meta["yahoo"])
        r = requests.get(url, params={"range": range_, "interval": interval},
                         headers=_UA, timeout=12)
        r.raise_for_status()
        payload = r.json()["chart"]["result"][0]
        ts = payload["timestamp"]
        q  = payload["indicators"]["quote"][0]
        df = pd.DataFrame({
            "open":   q.get("open"),
            "high":   q.get("high"),
            "low":    q.get("low"),
            "close":  q.get("close"),
            "volume": q.get("volume"),
        }, index=pd.to_datetime(ts, unit="s")).dropna()

        meta_block = payload["meta"]
        price = float(meta_block.get("regularMarketPrice") or df["close"].iloc[-1])
        prev_close = float(meta_block.get("chartPreviousClose") or df["close"].iloc[-2])
        change_pct = ((price - prev_close) / prev_close * 100.0) if prev_close else 0.0

        return MarketSnapshot(
            asset=asset, kind=meta["kind"], price=price,
            change_pct_24h=change_pct,
            volume=float(df["volume"].iloc[-1] or 0.0),
            currency=meta_block.get("currency", "USD"),
            ohlcv=df, source="Yahoo Finance",
            fetched_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        )

    # ── CoinGecko (crypto fallback) ───────────────────────────────────────────
    def _from_coingecko(self, asset: str, meta: dict) -> MarketSnapshot:
        r = requests.get(
            _COINGECKO_URL.format(id=meta["coingecko"]),
            params={"vs_currency": "usd", "days": "90", "interval": "daily"},
            headers=_UA, timeout=12,
        )
        r.raise_for_status()
        data = r.json()
        prices  = data.get("prices",       [])
        volumes = data.get("total_volumes", [])
        if not prices:
            raise RuntimeError("CoinGecko returned no price data")

        df = pd.DataFrame({
            "close":  [p[1] for p in prices],
            "volume": [v[1] for v in volumes] if volumes else [0] * len(prices),
        }, index=pd.to_datetime([p[0] for p in prices], unit="ms"))
        df["open"] = df["close"].shift(1).fillna(df["close"])
        df["high"] = df[["open", "close"]].max(axis=1)
        df["low"]  = df[["open", "close"]].min(axis=1)
        df = df[["open", "high", "low", "close", "volume"]]

        price = float(df["close"].iloc[-1])
        prev  = float(df["close"].iloc[-2]) if len(df) > 1 else price
        change_pct = ((price - prev) / prev * 100.0) if prev else 0.0

        return MarketSnapshot(
            asset=asset, kind=meta["kind"], price=price,
            change_pct_24h=change_pct, volume=float(df["volume"].iloc[-1] or 0.0),
            currency="USD", ohlcv=df, source="CoinGecko",
            fetched_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        )


# ─────────────────────────────────────────────────────────────────────────────
#  2. TAAgent — RSI / MACD / SMA / EMA
# ─────────────────────────────────────────────────────────────────────────────

class TAAgent:
    """Computes classical technical indicators from a close-price series.

    Pure pandas implementation — no TA-Lib dependency.
    """

    def analyze(self, snap: MarketSnapshot) -> TechnicalAnalysis:
        close = snap.ohlcv["close"].astype(float)
        if len(close) < 30:
            raise ValueError("Need at least 30 data points for TA")

        rsi = self._rsi(close, period=14).iloc[-1]
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(min(50, len(close))).mean().iloc[-1]
        ema_12 = close.ewm(span=12, adjust=False).mean().iloc[-1]
        ema_26 = close.ewm(span=26, adjust=False).mean().iloc[-1]

        macd_line   = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_v   = float(macd_line.iloc[-1])
        signal_v = float(signal_line.iloc[-1])

        if macd_v > signal_v and macd_v > 0:
            macd_state = "bullish"
        elif macd_v < signal_v and macd_v < 0:
            macd_state = "bearish"
        else:
            macd_state = "neutral"

        # Trend: compare last close to SMA_20
        last = float(close.iloc[-1])
        if last > sma_20 * 1.01:
            trend = "up"
        elif last < sma_20 * 0.99:
            trend = "down"
        else:
            trend = "sideways"

        return TechnicalAnalysis(
            asset=snap.asset,
            rsi=float(rsi),
            sma_20=float(sma_20),
            sma_50=float(sma_50),
            ema_12=float(ema_12),
            ema_26=float(ema_26),
            macd=macd_v,
            macd_signal=signal_v,
            macd_state=macd_state,
            trend=trend,
        )

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
        rs = gain / loss.replace(0, math.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)


# ─────────────────────────────────────────────────────────────────────────────
#  3. NewsAgent — DuckDuckGo headlines + keyword sentiment
# ─────────────────────────────────────────────────────────────────────────────

# Lightweight lexicon — good enough for headline-level sentiment without an
# extra ML dependency. Extend freely.
_POS_WORDS = {
    "surge", "soar", "rally", "gain", "rise", "rises", "rising", "bull", "bullish",
    "record", "high", "growth", "beats", "beat", "upgrade", "upgraded", "outperform",
    "approve", "approved", "boost", "strong", "strength", "positive", "profit",
    "profitable", "rebound", "breakthrough", "milestone", "buy",
}
_NEG_WORDS = {
    "plunge", "plummet", "crash", "drop", "drops", "fall", "falls", "falling",
    "bear", "bearish", "loss", "losses", "low", "lows", "miss", "missed",
    "downgrade", "downgraded", "underperform", "weak", "weakness", "negative",
    "decline", "declines", "lawsuit", "ban", "banned", "hack", "exploit",
    "fraud", "selloff", "sell-off", "concerns", "fear", "panic",
}

_WORD_RE = re.compile(r"[a-zA-Z\-]+")


class NewsAgent:
    """Pulls recent headlines via DuckDuckGo, scores sentiment by lexicon."""

    def get_sentiment(self, asset: str, *, max_results: int = 8) -> NewsSentiment:
        query = self._query_for(asset)
        headlines: list[dict] = []
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                for hit in ddgs.news(query, max_results=max_results) or []:
                    title = (hit.get("title") or "").strip()
                    if not title:
                        continue
                    s = self._score_text(title)
                    headlines.append({
                        "title":  title,
                        "url":    hit.get("url", ""),
                        "source": hit.get("source", ""),
                        "score":  s,
                    })
        except Exception as exc:
            headlines.append({"title": f"(news fetch failed: {exc})", "url": "",
                              "source": "", "score": 0.0})

        if not headlines:
            return NewsSentiment(asset=asset, sentiment="neutral", score=0.0,
                                 headlines=[])

        avg = sum(h["score"] for h in headlines) / max(1, len(headlines))
        if avg >= 0.15:
            label = "positive"
        elif avg <= -0.15:
            label = "negative"
        else:
            label = "neutral"
        return NewsSentiment(asset=asset, sentiment=label, score=round(avg, 3),
                             headlines=headlines)

    @staticmethod
    def _query_for(asset: str) -> str:
        meta = ASSET_UNIVERSE.get(asset.upper(), {})
        if meta.get("kind") == "crypto":
            return f"{asset} cryptocurrency news"
        return f"{asset} stock news"

    @staticmethod
    def _score_text(text: str) -> float:
        words = [w.lower() for w in _WORD_RE.findall(text)]
        pos = sum(1 for w in words if w in _POS_WORDS)
        neg = sum(1 for w in words if w in _NEG_WORDS)
        if pos == 0 and neg == 0:
            return 0.0
        return (pos - neg) / (pos + neg)


# ─────────────────────────────────────────────────────────────────────────────
#  4. SignalAgent — combines TA + sentiment into BUY/SELL/HOLD
# ─────────────────────────────────────────────────────────────────────────────

class SignalAgent:
    """Rule-based signal generator. Transparent, explainable, no LLM call.

    Confidence is a 0..100 score derived from how many sub-signals agree.
    """

    def generate(self, snap: MarketSnapshot, ta: TechnicalAnalysis,
                 news: NewsSentiment) -> TradeSignal:
        buy_score = 0
        sell_score = 0
        rationale: list[str] = []

        # ── RSI ───────────────────────────────────────────────────────────────
        if ta.rsi < 30:
            buy_score += 2
            rationale.append(f"RSI {ta.rsi:.1f} → oversold (buy signal)")
        elif ta.rsi > 70:
            sell_score += 2
            rationale.append(f"RSI {ta.rsi:.1f} → overbought (sell signal)")
        else:
            rationale.append(f"RSI {ta.rsi:.1f} → neutral zone")

        # ── MACD ──────────────────────────────────────────────────────────────
        if ta.macd_state == "bullish":
            buy_score += 1
            rationale.append("MACD bullish (12-EMA above 26-EMA + signal)")
        elif ta.macd_state == "bearish":
            sell_score += 1
            rationale.append("MACD bearish (12-EMA below 26-EMA + signal)")
        else:
            rationale.append("MACD neutral")

        # ── Trend (SMA20) ─────────────────────────────────────────────────────
        if ta.trend == "up":
            buy_score += 1
            rationale.append(f"Price above SMA20 ({ta.sma_20:.2f}) → uptrend")
        elif ta.trend == "down":
            sell_score += 1
            rationale.append(f"Price below SMA20 ({ta.sma_20:.2f}) → downtrend")
        else:
            rationale.append("Price near SMA20 → sideways")

        # ── 24h momentum ──────────────────────────────────────────────────────
        if snap.change_pct_24h > 3:
            buy_score += 1
            rationale.append(f"+{snap.change_pct_24h:.2f}% 24h momentum")
        elif snap.change_pct_24h < -3:
            sell_score += 1
            rationale.append(f"{snap.change_pct_24h:.2f}% 24h drop")

        # ── News sentiment ────────────────────────────────────────────────────
        if news.sentiment == "positive":
            buy_score += 1
            rationale.append(f"News sentiment positive ({news.score:+.2f})")
        elif news.sentiment == "negative":
            sell_score += 1
            rationale.append(f"News sentiment negative ({news.score:+.2f})")
        else:
            rationale.append("News sentiment neutral")

        # ── Decide ────────────────────────────────────────────────────────────
        total = buy_score + sell_score
        if buy_score > sell_score and buy_score >= 2:
            signal = "BUY"
            confidence = int(round(buy_score / max(total, 1) * 100))
        elif sell_score > buy_score and sell_score >= 2:
            signal = "SELL"
            confidence = int(round(sell_score / max(total, 1) * 100))
        else:
            signal = "HOLD"
            confidence = 50 + (10 if total == 0 else 0)

        return TradeSignal(asset=snap.asset, signal=signal,
                           confidence=min(confidence, 95), rationale=rationale)


# ─────────────────────────────────────────────────────────────────────────────
#  Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class QuantAgentModule:
    """Runs the full agent pipeline for a single asset."""

    def __init__(self) -> None:
        self.market = MarketAgent()
        self.ta     = TAAgent()
        self.news   = NewsAgent()
        self.signal = SignalAgent()

    def run(self, asset: str) -> dict[str, Any]:
        snap = self.market.fetch(asset)
        ta   = self.ta.analyze(snap)
        news = self.news.get_sentiment(asset)
        sig  = self.signal.generate(snap, ta, news)
        return {
            "market":    snap,
            "technical": ta,
            "news":      news,
            "signal":    sig,
        }
