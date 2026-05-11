# MediBot + PSX-Sense

A two-in-one Streamlit chatbot. One tab is a retrieval-augmented medical reference assistant (PDFs → FAISS → Groq Llama). The other tab is a live Pakistan Stock Exchange analyst (scraped PSX data → Groq Llama) with WhatsApp alerts driven by a background worker. Both tabs run entirely on the free Groq tier.

## Features

### 🩺 Medical (RAG)
- Multiple Groq models — Llama 3.3 70B, Llama 3.1 8B, Gemma 2 9B, Llama 3 70B
- Streaming responses, source citations (file + page), 0–8 turns of rolling memory
- Strict context-only mode or context+general-knowledge mode
- Runtime PDF upload — chunked, embedded, merged into FAISS
- Clear / export chat as Markdown
- Adjustable top-k (1–8) and temperature (0.0–1.0)

### 📈 Stocks (Live PSX)
- Live PSX market-watch scraped from `dps.psx.com.pk` (cached 60s)
- Top gainers / losers / volume leaders panels
- Personal watchlist with quote table
- Groq Llama streaming chat — grounded in live PSX data injected into the system prompt
- Watchlist alert rules: `rise_pct`, `drop_pct`, `price_above`, `price_below`
- **WhatsApp delivery via Meta Cloud API** (free 1000 conversations/month, headless, official)
- **Broadcast list** — add multiple recipients; one alert fans out to all (Cloud API can't post to groups)
- **Daily briefings 3× per day** (Open / Midday / Close) — configurable times in PKT
- **Instant briefing** button — push a live market summary on demand
- Background `alert_worker.py` poller — fires alerts during market hours (Mon–Fri 09:30–15:30 PKT)
- Per-rule cooldown so a triggered alert doesn't spam every cycle
- "Not investment advice" banner on every chat reply

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=...                # used by both tabs

# WhatsApp Cloud API (Stocks tab only)
WHATSAPP_PHONE_NUMBER_ID=...    # the bot's sending-number ID
WHATSAPP_ACCESS_TOKEN=...       # temporary 24h token, or a permanent System User token
```

- Groq key: https://console.groq.com/keys (free tier)
- Meta WhatsApp Cloud API setup is documented below.

## Run

### Streamlit UI

```bash
streamlit run medibot.py
```

Switch between **Medical (RAG)** and **Stocks (Live PSX)** in the sidebar.

### PSX → WhatsApp alert worker

```bash
python alert_worker.py
```

Options:
- `--interval 120` — poll every 2 minutes (default 180s)
- `--off-hours`   — poll outside PSX market hours
- `--once`        — single cycle then exit (useful for `cron` / Task Scheduler)

### Meta WhatsApp Cloud API setup (one-time)

1. Go to https://developers.facebook.com/ → **Create App** → use case **Other** → **Business** → name it anything (e.g. "PSX-Sense").
2. In the app's left nav, click **Add Product** → **WhatsApp** → **Set up**.
3. Open **WhatsApp → API Setup**. You'll see:
   - **From**: a Meta-provided test phone number with its **Phone number ID** — copy this into `WHATSAPP_PHONE_NUMBER_ID`.
   - **Temporary access token** (24h) — copy into `WHATSAPP_ACCESS_TOKEN`.
4. Under **To**, add each recipient's WhatsApp number. Meta texts them an OTP — enter it to verify.
5. From each recipient's phone, **send "hi" to the bot's WhatsApp number** (the "From" number). This opens the 24-hour freeform window so the bot can message them.
6. In Streamlit (Stocks mode), add the same numbers under **WhatsApp recipients** and hit **📨 Test** next to one to confirm the connection.
7. Enable **Daily briefings** with three time slots (default 09:30 / 12:30 / 15:30 PKT).
8. Add **Threshold alerts** (rule + threshold) as desired.
9. Launch `python alert_worker.py` in a separate terminal — leave it running.

### Cloud API limits to know
- **No group sending.** Cloud API cannot post into WhatsApp groups; it fans out one-by-one to a recipient list.
- **24-hour window.** Freeform text only works for 24h after the recipient's last inbound message. After that, you must use a pre-approved template (`send_template` in `src/whatsapp_alerts.py`). Easiest fix: text "hi" to the bot once a day.
- **Test number** can message up to 5 verified recipients. For more, register a real business phone number in Meta Business Manager.
- **Temporary token expires every 24h.** For long-running deployments, generate a permanent System User token in Meta Business Settings.

## Index the medical PDFs

Drop PDFs into `Data/` then run:

```bash
python app.py
```

This builds `vectorstore/db_faiss/`. You can also upload PDFs at runtime from the Streamlit sidebar (Medical mode).

## CLI version (medical only)

```bash
python connect_memory_withllm.py
```

## Project layout

```
medibot.py                  Streamlit UI (Medical + Stocks)
app.py                      Build the FAISS index from Data/
alert_worker.py             Background PSX poller → WhatsApp dispatcher
connect_memory_withllm.py   CLI Q&A loop (medical only)
src/
  helper.py                 FAISS, Groq, RAG chain, PDF ingestion
  prompt.py                 System prompts + suggested questions + PSX tickers
  stocks.py                 PSX scraper, watchlist persistence, alert engine
  stock_chat.py             Groq streaming chat for the stock module
  whatsapp_alerts.py        Meta WhatsApp Cloud API client
Data/                       PDFs to be indexed
data/
  watchlist.json            Persisted watchlist + alert rules + cooldowns
vectorstore/db_faiss/       Persisted FAISS index
```

## Disclaimers

- **Medical:** for educational/reference use only — not a substitute for professional medical advice. Always consult a qualified healthcare provider.
- **Stocks:** PSX-Sense is a data-analysis tool — *not* investment advice. It explains past and present numbers; it does not predict future prices. Markets are volatile; past performance does not predict future returns. Consult a SECP-licensed advisor before trading.
