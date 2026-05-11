from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import find_dotenv, load_dotenv

from src.helper import (
    AVAILABLE_MODELS,
    PDFIngestError,
    build_chain,
    build_llm,
    history_as_messages,
    ingest_uploaded_pdfs,
    load_vectorstore,
)
from src.prompt import (
    ASSISTED_SYSTEM_PROMPT,
    STRICT_SYSTEM_PROMPT,
    STOCK_SYSTEM_PROMPT,
    SUGGESTED_QUESTIONS,
    STOCK_SUGGESTED_QUESTIONS,
    PSX_TICKERS,
)
from src.stock_chat import (
    STOCK_LLM_MODELS,
    build_system_prompt,
    history_for_chat,
    stream_chat,
)
from src import stocks
from src.whatsapp_alerts import WhatsAppError, credentials_ok, send_message, send_to_many

load_dotenv(find_dotenv())

st.set_page_config(
    page_title="MediBot + PSX-Sense",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- styling ----------
st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; margin-bottom: 0.2rem; }
    .subtle { color: #888; font-size: 0.9rem; }
    .disclaimer {
        background: #fff4e5; border-left: 4px solid #ff9800;
        padding: 0.6rem 0.9rem; border-radius: 4px; font-size: 0.85rem; color: #5d3a00;
        margin-bottom: 1rem;
    }
    .disclaimer-stocks {
        background: #eaf4ff; border-left: 4px solid #1976d2;
        padding: 0.6rem 0.9rem; border-radius: 4px; font-size: 0.85rem; color: #0b3a66;
        margin-bottom: 1rem;
    }
    .chip {
        display: inline-block; padding: 4px 10px; margin: 3px;
        background: #eef3ff; border-radius: 999px; font-size: 0.85rem;
        border: 1px solid #d6e0ff;
    }
    .source-box {
        background: #f7f7f9; border-radius: 6px; padding: 8px 12px;
        font-size: 0.85rem; margin-top: 4px;
    }
    .metric-up   { color: #137333; font-weight: 600; }
    .metric-down { color: #c5221f; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- session state ----------
def init_state():
    defaults = {
        "domain": "Medical (RAG)",
        # medical
        "messages_medical": [],
        "model_label": next(iter(AVAILABLE_MODELS)),
        "temperature": 0.4,
        "top_k": 3,
        "strict_mode": True,
        "memory_turns": 3,
        "last_sources": [],
        # stocks
        "messages_stocks": [],
        "stock_model_label": next(iter(STOCK_LLM_MODELS)),
        "stock_temperature": 0.3,
        "stocks_memory_turns": 3,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


@st.cache_resource(show_spinner=False)
def cached_vectorstore():
    return load_vectorstore()


@st.cache_data(ttl=60, show_spinner=False)
def cached_market_watch():
    """Fetch + cache the full PSX snapshot for 60s.

    `stocks.fetch_market_watch` has its own in-process cache, but Streamlit's
    cache_data lets the Quote list survive across reruns of the same session.
    """
    return stocks.fetch_market_watch()


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.session_state.domain = st.radio(
        "🧭 Domain",
        ["Medical (RAG)", "Stocks (Live PSX)"],
        index=["Medical (RAG)", "Stocks (Live PSX)"].index(st.session_state.domain),
        horizontal=True,
    )
    st.divider()

    if st.session_state.domain == "Medical (RAG)":
        st.header("⚙️ Medical settings")

        st.session_state.model_label = st.selectbox(
            "Groq model",
            options=list(AVAILABLE_MODELS.keys()),
            index=list(AVAILABLE_MODELS.keys()).index(st.session_state.model_label),
        )
        st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature, 0.1)
        st.session_state.top_k = st.slider("Retrieved chunks (top-k)", 1, 8, st.session_state.top_k)
        st.session_state.memory_turns = st.slider("Conversation memory (turns)", 0, 8, st.session_state.memory_turns)
        st.session_state.strict_mode = st.toggle(
            "Strict context-only mode",
            value=st.session_state.strict_mode,
            help="When on, MediBot only answers from your uploaded reference material.",
        )

        st.divider()
        st.subheader("📄 Add documents")
        uploads = st.file_uploader(
            "Upload medical PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="Files are chunked and merged into the FAISS index.",
        )
        if uploads and st.button("Ingest uploaded PDFs", use_container_width=True):
            try:
                with st.spinner(f"Embedding {len(uploads)} file(s)…"):
                    added = ingest_uploaded_pdfs(uploads)
                    cached_vectorstore.clear()
                st.success(f"Added {added} chunks to the index.")
            except PDFIngestError as e:
                st.error(f"Ingest failed: {e}")
            except Exception as e:
                st.error(f"Unexpected error during ingest: {e}")

        st.divider()
        try:
            db = cached_vectorstore()
            n_docs = db.index.ntotal
            st.caption(f"📚 {n_docs:,} indexed chunks")
        except Exception:
            st.caption("📚 Index not loaded")
        st.caption(f"🤖 Groq model: `{AVAILABLE_MODELS[st.session_state.model_label]}`")

    else:  # Stocks
        st.header("📈 Stocks settings")

        # Load watchlist once for the whole sidebar render.
        watchlist = stocks.Watchlist.load()

        # ----- 1. Recipients (broadcast list) -----
        st.subheader("📲 WhatsApp recipients")

        if not credentials_ok():
            st.warning(
                "Cloud API credentials missing. Add `WHATSAPP_PHONE_NUMBER_ID` and "
                "`WHATSAPP_ACCESS_TOKEN` to your `.env` (see Readme)."
            )

        st.caption(
            "Cloud API can't post into WhatsApp groups (Meta restriction). "
            "Add each number here and the bot will fan out to all of them."
        )

        # Add-recipient row
        with st.form("add_recipient_form", clear_on_submit=True):
            ac1, ac2 = st.columns([4, 1])
            new_phone = ac1.text_input(
                "Add number (E.164, e.g. +923001234567)",
                value="",
                placeholder="+92300...",
                label_visibility="collapsed",
            )
            add_clicked = ac2.form_submit_button("➕ Add", use_container_width=True)
            if add_clicked:
                phone = (new_phone or "").strip()
                if not phone.startswith("+"):
                    st.error("Phone must start with '+' and country code.")
                elif phone in watchlist.recipients:
                    st.warning("Already in list.")
                else:
                    watchlist.add_recipient(phone)
                    watchlist.save()
                    st.success(f"Added {phone}")
                    st.rerun()

        # Existing recipients list
        if watchlist.recipients:
            for i, ph in enumerate(watchlist.recipients):
                rcols = st.columns([6, 2, 1])
                rcols[0].markdown(f"`{ph}`")
                if rcols[1].button("📨 Test", key=f"rcp-test-{i}", use_container_width=True,
                                   help="Send a one-off connection test to this number."):
                    try:
                        with st.spinner(f"Sending test to {ph}…"):
                            send_message(ph, "✅ PSX-Sense test — Cloud API connection works.")
                        st.success(f"Sent to {ph}")
                    except WhatsAppError as e:
                        st.error(f"{e}")
                if rcols[2].button("🗑", key=f"rcp-del-{i}",
                                   help="Remove this number from the broadcast list."):
                    watchlist.remove_recipient(ph)
                    watchlist.save()
                    st.rerun()
        else:
            st.caption("No recipients yet. Add a number above.")

        # ----- 2. Instant briefing -----
        st.divider()
        st.subheader("⚡ Instant briefing")
        if st.button("📤 Send briefing now (to all)", use_container_width=True,
                     help="Generate a live PSX summary and push it to every recipient."):
            if not watchlist.recipients:
                st.error("Add at least one recipient first.")
            else:
                try:
                    with st.spinner("Fetching PSX data + sending…"):
                        quotes = stocks.fetch_market_watch(force=True)
                        msg = stocks.generate_briefing(quotes, watchlist, slot="")
                        sent, errors = send_to_many(watchlist.recipients, msg)
                    if sent:
                        st.success(f"Briefing sent to {sent}/{len(watchlist.recipients)} recipient(s).")
                    for err in errors:
                        st.error(err)
                except Exception as e:
                    st.error(f"Briefing failed: {e}")

        # ----- 3. Daily scheduled briefings (3x/day) -----
        st.divider()
        st.subheader("⏰ Daily briefings (3× per day)")

        bri_enabled = st.toggle(
            "Enable daily WhatsApp briefings",
            value=watchlist.briefings.enabled,
            help="Sends a PSX summary at each of the three times below, every weekday. "
                 "Requires `python alert_worker.py` running.",
        )

        # Pad/truncate to exactly 3 slots so the UI always shows three pickers.
        times = (watchlist.briefings.times + list(stocks.DEFAULT_BRIEFING_TIMES))[:3]
        from datetime import time as _time
        def _parse(t: str) -> _time:
            try:
                hh, mm = (int(x) for x in t.split(":"))
                return _time(hh, mm)
            except Exception:
                return _time(9, 30)

        tcols = st.columns(3)
        new_times: list[str] = []
        for i, (col, label) in enumerate(zip(tcols, ["Open", "Midday", "Close"])):
            t = col.time_input(label, value=_parse(times[i]), key=f"bri-time-{i}", step=300)
            new_times.append(f"{t.hour:02d}:{t.minute:02d}")

        if (bri_enabled != watchlist.briefings.enabled) or (new_times != watchlist.briefings.times):
            watchlist.briefings.enabled = bri_enabled
            watchlist.briefings.times = new_times
            watchlist.save()

        if watchlist.briefings.enabled and watchlist.briefings.last_sent:
            # Show today's sent slots
            today = datetime.now().strftime("%Y-%m-%d")
            sent_today = [k.split("|")[1] for k in watchlist.briefings.last_sent if k.startswith(today)]
            if sent_today:
                st.caption(f"✅ Sent today: {', '.join(sorted(sent_today))}")

        # ----- 4. Alert rules -----
        st.divider()
        st.subheader("🎯 Threshold alerts")
        with st.expander("➕ Add / edit alert rules", expanded=False):
            with st.form("add_rule_form", clear_on_submit=True):
                c1, c2, c3 = st.columns([2, 2, 2])
                new_sym = c1.selectbox(
                    "Symbol",
                    options=PSX_TICKERS,
                    index=0,
                    help="Any PSX ticker. Use the 'custom' field for less common tickers.",
                )
                custom_sym = c1.text_input("...or custom", value="").strip().upper()
                new_rule = c2.selectbox(
                    "Rule",
                    options=["rise_pct", "drop_pct", "price_above", "price_below"],
                )
                new_thr = c3.number_input("Threshold", min_value=0.0, value=3.0, step=0.5)
                submitted = st.form_submit_button("Add rule", use_container_width=True)
                if submitted:
                    sym = (custom_sym or new_sym).upper()
                    rule = stocks.AlertRule(symbol=sym, rule=new_rule, threshold=float(new_thr))
                    watchlist.rules.append(rule)
                    watchlist.save()
                    st.success(f"Added: {rule.describe()}")
                    st.rerun()

            if watchlist.rules:
                st.markdown("**Current rules**")
                for i, r in enumerate(watchlist.rules):
                    cols = st.columns([6, 2, 1])
                    cols[0].markdown(f"`{r.symbol}` — {r.describe()}")
                    new_enabled = cols[1].toggle("On", value=r.enabled, key=f"rule-on-{i}", label_visibility="collapsed")
                    if new_enabled != r.enabled:
                        watchlist.rules[i].enabled = new_enabled
                        watchlist.save()
                    if cols[2].button("🗑", key=f"rule-del-{i}"):
                        watchlist.rules.pop(i)
                        watchlist.save()
                        st.rerun()
            else:
                st.caption("No alert rules yet. Add one above.")

        # ----- 5. LLM settings (moved to bottom — less frequently changed) -----
        st.divider()
        st.subheader("🤖 LLM")
        st.session_state.stock_model_label = st.selectbox(
            "Groq model",
            options=list(STOCK_LLM_MODELS.keys()),
            index=list(STOCK_LLM_MODELS.keys()).index(st.session_state.stock_model_label),
        )
        st.session_state.stock_temperature = st.slider(
            "Temperature", 0.0, 1.0, st.session_state.stock_temperature, 0.1
        )
        st.session_state.stocks_memory_turns = st.slider(
            "Conversation memory (turns)", 0, 8, st.session_state.stocks_memory_turns
        )

        st.divider()
        st.caption(f"📡 Market is **{'OPEN' if stocks.market_is_open() else 'CLOSED'}** (PKT)")
        st.caption(f"🤖 Groq: `{STOCK_LLM_MODELS[st.session_state.stock_model_label]}`")
        st.caption("🛰  Run `python alert_worker.py` separately for live WhatsApp alerts + briefings.")

    # Shared chat controls
    st.divider()
    st.subheader("💬 Chat")
    domain_msg_key = "messages_medical" if st.session_state.domain == "Medical (RAG)" else "messages_stocks"
    c1, c2 = st.columns(2)
    if c1.button("🧹 Clear", use_container_width=True):
        st.session_state[domain_msg_key] = []
        st.session_state.last_sources = []
        st.rerun()
    if st.session_state[domain_msg_key]:
        transcript = "\n\n".join(
            f"**{m['role'].title()}**: {m['content']}" for m in st.session_state[domain_msg_key]
        )
        c2.download_button(
            "⬇️ Export",
            data=f"# Chat — {datetime.now():%Y-%m-%d %H:%M}\n\n{transcript}",
            file_name=f"chat-{datetime.now():%Y%m%d-%H%M%S}.md",
            mime="text/markdown",
            use_container_width=True,
        )


# ============================================================================
# MAIN — Medical mode
# ============================================================================
def render_medical():
    st.markdown('<div class="main-header">🩺 MediBot</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Retrieval-augmented medical reference chatbot — Groq + FAISS</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="disclaimer">⚠️ <strong>Not medical advice.</strong> '
        'MediBot is for educational reference only. Always consult a qualified healthcare professional '
        'for diagnosis, treatment, or medication decisions.</div>',
        unsafe_allow_html=True,
    )

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        st.error("**GROQ_API_KEY is not set.** Get a free key at https://console.groq.com/keys and add it to your `.env` file.")
        st.stop()

    try:
        vectorstore = cached_vectorstore()
    except Exception as e:
        st.error(f"Could not load FAISS index at `vectorstore/db_faiss`. Run `python app.py` first.\n\n{e}")
        st.stop()

    messages = st.session_state.messages_medical

    if not messages:
        st.subheader("Try a starter question")
        cols = st.columns(2)
        for i, q in enumerate(SUGGESTED_QUESTIONS):
            if cols[i % 2].button(q, key=f"med-starter-{i}", use_container_width=True):
                st.session_state._queued_prompt = q
                st.rerun()
    else:
        for msg in messages:
            avatar = "🧑" if msg["role"] == "user" else "🩺"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

    queued = st.session_state.pop("_queued_prompt", None)
    user_input = st.chat_input("Ask a medical reference question…") or queued
    if not user_input:
        return

    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)
    messages.append({"role": "user", "content": user_input})

    system_prompt = STRICT_SYSTEM_PROMPT if st.session_state.strict_mode else ASSISTED_SYSTEM_PROMPT
    model_id = AVAILABLE_MODELS[st.session_state.model_label]
    llm = build_llm(model_id, st.session_state.temperature, groq_api_key, streaming=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": st.session_state.top_k})
    retrieve_step, answer_chain = build_chain(system_prompt, llm, retriever)

    chat_history = history_as_messages(messages[:-1], max_turns=st.session_state.memory_turns)

    with st.chat_message("assistant", avatar="🩺"):
        try:
            with st.spinner("Retrieving relevant context…"):
                retrieved = retrieve_step.invoke({
                    "question": user_input,
                    "chat_history": chat_history,
                })
            st.session_state.last_sources = retrieved["_docs"]

            stream = answer_chain.stream({
                "context": retrieved["context"],
                "question": retrieved["question"],
                "chat_history": retrieved["chat_history"],
            })
            full_response = st.write_stream(stream)
            messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(f"Generation failed: {e}")
            messages.pop()
            st.stop()

    if st.session_state.last_sources:
        with st.expander(f"📖 Sources ({len(st.session_state.last_sources)} chunks)"):
            for i, doc in enumerate(st.session_state.last_sources, start=1):
                src = Path(doc.metadata.get("source", "unknown")).name
                page = doc.metadata.get("page")
                page_str = f", page {page + 1}" if page is not None else ""
                st.markdown(f"**Source {i}** — `{src}`{page_str}")
                st.markdown(
                    f'<div class="source-box">{doc.page_content[:600]}{"…" if len(doc.page_content) > 600 else ""}</div>',
                    unsafe_allow_html=True,
                )


# ============================================================================
# MAIN — Stocks mode
# ============================================================================
def render_stocks():
    st.markdown('<div class="main-header">📈 PSX-Sense</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Live Pakistan Stock Exchange analyst — powered by Groq Llama</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="disclaimer-stocks">ℹ️ <strong>Not investment advice.</strong> '
        'PSX-Sense analyses live data and explains what the numbers show. '
        'It does not predict future prices, recommend trades, or replace a SECP-licensed advisor.</div>',
        unsafe_allow_html=True,
    )

    if not os.environ.get("GROQ_API_KEY"):
        st.error(
            "**GROQ_API_KEY is not set.** Get a free key at https://console.groq.com/keys "
            "and add `GROQ_API_KEY=...` to your `.env` file."
        )
        st.stop()

    # Fetch live data.
    try:
        with st.spinner("Fetching live PSX snapshot…"):
            quotes = cached_market_watch()
    except Exception as e:
        st.error(f"Could not fetch PSX data: {e}")
        st.info(
            "PSX endpoint may be unreachable from your network. "
            "Try again in a minute, or check https://dps.psx.com.pk/market-watch in your browser."
        )
        st.stop()

    # Top-level metrics row.
    movers = stocks.top_movers(5)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Symbols", f"{len(quotes):,}")
    c2.metric("Top gainer",
              movers["gainers"][0].symbol if movers["gainers"] else "—",
              f"{movers['gainers'][0].change_pct:+.2f}%" if movers["gainers"] else "")
    c3.metric("Top loser",
              movers["losers"][0].symbol if movers["losers"] else "—",
              f"{movers['losers'][0].change_pct:+.2f}%" if movers["losers"] else "")
    c4.metric("Market", "OPEN" if stocks.market_is_open() else "CLOSED")

    # Watchlist quick view.
    watchlist = stocks.Watchlist.load()
    if watchlist.symbols():
        with st.expander(f"⭐ Watchlist ({len(watchlist.symbols())} symbols)", expanded=True):
            sym_set = set(watchlist.symbols())
            wl_quotes = [q for q in quotes if q.symbol in sym_set]
            if wl_quotes:
                rows = [{
                    "Symbol":  q.symbol,
                    "Indices": q.indices,
                    "Price":   f"{q.price:,.2f}",
                    "Change":  f"{q.change:+.2f}",
                    "Change%": f"{q.change_pct:+.2f}%",
                    "Volume":  f"{q.volume:,}",
                } for q in wl_quotes]
                st.dataframe(rows, use_container_width=True, hide_index=True)
            missing = sym_set - {q.symbol for q in wl_quotes}
            if missing:
                st.caption(f"Not found in today's snapshot: {', '.join(sorted(missing))}")

    with st.expander("🔥 Top movers", expanded=False):
        tab1, tab2, tab3 = st.tabs(["📈 Gainers", "📉 Losers", "📊 By volume"])
        for tab, key in [(tab1, "gainers"), (tab2, "losers"), (tab3, "by_volume")]:
            with tab:
                rows = [{
                    "Symbol":  q.symbol,
                    "Indices": q.indices,
                    "Price":   f"{q.price:,.2f}",
                    "Change%": f"{q.change_pct:+.2f}%",
                    "Volume":  f"{q.volume:,}",
                } for q in stocks.top_movers(10)[key]]
                st.dataframe(rows, use_container_width=True, hide_index=True)

    st.divider()

    # Chat.
    messages = st.session_state.messages_stocks
    if not messages:
        st.subheader("Try a starter question")
        cols = st.columns(2)
        for i, q in enumerate(STOCK_SUGGESTED_QUESTIONS):
            if cols[i % 2].button(q, key=f"stk-starter-{i}", use_container_width=True):
                st.session_state._queued_prompt = q
                st.rerun()
    else:
        for msg in messages:
            avatar = "🧑" if msg["role"] == "user" else "📈"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

    queued = st.session_state.pop("_queued_prompt", None)
    user_input = st.chat_input("Ask about PSX stocks, sectors, or your watchlist…") or queued
    if not user_input:
        return

    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)
    messages.append({"role": "user", "content": user_input})

    # Inject live data scoped to user's watchlist (so the LLM has the context it needs).
    psx_context = stocks.format_for_llm(quotes, symbols=watchlist.symbols() or None, top_n=10)
    system_prompt = build_system_prompt(STOCK_SYSTEM_PROMPT, psx_context)
    history = history_for_chat(messages[:-1], max_turns=st.session_state.stocks_memory_turns)

    with st.chat_message("assistant", avatar="📈"):
        try:
            stream = stream_chat(
                system_prompt=system_prompt,
                history=history,
                user_message=user_input,
                model=STOCK_LLM_MODELS[st.session_state.stock_model_label],
                temperature=st.session_state.stock_temperature,
            )
            full_response = st.write_stream(stream)
            messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(f"Groq call failed: {e}")
            messages.pop()
            st.stop()


# ============================================================================
# Route
# ============================================================================
if st.session_state.domain == "Medical (RAG)":
    render_medical()
else:
    render_stocks()
