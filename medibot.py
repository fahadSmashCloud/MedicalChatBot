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
    JOB_SYSTEM_PROMPT,
    JOB_SUGGESTED_QUESTIONS,
    ROADMAP_SYSTEM_PROMPT,
    ROADMAP_SUGGESTED_QUESTIONS,
)
from src.stock_chat import (
    STOCK_LLM_MODELS,
    build_system_prompt,
    history_for_chat,
    stream_chat,
)
from src import stocks, jobs, roadmap

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
    .source-box {
        background: #f7f7f9; border-radius: 6px; padding: 8px 12px;
        font-size: 0.85rem; margin-top: 4px;
    }
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
        "top_k": 4,
        "strict_mode": False,    # default to assisted for a friendlier first run
        "memory_turns": 3,
        "last_sources": [],
        # stocks
        "messages_stocks": [],
        "stock_model_label": next(iter(STOCK_LLM_MODELS)),
        "stock_temperature": 0.3,
        "stocks_memory_turns": 3,
        "stock_watchlist": [],   # simple list of PSX tickers
        # jobs
        "messages_jobs": [],
        "job_model_label": next(iter(STOCK_LLM_MODELS)),
        "job_temperature": 0.4,
        "jobs_memory_turns": 3,
        "job_query":       "senior full stack engineer",
        "job_min_salary":  120000,
        "job_remote_only": False,
        "job_sources":     ["RemoteOK", "Remotive", "Arbeitnow"],
        "job_results":     [],
        "job_errors":      {},
        "user_profile":    "",
        # roadmap
        "messages_roadmap": [],
        "roadmap_model_label": next(iter(STOCK_LLM_MODELS)),
        "roadmap_temperature": 0.4,
        "roadmap_memory_turns": 3,
        "roadmap_tracks":  None,   # lazily loaded
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
    return stocks.fetch_market_watch()


# ============================================================================
# SIDEBAR
# ============================================================================
DOMAINS = ["Medical (RAG)", "Stocks (Live PSX)", "Jobs Finder", "Career Roadmap"]

with st.sidebar:
    st.session_state.domain = st.radio(
        "🧭 Domain",
        DOMAINS,
        index=DOMAINS.index(st.session_state.domain) if st.session_state.domain in DOMAINS else 0,
    )
    st.divider()

    if st.session_state.domain == "Medical (RAG)":
        st.header("⚙️ Medical settings")

        st.session_state.model_label = st.selectbox(
            "Model",
            options=list(AVAILABLE_MODELS.keys()),
            index=list(AVAILABLE_MODELS.keys()).index(st.session_state.model_label),
            help="DeepSeek R1 Distill 70B is the strongest open-source option.",
        )
        st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature, 0.1)
        st.session_state.top_k = st.slider("Retrieved chunks (top-k)", 1, 8, st.session_state.top_k)
        st.session_state.memory_turns = st.slider("Conversation memory (turns)", 0, 8, st.session_state.memory_turns)
        st.session_state.strict_mode = st.toggle(
            "Strict context-only mode",
            value=st.session_state.strict_mode,
            help="When on, answers come only from your uploaded reference material. When off (default), the bot may supplement with general knowledge — clearly marked.",
        )

        st.divider()
        st.subheader("📄 Add documents")
        uploads = st.file_uploader(
            "Upload reference PDFs",
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
        st.caption(f"🤖 Model: `{AVAILABLE_MODELS[st.session_state.model_label]}`")

    elif st.session_state.domain == "Stocks (Live PSX)":
        st.header("📈 Stocks settings")

        st.session_state.stock_model_label = st.selectbox(
            "Model",
            options=list(STOCK_LLM_MODELS.keys()),
            index=list(STOCK_LLM_MODELS.keys()).index(st.session_state.stock_model_label),
            help="DeepSeek R1 Distill 70B is the strongest open-source option for analysis.",
        )
        st.session_state.stock_temperature = st.slider(
            "Temperature", 0.0, 1.0, st.session_state.stock_temperature, 0.1
        )
        st.session_state.stocks_memory_turns = st.slider(
            "Conversation memory (turns)", 0, 8, st.session_state.stocks_memory_turns
        )

        st.divider()
        st.subheader("⭐ Your watchlist")
        st.caption("Tickers you add here get prioritised when the LLM answers questions.")

        with st.form("add_watch_form", clear_on_submit=True):
            wc1, wc2 = st.columns([3, 2])
            picked = wc1.selectbox(
                "Pick or type a ticker",
                options=PSX_TICKERS,
                index=0,
                label_visibility="collapsed",
            )
            custom = wc2.text_input("custom", value="", placeholder="or type...",
                                    label_visibility="collapsed").strip().upper()
            if st.form_submit_button("➕ Add to watchlist", use_container_width=True):
                sym = custom or picked
                if sym and sym not in st.session_state.stock_watchlist:
                    st.session_state.stock_watchlist.append(sym)
                    st.rerun()

        if st.session_state.stock_watchlist:
            for i, sym in enumerate(st.session_state.stock_watchlist):
                cols = st.columns([6, 1])
                cols[0].markdown(f"`{sym}`")
                if cols[1].button("🗑", key=f"wl-del-{i}"):
                    st.session_state.stock_watchlist.pop(i)
                    st.rerun()
        else:
            st.caption("No watchlist symbols yet.")

        st.divider()
        st.caption(f"📡 Market is **{'OPEN' if stocks.market_is_open() else 'CLOSED'}** (PKT)")
        st.caption(f"🤖 Model: `{STOCK_LLM_MODELS[st.session_state.stock_model_label]}`")

    elif st.session_state.domain == "Jobs Finder":
        st.header("💼 Jobs Finder")

        st.session_state.job_model_label = st.selectbox(
            "LLM (for fit analysis)",
            options=list(STOCK_LLM_MODELS.keys()),
            index=list(STOCK_LLM_MODELS.keys()).index(st.session_state.job_model_label),
        )
        st.session_state.job_temperature = st.slider(
            "Temperature", 0.0, 1.0, st.session_state.job_temperature, 0.1,
            key="job_temp_slider",
        )

        st.divider()
        st.subheader("🔍 Search filters")
        st.session_state.job_query = st.text_input(
            "Keywords / role",
            value=st.session_state.job_query,
            placeholder="e.g. senior full stack, data engineer",
        )
        st.session_state.job_min_salary = st.number_input(
            "Min salary (USD/yr)",
            min_value=0, max_value=500000, step=10000,
            value=st.session_state.job_min_salary,
            help="Filters out jobs below this. Unknown salaries are kept if the title is senior/lead/staff.",
        )
        st.session_state.job_remote_only = st.toggle(
            "Remote only", value=st.session_state.job_remote_only,
        )

        all_sources = ["RemoteOK", "Remotive", "Arbeitnow", "Adzuna", "JSearch"]
        st.session_state.job_sources = st.multiselect(
            "Sources",
            options=all_sources,
            default=st.session_state.job_sources,
            help="Adzuna requires ADZUNA_APP_ID + ADZUNA_APP_KEY in .env. "
                 "JSearch (covers LinkedIn/Indeed/Glassdoor) requires RAPIDAPI_KEY.",
        )

        if st.button("🔎 Search jobs", use_container_width=True, type="primary"):
            with st.spinner("Querying job boards…"):
                results, errs = jobs.fetch_jobs(
                    query=st.session_state.job_query,
                    min_salary_usd=st.session_state.job_min_salary,
                    remote_only=st.session_state.job_remote_only,
                    sources=st.session_state.job_sources,
                )
                st.session_state.job_results = results
                st.session_state.job_errors = errs

        st.divider()
        st.subheader("👤 Your profile")
        st.caption("Pasted profile / resume bullets feed the LLM fit-analysis.")
        st.session_state.user_profile = st.text_area(
            "Profile",
            value=st.session_state.user_profile,
            height=180,
            placeholder="Sr. Full Stack Engineer, 8 yrs.\n"
                        "Stack: Python, FastAPI, React, Postgres, AWS.\n"
                        "Interested in: data eng, AI/LLM, Snowflake, remote.\n"
                        "Salary target: $140k+ USD remote.",
            label_visibility="collapsed",
        )

        st.divider()
        st.caption(f"🤖 Model: `{STOCK_LLM_MODELS[st.session_state.job_model_label]}`")

    else:  # Career Roadmap
        st.header("🎯 Career Roadmap")

        st.session_state.roadmap_model_label = st.selectbox(
            "LLM (for coaching)",
            options=list(STOCK_LLM_MODELS.keys()),
            index=list(STOCK_LLM_MODELS.keys()).index(st.session_state.roadmap_model_label),
        )
        st.session_state.roadmap_temperature = st.slider(
            "Temperature", 0.0, 1.0, st.session_state.roadmap_temperature, 0.1,
            key="roadmap_temp_slider",
        )

        if st.session_state.roadmap_tracks is None:
            st.session_state.roadmap_tracks = roadmap.load_roadmap()

        tracks = st.session_state.roadmap_tracks
        done, total = roadmap.overall_progress(tracks)
        st.divider()
        st.metric("Overall progress", f"{done} / {total}",
                  f"{(done / total * 100) if total else 0:.0f}%")
        st.progress(done / total if total else 0)

        if st.button("💾 Save progress", use_container_width=True):
            roadmap.save_roadmap(tracks)
            st.success("Saved to data/roadmap.json")

        if st.button("↺ Reset to defaults", use_container_width=True):
            if roadmap.ROADMAP_PATH.exists():
                roadmap.ROADMAP_PATH.unlink()
            st.session_state.roadmap_tracks = roadmap.load_roadmap()
            st.rerun()

        st.caption(f"🤖 Model: `{STOCK_LLM_MODELS[st.session_state.roadmap_model_label]}`")

    # Shared chat controls
    st.divider()
    st.subheader("💬 Chat")
    _msg_key_map = {
        "Medical (RAG)":     "messages_medical",
        "Stocks (Live PSX)": "messages_stocks",
        "Jobs Finder":       "messages_jobs",
        "Career Roadmap":    "messages_roadmap",
    }
    domain_msg_key = _msg_key_map[st.session_state.domain]
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
# MAIN — Medical / RAG mode
# ============================================================================
def render_medical():
    st.markdown('<div class="main-header">🩺 MediBot — your private RAG library</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Upload books or PDFs, ask anything — retrieval-augmented chat with citations.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="disclaimer">⚠️ <strong>Not professional advice.</strong> '
        'For medical, legal, or financial decisions, always consult a qualified professional.</div>',
        unsafe_allow_html=True,
    )

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        st.error("**GROQ_API_KEY is not set.** Get a free key at https://console.groq.com/keys and add it to your `.env` file.")
        st.stop()

    try:
        vectorstore = cached_vectorstore()
    except Exception as e:
        st.error(f"Could not load FAISS index at `vectorstore/db_faiss`. Run `python app.py` first, or upload PDFs in the sidebar.\n\n{e}")
        st.stop()

    messages = st.session_state.messages_medical

    if not messages:
        st.subheader("Try a starter question — or just say hi")
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
    user_input = st.chat_input("Ask a question, or just chat…") or queued
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
            with st.spinner("Thinking…"):
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
    st.markdown('<div class="subtle">Live Pakistan Stock Exchange analyst — chat grounded in real-time market data.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="disclaimer-stocks">ℹ️ <strong>Not investment advice.</strong> '
        'PSX-Sense analyses live data and explains what the numbers show. '
        'It does not predict future prices or recommend trades.</div>',
        unsafe_allow_html=True,
    )

    if not os.environ.get("GROQ_API_KEY"):
        st.error(
            "**GROQ_API_KEY is not set.** Get a free key at https://console.groq.com/keys "
            "and add `GROQ_API_KEY=...` to your `.env` file."
        )
        st.stop()

    try:
        with st.spinner("Fetching live PSX snapshot…"):
            quotes = cached_market_watch()
    except Exception as e:
        st.error(f"Could not fetch PSX data: {e}")
        st.info("PSX endpoint may be unreachable. Try again in a minute, or check https://dps.psx.com.pk/market-watch in your browser.")
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

    watchlist_symbols = st.session_state.stock_watchlist
    if watchlist_symbols:
        with st.expander(f"⭐ Your watchlist ({len(watchlist_symbols)} symbols)", expanded=True):
            sym_set = {s.upper() for s in watchlist_symbols}
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

    messages = st.session_state.messages_stocks
    if not messages:
        st.subheader("Try a starter question — or just say hi")
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
    user_input = st.chat_input("Ask about PSX stocks, sectors, or your watchlist — or just chat…") or queued
    if not user_input:
        return

    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)
    messages.append({"role": "user", "content": user_input})

    psx_context = stocks.format_for_llm(quotes, symbols=watchlist_symbols or None, top_n=10)
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
# MAIN — Jobs Finder
# ============================================================================
def render_jobs():
    st.markdown('<div class="main-header">💼 Jobs Finder</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtle">Premium global roles aggregated from RemoteOK, Remotive, '
        'Arbeitnow, Adzuna, and JSearch (LinkedIn/Indeed/Glassdoor proxy).</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="disclaimer-stocks">ℹ️ <strong>Why no direct LinkedIn scraping?</strong> '
        'LinkedIn aggressively bans scraper IPs. Enable JSearch (RapidAPI free tier) in '
        'the sidebar to legally pull LinkedIn / Indeed / Glassdoor listings.</div>',
        unsafe_allow_html=True,
    )

    if not os.environ.get("GROQ_API_KEY"):
        st.error("**GROQ_API_KEY is not set.** Add it to your `.env` file to enable chat.")
        st.stop()

    results = st.session_state.job_results
    errors = st.session_state.job_errors

    if errors:
        with st.expander(f"⚠️ {len(errors)} source(s) failed", expanded=False):
            for src, msg in errors.items():
                st.warning(f"**{src}** — {msg}")

    if not results:
        st.info("👉 Set your filters in the sidebar and click **Search jobs** to load postings.")
    else:
        st.subheader(f"📋 {len(results)} matches — sorted by salary midpoint")
        rows = [{
            "Title":    j.title,
            "Company":  j.company,
            "Location": j.location + (" 🌍" if j.remote else ""),
            "Salary":   j.salary_str,
            "Source":   j.source,
            "Posted":   j.posted or "—",
            "URL":      j.url,
        } for j in results[:50]]
        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
            column_config={"URL": st.column_config.LinkColumn("Apply", display_text="Open ↗")},
        )

    st.divider()
    messages = st.session_state.messages_jobs

    if not messages:
        st.subheader("Ask JobScout — or just say hi")
        cols = st.columns(2)
        for i, q in enumerate(JOB_SUGGESTED_QUESTIONS):
            if cols[i % 2].button(q, key=f"job-starter-{i}", use_container_width=True):
                st.session_state._queued_prompt = q
                st.rerun()
    else:
        for msg in messages:
            avatar = "🧑" if msg["role"] == "user" else "💼"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

    queued = st.session_state.pop("_queued_prompt", None)
    user_input = st.chat_input("Ask about the postings, fit, salary, or skill gaps…") or queued
    if not user_input:
        return

    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)
    messages.append({"role": "user", "content": user_input})

    jobs_ctx = jobs.format_jobs_for_llm(results, limit=15)
    profile = st.session_state.user_profile.strip() or "(not provided)"
    system_prompt = JOB_SYSTEM_PROMPT.format(
        jobs_data=jobs_ctx,
        user_profile=profile,
        today=datetime.now().strftime("%Y-%m-%d"),
    )
    history = history_for_chat(messages[:-1], max_turns=st.session_state.jobs_memory_turns)

    with st.chat_message("assistant", avatar="💼"):
        try:
            stream = stream_chat(
                system_prompt=system_prompt,
                history=history,
                user_message=user_input,
                model=STOCK_LLM_MODELS[st.session_state.job_model_label],
                temperature=st.session_state.job_temperature,
            )
            full_response = st.write_stream(stream)
            messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(f"Groq call failed: {e}")
            messages.pop()
            st.stop()


# ============================================================================
# MAIN — Career Roadmap
# ============================================================================
def render_roadmap():
    st.markdown('<div class="main-header">🎯 Top-1% Engineer Roadmap</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtle">Personal skill-tracks across data engineering, '
        'data analysis, Python, Oracle, Odoo, system design, cloud, AI/ML, DevOps, and craft.</div>',
        unsafe_allow_html=True,
    )

    if not os.environ.get("GROQ_API_KEY"):
        st.error("**GROQ_API_KEY is not set.** Add it to your `.env` file to enable chat.")
        st.stop()

    if st.session_state.roadmap_tracks is None:
        st.session_state.roadmap_tracks = roadmap.load_roadmap()
    tracks = st.session_state.roadmap_tracks

    # Per-track progress strip
    cols = st.columns(min(len(tracks), 5))
    for i, t in enumerate(tracks[:5]):
        cols[i].metric(t.name, f"{t.done_count}/{t.total}", f"{t.progress * 100:.0f}%")
    if len(tracks) > 5:
        cols2 = st.columns(min(len(tracks) - 5, 5))
        for i, t in enumerate(tracks[5:10]):
            cols2[i].metric(t.name, f"{t.done_count}/{t.total}", f"{t.progress * 100:.0f}%")

    st.divider()

    # Tabs per track
    tab_objs = st.tabs([f"{t.name} ({t.done_count}/{t.total})" for t in tracks])
    for tab, t in zip(tab_objs, tracks):
        with tab:
            st.caption(t.description)
            st.progress(t.progress)
            for m in t.milestones:
                col_check, col_notes = st.columns([5, 4])
                new_done = col_check.checkbox(
                    f"**[{m.level}]** {m.title}",
                    value=m.done,
                    key=f"ms-{m.id}",
                )
                if new_done != m.done:
                    m.done = new_done
                    roadmap.save_roadmap(tracks)

                new_notes = col_notes.text_input(
                    "notes",
                    value=m.notes,
                    key=f"notes-{m.id}",
                    placeholder="optional notes / resources",
                    label_visibility="collapsed",
                )
                if new_notes != m.notes:
                    m.notes = new_notes
                    roadmap.save_roadmap(tracks)

    st.divider()
    messages = st.session_state.messages_roadmap

    if not messages:
        st.subheader("Ask RoadmapCoach — or just say hi")
        cols = st.columns(2)
        for i, q in enumerate(ROADMAP_SUGGESTED_QUESTIONS):
            if cols[i % 2].button(q, key=f"rm-starter-{i}", use_container_width=True):
                st.session_state._queued_prompt = q
                st.rerun()
    else:
        for msg in messages:
            avatar = "🧑" if msg["role"] == "user" else "🎯"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

    queued = st.session_state.pop("_queued_prompt", None)
    user_input = st.chat_input("Ask what to learn next, sequence your studies, or just chat…") or queued
    if not user_input:
        return

    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)
    messages.append({"role": "user", "content": user_input})

    roadmap_ctx = roadmap.format_for_llm(tracks)
    system_prompt = ROADMAP_SYSTEM_PROMPT.format(
        roadmap_data=roadmap_ctx,
        today=datetime.now().strftime("%Y-%m-%d"),
    )
    history = history_for_chat(messages[:-1], max_turns=st.session_state.roadmap_memory_turns)

    with st.chat_message("assistant", avatar="🎯"):
        try:
            stream = stream_chat(
                system_prompt=system_prompt,
                history=history,
                user_message=user_input,
                model=STOCK_LLM_MODELS[st.session_state.roadmap_model_label],
                temperature=st.session_state.roadmap_temperature,
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
_ROUTES = {
    "Medical (RAG)":     render_medical,
    "Stocks (Live PSX)": render_stocks,
    "Jobs Finder":       render_jobs,
    "Career Roadmap":    render_roadmap,
}
_ROUTES.get(st.session_state.domain, render_medical)()
