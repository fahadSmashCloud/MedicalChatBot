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
    RESUME_ANALYSIS_SYSTEM_PROMPT,
    RESUME_CHAT_SYSTEM_PROMPT,
    RESUME_SUGGESTED_QUESTIONS,
    CODE_SYSTEM_PROMPT,
    CODE_SUGGESTED_QUESTIONS,
    INTERVIEW_QUESTION_SYSTEM_PROMPT,
    INTERVIEW_EVAL_SYSTEM_PROMPT,
    INTERVIEW_CHAT_SYSTEM_PROMPT,
    INTERVIEW_SUGGESTED_QUESTIONS,
    AGENT_SUGGESTED_TASKS,
)
from src.stock_chat import (
    STOCK_LLM_MODELS,
    build_system_prompt,
    call_groq_json,
    history_for_chat,
    stream_chat,
)
from src import stocks, jobs, roadmap
from src import resume_analyzer, code_assistant, interview_guide
from src import auth as auth_module
from src import agent_core, agent_tools
from src import quant_agent

load_dotenv(find_dotenv())

st.set_page_config(
    page_title="AI Workbench — MediBot · PSX · Jobs · Roadmap · Resume · Code · Interview",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# CSS — production-grade, dark-mode compatible
# ============================================================================
st.markdown(
    """
    <style>
    /* ---- typography & layout ---- */
    .main-header {
        font-size: 2rem; font-weight: 800; letter-spacing: -0.5px;
        margin-bottom: 0.15rem;
        background: linear-gradient(90deg, #4f8ef7, #a855f7);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .subtle { color: #888; font-size: 0.88rem; margin-bottom: 0.5rem; }

    /* ---- status / disclaimer banners ---- */
    .disclaimer {
        background: #fff4e5; border-left: 4px solid #ff9800;
        padding: 0.55rem 0.9rem; border-radius: 6px;
        font-size: 0.83rem; color: #5d3a00; margin-bottom: 1rem;
    }
    .disclaimer-stocks {
        background: #eaf4ff; border-left: 4px solid #1976d2;
        padding: 0.55rem 0.9rem; border-radius: 6px;
        font-size: 0.83rem; color: #0b3a66; margin-bottom: 1rem;
    }
    .disclaimer-green {
        background: #f0fdf4; border-left: 4px solid #16a34a;
        padding: 0.55rem 0.9rem; border-radius: 6px;
        font-size: 0.83rem; color: #14532d; margin-bottom: 1rem;
    }
    .disclaimer-purple {
        background: #faf5ff; border-left: 4px solid #a855f7;
        padding: 0.55rem 0.9rem; border-radius: 6px;
        font-size: 0.83rem; color: #4c1d95; margin-bottom: 1rem;
    }

    /* ---- source / code boxes ---- */
    .source-box {
        background: #f7f7f9; border-radius: 6px; padding: 8px 12px;
        font-size: 0.83rem; margin-top: 4px; border: 1px solid #e8e8ee;
    }
    .score-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px; padding: 1.2rem; color: white;
        text-align: center; margin-bottom: 1rem;
    }
    .score-card h1 { color: white !important; font-size: 3rem; margin: 0; }
    .score-card p { color: rgba(255,255,255,0.85); margin: 0; }

    /* ---- badge pills for milestone levels ---- */
    .badge-core     { background:#dbeafe; color:#1d4ed8; border-radius:4px;
                      padding:1px 6px; font-size:0.75rem; font-weight:600; }
    .badge-advanced { background:#fef9c3; color:#92400e; border-radius:4px;
                      padding:1px 6px; font-size:0.75rem; font-weight:600; }
    .badge-expert   { background:#fce7f3; color:#9d174d; border-radius:4px;
                      padding:1px 6px; font-size:0.75rem; font-weight:600; }

    /* ---- language preset pill buttons ---- */
    .lang-pill {
        display:inline-block; padding:4px 14px; margin:3px;
        border-radius:20px; font-size:0.85rem; font-weight:600;
        cursor:pointer; border:2px solid transparent;
        transition: all 0.15s ease;
    }
    .lang-pill:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }

    /* ---- market sentiment ---- */
    .sentiment-bullish { color: #16a34a; font-weight: 700; font-size: 1.2rem; }
    .sentiment-bearish { color: #dc2626; font-weight: 700; font-size: 1.2rem; }
    .sentiment-neutral { color: #d97706; font-weight: 700; font-size: 1.2rem; }

    /* ---- interview guider ---- */
    .interview-q-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #0f2340 100%);
        border-radius: 12px; padding: 1.4rem 1.6rem; color: white;
        margin-bottom: 1rem; border-left: 5px solid #4f8ef7;
    }
    .interview-q-card h3 { color: white !important; margin: 0 0 0.5rem 0; font-size: 1rem; }
    .interview-q-card p  { color: rgba(255,255,255,0.9); font-size: 1.05rem; margin: 0; line-height: 1.6; }
    .interview-q-meta   { color: rgba(255,255,255,0.6); font-size: 0.8rem; margin-top: 0.6rem; }

    .score-green  { background: linear-gradient(135deg, #16a34a, #15803d); }
    .score-yellow { background: linear-gradient(135deg, #d97706, #b45309); }
    .score-red    { background: linear-gradient(135deg, #dc2626, #b91c1c); }

    .concept-chip {
        display: inline-block; background: rgba(79,142,247,0.15);
        border: 1px solid rgba(79,142,247,0.4); border-radius: 20px;
        padding: 2px 10px; font-size: 0.78rem; color: #4f8ef7;
        margin: 2px;
    }
    .disclaimer-blue {
        background: #eff6ff; border-left: 4px solid #3b82f6;
        padding: 0.55rem 0.9rem; border-radius: 6px;
        font-size: 0.83rem; color: #1e3a5f; margin-bottom: 1rem;
    }

    /* ---- auth pages ---- */
    .auth-card {
        max-width: 420px; margin: 4rem auto; padding: 2.5rem 2rem;
        background: white; border-radius: 16px;
        box-shadow: 0 8px 40px rgba(0,0,0,0.10);
        border: 1px solid #e8e8ee;
    }
    .auth-logo {
        text-align: center; font-size: 2.8rem; margin-bottom: 0.3rem;
    }
    .auth-title {
        text-align: center; font-size: 1.4rem; font-weight: 700;
        margin-bottom: 0.1rem; color: #1a1a2e;
    }
    .auth-subtitle {
        text-align: center; font-size: 0.85rem; color: #888;
        margin-bottom: 1.6rem;
    }
    .role-badge-superadmin {
        display: inline-block; background: linear-gradient(90deg,#4f8ef7,#a855f7);
        color: white; border-radius: 20px; padding: 2px 12px;
        font-size: 0.75rem; font-weight: 700; letter-spacing: 0.5px;
    }
    .role-badge-user {
        display: inline-block; background: #f0fdf4; color: #16a34a;
        border: 1px solid #bbf7d0; border-radius: 20px; padding: 2px 12px;
        font-size: 0.75rem; font-weight: 600;
    }

    /* ---- misc ---- */
    div[data-testid="stMetricValue"] { font-size: 1.4rem !important; }

    /* ---- Agentic AI step cards ---- */
    .agent-step {
        border-radius: 8px; padding: 0.7rem 1rem;
        margin-bottom: 0.55rem; font-size: 0.88rem; line-height: 1.5;
        border-left: 4px solid transparent;
    }
    .agent-thought {
        background: #1e2a3a; border-left-color: #4f8ef7;
    }
    .agent-action {
        background: #1a2a1a; border-left-color: #4caf50;
    }
    .agent-observation {
        background: #2a2a1a; border-left-color: #ff9800;
        white-space: pre-wrap; font-family: monospace; font-size: 0.82rem;
    }
    .agent-final {
        background: #2a1a3a; border-left-color: #a855f7;
        font-weight: 500;
    }
    .agent-error {
        background: #3a1a1a; border-left-color: #ef4444;
    }
    .agent-step-label {
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.8px;
        text-transform: uppercase; opacity: 0.7; margin-bottom: 0.25rem;
    }
    .agent-tool-badge {
        display: inline-block; background: rgba(79,142,247,0.15);
        border: 1px solid rgba(79,142,247,0.35); border-radius: 12px;
        padding: 1px 9px; font-size: 0.72rem; font-weight: 600;
        color: #4f8ef7; margin-left: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# Session state
# ============================================================================
def init_state():
    defaults = {
        "domain": "Medical (RAG)",
        # medical
        "messages_medical": [],
        "model_label": next(iter(AVAILABLE_MODELS)),
        "temperature": 0.4,
        "top_k": 4,
        "strict_mode": False,
        "memory_turns": 3,
        "last_sources": [],
        # stocks
        "messages_stocks": [],
        "stock_model_label": next(iter(STOCK_LLM_MODELS)),
        "stock_temperature": 0.3,
        "stocks_memory_turns": 3,
        "stock_watchlist": [],
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
        "roadmap_tracks":  None,
        # resume
        "messages_resume": [],
        "resume_model_label": next(iter(STOCK_LLM_MODELS)),
        "resume_temperature": 0.3,
        "resume_memory_turns": 3,
        "resume_text": "",           # extracted plain text
        "resume_analysis": None,     # parsed dict
        "resume_analysis_md": "",    # rendered markdown
        "resume_job_desc": "",       # optional JD for gap analysis
        # code assistant
        "messages_code": [],
        "code_model_label": next(iter(STOCK_LLM_MODELS)),
        "code_temperature": 0.4,
        "code_memory_turns": 4,
        "code_language": "Python",   # selected language preset
        # auth
        "auth_user": None,          # dict with id/name/email/role, or None
        "auth_page": "login",       # "login" | "register"
        "auth_change_pw": False,    # show change-password form in sidebar
        # interview guider
        "messages_interview": [],
        "interview_model_label": next(iter(STOCK_LLM_MODELS)),
        "interview_temperature": 0.7,
        "interview_topic": "Data Structures & Algorithms",
        "interview_difficulty_label": "Senior (5–8 yrs)",
        "interview_n_questions": 5,
        "interview_focus": "",
        "interview_questions": [],       # list[InterviewQuestion]
        "interview_evaluations": {},     # {question_id: AnswerEvaluation}
        "interview_current_q": 0,        # index into questions list
        "interview_show_hint": False,
        "interview_session_done": False,
        "interview_pending_answer": "",  # textarea draft
        # agentic AI
        "agent_model_label":   next(iter(STOCK_LLM_MODELS)),
        "agent_max_iter":      8,
        "agent_steps":         [],       # list[AgentStep] from last run
        "agent_task":          "",       # last task string
        "agent_enabled_tools": [t.name for t in agent_tools.TOOLS],
        # quant agent
        "quant_asset":   "BTC",
        "quant_result":  None,
        "quant_error":   "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()
auth_module.init_db()   # create tables + seed superadmin on first run


@st.cache_resource(show_spinner=False)
def cached_vectorstore():
    return load_vectorstore()


@st.cache_data(ttl=60, show_spinner=False)
def cached_market_watch():
    return stocks.fetch_market_watch()


# ============================================================================
# AUTH — Login / Register pages
# ============================================================================
def render_login():
    st.markdown(
        '<div class="auth-logo">🚀</div>'
        '<div class="auth-title">AI Workbench</div>'
        '<div class="auth-subtitle">Sign in to continue</div>',
        unsafe_allow_html=True,
    )
    with st.form("login_form"):
        email    = st.text_input("Email", placeholder="you@example.com")
        password = st.text_input("Password", type="password", placeholder="••••••••")
        submitted = st.form_submit_button("Sign In", use_container_width=True, type="primary")

    if submitted:
        if not email or not password:
            st.error("Please enter your email and password.")
        else:
            ok, msg, user = auth_module.login_user(email, password)
            if ok:
                st.session_state.auth_user = user
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Don't have an account? Register", use_container_width=True):
        st.session_state.auth_page = "register"
        st.rerun()


def render_register():
    st.markdown(
        '<div class="auth-logo">🚀</div>'
        '<div class="auth-title">Create Account</div>'
        '<div class="auth-subtitle">Join AI Workbench — free forever</div>',
        unsafe_allow_html=True,
    )
    with st.form("register_form"):
        name     = st.text_input("Full name", placeholder="Your Name")
        email    = st.text_input("Email", placeholder="you@example.com")
        password = st.text_input("Password", type="password",
                                  placeholder="Min 8 chars, upper + lower + digit")
        confirm  = st.text_input("Confirm password", type="password", placeholder="••••••••")
        submitted = st.form_submit_button("Create Account", use_container_width=True, type="primary")

    if submitted:
        if password != confirm:
            st.error("Passwords do not match.")
        else:
            ok, msg = auth_module.register_user(name, email, password)
            if ok:
                st.success(msg)
                st.session_state.auth_page = "login"
                st.rerun()
            else:
                st.error(msg)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Already have an account? Sign In", use_container_width=True):
        st.session_state.auth_page = "login"
        st.rerun()


def render_auth_gate():
    """Show login or register centered on the page — blocks all app content."""
    # Narrow centered column trick
    _, col, _ = st.columns([1, 1.6, 1])
    with col:
        if st.session_state.auth_page == "register":
            render_register()
        else:
            render_login()


# ============================================================================
# ADMIN PANEL (superadmin only)
# ============================================================================
def render_admin_panel():
    st.markdown('<div class="main-header">🛡️ Super Admin Panel</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Manage all users, roles, and accounts.</div>', unsafe_allow_html=True)

    users = auth_module.list_users()
    total = len(users)
    admins = sum(1 for u in users if u["role"] == "superadmin")

    m1, m2, m3 = st.columns(3)
    m1.metric("Total users", total)
    m2.metric("Super admins", admins)
    m3.metric("Regular users", total - admins)

    st.divider()
    st.subheader("👥 All users")

    for u in users:
        is_me = u["id"] == st.session_state.auth_user["id"]
        with st.expander(
            f"{'👑' if u['role'] == 'superadmin' else '👤'} {u['name']} — {u['email']}"
            + (" *(you)*" if is_me else ""),
            expanded=False,
        ):
            c1, c2, c3 = st.columns([2, 2, 1])
            c1.markdown(f"**Role:** `{u['role']}`")
            c2.markdown(f"**Joined:** {u['created_at'][:10]}")
            c3.markdown(f"**Last login:** {(u['last_login'] or '—')[:10]}")

            st.markdown("---")
            col_role, col_pw, col_del = st.columns(3)

            # Change role
            with col_role:
                new_role = st.selectbox(
                    "Role", auth_module.ROLES,
                    index=auth_module.ROLES.index(u["role"]),
                    key=f"role_{u['id']}",
                )
                if st.button("Update role", key=f"upd_role_{u['id']}", use_container_width=True):
                    ok, msg = auth_module.set_user_role(u["id"], new_role)
                    (st.success if ok else st.error)(msg)
                    st.rerun()

            # Reset password
            with col_pw:
                new_pw = st.text_input("New password", type="password",
                                        key=f"rpw_{u['id']}", placeholder="Force-reset")
                if st.button("Reset password", key=f"do_rpw_{u['id']}", use_container_width=True):
                    if new_pw:
                        ok, msg = auth_module.admin_reset_password(u["id"], new_pw)
                        (st.success if ok else st.error)(msg)
                    else:
                        st.warning("Enter a new password first.")

            # Delete
            with col_del:
                st.markdown("<br>", unsafe_allow_html=True)
                if not is_me:
                    if st.button("🗑 Delete", key=f"del_{u['id']}", use_container_width=True):
                        ok, msg = auth_module.delete_user(u["id"])
                        (st.success if ok else st.error)(msg)
                        st.rerun()
                else:
                    st.caption("*(cannot delete yourself)*")


# ============================================================================
# SIDEBAR
# ============================================================================
DOMAINS = [
    "Medical (RAG)",
    "Stocks (Live PSX)",
    "Jobs Finder",
    "Career Roadmap",
    "Resume Analyzer",
    "Code Assistant",
    "Interview Guider",
    "🤖 Agentic AI",
    "📈 Quant Agent",
]
ADMIN_DOMAINS = DOMAINS + ["🛡️ Admin Panel"]

# Regular (non-superadmin) users are restricted to these three domains.
USER_DOMAINS = [
    "Medical (RAG)",
    "Career Roadmap",
    "📈 Quant Agent",
]

# ── Auth gate — block everything if not signed in ───────────────────────────
if not st.session_state.auth_user:
    render_auth_gate()
    st.stop()

# ── Signed in from here onward ───────────────────────────────────────────────
_current_user = st.session_state.auth_user
_is_superadmin = _current_user["role"] == "superadmin"

with st.sidebar:
    # -- User info strip ------------------------------------------------------
    role_badge = (
        '<span class="role-badge-superadmin">👑 Super Admin</span>'
        if _is_superadmin else
        '<span class="role-badge-user">👤 User</span>'
    )
    st.markdown(
        f"**{_current_user['name']}**&nbsp; {role_badge}",
        unsafe_allow_html=True,
    )
    st.caption(_current_user["email"])

    col_logout, col_pw = st.columns(2)
    if col_logout.button("🚪 Sign out", use_container_width=True):
        st.session_state.auth_user = None
        st.session_state.auth_page = "login"
        st.rerun()
    if col_pw.button("🔑 Password", use_container_width=True):
        st.session_state.auth_change_pw = not st.session_state.get("auth_change_pw", False)
        st.rerun()

    if st.session_state.get("auth_change_pw"):
        with st.form("change_pw_form"):
            old_pw  = st.text_input("Current password", type="password")
            new_pw  = st.text_input("New password", type="password")
            conf_pw = st.text_input("Confirm new password", type="password")
            if st.form_submit_button("Change password", use_container_width=True):
                if new_pw != conf_pw:
                    st.error("Passwords don't match.")
                else:
                    ok, msg = auth_module.change_password(_current_user["id"], old_pw, new_pw)
                    (st.success if ok else st.error)(msg)
                    if ok:
                        st.session_state.auth_change_pw = False
                        st.rerun()

    st.divider()

    # -- Health status strip --------------------------------------------------
    api_ok = bool(os.environ.get("GROQ_API_KEY"))
    idx_ok = Path("vectorstore/db_faiss").exists()
    st.caption(
        f"{'🟢' if api_ok else '🔴'} Groq API &nbsp;|&nbsp; "
        f"{'🟢' if idx_ok else '🟡'} FAISS index &nbsp;|&nbsp; "
        f"{'🟢' if os.environ.get('RAPIDAPI_KEY') else '⚪'} JSearch &nbsp;|&nbsp; "
        f"{'🟢' if os.environ.get('ADZUNA_APP_ID') else '⚪'} Adzuna"
    )
    st.divider()

    _domain_list = ADMIN_DOMAINS if _is_superadmin else USER_DOMAINS
    if st.session_state.domain not in _domain_list:
        st.session_state.domain = "Medical (RAG)"
    st.session_state.domain = st.radio(
        "🧭 Domain",
        _domain_list,
        index=_domain_list.index(st.session_state.domain),
    )
    st.divider()

    # -------------------------------------------------------------------------
    if st.session_state.domain == "Medical (RAG)":
        st.header("⚙️ Medical settings")
        st.session_state.model_label = st.selectbox(
            "Model",
            options=list(AVAILABLE_MODELS.keys()),
            index=list(AVAILABLE_MODELS.keys()).index(st.session_state.model_label),
        )
        st.session_state.temperature  = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature, 0.1)
        st.session_state.top_k        = st.slider("Retrieved chunks (top-k)", 1, 8, st.session_state.top_k)
        st.session_state.memory_turns = st.slider("Conversation memory (turns)", 0, 8, st.session_state.memory_turns)
        st.session_state.strict_mode  = st.toggle("Strict context-only mode", value=st.session_state.strict_mode,
            help="On = answers from your PDFs only. Off = may supplement with general knowledge.")
        st.divider()
        st.subheader("📄 Add documents")
        uploads = st.file_uploader("Upload reference PDFs", type=["pdf"], accept_multiple_files=True)
        if uploads and st.button("Ingest uploaded PDFs", use_container_width=True):
            try:
                with st.spinner(f"Embedding {len(uploads)} file(s)…"):
                    added = ingest_uploaded_pdfs(uploads)
                    cached_vectorstore.clear()
                st.success(f"Added {added} chunks to the index.")
            except PDFIngestError as e:
                st.error(f"Ingest failed: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
        st.divider()
        try:
            n = cached_vectorstore().index.ntotal
            st.caption(f"📚 {n:,} indexed chunks")
        except Exception:
            st.caption("📚 Index not loaded")
        st.caption(f"🤖 `{AVAILABLE_MODELS[st.session_state.model_label]}`")

    # -------------------------------------------------------------------------
    elif st.session_state.domain == "Stocks (Live PSX)":
        st.header("📈 Stocks settings")
        st.session_state.stock_model_label   = st.selectbox("Model", list(STOCK_LLM_MODELS.keys()),
            index=list(STOCK_LLM_MODELS.keys()).index(st.session_state.stock_model_label))
        st.session_state.stock_temperature   = st.slider("Temperature", 0.0, 1.0, st.session_state.stock_temperature, 0.1)
        st.session_state.stocks_memory_turns = st.slider("Conversation memory (turns)", 0, 8, st.session_state.stocks_memory_turns)
        st.divider()
        st.subheader("⭐ Your watchlist")
        st.caption("Tickers added here get prioritised in LLM context.")
        with st.form("add_watch_form", clear_on_submit=True):
            wc1, wc2 = st.columns([3, 2])
            picked = wc1.selectbox("Pick or type a ticker", options=PSX_TICKERS, index=0, label_visibility="collapsed")
            custom = wc2.text_input("custom", value="", placeholder="or type...", label_visibility="collapsed").strip().upper()
            if st.form_submit_button("➕ Add to watchlist", use_container_width=True):
                sym = custom or picked
                if sym and sym not in st.session_state.stock_watchlist:
                    st.session_state.stock_watchlist.append(sym)
                    st.rerun()
        if st.session_state.stock_watchlist:
            for i, sym in enumerate(st.session_state.stock_watchlist):
                c1, c2 = st.columns([6, 1])
                c1.markdown(f"`{sym}`")
                if c2.button("🗑", key=f"wl-del-{i}"):
                    st.session_state.stock_watchlist.pop(i)
                    st.rerun()
        else:
            st.caption("No watchlist symbols yet.")
        st.divider()
        st.caption(f"📡 Market: **{'OPEN' if stocks.market_is_open() else 'CLOSED'}** (PKT)")
        st.caption(f"🤖 `{STOCK_LLM_MODELS[st.session_state.stock_model_label]}`")

    # -------------------------------------------------------------------------
    elif st.session_state.domain == "Jobs Finder":
        st.header("💼 Jobs Finder")
        st.session_state.job_model_label = st.selectbox("LLM (for fit analysis)",
            list(STOCK_LLM_MODELS.keys()),
            index=list(STOCK_LLM_MODELS.keys()).index(st.session_state.job_model_label))
        st.session_state.job_temperature = st.slider("Temperature", 0.0, 1.0,
            st.session_state.job_temperature, 0.1, key="job_temp_slider")
        st.divider()
        st.subheader("🔍 Search filters")
        st.session_state.job_query       = st.text_input("Keywords / role", value=st.session_state.job_query,
            placeholder="e.g. senior full stack, data engineer")
        st.session_state.job_min_salary  = st.number_input("Min salary (USD/yr)", min_value=0,
            max_value=500000, step=10000, value=st.session_state.job_min_salary)
        st.session_state.job_remote_only = st.toggle("Remote only", value=st.session_state.job_remote_only)
        st.session_state.job_sources     = st.multiselect("Sources",
            options=["RemoteOK", "Remotive", "Arbeitnow", "Adzuna", "JSearch"],
            default=st.session_state.job_sources,
            help="Adzuna needs ADZUNA_APP_ID + ADZUNA_APP_KEY. JSearch needs RAPIDAPI_KEY.")
        if st.button("🔎 Search jobs", use_container_width=True, type="primary"):
            with st.spinner("Querying job boards…"):
                results, errs = jobs.fetch_jobs(
                    query=st.session_state.job_query,
                    min_salary_usd=st.session_state.job_min_salary,
                    remote_only=st.session_state.job_remote_only,
                    sources=st.session_state.job_sources,
                )
                st.session_state.job_results = results
                st.session_state.job_errors  = errs
        st.divider()
        st.subheader("👤 Your profile")
        st.caption("Paste resume bullets — used for LLM fit analysis.")
        st.session_state.user_profile = st.text_area("Profile",
            value=st.session_state.user_profile, height=180,
            placeholder="Sr. Full Stack Engineer, 8 yrs.\nStack: Python, FastAPI, React, Postgres, AWS.\n"
                        "Interested in: data eng, AI/LLM, Snowflake, remote.\nSalary target: $140k+ USD.",
            label_visibility="collapsed")
        st.divider()
        st.caption(f"🤖 `{STOCK_LLM_MODELS[st.session_state.job_model_label]}`")

    # -------------------------------------------------------------------------
    elif st.session_state.domain == "Career Roadmap":
        st.header("🎯 Career Roadmap")
        st.session_state.roadmap_model_label = st.selectbox("LLM (for coaching)",
            list(STOCK_LLM_MODELS.keys()),
            index=list(STOCK_LLM_MODELS.keys()).index(st.session_state.roadmap_model_label))
        st.session_state.roadmap_temperature = st.slider("Temperature", 0.0, 1.0,
            st.session_state.roadmap_temperature, 0.1, key="roadmap_temp_slider")
        if st.session_state.roadmap_tracks is None:
            st.session_state.roadmap_tracks = roadmap.load_roadmap()
        tracks = st.session_state.roadmap_tracks
        done, total = roadmap.overall_progress(tracks)
        st.divider()
        st.metric("Overall progress", f"{done} / {total}", f"{(done / total * 100) if total else 0:.0f}%")
        st.progress(done / total if total else 0)
        if st.button("💾 Save progress", use_container_width=True):
            roadmap.save_roadmap(tracks)
            st.success("Saved to data/roadmap.json")
        if st.button("↺ Reset to defaults", use_container_width=True):
            if roadmap.ROADMAP_PATH.exists():
                roadmap.ROADMAP_PATH.unlink()
            st.session_state.roadmap_tracks = roadmap.load_roadmap()
            st.rerun()
        st.caption(f"🤖 `{STOCK_LLM_MODELS[st.session_state.roadmap_model_label]}`")

    # -------------------------------------------------------------------------
    elif st.session_state.domain == "Resume Analyzer":
        st.header("📄 Resume Analyzer")
        st.session_state.resume_model_label = st.selectbox("LLM (for analysis)",
            list(STOCK_LLM_MODELS.keys()),
            index=list(STOCK_LLM_MODELS.keys()).index(st.session_state.resume_model_label))
        st.session_state.resume_temperature = st.slider("Chat temperature", 0.0, 1.0,
            st.session_state.resume_temperature, 0.1, key="resume_temp_slider")
        st.divider()
        st.subheader("📎 Upload resume")
        resume_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
        st.session_state.resume_job_desc = st.text_area(
            "Job description (optional — enables keyword gap analysis)",
            value=st.session_state.resume_job_desc, height=120,
            placeholder="Paste a job posting here to get missing-keyword analysis…",
            label_visibility="visible")
        if resume_file and st.button("🔍 Analyse resume", use_container_width=True, type="primary"):
            with st.spinner("Extracting text…"):
                try:
                    text = resume_analyzer.extract_resume_text(resume_file)
                    st.session_state.resume_text = text
                except ValueError as e:
                    st.error(str(e))
                    st.stop()
            with st.spinner("Running AI analysis (10–20 s)…"):
                try:
                    raw = call_groq_json(
                        system_prompt=RESUME_ANALYSIS_SYSTEM_PROMPT,
                        user_prompt=resume_analyzer.build_analysis_prompt(
                            text, st.session_state.resume_job_desc
                        ),
                        model=STOCK_LLM_MODELS[st.session_state.resume_model_label],
                    )
                    parsed = resume_analyzer.parse_analysis(raw)
                    st.session_state.resume_analysis    = parsed
                    st.session_state.resume_analysis_md = resume_analyzer.format_analysis_as_markdown(parsed)
                    st.session_state.messages_resume    = []   # fresh chat after new upload
                    st.success("Analysis complete! See the main panel.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
        if st.session_state.resume_text and st.button("🗑 Clear resume", use_container_width=True):
            for k in ("resume_text", "resume_analysis", "resume_analysis_md", "messages_resume"):
                st.session_state[k] = "" if isinstance(st.session_state[k], str) else ([] if isinstance(st.session_state[k], list) else None)
            st.rerun()
        st.divider()
        st.caption(f"🤖 `{STOCK_LLM_MODELS[st.session_state.resume_model_label]}`")

    # -------------------------------------------------------------------------
    elif st.session_state.domain == "Interview Guider":
        st.header("🎤 Interview Guider")
        st.session_state.interview_model_label = st.selectbox(
            "LLM (question & eval)",
            list(STOCK_LLM_MODELS.keys()),
            index=list(STOCK_LLM_MODELS.keys()).index(st.session_state.interview_model_label),
            key="interview_model_select",
        )
        st.session_state.interview_temperature = st.slider(
            "Creativity", 0.3, 1.0, st.session_state.interview_temperature, 0.1,
            help="Higher = more varied question phrasing",
            key="interview_temp_slider",
        )
        st.divider()
        st.subheader("📋 Session setup")
        topic_options = list(interview_guide.INTERVIEW_TOPICS.keys())
        st.session_state.interview_topic = st.selectbox(
            "Topic",
            topic_options,
            index=topic_options.index(st.session_state.interview_topic)
                  if st.session_state.interview_topic in topic_options else 0,
            format_func=lambda k: f"{interview_guide.INTERVIEW_TOPICS[k]['icon']} {k}",
            key="interview_topic_select",
        )
        diff_options = list(interview_guide.DIFFICULTY_LEVELS.keys())
        if st.session_state.interview_difficulty_label not in diff_options:
            st.session_state.interview_difficulty_label = diff_options[2]
        st.session_state.interview_difficulty_label = st.selectbox(
            "Difficulty",
            diff_options,
            index=diff_options.index(st.session_state.interview_difficulty_label),
            key="interview_diff_select",
        )
        st.session_state.interview_n_questions = st.select_slider(
            "# Questions", options=[3, 5, 8, 10],
            value=st.session_state.interview_n_questions,
            key="interview_nq_slider",
        )
        st.session_state.interview_focus = st.text_input(
            "Focus area (optional)",
            value=st.session_state.interview_focus,
            placeholder="e.g. DP on trees, async Python, RAG chunking",
            key="interview_focus_input",
        )
        st.divider()
        if st.button("🚀 Generate Interview", use_container_width=True, type="primary", key="interview_gen_btn"):
            topic = st.session_state.interview_topic
            topic_data = interview_guide.INTERVIEW_TOPICS[topic]
            difficulty = interview_guide.DIFFICULTY_LEVELS[st.session_state.interview_difficulty_label]
            with st.spinner(f"Generating {st.session_state.interview_n_questions} {difficulty} questions…"):
                try:
                    raw = call_groq_json(
                        system_prompt=INTERVIEW_QUESTION_SYSTEM_PROMPT,
                        user_prompt=interview_guide.build_question_prompt(
                            topic=topic,
                            subtopics=topic_data["subtopics"],
                            difficulty=difficulty,
                            n_questions=st.session_state.interview_n_questions,
                            focus_areas=st.session_state.interview_focus,
                        ),
                        model=STOCK_LLM_MODELS[st.session_state.interview_model_label],
                    )
                    questions = interview_guide.parse_questions(raw, difficulty=difficulty)
                    st.session_state.interview_questions   = questions
                    st.session_state.interview_evaluations = {}
                    st.session_state.interview_current_q  = 0
                    st.session_state.interview_show_hint  = False
                    st.session_state.interview_session_done = False
                    st.session_state.messages_interview    = []
                    st.rerun()
                except Exception as e:
                    st.error(f"Question generation failed: {e}")

        questions = st.session_state.interview_questions
        evaluations = st.session_state.interview_evaluations
        if questions:
            done_count = len(evaluations)
            st.divider()
            st.metric("Progress", f"{done_count} / {len(questions)}", "answered")
            st.progress(done_count / len(questions))
            if st.button("🗑 Reset session", use_container_width=True, key="interview_reset_btn"):
                for k in ("interview_questions", "interview_evaluations", "messages_interview"):
                    st.session_state[k] = [] if isinstance(st.session_state[k], list) else {}
                st.session_state.interview_current_q    = 0
                st.session_state.interview_session_done = False
                st.rerun()
        st.divider()
        st.caption(f"🤖 `{STOCK_LLM_MODELS[st.session_state.interview_model_label]}`")

    elif st.session_state.domain == "🤖 Agentic AI":
        st.header("🤖 Agentic AI")
        st.session_state.agent_model_label = st.selectbox(
            "LLM",
            list(STOCK_LLM_MODELS.keys()),
            index=list(STOCK_LLM_MODELS.keys()).index(st.session_state.agent_model_label),
            key="agent_model_sel",
        )
        st.session_state.agent_max_iter = st.slider(
            "Max iterations", 1, 15,
            st.session_state.agent_max_iter, key="agent_max_iter_slider",
            help="Hard cap on how many Reason→Act→Observe loops the agent runs.",
        )
        st.divider()
        st.subheader("🔧 Enable tools")
        all_tool_names = [t.name for t in agent_tools.TOOLS]
        selected_tools = []
        for tname in all_tool_names:
            spec = agent_tools.TOOLS_BY_NAME[tname]
            checked = tname in st.session_state.agent_enabled_tools
            if st.checkbox(tname, value=checked, key=f"tool_chk_{tname}",
                           help=spec.description[:120]):
                selected_tools.append(tname)
        st.session_state.agent_enabled_tools = selected_tools
        st.divider()
        if st.button("🗑 Clear results", use_container_width=True, key="agent_clear_btn"):
            st.session_state.agent_steps = []
            st.session_state.agent_task  = ""
            st.rerun()
        st.caption(f"🤖 `{STOCK_LLM_MODELS[st.session_state.agent_model_label]}`")

    elif st.session_state.domain == "📈 Quant Agent":
        st.header("📈 Quant Agent")
        asset_options = list(quant_agent.ASSET_UNIVERSE.keys())
        if st.session_state.quant_asset not in asset_options:
            st.session_state.quant_asset = asset_options[0]
        st.session_state.quant_asset = st.selectbox(
            "Asset",
            asset_options,
            index=asset_options.index(st.session_state.quant_asset),
            format_func=lambda a: f"{a}  ·  {quant_agent.ASSET_UNIVERSE[a]['kind']}",
            key="quant_asset_sel",
        )
        st.divider()
        if st.button("🚀 Run Analysis", use_container_width=True, type="primary",
                     key="quant_run_btn"):
            st.session_state.quant_result = None
            st.session_state.quant_error  = ""
            try:
                with st.spinner(f"Running 4-agent pipeline for {st.session_state.quant_asset}…"):
                    st.session_state.quant_result = quant_agent.QuantAgentModule().run(
                        st.session_state.quant_asset
                    )
            except Exception as exc:
                st.session_state.quant_error = str(exc)
            st.rerun()
        if st.button("🗑 Clear results", use_container_width=True, key="quant_clear_btn"):
            st.session_state.quant_result = None
            st.session_state.quant_error  = ""
            st.rerun()
        st.divider()
        st.caption(
            "Pipeline: Market → TA → News → Signal. "
            "Sources: Yahoo Finance / CoinGecko / DuckDuckGo."
        )

    else:  # Code Assistant
        st.header("💻 Code Assistant")
        st.session_state.code_model_label = st.selectbox("LLM",
            list(STOCK_LLM_MODELS.keys()),
            index=list(STOCK_LLM_MODELS.keys()).index(st.session_state.code_model_label))
        st.session_state.code_temperature = st.slider("Temperature", 0.0, 1.0,
            st.session_state.code_temperature, 0.1, key="code_temp_slider")
        st.session_state.code_memory_turns = st.slider("Memory (turns)", 0, 8,
            st.session_state.code_memory_turns, key="code_mem_slider")
        st.divider()
        st.subheader("🌐 Language / stack")
        lang_options = list(code_assistant.LANGUAGE_PRESETS.keys())
        st.session_state.code_language = st.selectbox(
            "Focus area",
            options=lang_options,
            index=lang_options.index(st.session_state.code_language),
            format_func=lambda k: f"{code_assistant.LANGUAGE_PRESETS[k]['icon']} {k}",
        )
        selected = code_assistant.LANGUAGE_PRESETS[st.session_state.code_language]
        st.caption(f"Focus: {selected['focus'][:120]}…")
        st.divider()
        st.subheader("⚡ Task shortcuts")
        st.caption("Tap to pre-fill the chat input.")
        for label, template in code_assistant.TASK_SHORTCUTS:
            if st.button(label, use_container_width=True, key=f"shortcut-{label}"):
                st.session_state._queued_prompt = template
                st.rerun()
        st.divider()
        st.caption(f"🤖 `{STOCK_LLM_MODELS[st.session_state.code_model_label]}`")

    # ---- Shared chat controls -----------------------------------------------
    st.divider()
    st.subheader("💬 Chat")
    _msg_key_map = {
        "Medical (RAG)":     "messages_medical",
        "Stocks (Live PSX)": "messages_stocks",
        "Jobs Finder":       "messages_jobs",
        "Career Roadmap":    "messages_roadmap",
        "Resume Analyzer":   "messages_resume",
        "Code Assistant":    "messages_code",
        "Interview Guider":  "messages_interview",
        "🤖 Agentic AI":    None,   # agent has its own step-based UI, no chat buffer
        "📈 Quant Agent":   None,   # quant has its own dashboard UI, no chat buffer
        "🛡️ Admin Panel":   None,
    }
    domain_msg_key = _msg_key_map.get(st.session_state.domain)
    c1, c2 = st.columns(2)
    if domain_msg_key and c1.button("🧹 Clear", use_container_width=True):
        st.session_state[domain_msg_key] = []
        st.session_state.last_sources = []
        st.rerun()
    if domain_msg_key and st.session_state[domain_msg_key]:
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
# MAIN — Medical / RAG
# ============================================================================
def render_medical():
    st.markdown('<div class="main-header">🩺 MediBot — private RAG library</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Upload books or PDFs, ask anything — retrieval-augmented chat with citations.</div>', unsafe_allow_html=True)
    st.markdown('<div class="disclaimer">⚠️ <strong>Not professional advice.</strong> For medical, legal, or financial decisions, always consult a qualified professional.</div>', unsafe_allow_html=True)

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        st.error("**GROQ_API_KEY is not set.** Get a free key at https://console.groq.com/keys and add it to your `.env` file.")
        st.stop()

    try:
        vectorstore = cached_vectorstore()
    except Exception as e:
        st.error(f"Could not load FAISS index. Run `python app.py` first, or upload PDFs in the sidebar.\n\n{e}")
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
            with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🩺"):
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
                retrieved = retrieve_step.invoke({"question": user_input, "chat_history": chat_history})
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
                src  = Path(doc.metadata.get("source", "unknown")).name
                page = doc.metadata.get("page")
                page_str = f", page {page + 1}" if page is not None else ""
                st.markdown(f"**Source {i}** — `{src}`{page_str}")
                st.markdown(
                    f'<div class="source-box">{doc.page_content[:600]}{"…" if len(doc.page_content) > 600 else ""}</div>',
                    unsafe_allow_html=True,
                )


# ============================================================================
# MAIN — Stocks
# ============================================================================
def render_stocks():
    st.markdown('<div class="main-header">📈 PSX-Sense</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Live Pakistan Stock Exchange analyst — chat grounded in real-time market data.</div>', unsafe_allow_html=True)
    st.markdown('<div class="disclaimer-stocks">ℹ️ <strong>Not investment advice.</strong> PSX-Sense analyses live data and explains what the numbers show. It does not predict future prices or recommend trades.</div>', unsafe_allow_html=True)

    if not os.environ.get("GROQ_API_KEY"):
        st.error("**GROQ_API_KEY is not set.**")
        st.stop()

    try:
        with st.spinner("Fetching live PSX snapshot…"):
            quotes = cached_market_watch()
    except Exception as e:
        st.error(f"Could not fetch PSX data: {e}")
        st.info("PSX endpoint may be unreachable. Try again in a minute.")
        st.stop()

    movers = stocks.top_movers(5)

    # --- Metrics row ---------------------------------------------------------
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Symbols", f"{len(quotes):,}")
    c2.metric("Top gainer",
              movers["gainers"][0].symbol if movers["gainers"] else "—",
              f"{movers['gainers'][0].change_pct:+.2f}%" if movers["gainers"] else "")
    c3.metric("Top loser",
              movers["losers"][0].symbol if movers["losers"] else "—",
              f"{movers['losers'][0].change_pct:+.2f}%" if movers["losers"] else "")
    c4.metric("Market", "OPEN 🟢" if stocks.market_is_open() else "CLOSED 🔴")

    # --- Market Sentiment widget (AI-powered) --------------------------------
    gainers_n = len(movers["gainers"])
    losers_n  = len(movers["losers"])
    ratio = gainers_n / max(gainers_n + losers_n, 1)
    if ratio >= 0.6:
        sentiment_label, sentiment_css = "Bullish 🐂", "sentiment-bullish"
    elif ratio <= 0.4:
        sentiment_label, sentiment_css = "Bearish 🐻", "sentiment-bearish"
    else:
        sentiment_label, sentiment_css = "Mixed / Neutral ⚖️", "sentiment-neutral"
    c5.metric("Sentiment", sentiment_label)

    # --- Watchlist -----------------------------------------------------------
    watchlist_symbols = st.session_state.stock_watchlist
    if watchlist_symbols:
        with st.expander(f"⭐ Your watchlist ({len(watchlist_symbols)} symbols)", expanded=True):
            sym_set  = {s.upper() for s in watchlist_symbols}
            wl_quotes = [q for q in quotes if q.symbol in sym_set]
            if wl_quotes:
                rows = [{"Symbol": q.symbol, "Indices": q.indices,
                         "Price": f"{q.price:,.2f}", "Change": f"{q.change:+.2f}",
                         "Change%": f"{q.change_pct:+.2f}%", "Volume": f"{q.volume:,}"}
                        for q in wl_quotes]
                st.dataframe(rows, use_container_width=True, hide_index=True)
            missing = sym_set - {q.symbol for q in wl_quotes}
            if missing:
                st.caption(f"Not in today's snapshot: {', '.join(sorted(missing))}")

    # --- Top movers ----------------------------------------------------------
    with st.expander("🔥 Top movers", expanded=False):
        tab1, tab2, tab3 = st.tabs(["📈 Gainers", "📉 Losers", "📊 By volume"])
        for tab, key in [(tab1, "gainers"), (tab2, "losers"), (tab3, "by_volume")]:
            with tab:
                rows = [{"Symbol": q.symbol, "Indices": q.indices,
                         "Price": f"{q.price:,.2f}", "Change%": f"{q.change_pct:+.2f}%",
                         "Volume": f"{q.volume:,}"}
                        for q in stocks.top_movers(10)[key]]
                st.dataframe(rows, use_container_width=True, hide_index=True)

    st.divider()

    # --- Chat ----------------------------------------------------------------
    messages = st.session_state.messages_stocks
    if not messages:
        st.subheader("Ask PSX-Sense — or just say hi")
        cols = st.columns(2)
        for i, q in enumerate(STOCK_SUGGESTED_QUESTIONS):
            if cols[i % 2].button(q, key=f"stk-starter-{i}", use_container_width=True):
                st.session_state._queued_prompt = q
                st.rerun()
    else:
        for msg in messages:
            with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "📈"):
                st.markdown(msg["content"])

    queued = st.session_state.pop("_queued_prompt", None)
    user_input = st.chat_input("Ask about PSX stocks, sectors, or your watchlist…") or queued
    if not user_input:
        return

    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)
    messages.append({"role": "user", "content": user_input})

    psx_context   = stocks.format_for_llm(quotes, symbols=watchlist_symbols or None, top_n=10)
    system_prompt = build_system_prompt(STOCK_SYSTEM_PROMPT, psx_context)
    history       = history_for_chat(messages[:-1], max_turns=st.session_state.stocks_memory_turns)

    with st.chat_message("assistant", avatar="📈"):
        try:
            full_response = st.write_stream(stream_chat(
                system_prompt=system_prompt, history=history, user_message=user_input,
                model=STOCK_LLM_MODELS[st.session_state.stock_model_label],
                temperature=st.session_state.stock_temperature,
            ))
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
    st.markdown('<div class="subtle">Premium global roles aggregated from RemoteOK, Remotive, Arbeitnow, Adzuna, and JSearch (LinkedIn/Indeed/Glassdoor proxy).</div>', unsafe_allow_html=True)
    st.markdown('<div class="disclaimer-stocks">ℹ️ <strong>Why no direct LinkedIn scraping?</strong> LinkedIn aggressively bans scraper IPs. Enable JSearch (RapidAPI free tier) in the sidebar to legally pull LinkedIn / Indeed / Glassdoor listings.</div>', unsafe_allow_html=True)

    if not os.environ.get("GROQ_API_KEY"):
        st.error("**GROQ_API_KEY is not set.**")
        st.stop()

    results = st.session_state.job_results
    errors  = st.session_state.job_errors

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
            rows, use_container_width=True, hide_index=True,
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
            with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "💼"):
                st.markdown(msg["content"])

    queued = st.session_state.pop("_queued_prompt", None)
    user_input = st.chat_input("Ask about the postings, fit, salary, or skill gaps…") or queued
    if not user_input:
        return

    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)
    messages.append({"role": "user", "content": user_input})

    jobs_ctx      = jobs.format_jobs_for_llm(results, limit=15)
    profile       = st.session_state.user_profile.strip() or "(not provided)"
    system_prompt = JOB_SYSTEM_PROMPT.format(jobs_data=jobs_ctx, user_profile=profile,
                                              today=datetime.now().strftime("%Y-%m-%d"))
    history       = history_for_chat(messages[:-1], max_turns=st.session_state.jobs_memory_turns)

    with st.chat_message("assistant", avatar="💼"):
        try:
            full_response = st.write_stream(stream_chat(
                system_prompt=system_prompt, history=history, user_message=user_input,
                model=STOCK_LLM_MODELS[st.session_state.job_model_label],
                temperature=st.session_state.job_temperature,
            ))
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
    st.markdown('<div class="subtle">Personal skill-tracks — data engineering, Python, Oracle, Odoo, cloud, AI/ML, DevOps, and more.</div>', unsafe_allow_html=True)

    if not os.environ.get("GROQ_API_KEY"):
        st.error("**GROQ_API_KEY is not set.**")
        st.stop()

    if st.session_state.roadmap_tracks is None:
        st.session_state.roadmap_tracks = roadmap.load_roadmap()
    tracks = st.session_state.roadmap_tracks

    cols = st.columns(min(len(tracks), 5))
    for i, t in enumerate(tracks[:5]):
        cols[i].metric(t.name, f"{t.done_count}/{t.total}", f"{t.progress * 100:.0f}%")
    if len(tracks) > 5:
        cols2 = st.columns(min(len(tracks) - 5, 5))
        for i, t in enumerate(tracks[5:10]):
            cols2[i].metric(t.name, f"{t.done_count}/{t.total}", f"{t.progress * 100:.0f}%")

    st.divider()
    tab_objs = st.tabs([f"{t.name} ({t.done_count}/{t.total})" for t in tracks])
    for tab, t in zip(tab_objs, tracks):
        with tab:
            st.caption(t.description)
            st.progress(t.progress)
            for m in t.milestones:
                col_check, col_notes = st.columns([5, 4])
                new_done = col_check.checkbox(f"**[{m.level}]** {m.title}", value=m.done, key=f"ms-{m.id}")
                if new_done != m.done:
                    m.done = new_done
                    roadmap.save_roadmap(tracks)
                new_notes = col_notes.text_input("notes", value=m.notes, key=f"notes-{m.id}",
                    placeholder="optional notes / resources", label_visibility="collapsed")
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
            with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🎯"):
                st.markdown(msg["content"])

    queued = st.session_state.pop("_queued_prompt", None)
    user_input = st.chat_input("Ask what to learn next, sequence your studies, or just chat…") or queued
    if not user_input:
        return

    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)
    messages.append({"role": "user", "content": user_input})

    roadmap_ctx   = roadmap.format_for_llm(tracks)
    system_prompt = ROADMAP_SYSTEM_PROMPT.format(roadmap_data=roadmap_ctx,
                                                  today=datetime.now().strftime("%Y-%m-%d"))
    history       = history_for_chat(messages[:-1], max_turns=st.session_state.roadmap_memory_turns)

    with st.chat_message("assistant", avatar="🎯"):
        try:
            full_response = st.write_stream(stream_chat(
                system_prompt=system_prompt, history=history, user_message=user_input,
                model=STOCK_LLM_MODELS[st.session_state.roadmap_model_label],
                temperature=st.session_state.roadmap_temperature,
            ))
            messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(f"Groq call failed: {e}")
            messages.pop()
            st.stop()


# ============================================================================
# MAIN — Resume Analyzer
# ============================================================================
def render_resume():
    st.markdown('<div class="main-header">📄 Resume Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Upload your PDF resume → get an ATS score, skill gap analysis, STAR bullet rewrites, and a coaching chat.</div>', unsafe_allow_html=True)
    st.markdown('<div class="disclaimer-purple">🤖 Powered by Groq LLMs. Analysis is AI-generated — treat it as a second opinion, not ground truth. Upload resume and optionally paste a job description in the sidebar.</div>', unsafe_allow_html=True)

    if not os.environ.get("GROQ_API_KEY"):
        st.error("**GROQ_API_KEY is not set.**")
        st.stop()

    # -- Analysis result panel ------------------------------------------------
    if st.session_state.resume_analysis_md:
        a = st.session_state.resume_analysis or {}
        ats = a.get("ats_score", 0)
        overall = a.get("overall_score", 0)

        col_score, col_detail = st.columns([1, 3])
        with col_score:
            color = "#16a34a" if ats >= 75 else "#d97706" if ats >= 50 else "#dc2626"
            st.markdown(
                f'<div class="score-card">'
                f'<p>ATS Score</p><h1>{ats}</h1><p style="font-size:0.8rem">/ 100</p>'
                f'<hr style="border-color:rgba(255,255,255,0.3)">'
                f'<p>Overall</p><h1 style="font-size:2rem">{overall}</h1><p style="font-size:0.8rem">/ 10</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
            skills = a.get("skills_detected", [])
            if skills:
                st.markdown("**Skills detected:**")
                st.markdown("  ".join(f"`{s}`" for s in skills[:20]))

        with col_detail:
            tabs = st.tabs(["📊 Full Analysis", "✅ Strengths", "⚠️ Weaknesses", "💡 Suggestions", "🚀 Rewrites"])
            with tabs[0]:
                st.markdown(st.session_state.resume_analysis_md)
            with tabs[1]:
                for s in a.get("strengths", []):
                    st.success(s)
            with tabs[2]:
                for w in a.get("weaknesses", []):
                    st.warning(w)
            with tabs[3]:
                for sg in a.get("improvement_suggestions", []):
                    st.markdown(f"**{sg.get('section', '')}** — {sg.get('suggestion', '')}")
            with tabs[4]:
                for b in a.get("impact_bullets", []):
                    st.info(b)

        st.divider()

        # -- Quick actions ----------------------------------------------------
        st.subheader("⚡ Quick actions")
        qa1, qa2, qa3 = st.columns(3)
        if qa1.button("✉️ Generate cover letter", use_container_width=True):
            st.session_state._queued_prompt = resume_analyzer.cover_letter_prompt(
                st.session_state.resume_text, "Senior Software Engineer", "your target company"
            )
            st.rerun()
        if qa2.button("🎤 Interview prep questions", use_container_width=True):
            st.session_state._queued_prompt = resume_analyzer.interview_prep_prompt(
                st.session_state.resume_text, "Senior Software Engineer"
            )
            st.rerun()
        if qa3.button("📝 Rewrite weak bullets", use_container_width=True):
            st.session_state._queued_prompt = "Rewrite my 5 weakest bullet points in strong STAR format with quantified outcomes."
            st.rerun()

        st.divider()
    else:
        st.info("👈 Upload your resume PDF in the sidebar and click **Analyse resume** to get started.")
        st.markdown("""
**What you'll get:**
- 🎯 ATS compatibility score (0–100)
- ⭐ Overall quality score (0–10)
- ✅ Key strengths & ⚠️ weaknesses
- 🛠️ All detected skills + 🚨 missing keywords (if you paste a JD)
- 💡 Section-by-section improvement suggestions
- 🚀 Up to 5 weak bullets rewritten in STAR format
- 💬 Follow-up chat to draft cover letters, prep for interviews, tailor for specific roles
        """)
        return

    # -- Chat -----------------------------------------------------------------
    messages = st.session_state.messages_resume
    if not messages:
        st.subheader("Ask ResumeAI — or just say hi")
        cols = st.columns(2)
        for i, q in enumerate(RESUME_SUGGESTED_QUESTIONS):
            if cols[i % 2].button(q, key=f"res-starter-{i}", use_container_width=True):
                st.session_state._queued_prompt = q
                st.rerun()
    else:
        for msg in messages:
            with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "📄"):
                st.markdown(msg["content"])

    queued = st.session_state.pop("_queued_prompt", None)
    user_input = st.chat_input("Ask for bullet rewrites, a cover letter, interview prep…") or queued
    if not user_input:
        return

    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)
    messages.append({"role": "user", "content": user_input})

    resume_text = st.session_state.resume_text[:4000]
    analysis_summary = st.session_state.resume_analysis_md[:1500] if st.session_state.resume_analysis_md else ""
    system_prompt = RESUME_CHAT_SYSTEM_PROMPT.format(
        resume_text=resume_text,
        analysis_summary=analysis_summary,
    )
    history = history_for_chat(messages[:-1], max_turns=st.session_state.resume_memory_turns)

    with st.chat_message("assistant", avatar="📄"):
        try:
            full_response = st.write_stream(stream_chat(
                system_prompt=system_prompt, history=history, user_message=user_input,
                model=STOCK_LLM_MODELS[st.session_state.resume_model_label],
                temperature=st.session_state.resume_temperature,
            ))
            messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(f"Groq call failed: {e}")
            messages.pop()
            st.stop()


# ============================================================================
# MAIN — Code Assistant
# ============================================================================
def render_code():
    preset = code_assistant.LANGUAGE_PRESETS[st.session_state.code_language]

    st.markdown(
        f'<div class="main-header">{preset["icon"]} Code Assistant — {st.session_state.code_language}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="subtle">CodeSensei — senior engineer and code reviewer. '
        f'Stack focus: {preset["focus"][:120]}…</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="disclaimer-green">💡 <strong>How to use:</strong> Pick a language preset in the sidebar, '
        'use the task shortcut buttons to pre-fill templates, or just start typing. '
        'CodeSensei always uses typed code blocks, explains root causes, and gives opinionated recommendations.</div>',
        unsafe_allow_html=True,
    )

    if not os.environ.get("GROQ_API_KEY"):
        st.error("**GROQ_API_KEY is not set.**")
        st.stop()

    # -- Language preset pills (visual only) ----------------------------------
    pills_html = " ".join(
        f'<span class="lang-pill" style="background:{"#eff6ff" if k == st.session_state.code_language else "#f5f5f5"}; '
        f'border-color:{"#3b82f6" if k == st.session_state.code_language else "#e5e5e5"}; '
        f'color:{"#1d4ed8" if k == st.session_state.code_language else "#555"};">'
        f'{v["icon"]} {k}</span>'
        for k, v in code_assistant.LANGUAGE_PRESETS.items()
    )
    st.markdown(pills_html, unsafe_allow_html=True)
    st.caption("Change the focus in the sidebar → Language / stack selector.")

    st.divider()

    # -- Task shortcut quick-action row ---------------------------------------
    shortcut_cols = st.columns(4)
    for i, (label, template) in enumerate(code_assistant.TASK_SHORTCUTS[:4]):
        if shortcut_cols[i % 4].button(label, use_container_width=True, key=f"qs-{i}"):
            st.session_state._queued_prompt = template
            st.rerun()

    st.divider()

    # -- Chat -----------------------------------------------------------------
    messages = st.session_state.messages_code
    if not messages:
        st.subheader("Ask CodeSensei — or tap a shortcut above")
        cols = st.columns(2)
        for i, q in enumerate(CODE_SUGGESTED_QUESTIONS):
            if cols[i % 2].button(q, key=f"code-starter-{i}", use_container_width=True):
                st.session_state._queued_prompt = q
                st.rerun()
    else:
        for msg in messages:
            with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else preset["icon"]):
                st.markdown(msg["content"])

    queued = st.session_state.pop("_queued_prompt", None)
    user_input = st.chat_input(f"Ask {st.session_state.code_language} questions, paste code to review, or use a shortcut…") or queued
    if not user_input:
        return

    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)
    messages.append({"role": "user", "content": user_input})

    system_prompt = CODE_SYSTEM_PROMPT.format(focus=preset["focus"])
    history       = history_for_chat(messages[:-1], max_turns=st.session_state.code_memory_turns)

    with st.chat_message("assistant", avatar=preset["icon"]):
        try:
            full_response = st.write_stream(stream_chat(
                system_prompt=system_prompt, history=history, user_message=user_input,
                model=STOCK_LLM_MODELS[st.session_state.code_model_label],
                temperature=st.session_state.code_temperature,
            ))
            messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(f"Groq call failed: {e}")
            messages.pop()
            st.stop()


# ============================================================================
# MAIN — Interview Guider
# ============================================================================
def render_interview():
    topic      = st.session_state.interview_topic
    topic_data = interview_guide.INTERVIEW_TOPICS.get(topic, {})
    icon       = topic_data.get("icon", "🎤")

    st.markdown(
        f'<div class="main-header">{icon} Interview Guider — {topic}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="subtle">{topic_data.get("description", "")}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="disclaimer-blue">🤖 <strong>How it works:</strong> '
        'The LLM generates hard, nuanced questions. You type your answer. '
        'A <strong>PyTorch sentence-transformer</strong> scores semantic similarity '
        'against the ideal answer, then the LLM grades depth, completeness, and communication. '
        'Use the coaching chat at the bottom to go deeper on any topic.</div>',
        unsafe_allow_html=True,
    )

    if not os.environ.get("GROQ_API_KEY"):
        st.error("**GROQ_API_KEY is not set.**")
        st.stop()

    questions   = st.session_state.interview_questions
    evaluations = st.session_state.interview_evaluations   # {q.id: AnswerEvaluation}

    # ── No questions yet: feature overview ──────────────────────────────────
    if not questions:
        st.info("👈 Configure your session in the sidebar and click **Generate Interview** to start.")
        st.markdown("""
**What makes this different:**
- 🧮 **8 topic areas** — DSA, System Design, Python, SQL, ML/AI, Behavioral, API Design, DevOps
- 🎯 **4 difficulty levels** — Junior → Staff/Principal
- 🤖 **Dual scoring**: LLM grades depth & logic; PyTorch sentence-transformers grade semantic similarity — two independent signals
- 💡 **Hint system** — reveals expected concepts without spoiling the ideal answer
- 📖 **Ideal answer reveal** — see exactly what a top candidate would say
- 🔍 **Follow-up probes** — know what the interviewer asks next to expose gaps
- 📊 **Session summary** — your study priorities ranked by frequency of gaps
- 💬 **Coaching chat** — ask InterviewCoach to go deeper on any missed concept
        """)

        st.divider()
        st.subheader("📚 Available Topics")
        topic_cols = st.columns(4)
        for i, (t_name, t_data) in enumerate(interview_guide.INTERVIEW_TOPICS.items()):
            with topic_cols[i % 4]:
                st.markdown(
                    f"**{t_data['icon']} {t_name}**  \n"
                    f"<span style='font-size:0.8rem;color:#888'>{t_data['description']}</span>",
                    unsafe_allow_html=True,
                )
        return

    # ── Session complete: summary + chat ────────────────────────────────────
    if st.session_state.interview_session_done or len(evaluations) == len(questions):
        st.session_state.interview_session_done = True
        summary_md = interview_guide.session_summary_md(questions, evaluations)
        st.markdown(summary_md)
        st.divider()

        col_restart, col_harder = st.columns(2)
        if col_restart.button("🔄 New session (same topic)", use_container_width=True):
            st.session_state.interview_questions    = []
            st.session_state.interview_evaluations  = {}
            st.session_state.interview_current_q   = 0
            st.session_state.interview_session_done = False
            st.rerun()
        if col_harder.button("⬆️ Try harder difficulty", use_container_width=True):
            diff_keys = list(interview_guide.DIFFICULTY_LEVELS.keys())
            cur_idx   = diff_keys.index(st.session_state.interview_difficulty_label) if st.session_state.interview_difficulty_label in diff_keys else 2
            st.session_state.interview_difficulty_label = diff_keys[min(cur_idx + 1, len(diff_keys) - 1)]
            st.session_state.interview_questions    = []
            st.session_state.interview_evaluations  = {}
            st.session_state.interview_current_q   = 0
            st.session_state.interview_session_done = False
            st.rerun()

    else:
        # ── Active question ─────────────────────────────────────────────────
        cur_idx = st.session_state.interview_current_q
        if cur_idx >= len(questions):
            cur_idx = len(questions) - 1
            st.session_state.interview_current_q = cur_idx

        q = questions[cur_idx]
        already_evaluated = q.id in evaluations

        # Progress bar + navigation row
        prog_col, nav_left, nav_right = st.columns([6, 1, 1])
        prog_col.progress((cur_idx) / len(questions), text=f"Question {cur_idx + 1} of {len(questions)}")
        if nav_left.button("◀", disabled=(cur_idx == 0), key="q_prev"):
            st.session_state.interview_current_q = cur_idx - 1
            st.session_state.interview_show_hint = False
            st.rerun()
        if nav_right.button("▶", disabled=(cur_idx == len(questions) - 1), key="q_next"):
            st.session_state.interview_current_q = cur_idx + 1
            st.session_state.interview_show_hint = False
            st.rerun()

        # Question card
        diff_label = st.session_state.interview_difficulty_label
        st.markdown(
            f'<div class="interview-q-card">'
            f'<h3>Q{cur_idx + 1} · {q.category} · <em>{diff_label}</em></h3>'
            f'<p>{q.question}</p>'
            f'<div class="interview-q-meta">Expected concepts: '
            + " ".join(f'<span class="concept-chip">{c}</span>' for c in q.expected_concepts)
            + "</div></div>",
            unsafe_allow_html=True,
        )

        # Follow-up probes (collapsed)
        if q.follow_ups:
            with st.expander("🔍 Potential follow-up probes from the interviewer", expanded=False):
                for fp in q.follow_ups:
                    st.markdown(f"- _{fp}_")

        # Already answered: show result
        if already_evaluated:
            ev = evaluations[q.id]
            st.success("✅ Answer submitted — evaluation below.")
            with st.expander("📊 Your answer", expanded=False):
                st.markdown(f"_{ev.user_answer}_")
            st.markdown(interview_guide.format_evaluation_md(ev))
            st.divider()
            if cur_idx < len(questions) - 1:
                if st.button("Next question →", type="primary", use_container_width=True, key="q_next_after_eval"):
                    st.session_state.interview_current_q = cur_idx + 1
                    st.session_state.interview_show_hint = False
                    st.rerun()
            else:
                if st.button("🏁 Finish & see summary", type="primary", use_container_width=True, key="q_finish"):
                    st.session_state.interview_session_done = True
                    st.rerun()

        else:
            # Answer input area
            user_answer = st.text_area(
                "Your answer",
                height=200,
                placeholder="Type your answer here. Aim for 3–6 sentences. Show your reasoning, trade-offs, and real-world experience.",
                key=f"answer_input_{cur_idx}",
            )

            btn_col1, btn_col2, btn_col3 = st.columns(3)
            submit_clicked = btn_col1.button("✅ Submit Answer", type="primary", use_container_width=True, key=f"submit_{cur_idx}")
            hint_clicked   = btn_col2.button(
                "💡 Hint" if not st.session_state.interview_show_hint else "🙈 Hide hint",
                use_container_width=True, key=f"hint_{cur_idx}",
            )
            skip_clicked   = btn_col3.button("⏭ Skip", use_container_width=True, key=f"skip_{cur_idx}")

            if hint_clicked:
                st.session_state.interview_show_hint = not st.session_state.interview_show_hint
                st.rerun()

            if st.session_state.interview_show_hint:
                st.info(interview_guide.build_hint_text(q))

            if skip_clicked:
                st.session_state.interview_current_q = min(cur_idx + 1, len(questions) - 1)
                st.session_state.interview_show_hint = False
                st.rerun()

            if submit_clicked:
                if not (user_answer or "").strip():
                    st.warning("Please type an answer before submitting.")
                else:
                    with st.spinner("Evaluating with Groq LLM…"):
                        try:
                            raw_eval = call_groq_json(
                                system_prompt=INTERVIEW_EVAL_SYSTEM_PROMPT,
                                user_prompt=interview_guide.build_eval_prompt(q, user_answer),
                                model=STOCK_LLM_MODELS[st.session_state.interview_model_label],
                            )
                            ev = interview_guide.parse_evaluation(raw_eval, q.id, user_answer)
                        except Exception as e:
                            st.error(f"Evaluation failed: {e}")
                            st.stop()

                    with st.spinner("Computing semantic similarity (PyTorch)…"):
                        ev.semantic_score = interview_guide.compute_semantic_score(
                            user_answer, ev.ideal_answer
                        )

                    evaluations[q.id] = ev
                    st.session_state.interview_evaluations = evaluations
                    st.session_state.interview_show_hint   = False
                    st.rerun()

    # ── Coaching chat ────────────────────────────────────────────────────────
    st.divider()
    messages = st.session_state.messages_interview
    if not messages:
        st.subheader("💬 Ask InterviewCoach — go deeper on any topic")
        chat_cols = st.columns(2)
        for i, q_text in enumerate(INTERVIEW_SUGGESTED_QUESTIONS):
            if chat_cols[i % 2].button(q_text, key=f"iv-starter-{i}", use_container_width=True):
                st.session_state._queued_prompt = q_text
                st.rerun()
    else:
        for msg in messages:
            with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🎤"):
                st.markdown(msg["content"])

    queued     = st.session_state.pop("_queued_prompt", None)
    user_input = st.chat_input("Ask InterviewCoach to explain a concept, suggest resources, or run a drill…") or queued
    if not user_input:
        return

    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)
    messages.append({"role": "user", "content": user_input})

    # Build context from session: topic, difficulty, gaps from evaluations
    session_ctx = ""
    if questions:
        session_ctx = (
            f"\n\nCurrent session: Topic={topic}, Difficulty={st.session_state.interview_difficulty_label}, "
            f"{len(evaluations)}/{len(questions)} answered."
        )
    if evaluations:
        all_gaps = []
        for ev in evaluations.values():
            all_gaps.extend(ev.gaps)
        if all_gaps:
            session_ctx += f"\nCandidate gaps identified: {', '.join(set(all_gaps[:8]))}."

    system_prompt = INTERVIEW_CHAT_SYSTEM_PROMPT + session_ctx
    history       = history_for_chat(messages[:-1], max_turns=4)

    with st.chat_message("assistant", avatar="🎤"):
        try:
            full_response = st.write_stream(stream_chat(
                system_prompt=system_prompt,
                history=history,
                user_message=user_input,
                model=STOCK_LLM_MODELS[st.session_state.interview_model_label],
                temperature=0.5,
            ))
            messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(f"Groq call failed: {e}")
            messages.pop()
            st.stop()


# ============================================================================
# MAIN — Agentic AI
# ============================================================================

_STEP_CONFIG = {
    "thought":     ("💭 Thought",     "agent-thought"),
    "action":      ("🔧 Action",      "agent-action"),
    "observation": ("👁 Observation", "agent-observation"),
    "final":       ("✅ Final Answer","agent-final"),
    "error":       ("❌ Error",       "agent-error"),
}


def _render_agent_step(step: agent_core.AgentStep) -> None:
    """Render one AgentStep as a styled HTML card."""
    label_text, css_class = _STEP_CONFIG.get(
        step.step_type, (step.step_type.title(), "agent-step")
    )
    badge = (
        f'<span class="agent-tool-badge">{step.tool_name}</span>'
        if step.tool_name else ""
    )
    # Escape HTML in content but preserve markdown code fences visually
    import html as _html
    safe_content = _html.escape(step.content)
    st.markdown(
        f'<div class="agent-step {css_class}">'
        f'<div class="agent-step-label">{label_text}{badge}</div>'
        f'<div style="white-space:pre-wrap;">{safe_content}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_agent() -> None:
    st.markdown(
        '<div class="main-header">🤖 Agentic AI</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtle">ReAct agent — Reason · Act · Observe · loop. '
        'Powered by Groq function calling + real-world tools.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="disclaimer-blue">'
        '🔬 <strong>How it works:</strong> '
        'The agent uses the <strong>ReAct pattern</strong> (Yao et al., 2022): '
        'it <em>reasons</em> about the task, <em>selects a tool</em>, <em>observes</em> '
        'the result, then loops until it has a confident final answer. '
        'Each step is shown below in real-time. '
        'The REST API (<code>api/main.py</code>) exposes this same agent over HTTP.'
        '</div>',
        unsafe_allow_html=True,
    )

    if not os.environ.get("GROQ_API_KEY"):
        st.error("**GROQ_API_KEY is not set.** Add it to your .env file.")
        return

    # ── Task input ────────────────────────────────────────────────────────────
    st.divider()

    # Example tasks as clickable chips
    if not st.session_state.agent_steps:
        st.subheader("💡 Try an example task")
        cols = st.columns(2)
        for i, example in enumerate(AGENT_SUGGESTED_TASKS):
            if cols[i % 2].button(example, key=f"agent-ex-{i}", use_container_width=True):
                st.session_state._agent_task_input = example
                st.rerun()

    task_value = st.session_state.pop("_agent_task_input", st.session_state.agent_task)
    task = st.text_area(
        "Task for the agent",
        value=task_value,
        height=100,
        placeholder="What do you want the agent to research, calculate, or find?",
        key="agent_task_input_area",
    )

    enabled_tools = st.session_state.agent_enabled_tools
    if not enabled_tools:
        st.warning("No tools selected. Enable at least one tool in the sidebar.")

    run_clicked = st.button(
        "▶ Run Agent",
        type="primary",
        use_container_width=True,
        disabled=not task.strip() or not enabled_tools,
    )

    if run_clicked and task.strip():
        st.session_state.agent_steps = []
        st.session_state.agent_task  = task.strip()

        model = STOCK_LLM_MODELS[st.session_state.agent_model_label]
        max_iter = st.session_state.agent_max_iter

        # Run agent — collect all steps, show a spinner (steps display after)
        steps: list[agent_core.AgentStep] = []
        progress_bar = st.progress(0, text="Agent starting…")

        with st.spinner("Agent working…"):
            gen = agent_core.run_agent(
                task=task.strip(),
                model=model,
                max_iterations=max_iter,
                enabled_tools=enabled_tools if enabled_tools else None,
            )
            action_count = 0
            for step in gen:
                steps.append(step)
                if step.step_type == "action":
                    action_count += 1
                    pct = min(action_count / max_iter, 0.95)
                    progress_bar.progress(pct, text=f"Step {action_count} / {max_iter} max…")

        progress_bar.progress(1.0, text="Done.")
        st.session_state.agent_steps = steps
        st.rerun()

    # ── Display stored steps ──────────────────────────────────────────────────
    steps = st.session_state.agent_steps
    if not steps:
        return

    task_label = st.session_state.agent_task
    st.divider()
    st.subheader(f"🔍 Reasoning trace — *{task_label[:80]}{'…' if len(task_label) > 80 else ''}*")

    action_count = sum(1 for s in steps if s.step_type == "action")
    final_steps  = [s for s in steps if s.step_type == "final"]
    error_steps  = [s for s in steps if s.step_type == "error"]

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Tool calls", action_count)
    col_b.metric("Total steps", len(steps))
    col_c.metric("Status", "✅ Done" if final_steps else ("❌ Error" if error_steps else "⏳"))

    st.markdown("---")
    for step in steps:
        _render_agent_step(step)

    # ── Final answer highlight ─────────────────────────────────────────────────
    if final_steps:
        st.divider()
        st.subheader("📋 Final Answer")
        st.markdown(final_steps[-1].content)

    # ── API usage hint ─────────────────────────────────────────────────────────
    with st.expander("🔌 Use this agent via REST API"):
        st.markdown(
            "Start the FastAPI server from the project root:\n"
            "```bash\n"
            "uvicorn api.main:app --reload --port 8000\n"
            "```\n"
            "Then call the agent:\n"
            "```bash\n"
            "curl -X POST http://localhost:8000/agent/run \\\n"
            '  -H "Content-Type: application/json" \\\n'
            f'  -d \'{{"task": "{task_label[:60]}...", "max_iterations": {st.session_state.agent_max_iter}}}\'\n'
            "```\n"
            "Interactive docs: **http://localhost:8000/docs**"
        )


# ============================================================================
# MAIN — Quant Agent
# ============================================================================
def render_quant() -> None:
    st.markdown('<div class="main-header">📈 AI Quant Agent</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtle">A 4-agent pipeline that fetches market data, '
        'computes technical indicators, scores news sentiment, and generates a '
        'BUY / SELL / HOLD signal — for stocks and crypto.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="disclaimer">⚠️ <strong>Educational only — not financial advice.</strong> '
        'Signals are rule-based heuristics, not predictions.</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.quant_error:
        st.error(f"Pipeline failed: {st.session_state.quant_error}")
        return

    result = st.session_state.quant_result
    if not result:
        st.info(
            "👉 Choose an asset in the sidebar and click **Run Analysis** "
            "to launch the agent pipeline."
        )
        st.markdown(
            "**Pipeline stages**\n\n"
            "1. **MarketAgent** — live price + 3-month OHLCV from Yahoo Finance "
            "(CoinGecko fallback for crypto).\n"
            "2. **TAAgent** — RSI(14), SMA(20/50), EMA(12/26), MACD + signal line.\n"
            "3. **NewsAgent** — recent headlines via DuckDuckGo, scored with a "
            "finance-sentiment lexicon.\n"
            "4. **SignalAgent** — combines the three streams into a "
            "rule-based BUY / SELL / HOLD with a confidence score and an "
            "explainable rationale."
        )
        return

    snap = result["market"]
    ta   = result["technical"]
    news = result["news"]
    sig  = result["signal"]

    # ── Final signal card ─────────────────────────────────────────────────────
    sig_color = {"BUY": "#16a34a", "SELL": "#dc2626", "HOLD": "#ca8a04"}[sig.signal]
    st.markdown(
        f"""
        <div style="border:2px solid {sig_color}; border-radius:12px;
                    padding:18px 22px; margin: 6px 0 18px 0;
                    background: rgba(255,255,255,0.03);">
          <div style="font-size:14px; opacity:0.8;">Final signal for <b>{snap.asset}</b></div>
          <div style="font-size:42px; font-weight:800; color:{sig_color};
                      letter-spacing:1px; line-height:1.1;">{sig.signal}</div>
          <div style="font-size:14px; opacity:0.9;">Confidence: <b>{sig.confidence}%</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Market summary ────────────────────────────────────────────────────────
    st.subheader("💹 Market summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Price", f"{snap.price:,.2f} {snap.currency}",
              f"{snap.change_pct_24h:+.2f}% 24h")
    m2.metric("Volume", f"{snap.volume:,.0f}")
    m3.metric("Kind", snap.kind.title())
    m4.metric("Source", snap.source)
    if not snap.ohlcv.empty:
        st.line_chart(snap.ohlcv["close"], height=220)

    # ── Technical analysis ────────────────────────────────────────────────────
    st.subheader("📊 Technical analysis")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("RSI(14)", f"{ta.rsi:.1f}",
              "oversold" if ta.rsi < 30 else ("overbought" if ta.rsi > 70 else "neutral"))
    t2.metric("MACD", f"{ta.macd:+.3f}", ta.macd_state)
    t3.metric("SMA 20 / 50", f"{ta.sma_20:,.2f} / {ta.sma_50:,.2f}")
    t4.metric("Trend", ta.trend.upper())

    # ── News sentiment ────────────────────────────────────────────────────────
    st.subheader("📰 News sentiment")
    sent_color = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}[news.sentiment]
    st.markdown(
        f"{sent_color} **{news.sentiment.title()}** "
        f"(avg score: `{news.score:+.2f}`, {len(news.headlines)} headlines)"
    )
    for h in news.headlines[:6]:
        tag = "🟢" if h["score"] > 0 else ("🔴" if h["score"] < 0 else "⚪")
        if h.get("url"):
            st.markdown(f"- {tag} [{h['title']}]({h['url']})  "
                        f"<span style='opacity:0.6'>· {h.get('source','')}</span>",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"- {tag} {h['title']}")

    # ── Rationale ─────────────────────────────────────────────────────────────
    st.subheader("🧠 Why this signal?")
    for r in sig.rationale:
        st.markdown(f"- {r}")

    # ── Raw JSON (collapsible) ────────────────────────────────────────────────
    with st.expander("🔍 Raw pipeline output (JSON)"):
        import json as _json
        st.code(
            _json.dumps({
                "market": {
                    "asset": snap.asset, "kind": snap.kind, "price": snap.price,
                    "change_pct_24h": snap.change_pct_24h, "volume": snap.volume,
                    "currency": snap.currency, "source": snap.source,
                    "fetched_at": snap.fetched_at,
                },
                "technical": {
                    "rsi": ta.rsi, "sma_20": ta.sma_20, "sma_50": ta.sma_50,
                    "ema_12": ta.ema_12, "ema_26": ta.ema_26,
                    "macd": ta.macd, "macd_signal": ta.macd_signal,
                    "macd_state": ta.macd_state, "trend": ta.trend,
                },
                "news": {
                    "sentiment": news.sentiment, "score": news.score,
                    "headlines": news.headlines,
                },
                "signal": {
                    "signal": sig.signal, "confidence": sig.confidence,
                    "rationale": sig.rationale,
                },
            }, indent=2, default=str),
            language="json",
        )


# ============================================================================
# Route
# ============================================================================
_ROUTES = {
    "Medical (RAG)":     render_medical,
    "Stocks (Live PSX)": render_stocks,
    "Jobs Finder":       render_jobs,
    "Career Roadmap":    render_roadmap,
    "Resume Analyzer":   render_resume,
    "Code Assistant":    render_code,
    "Interview Guider":  render_interview,
    "🤖 Agentic AI":    render_agent,
    "📈 Quant Agent":   render_quant,
    "🛡️ Admin Panel":   render_admin_panel,
}
_ROUTES.get(st.session_state.domain, render_medical)()
