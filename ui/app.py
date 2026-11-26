# ui/app.py
"""ChatGPT-Style Interface for Agentic RE-SDLC Advisor (Google Gemini + Local RAG).
Features:
- Clean dark theme with sidebar session persistence (PostgreSQL with SQLite fallback)
- Zero-Trust security guardrails, PII masking, and financial domain gating
- Local Qdrant vector RAG grounding + in-memory file attachment (PDF, DOCX, TXT, MD)
- Dynamic LLM-generated SRS, clarification questions, risk scoring, and PDF export
- Human-in-the-loop verification gates + ranked SDLC recommendation & AI critique
"""

import io
import json
import os
import pathlib
import sys
import uuid
import pandas as pd
import streamlit as st

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.llm_client import get_active_model_name
from agents.sdlc_agent import critique_human_sdlc_selection, recommend_sdlc_structured
from db.database import (
    authenticate_user,
    delete_chat_session,
    get_user_chat_sessions,
    init_db,
    load_chat_session,
    register_user,
    save_chat_session_state,
)
from pdf_generator import generate_pdf_report
from rag.retriever import get_rag_stats
from requirements_analysis import analyze_requirement

try:
    init_db()
except Exception as _e:
    print(f"[WARN] Database initialization: {_e}")


def _safe_generate_pdf(data: dict, out_path: str = "output/requirements_analysis.pdf") -> bool:
    try:
        generate_pdf_report(data, out_path)
        return True
    except Exception as exc:
        print(f"[WARN] PDF generation warning: {exc}")
        return False


st.set_page_config(
    page_title="Agentic RE-SDLC Advisor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# DARK THEME & HIGH-CONTRAST CSS
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    html, body, .stApp {
        background-color: #0D0D0D !important;
        color: #ECECEC !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", Helvetica, Arial, sans-serif !important;
    }
    .block-container {
        max-width: 100% !important;
        padding-top: 1.5rem !important;
        padding-bottom: 6.5rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
    }
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        visibility: visible !important;
        z-index: 1000 !important;
    }
    #MainMenu, .stDeployButton, div[data-testid="stToolbarActions"] { visibility: hidden !important; }
    footer { visibility: hidden !important; }
    [data-testid="collapsedControl"] {
        position: fixed !important;
        top: 14px !important;
        left: 16px !important;
        visibility: visible !important;
        display: flex !important;
        z-index: 999999 !important;
    }
    [data-testid="collapsedControl"] button,
    [data-testid="stSidebarCollapseButton"],
    button[aria-label="Expand sidebar"],
    button[aria-label="Collapse sidebar"] {
        visibility: visible !important;
        display: flex !important;
        color: #FFFFFF !important;
        background-color: #1E1E1E !important;
        border: 1px solid #383838 !important;
        border-radius: 6px !important;
        cursor: pointer !important;
        padding: 4px 8px !important;
    }
    [data-testid="collapsedControl"] svg,
    [data-testid="stSidebarCollapseButton"] svg {
        fill: #FFFFFF !important;
        stroke: #FFFFFF !important;
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #171717 !important;
        border-right: 1px solid #262626 !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.2rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        letter-spacing: -0.25px !important;
    }
    h1 { font-size: 22px !important; margin-top: 24px !important; margin-bottom: 12px !important; border-bottom: 1px solid #262626 !important; padding-bottom: 8px !important; }
    h2 { font-size: 17.5px !important; margin-top: 26px !important; margin-bottom: 10px !important; border-bottom: 1px solid #222222 !important; padding-bottom: 6px !important; }
    h3 { font-size: 15px !important; margin-top: 16px !important; margin-bottom: 6px !important; }
    .stMarkdown p, .stMarkdown li { color: #D6D6D6 !important; font-size: 14px !important; line-height: 1.65 !important; }
    strong { color: #FFFFFF !important; }
    hr { border-color: #262626 !important; margin: 22px 0 !important; }
    blockquote { border-left: 3px solid #444444 !important; padding-left: 14px !important; color: #B4B4B4 !important; margin: 12px 0 !important; }
    div[data-testid="stChatMessage"] { background-color: transparent !important; border: none !important; box-shadow: none !important; padding: 12px 0px !important; }
    div[data-testid="stChatMessage"] > div { padding: 0 !important; }
    div[data-testid="stChatMessageAvatar"] { background-color: #212121 !important; border: 1px solid #333333 !important; color: #FFFFFF !important; }
    div.stButton > button {
        background-color: #212121 !important;
        border: 1px solid #383838 !important;
        border-radius: 8px !important;
        font-size: 13.5px !important;
        padding: 8px 16px !important;
    }
    div.stButton > button, div.stButton > button *, div.stButton > button p, div.stButton > button span {
        color: #FFFFFF !important;
        font-weight: 500 !important;
    }
    div.stButton > button:hover { background-color: #2F2F2F !important; border-color: #555555 !important; }
    div[class*="st-key-dbsess_"] button {
        background: transparent !important; border: none !important; box-shadow: none !important;
        padding: 4px 0px !important; width: 100% !important; justify-content: flex-start !important; text-align: left !important;
    }
    div[class*="st-key-dbsess_"] button p {
        color: #D1D5DB !important; font-size: 13.5px !important; font-weight: 400 !important;
        white-space: nowrap !important; overflow: hidden !important; text-overflow: ellipsis !important; width: 100% !important;
    }
    div[class*="st-key-dbsess_"] button:hover p { color: #FFFFFF !important; text-decoration: underline !important; }
    div[class*="st-key-del_"] button {
        background: transparent !important; border: none !important; box-shadow: none !important;
        padding: 0 !important; min-height: 24px !important; height: 24px !important; width: 24px !important; opacity: 0.6 !important;
    }
    div[class*="st-key-del_"] button:hover { background-color: #2D1414 !important; opacity: 1 !important; }
    div.stButton > button[kind="primary"] { background-color: #FFFFFF !important; border: 1px solid #FFFFFF !important; border-radius: 8px !important; }
    div.stButton > button[kind="primary"], div.stButton > button[kind="primary"] *, div.stButton > button[kind="primary"] p {
        color: #000000 !important; font-weight: 600 !important;
    }
    div.stButton > button[kind="primary"]:hover, div.stButton > button[kind="primary"]:hover * {
        background-color: #E5E5E5 !important; border-color: #E5E5E5 !important; color: #000000 !important;
    }
    div.stDownloadButton > button { background-color: #212121 !important; border: 1px solid #383838 !important; border-radius: 8px !important; }
    div.stDownloadButton > button, div.stDownloadButton > button * { color: #FFFFFF !important; font-weight: 500 !important; }
    input, textarea, select, div[data-baseweb="select"] {
        background-color: #1E1E1E !important; color: #EDEDED !important; border: 1px solid #333333 !important; border-radius: 8px !important;
    }
    div[data-testid="stChatInput"] {
        background-color: #212121 !important; border: 1px solid #383838 !important; border-radius: 24px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.45) !important;
    }
    .file-pill {
        display: inline-flex; align-items: center; background-color: #1F1F1F;
        border: 1px solid #383838; border-radius: 8px; padding: 6px 12px;
        margin-bottom: 8px; font-size: 12.5px; color: #ECECEC;
    }
</style>
""", unsafe_allow_html=True)


def extract_text_from_upload(uploaded_file) -> str:
    """Extract plain text from uploaded PDF, DOCX, TXT, or MD directly in memory."""
    if uploaded_file is None:
        return ""
    fname = uploaded_file.name.lower()
    content = uploaded_file.getvalue()

    if fname.endswith((".txt", ".md")):
        return content.decode("utf-8", errors="ignore")
    if fname.endswith(".docx"):
        try:
            import docx
            doc = docx.Document(io.BytesIO(content))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception:
            return ""
    if fname.endswith(".pdf"):
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(content))
            pages = [p.extract_text() or "" for p in reader.pages]
            return "\n".join(p for p in pages if p.strip())
        except Exception:
            return ""
    return ""


def _run_pipeline_interactive(prompt: str) -> dict:
    """Execute the multi-agent pipeline with live step updates."""
    with st.status("Executing Multi-Agent Analysis Pipeline...", expanded=True) as status_box:
        def on_step(agent_label, message):
            status_box.write(f"**[{agent_label}]**: {message}")

        try:
            analysis = analyze_requirement(prompt, step_callback=on_step)
        except Exception as exc:
            status_box.update(label=f"Pipeline Error: {exc}", state="error", expanded=True)
            return {
                "is_error": True,
                "error_message": str(exc),
                "original_requirement": prompt,
            }

        if analysis.get("is_out_of_domain", False):
            status_box.update(label="Guardrail Gate: Outside Financial Domain", state="error", expanded=False)
        elif analysis.get("is_security_violation", False):
            status_box.update(label="Guardrail Gate: Security Violation Blocked", state="error", expanded=False)
        elif analysis.get("is_conversational", False):
            status_box.update(label="Conversational Intent Verified", state="complete", expanded=False)
        else:
            status_box.write("**[Artefact Agent]**: Compiling SRS PDF report...")
            _safe_generate_pdf(analysis, "output/requirements_analysis.pdf")
            status_box.update(label="Multi-Agent Analysis Complete", state="complete", expanded=False)

    return analysis


def render_pipeline_trace_expander(analysis: dict, sdlc_rec: dict = None):
    """Render live multi-agent trace using real runtime telemetry."""
    trace = analysis.get("agent_trace") or analysis.get("pipeline_trace", [])
    if not trace:
        return

    agents_by_id = {a.get("agent_id"): a for a in trace}
    reqs_count = len(analysis.get("requirements_list", []))
    cats_count = len(analysis.get("categories_breakdown", {}))
    risk_score = analysis.get("risk_score_pct", 65)
    risk_lvl = analysis.get("risk_level", "MEDIUM")
    domain = analysis.get("domain", "Financial Systems")
    model_name = analysis.get("llm_model") or get_active_model_name()
    rag_stats = analysis.get("rag_stats") or {}
    rag_chunks = rag_stats.get("chunks_count", 0)
    masked_pii = analysis.get("masked_pii") or []

    sdlc_display = "STANDBY (Gate 1)"
    if sdlc_rec and sdlc_rec.get("recommended_model"):
        sdlc_display = sdlc_rec.get("recommended_model")

    with st.expander("Multi-Agent Execution & Reasoning Trace", expanded=False):
        st.markdown(f"""
        <div style='background-color:#161618; border:1px solid #2B2B30; border-radius:10px; padding:14px 18px; margin-bottom:16px;'>
            <div style='display:flex; flex-wrap:wrap; gap:12px; font-size:13px;'>
                <div style='background:#1E1E22; border:1px solid #333338; border-radius:6px; padding:6px 12px;'>
                    <span style='color:#A78BFA; font-weight:600;'>Guardrails:</span> <b style='color:#10A37F;'>PASSED</b>
                </div>
                <div style='background:#1E1E22; border:1px solid #333338; border-radius:6px; padding:6px 12px;'>
                    <span style='color:#38BDF8; font-weight:600;'>Local RAG:</span> <b style='color:#38BDF8;'>{rag_chunks} chunks indexed</b>
                </div>
                <div style='background:#1E1E22; border:1px solid #333338; border-radius:6px; padding:6px 12px;'>
                    <span style='color:#34D399; font-weight:600;'>LLM Engine:</span> <b style='color:#34D399;'>{model_name}</b>
                </div>
                <div style='background:#1E1E22; border:1px solid #333338; border-radius:6px; padding:6px 12px;'>
                    <span style='color:#FBBF24; font-weight:600;'>Extracted:</span> <b style='color:#FBBF24;'>{reqs_count} Reqs / {cats_count} Categories</b>
                </div>
                <div style='background:#1E1E22; border:1px solid #333338; border-radius:6px; padding:6px 12px;'>
                    <span style='color:#EC4899; font-weight:600;'>SDLC Agent:</span> <b style='color:#ECECEC;'>{sdlc_display}</b>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        tabs = st.tabs([
            "Guardrails & Privacy",
            "RAG Grounding",
            "Extraction & Clarifications",
            "Risk & Traceability",
            "SDLC Selection",
        ])

        with tabs[0]:
            sup = agents_by_id.get("supervisor", {})
            st.markdown("#### Coordinator & Guardrail Agent")
            st.markdown(f"*{sup.get('output_summary', 'Input validated.')}*")
            c1, c2, c3 = st.columns(3)
            c1.metric("Security Filter", "PASSED")
            c2.metric("Financial Domain", domain[:26])
            c3.metric("PII Masked", str(len(masked_pii)) if masked_pii else "0 items")
            if masked_pii:
                st.caption(f"Redacted before external API call: {', '.join(masked_pii)}")

        with tabs[1]:
            com = agents_by_id.get("compliance", {})
            st.markdown("#### Local RAG Grounding Agent")
            st.markdown(f"*{com.get('output_summary', 'Vector search completed.')}*")
            c1, c2, c3 = st.columns(3)
            c1.metric("Qdrant Mode", str(rag_stats.get("mode", "local"))[:24])
            c2.metric("Indexed Chunks", str(rag_chunks))
            c3.metric("Embedding Model", str(rag_stats.get("embedding_model", "BAAI/bge-small-en-v1.5"))[:22])

            clauses = com.get("details", {}).get("retrieved_clauses", [])
            if clauses:
                st.markdown("##### Retrieved Evidence Chunks:")
                for idx, cl in enumerate(clauses, 1):
                    st.markdown(f"**[{idx}] {cl.get('citation', 'KB')}** (`{cl.get('category', 'general')}` | Cosine Score: `{cl.get('score', 0.0)}`)")
                    if cl.get("preview"):
                        st.caption(f'"{cl.get("preview")}..."')
            else:
                st.info("No matching chunks found in local vector store. Run `python -m rag.ingest` or upload documents to populate RAG.")

            regs = analysis.get("regulations", [])
            if regs:
                st.markdown("##### Mapped Regulations & Standards:")
                for reg in regs:
                    st.markdown(f"- **{reg.get('regulation')}** ({reg.get('relevance')}): *{reg.get('statutory_citation')}*")

        with tabs[2]:
            ext = agents_by_id.get("extraction", {})
            st.markdown("#### Extraction, Classification & Clarification Agent")
            st.markdown(f"*{ext.get('output_summary', '')}*")
            st.markdown(f"- **API Model**: `{model_name}`")
            st.markdown(f"- **MoSCoW Baseline**: `{analysis.get('moscow_priority', 'Must Have')}` — *{analysis.get('priority_justification', '')}*")
            clarifications = analysis.get("clarification_questions", [])
            if clarifications:
                st.markdown("##### Open Ambiguities & Clarification Questions:")
                for q in clarifications:
                    st.markdown(f"- {q}")

        with tabs[3]:
            st.markdown("#### Risk, Controls & Traceability Agent")
            c1, c2 = st.columns(2)
            c1.metric("Risk Classification", f"{risk_lvl} ({risk_score}/100)")
            c2.metric("Requirements Extracted", f"{reqs_count}")
            for rf in analysis.get("risk_factors", []):
                st.markdown(f"- **Risk Factor**: {rf}")

            t_matrix = analysis.get("traceability_matrix", [])
            if t_matrix:
                st.markdown("##### Traceability Matrix:")
                st.dataframe(pd.DataFrame(t_matrix), use_container_width=True, hide_index=True)

            tech_controls = analysis.get("technical_controls", [])
            if tech_controls:
                st.markdown("##### Technical & Security Controls:")
                for tc in tech_controls:
                    st.markdown(f"- **{tc.get('control_id')} — {tc.get('title')}**: {tc.get('specification')} *(Verification: {tc.get('verification')})*")

        with tabs[4]:
            st.markdown("#### SDLC Selection Agent")
            if sdlc_rec and sdlc_rec.get("recommended_model"):
                st.success(f"Recommended Lifecycle: **{sdlc_rec.get('recommended_model')}** ({sdlc_rec.get('confidence_pct', 'N/A')}% suitability)")
                st.markdown(sdlc_rec.get("reasoning", ""))
            else:
                st.info("Pending Step 1 human approval of requirements.")


# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
defaults = {
    "authenticated_user": None,
    "current_session_id": str(uuid.uuid4()),
    "messages": [],
    "current_analysis": None,
    "requirements_approved": False,
    "sdlc_recommendation": None,
    "sdlc_critique": None,
    "sdlc_approved": False,
    "show_revision_box": False,
    "pending_prompt": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _sync_to_db():
    try:
        user = st.session_state.get("authenticated_user")
        if user and st.session_state.current_analysis:
            title = st.session_state.current_analysis.get("title", "Requirements Specification")[:100]
            save_chat_session_state(
                user_id=user["id"],
                session_id=st.session_state.current_session_id,
                title=title,
                messages=st.session_state.messages,
                analysis=st.session_state.current_analysis,
                sdlc=st.session_state.sdlc_recommendation,
                req_approved=st.session_state.requirements_approved,
                sdlc_approved=st.session_state.sdlc_approved,
            )
    except Exception as exc:
        print(f"[WARN] DB sync error: {exc}")


# -----------------------------------------------------------------------------
# AUTHENTICATION WALL
# -----------------------------------------------------------------------------
if st.session_state.authenticated_user is None:
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, col_l2, _ = st.columns([1, 2, 1])
    with col_l2:
        st.markdown("<div style='background: #161618; padding: 28px; border: 1px solid #2a2a2e; border-radius: 12px;'>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; margin-bottom: 2px; font-weight: 700;'>Agentic RE-SDLC</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #888888; font-size: 13px; margin-bottom: 20px;'>Financial Software Requirements Engineering & SDLC Platform</p>", unsafe_allow_html=True)

        tab_login, tab_signup = st.tabs(["Sign In", "Create Account"])

        with tab_login:
            login_email = st.text_input("Email", placeholder="architect@fintech.local", key="login_email")
            login_password = st.text_input("Password", type="password", key="login_pass")
            st.caption("Demo account: `architect@fintech.local` / `DemoPass123!`")
            if st.button("Sign In", type="primary", use_container_width=True, key="btn_signin"):
                if not login_email or not login_password:
                    st.error("Please enter email and password.")
                else:
                    u = authenticate_user(login_email, login_password)
                    if u:
                        st.session_state.authenticated_user = u
                        st.session_state.current_session_id = str(uuid.uuid4())
                        st.rerun()
                    else:
                        st.error("Invalid email or password.")

        with tab_signup:
            reg_name = st.text_input("Full Name", placeholder="e.g. Alex Verma", key="reg_name")
            reg_email = st.text_input("Email", placeholder="e.g. alex@company.com", key="reg_email")
            reg_pass = st.text_input("Password (min 6 chars)", type="password", key="reg_pass")
            reg_role = st.selectbox("Role", ["Principal Architect", "Business Analyst", "Compliance Officer", "Security Analyst"], key="reg_role")
            if st.button("Create Account & Sign In", type="primary", use_container_width=True, key="btn_signup"):
                res = register_user(reg_email, reg_pass, full_name=reg_name, role=reg_role)
                if res.get("success"):
                    st.session_state.authenticated_user = res["user"]
                    st.session_state.current_session_id = str(uuid.uuid4())
                    st.rerun()
                else:
                    st.error(res.get("error", "Registration failed."))

        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    u = st.session_state.authenticated_user or {}
    st.markdown("### Agentic RE-SDLC")
    st.markdown(
        f"<div style='background:#1c1c1f; padding:8px 12px; border-radius:8px; border:1px solid #2a2a2e; margin-bottom:12px; font-size:12px;'>"
        f"<b>{u.get('full_name', 'User')}</b><br>"
        f"<span style='color:#888;'>{u.get('email', '')}</span><br>"
        f"<span style='color:#4ade80;'>{u.get('role', '')}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    col_new, col_out = st.columns([2, 1])
    with col_new:
        if st.button("+ New chat", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.session_state.current_analysis = None
            st.session_state.requirements_approved = False
            st.session_state.sdlc_recommendation = None
            st.session_state.sdlc_critique = None
            st.session_state.sdlc_approved = False
            st.session_state.show_revision_box = False
            st.session_state.pending_prompt = None
            st.session_state.current_session_id = str(uuid.uuid4())
            st.rerun()
    with col_out:
        if st.button("Sign Out", use_container_width=True):
            st.session_state.authenticated_user = None
            st.session_state.messages = []
            st.session_state.current_analysis = None
            st.session_state.current_session_id = str(uuid.uuid4())
            st.rerun()

    try:
        user_db_sessions = get_user_chat_sessions(u["id"])
    except Exception:
        user_db_sessions = []

    if user_db_sessions:
        st.markdown("<p style='font-size:11px; color:#888888; font-weight:600; text-transform:uppercase; margin-top:16px; margin-bottom:8px;'>Recents</p>", unsafe_allow_html=True)
        for s in user_db_sessions:
            c_sess, c_del = st.columns([5, 1])
            with c_sess:
                if st.button(f"{s['title'][:22]}", key=f"dbsess_{s['id']}", use_container_width=True):
                    loaded = load_chat_session(s["id"], u["id"])
                    if loaded:
                        st.session_state.current_session_id = s["id"]
                        st.session_state.messages = loaded["messages"]
                        st.session_state.current_analysis = loaded["analysis"]
                        st.session_state.sdlc_recommendation = loaded["sdlc"]
                        st.session_state.sdlc_critique = None
                        st.session_state.requirements_approved = loaded["req_approved"]
                        st.session_state.sdlc_approved = loaded["sdlc_approved"]
                        st.session_state.show_revision_box = False
                        st.rerun()
            with c_del:
                if st.button("🗑", key=f"del_{s['id']}", help="Delete chat"):
                    delete_chat_session(s["id"], u["id"])
                    if st.session_state.current_session_id == s["id"]:
                        st.session_state.messages = []
                        st.session_state.current_analysis = None
                        st.session_state.current_session_id = str(uuid.uuid4())
                    st.rerun()

    st.markdown("---")
    st.markdown("<p style='font-size:11px; color:#888888; font-weight:600; text-transform:uppercase; margin-bottom:6px;'>Attach Document</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload specification or policy document",
        type=["pdf", "docx", "txt", "md"],
        label_visibility="collapsed",
        key="sidebar_doc_uploader",
    )
    if uploaded_file is not None:
        file_size_kb = round(len(uploaded_file.getvalue()) / 1024, 1)
        st.markdown(
            f"<div class='file-pill' style='margin-top:6px; margin-bottom:8px;'><b>{uploaded_file.name}</b> ({file_size_kb} KB)</div>",
            unsafe_allow_html=True,
        )
        if st.button("Analyze Document", use_container_width=True, type="primary"):
            doc_text = extract_text_from_upload(uploaded_file)
            st.session_state.pending_prompt = (
                f"Extract and classify software requirements from uploaded document '{uploaded_file.name}':\n\n{doc_text[:4500]}"
            )
            st.rerun()

    rag_info = get_rag_stats()
    st.markdown(
        f"<div style='margin-top:24px; font-size:11px; color:#666666;'>"
        f"Model: {get_active_model_name()}<br>"
        f"RAG: {rag_info.get('chunks_count', 0)} chunks ({rag_info.get('status', 'unknown')})"
        f"</div>",
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# MAIN VIEWPORT
# -----------------------------------------------------------------------------
if st.session_state.pending_prompt:
    active_prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    st.session_state.messages.append({"role": "user", "content": active_prompt})
    analysis = _run_pipeline_interactive(active_prompt)

    if analysis.get("is_error"):
        st.session_state.messages.append({"role": "assistant", "content": f"**Error**: {analysis.get('error_message')}"})
        st.session_state.current_analysis = None
    elif analysis.get("is_conversational"):
        st.session_state.messages.append({"role": "assistant", "content": analysis.get("conversational_response")})
        st.session_state.current_analysis = None
    elif analysis.get("is_out_of_domain"):
        st.session_state.messages.append({"role": "assistant", "content": analysis.get("domain_rejection_response")})
        st.session_state.current_analysis = None
    elif analysis.get("is_security_violation"):
        st.session_state.messages.append({"role": "assistant", "content": analysis.get("security_rejection_response")})
        st.session_state.current_analysis = None
    else:
        st.session_state.current_analysis = analysis
        st.session_state.requirements_approved = False
        st.session_state.sdlc_recommendation = None
        st.session_state.sdlc_critique = None
        st.session_state.sdlc_approved = False
        req_count = len(analysis.get("requirements_list", []))
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Generated **{req_count} structured requirements** for **{analysis.get('title')}**. Review the specification below to proceed.",
        })
        _sync_to_db()
    st.rerun()

if not st.session_state.messages and st.session_state.current_analysis is None:
    st.markdown("<div style='text-align:center; margin-top:45px; margin-bottom:36px;'>", unsafe_allow_html=True)
    st.markdown("<h1 style='font-size:26px; border:none; margin-bottom:6px;'>What financial system would you like to design?</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:14px; color:#888888;'>Extract RAG-grounded requirements, assess risk & compliance, and identify the optimal SDLC model.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Digital Banking & Customer Onboarding\n\nMobile banking with KYC verification, MFA, and audit trails", use_container_width=True):
            st.session_state.pending_prompt = "Design a digital banking onboarding and mobile account management platform with KYC verification, MFA, and full audit logging"
            st.rerun()
        if st.button("Real-Time Fraud Detection Engine\n\nTransaction risk scoring, anomaly alerts, and case management", use_container_width=True):
            st.session_state.pending_prompt = "Design a real-time payment fraud detection and AML alert management engine with low-latency scoring and analyst review workflows"
            st.rerun()
    with c2:
        if st.button("Digital Loan Origination System\n\nCredit scoring, underwriting rules, and repayment schedules", use_container_width=True):
            st.session_state.pending_prompt = "Design a digital loan origination and credit underwriting system with automated eligibility checks, transparent fee disclosures, and repayment tracking"
            st.rerun()
        if st.button("Payment Processing & Settlement Switch\n\nTokenized card/wallet payments, reconciliation, and high availability", use_container_width=True):
            st.session_state.pending_prompt = "Design a high-availability payment processing gateway with tokenization, idempotent transaction handling, and automated T+1 reconciliation"
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------------------------------------------------------
# STEP 1: REQUIREMENTS SPECIFICATION
# -----------------------------------------------------------------------------
analysis = st.session_state.current_analysis
if (
    analysis is not None
    and not analysis.get("is_error")
    and not analysis.get("is_conversational")
    and not analysis.get("is_out_of_domain")
    and not analysis.get("is_security_violation")
):
    reqs_list = analysis.get("requirements_list", [])
    risk_lvl = analysis.get("risk_level", "MEDIUM")
    risk_score = analysis.get("risk_score_pct", 65)
    risk_summary = analysis.get("risk_summary", "")

    with st.chat_message("assistant"):
        st.markdown(f"# {analysis.get('title', 'Software Requirements Specification')}")
        st.markdown(
            f"**Domain**: {analysis.get('domain')} &nbsp;|&nbsp; "
            f"**MoSCoW Baseline**: {analysis.get('moscow_priority', 'Must Have')} &nbsp;|&nbsp; "
            f"**Risk**: {risk_lvl} ({risk_score}/100) &nbsp;|&nbsp; "
            f"**Total Requirements**: {len(reqs_list)}"
        )
        if analysis.get("system_summary"):
            st.markdown(f"> **System Summary**: {analysis.get('system_summary')}")
        if risk_summary:
            st.markdown(f"> **Risk Rationale**: {risk_summary}")

        render_pipeline_trace_expander(analysis, sdlc_rec=st.session_state.get("sdlc_recommendation"))

        st.markdown("---")
        cat_breakdown = analysis.get("categories_breakdown", {})
        for idx, (cat_name, cat_reqs) in enumerate(cat_breakdown.items(), 1):
            st.markdown(f"## {idx}. {cat_name} ({len(cat_reqs)})")
            for r in cat_reqs:
                st.markdown(f"### {r.get('id')} — {r.get('title')}")
                st.markdown(r.get("description", ""))
                st.markdown(
                    f"- **Priority**: `{r.get('priority')}` | **Confidence**: `{r.get('confidence_score', 0.85)}` | "
                    f"**Reference**: *{r.get('regulation')}*\n"
                    f"- **Acceptance Criteria**: {r.get('acceptance_criteria')}"
                )

        clarifications = analysis.get("clarification_questions", [])
        if clarifications:
            st.markdown("---")
            st.markdown("## Open Ambiguities & Clarification Questions")
            for q_idx, q in enumerate(clarifications, 1):
                st.markdown(f"{q_idx}. {q}")

        st.markdown("---")
        col_pdf, col_json = st.columns([1, 1])
        with col_pdf:
            pdf_path = "output/requirements_analysis.pdf"
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label=f"Download SRS PDF ({len(reqs_list)} Requirements)",
                        data=f.read(),
                        file_name="requirements_analysis.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
        with col_json:
            st.download_button(
                label="Download Specification (JSON)",
                data=json.dumps(analysis, indent=2),
                file_name="requirements_specification.json",
                mime="application/json",
                use_container_width=True,
            )

        st.markdown("---")
        if not st.session_state.requirements_approved:
            st.markdown("## Step 1: Human Verification of Requirements")
            st.caption(f"Review the {len(reqs_list)} requirements above. Approve to run SDLC selection, request changes, or reset.")

            is_revising = st.session_state.get("show_revision_box", False)
            c_app, c_rev, c_rej = st.columns([1.3, 1.2, 0.8])
            with c_app:
                if st.button("Approve Requirements & Predict SDLC", type="primary", use_container_width=True):
                    st.session_state.requirements_approved = True
                    st.session_state.show_revision_box = False
                    _sync_to_db()
                    st.rerun()
            with c_rev:
                rev_label = "Cancel Revision" if is_revising else "Request Changes / Add Specs"
                if st.button(rev_label, use_container_width=True):
                    st.session_state.show_revision_box = not is_revising
                    st.rerun()
            with c_rej:
                if st.button("Reject & Reset", use_container_width=True):
                    st.session_state.current_analysis = None
                    st.session_state.requirements_approved = False
                    st.session_state.sdlc_recommendation = None
                    st.session_state.sdlc_critique = None
                    st.session_state.sdlc_approved = False
                    st.session_state.show_revision_box = False
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Specification discarded. What financial system would you like to analyze next?",
                    })
                    st.rerun()

            if is_revising:
                st.info("Type your requested changes or clarification answers in the chat input below.")
        else:
            st.markdown("**Requirements Verified and Approved by Human Reviewer**")

# -----------------------------------------------------------------------------
# STEP 2: SDLC RECOMMENDATION & HUMAN OVERRIDE
# -----------------------------------------------------------------------------
if st.session_state.requirements_approved and analysis is not None:
    with st.chat_message("assistant"):
        st.markdown("## Step 2: Adaptive SDLC Recommendation")

        if st.session_state.sdlc_recommendation is None:
            with st.status("Evaluating SDLC Decision Factors via Gemini + RAG...", expanded=True) as sdlc_status:
                try:
                    req_text = analysis.get("original_requirement", "")
                    rag_ctx = analysis.get("rag_context", "")
                    rec = recommend_sdlc_structured(req_text, rag_context=rag_ctx)
                    st.session_state.sdlc_recommendation = rec
                    st.session_state.sdlc_approved = True
                    _sync_to_db()
                    sdlc_status.update(label=f"Recommended: {rec.get('recommended_model')}", state="complete", expanded=False)
                    st.rerun()
                except Exception as exc:
                    st.error(f"SDLC Recommendation failed: {exc}")
                    st.stop()

        sdlc = st.session_state.sdlc_recommendation
        st.markdown(f"### Recommended SDLC Model: **{sdlc.get('recommended_model')}** ({sdlc.get('confidence_pct', 'N/A')}% fit)")
        st.markdown(f"**Rationale**: {sdlc.get('reasoning', '')}")

        factors = sdlc.get("decision_factors", {})
        if factors:
            cols = st.columns(len(factors))
            for idx, (fk, fv) in enumerate(factors.items()):
                cols[idx].metric(fk.replace("_", " ").title(), str(fv))

        ranked = sdlc.get("ranked_recommendations", [])
        if ranked:
            st.markdown("### Ranked SDLC Options:")
            for item in ranked:
                st.markdown(f"- **{item.get('model')}** (`{item.get('score_pct', '')}%`): {item.get('summary', '')}")

        alternatives = sdlc.get("alternatives_considered", [])
        if alternatives:
            st.markdown("### Rejected Alternatives:")
            for alt in alternatives:
                st.markdown(f"- **{alt.get('model')}**: {alt.get('why_not')}")

        workflow = sdlc.get("tailored_workflow", [])
        if workflow:
            st.markdown("### Tailored SDLC Workflow & Gates:")
            st.dataframe(pd.DataFrame(workflow), use_container_width=True, hide_index=True)

        with st.expander("Override SDLC Model & Request AI Critique (Human-in-the-Loop)", expanded=False):
            sdlc_catalog = [
                "Agile-DevSecOps Hybrid",
                "V-Model with DevSecOps Gates",
                "V-Model",
                "Waterfall",
                "Spiral Model",
                "Iterative / Prototyping Model",
                "Dual-Track Agile",
                "Incremental Model",
            ]
            curr_rec = sdlc.get("recommended_model", "Agile-DevSecOps Hybrid")
            curr_idx = sdlc_catalog.index(curr_rec) if curr_rec in sdlc_catalog else 0
            new_model = st.selectbox("Select Alternative SDLC Model:", sdlc_catalog, index=curr_idx, key="sdlc_switch_select")
            override_notes = st.text_input("Reason / Team Constraints (optional):", key="sdlc_override_notes")
            if st.button("Apply Override & Critique", type="primary"):
                critique = critique_human_sdlc_selection(
                    requirement=analysis.get("original_requirement", ""),
                    chosen_model=new_model,
                    human_feedback=override_notes,
                    rag_context=analysis.get("rag_context", ""),
                )
                sdlc["recommended_model"] = new_model
                sdlc["reasoning"] = critique.get("architectural_synthesis", f"Adopted {new_model} per architect decision.")
                st.session_state.sdlc_recommendation = sdlc
                st.session_state.sdlc_critique = critique
                st.session_state.sdlc_approved = True
                _sync_to_db()
                st.rerun()

        if st.session_state.sdlc_critique:
            crit = st.session_state.sdlc_critique
            st.info(
                f"**AI Governance Critique for {crit.get('chosen_model')}**\n\n"
                f"- **Strengths**: {crit.get('strengths_of_choice')}\n"
                f"- **Risks & Mitigations**: {crit.get('statutory_risks_and_mitigations')}\n"
                f"- **Governance Cadence**: {crit.get('governance_cadence')}"
            )


# -----------------------------------------------------------------------------
# CHAT INPUT BAR
# -----------------------------------------------------------------------------
user_prompt = st.chat_input("Describe your financial system or type a command...")

if user_prompt:
    raw_prompt = user_prompt.strip()
    clean_lower = raw_prompt.lower()

    reset_words = {"reset", "clear", "start over", "discard", "new chat", "new session", "start fresh"}
    if clean_lower in reset_words:
        st.session_state.messages = []
        st.session_state.current_analysis = None
        st.session_state.requirements_approved = False
        st.session_state.sdlc_recommendation = None
        st.session_state.sdlc_critique = None
        st.session_state.sdlc_approved = False
        st.session_state.show_revision_box = False
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Session reset. What financial software system would you like to analyze?",
        })
        st.rerun()

    doc_text = ""
    doc_name = ""
    if "sidebar_doc_uploader" in st.session_state and st.session_state.sidebar_doc_uploader is not None:
        u_file = st.session_state.sidebar_doc_uploader
        doc_text = extract_text_from_upload(u_file)
        doc_name = u_file.name

    if doc_text:
        display_msg = f"{raw_prompt}\n\n*(Attached: {doc_name})*"
        full_query = f"{raw_prompt}\n\n[ATTACHED DOCUMENT: {doc_name}]\n{doc_text[:4500]}"
    else:
        display_msg = raw_prompt
        full_query = raw_prompt

    st.session_state.messages.append({"role": "user", "content": display_msg})

    # Modification request to active specification
    is_modification = False
    if analysis is not None and not analysis.get("is_conversational") and not analysis.get("is_error"):
        mod_verbs = ("add ", "change ", "modify ", "update ", "revise ", "remove ", "include ", "adjust ", "answer:")
        if st.session_state.get("show_revision_box", False) or any(clean_lower.startswith(v) for v in mod_verbs):
            is_modification = True

    if is_modification:
        st.session_state.show_revision_box = False
        original_req = analysis.get("original_requirement", "")
        revised_query = f"{original_req}\n\n[USER REVISION / CLARIFICATION]: {full_query}"
        revised_analysis = _run_pipeline_interactive(revised_query)
        st.session_state.current_analysis = revised_analysis
        st.session_state.requirements_approved = False
        st.session_state.sdlc_recommendation = None
        st.session_state.sdlc_critique = None
        st.session_state.sdlc_approved = False
        new_count = len(revised_analysis.get("requirements_list", []))
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Specification updated with **{new_count} requirements** incorporating your revision.",
        })
        _sync_to_db()
        st.rerun()

    new_analysis = _run_pipeline_interactive(full_query)
    if new_analysis.get("is_error"):
        st.session_state.messages.append({"role": "assistant", "content": f"**Error**: {new_analysis.get('error_message')}"})
        st.session_state.current_analysis = None
    elif new_analysis.get("is_conversational"):
        st.session_state.messages.append({"role": "assistant", "content": new_analysis.get("conversational_response")})
        st.session_state.current_analysis = None
    elif new_analysis.get("is_out_of_domain"):
        st.session_state.messages.append({"role": "assistant", "content": new_analysis.get("domain_rejection_response")})
        st.session_state.current_analysis = None
    elif new_analysis.get("is_security_violation"):
        st.session_state.messages.append({"role": "assistant", "content": new_analysis.get("security_rejection_response")})
        st.session_state.current_analysis = None
    else:
        st.session_state.current_analysis = new_analysis
        st.session_state.requirements_approved = False
        st.session_state.sdlc_recommendation = None
        st.session_state.sdlc_critique = None
        st.session_state.sdlc_approved = False
        st.session_state.show_revision_box = False
        req_count = len(new_analysis.get("requirements_list", []))
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Generated **{req_count} structured requirements** for **{new_analysis.get('title')}**. Review below and approve to proceed to SDLC recommendation.",
        })
        _sync_to_db()
    st.rerun()
