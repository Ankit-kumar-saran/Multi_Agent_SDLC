# requirements_analysis.py
"""Lightweight Multi-Agent Orchestrator for Financial Requirements Engineering & SDLC Analysis.

Pipeline:
1. Conversational & Intent Routing
2. Zero-Trust Security Guardrails, Sensitive-Data Masking, & Financial Domain Gating
3. Local Qdrant RAG Grounding (BAAI/bge-small-en-v1.5)
4. Google Gemini API Extraction, Classification, Risk & Clarification Synthesis
5. Dynamic Traceability Matrix & Execution Trace Assembly
"""

from typing import Any, Dict, List
from agents.extraction_agent import extract_system_architecture
from agents.llm_client import get_active_model_name
from guardrails.input_filter import (
    check_domain_scope,
    is_safe as input_is_safe,
    mask_sensitive_data,
)
from rag.retriever import get_rag_stats, retrieve


def is_conversational(text: str) -> bool:
    """Detect whether input is casual conversation/greeting rather than a system requirement."""
    cleaned = text.strip().lower().rstrip("?!.")
    greetings = {
        "hi", "hello", "hey", "how are you", "how r u", "who are you", "who r u",
        "what can you do", "good morning", "good afternoon", "good evening",
        "help", "test", "sup", "whats up", "what is this", "explain yourself",
        "tell me about yourself", "hi there", "hello there", "thanks", "thank you",
    }
    if cleaned in greetings:
        return True
    if any(cleaned.startswith(g + " ") for g in ("hi", "hello", "hey", "good morning", "good evening")):
        return True
    return False


def get_conversational_response(user_text: str) -> str:
    """Return a concise, accurate conversational response."""
    text = user_text.lower()
    if "how are you" in text:
        return (
            "I'm ready to help! I am your **Agentic RE-SDLC Advisor** for financial software systems, "
            "powered by Google Gemini, local Qdrant RAG, and security guardrails.\n\n"
            "Describe a financial application or upload a document (.pdf, .docx, .txt, .md), and I will "
            "extract structured requirements, identify compliance & risk controls, surface clarification "
            "questions, generate an SRS PDF, and recommend an appropriate SDLC model."
        )
    return (
        "Hello! Welcome to the **Agentic RE-SDLC Advisor** for financial software engineering.\n\n"
        "You can describe any financial system (e.g., digital banking, loan origination, payment processing, "
        "fraud detection, insurance claims, trading/wealth, KYC onboarding, or regulatory reporting) or upload "
        "a requirements/policy document in the sidebar to generate:\n"
        "- **Structured Functional & Non-Functional Requirements** with acceptance criteria & confidence scores\n"
        "- **RAG-Grounded Compliance & Technical Controls**\n"
        "- **Context-Aware Clarification Questions** for ambiguities\n"
        "- **Downloadable SRS PDF & JSON Artefacts**\n"
        "- **Ranked SDLC Recommendation & Workflow Gates**"
    )


def _risk_color(level: str) -> str:
    return {
        "CRITICAL": "#EF4444",
        "HIGH": "#F97316",
        "MEDIUM": "#F59E0B",
        "LOW": "#10A37F",
    }.get(level.upper(), "#F59E0B")


def analyze_requirement(requirement_text: str, step_callback=None) -> Dict[str, Any]:
    """Run the end-to-end lightweight multi-agent RE pipeline for a financial requirement."""
    if not requirement_text or not requirement_text.strip():
        raise ValueError("Empty requirement text provided.")

    # 1. Conversational Routing
    if is_conversational(requirement_text):
        if step_callback:
            step_callback("Coordinator Agent", "Classified input as conversational inquiry.")
        conv_trace = [
            {
                "agent_id": "supervisor",
                "agent_name": "Coordinator & Guardrail Agent",
                "role": "Orchestration & Safety Gate",
                "status": "PASSED",
                "badge": "Orchestrator",
                "badge_color": "#A78BFA",
                "output_summary": "Classified input as conversational interaction.",
                "details": {"intent": "Conversational"},
            }
        ]
        return {
            "is_conversational": True,
            "conversational_response": get_conversational_response(requirement_text),
            "original_requirement": requirement_text.strip(),
            "pipeline_trace": conv_trace,
            "agent_trace": conv_trace,
        }

    # 2. Zero-Trust Security Guardrail
    if not input_is_safe(requirement_text):
        if step_callback:
            step_callback("Guardrail Agent", "Blocked input due to unsafe/adversarial pattern.")
        sec_trace = [
            {
                "agent_id": "supervisor",
                "agent_name": "Coordinator & Guardrail Agent",
                "role": "Orchestration & Safety Gate",
                "status": "REJECTED (Security Guardrail)",
                "badge": "Security Alert",
                "badge_color": "#EF4444",
                "output_summary": "Input blocked by security guardrails.",
                "details": {"guardrail_status": "Blocked"},
            }
        ]
        return {
            "is_security_violation": True,
            "security_rejection_response": (
                "**Security Guardrail Alert**: Your input was blocked because it contained potentially "
                "unsafe, adversarial, or prohibited commands."
            ),
            "original_requirement": requirement_text.strip(),
            "pipeline_trace": sec_trace,
            "agent_trace": sec_trace,
        }

    # 3. Sensitive Data Masking (before external LLM or RAG calls)
    masked_text, redacted_items = mask_sensitive_data(requirement_text.strip())

    # 4. Financial Domain Scope Check
    in_domain, detected_category, clarification = check_domain_scope(masked_text)
    if not in_domain:
        if step_callback:
            step_callback("Guardrail Agent", f"Out-of-scope domain intercepted: {detected_category}")
        domain_trace = [
            {
                "agent_id": "supervisor",
                "agent_name": "Coordinator & Guardrail Agent",
                "role": "Orchestration & Domain Gate",
                "status": "REJECTED (Domain Boundary)",
                "badge": "Scope Alert",
                "badge_color": "#F59E0B",
                "output_summary": f"Query classified outside financial domain ({detected_category}).",
                "details": {"detected_topic": detected_category},
            }
        ]
        rejection_msg = (
            f"### Domain Scope Alert: Outside Financial Software Scope\n\n"
            f"This request was classified under **{detected_category}**, whereas this platform is scoped "
            f"strictly to **financial software engineering** (banking, payments, lending, fraud/AML, "
            f"insurance, investments/trading, KYC, regulatory reporting, and financial analytics).\n\n"
            f"> **Guidance**: {clarification}"
        )
        return {
            "is_out_of_domain": True,
            "domain_rejection_response": rejection_msg,
            "original_requirement": masked_text,
            "pipeline_trace": domain_trace,
            "agent_trace": domain_trace,
        }

    if step_callback:
        mask_note = f" (masked: {', '.join(redacted_items)})" if redacted_items else ""
        step_callback("Guardrail Agent", f"Verified financial scope '{detected_category}'{mask_note}.")

    # 5. Local RAG Retrieval (Qdrant)
    rag_contexts: List[Dict[str, Any]] = []
    rag_context_str = ""
    rag_stats = get_rag_stats()
    try:
        rag_contexts = retrieve(masked_text, top_k=5)
        rag_context_str = "\n\n".join(
            f"[{c.get('citation', 'Knowledge Base')} | score={c.get('score', 0.0)}]:\n{c.get('text', '').strip()[:550]}"
            for c in rag_contexts
            if c.get("text")
        )
    except Exception as exc:
        print(f"[WARN] RAG retrieval error: {exc}")

    rag_hits_summary = [
        {
            "citation": c.get("citation", "Knowledge Base"),
            "category": c.get("category", "general"),
            "score": round(float(c.get("score", 0.0)), 4),
            "preview": (c.get("text") or "").strip()[:220],
        }
        for c in rag_contexts
    ]

    if step_callback:
        top_cite = rag_hits_summary[0]["citation"] if rag_hits_summary else "None (no indexed matches)"
        step_callback("Compliance & RAG Agent", f"Retrieved {len(rag_contexts)} knowledge base chunks (Top: {top_cite}).")

    # 6. Gemini API Extraction, Classification & Risk Analysis
    model_name = get_active_model_name()
    if step_callback:
        step_callback("Extraction & Classification Agent", f"Calling {model_name} to synthesize project-specific SRS...")

    arch_data = extract_system_architecture(masked_text, rag_context=rag_context_str)

    requirements_list = arch_data.get("requirements", [])
    project_title = arch_data.get("project_title", "Financial Software Specification")
    detected_domain = arch_data.get("domain", detected_category)

    categories_breakdown: Dict[str, List[Dict[str, Any]]] = {}
    for req in requirements_list:
        cat = req.get("category", "Functional")
        categories_breakdown.setdefault(cat, []).append(req)

    if step_callback:
        step_callback(
            "Extraction & Classification Agent",
            f"Generated {len(requirements_list)} requirements across {len(categories_breakdown)} categories.",
        )

    # 7. Dynamic Traceability Matrix
    traceability_matrix = []
    for idx, (cat_name, cat_items) in enumerate(categories_breakdown.items(), 1):
        req_ids = [it["id"] for it in cat_items]
        id_range = f"{req_ids[0]} - {req_ids[-1]}" if len(req_ids) > 1 else req_ids[0]
        traceability_matrix.append(
            {
                "req_id": id_range,
                "sub_module": f"{project_title} — {cat_name}",
                "sdlc_phase": f"Phase {idx}: {cat_name}",
                "target_artefact": f"{cat_name} Design & Test Suite",
                "verification_method": "Automated Tests & Compliance Review",
            }
        )

    risk_level = arch_data.get("risk_level", "MEDIUM")
    risk_score = arch_data.get("risk_score_pct", 65)
    risk_summary = arch_data.get("risk_summary", "")
    risk_factors = arch_data.get("risk_factors", [])
    clarifications = arch_data.get("clarification_questions", [])

    if step_callback:
        step_callback("Risk & Validation Agent", f"Assessed risk at {risk_score}/100 ({risk_level}); surfaced {len(clarifications)} clarification questions.")

    agent_trace = [
        {
            "agent_id": "supervisor",
            "agent_name": "Coordinator & Guardrail Agent",
            "role": "Security, PII Masking & Financial Scope Gate",
            "status": "PASSED",
            "badge": "Guardrails",
            "badge_color": "#A78BFA",
            "output_summary": f"Verified safe input in financial domain '{detected_category}'.",
            "details": {
                "guardrail_status": "PASSED",
                "detected_scope": detected_category,
                "masked_sensitive_items": redacted_items or ["None detected"],
            },
        },
        {
            "agent_id": "compliance",
            "agent_name": "RAG Grounding Agent",
            "role": "Local Qdrant Vector Retrieval",
            "status": "COMPLETED",
            "badge": "RAG",
            "badge_color": "#38BDF8",
            "output_summary": f"Retrieved {len(rag_contexts)} chunks from local Qdrant ({rag_stats.get('chunks_count', 0)} total indexed).",
            "details": {
                "vector_db_mode": rag_stats.get("mode", "local"),
                "chunks_indexed": rag_stats.get("chunks_count", 0),
                "embedding_model": rag_stats.get("embedding_model", "BAAI/bge-small-en-v1.5"),
                "retrieved_clauses": rag_hits_summary,
            },
        },
        {
            "agent_id": "extraction",
            "agent_name": "Extraction & Classification Agent",
            "role": f"Gemini API ({model_name})",
            "status": "COMPLETED",
            "badge": "LLM API",
            "badge_color": "#34D399",
            "output_summary": f"Extracted {len(requirements_list)} tailored requirements across {len(categories_breakdown)} categories.",
            "details": {
                "provider": "Google Gemini API",
                "model": model_name,
                "project_title": project_title,
                "moscow_priority": arch_data.get("moscow_priority", "Must Have"),
                "clarification_questions": clarifications,
            },
        },
        {
            "agent_id": "artefact",
            "agent_name": "Risk, Traceability & Artefact Agent",
            "role": "Risk Scoring, RTM & SRS Compiler",
            "status": "COMPLETED",
            "badge": "Artefacts",
            "badge_color": "#FBBF24",
            "output_summary": f"Risk {risk_score}/100 ({risk_level}); compiled {len(traceability_matrix)} traceability groups & PDF.",
            "details": {
                "risk_score_pct": risk_score,
                "risk_level": risk_level,
                "risk_factors": risk_factors,
                "traceability_rows": len(traceability_matrix),
            },
        },
        {
            "agent_id": "sdlc",
            "agent_name": "SDLC Selection Agent",
            "role": "Adaptive Lifecycle & Gate Recommendation",
            "status": "STANDBY",
            "badge": "Gate 1 Pending",
            "badge_color": "#EC4899",
            "output_summary": "Awaiting human approval of requirements before generating ranked SDLC recommendation.",
            "details": {
                "provider": "Google Gemini API",
                "model": model_name,
            },
        },
    ]

    return {
        "is_conversational": False,
        "title": project_title,
        "domain": detected_domain,
        "system_summary": arch_data.get("system_summary", ""),
        "stakeholders": arch_data.get("stakeholders", []),
        "original_requirement": masked_text,
        "masked_pii": redacted_items,
        "fine_tuned_extraction": arch_data,
        "total_requirements_count": len(requirements_list),
        "moscow_priority": arch_data.get("moscow_priority", "Must Have"),
        "priority_justification": arch_data.get("priority_justification", ""),
        "risk_level": risk_level,
        "risk_score_pct": risk_score,
        "risk_color": _risk_color(risk_level),
        "risk_summary": risk_summary,
        "risk_factors": risk_factors,
        "categories": list(categories_breakdown.keys()),
        "categories_breakdown": categories_breakdown,
        "requirements_list": requirements_list,
        "regulations": arch_data.get("applicable_regulations", []),
        "technical_controls": arch_data.get("technical_controls", []),
        "clarification_questions": clarifications,
        "traceability_matrix": traceability_matrix,
        "rag_citations": [str(c.get("citation", "Knowledge Base")) for c in rag_contexts],
        "rag_context": rag_context_str,
        "rag_stats": rag_stats,
        "llm_model": model_name,
        "pipeline_trace": agent_trace,
        "agent_trace": agent_trace,
    }
