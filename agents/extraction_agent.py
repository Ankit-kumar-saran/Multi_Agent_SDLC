# agents/extraction_agent.py
"""Extraction, Classification & Clarification Agent for Financial Software Systems.
Uses Google Gemini API via agents.llm_client, protected by Zero-Trust guardrails
and sensitive-data masking, and grounded in local Qdrant RAG context.
"""

from typing import Any, Dict, List
from agents.llm_client import generate_content, parse_json_safely
from guardrails.input_filter import is_safe as input_is_safe, mask_sensitive_data
from guardrails.output_filter import filter_output


def extract_and_classify(dialogue: str) -> Dict[str, Any]:
    """Extract a single structured requirement from stakeholder dialogue."""
    if not input_is_safe(dialogue):
        raise ValueError("Input rejected by security guardrails.")

    masked_dialogue, _ = mask_sensitive_data(dialogue)

    system_prompt = (
        "You are a Requirements Engineering & Classification Agent for financial software systems. "
        "Extract a structured requirement from the stakeholder input with MoSCoW prioritization, "
        "multi-label classification, acceptance criteria, and confidence scoring. "
        "Output ONLY valid JSON."
    )

    user_prompt = f"""Extract a structured requirement from this input:
"{masked_dialogue[:4000]}"

Return JSON with this exact schema:
{{
  "req_id": "REQ-01",
  "statement": "Clear, testable requirement statement",
  "categories": ["Functional", "Security"],
  "priority": "Must Have",
  "priority_justification": "Why this MoSCoW priority applies",
  "regulatory_citations": ["Relevant regulation, standard, or policy"],
  "acceptance_criteria": "Verifiable test criteria",
  "dependencies": ["Dependency 1"],
  "confidence_score": 0.92
}}"""

    raw = generate_content(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        json_mode=True,
        temperature=0.1,
        max_output_tokens=1024,
    )
    allowed, cleaned, reason = filter_output(raw, preserve_json=True)
    if not allowed:
        raise ValueError(f"Output rejected by guardrails: {reason}")

    fallback = {
        "req_id": "REQ-01",
        "statement": masked_dialogue[:250],
        "categories": ["Functional"],
        "priority": "Must Have",
        "priority_justification": "Core stakeholder requirement.",
        "regulatory_citations": [],
        "acceptance_criteria": "Passes functional and security verification.",
        "dependencies": [],
        "confidence_score": 0.75,
    }
    parsed = parse_json_safely(cleaned, fallback=fallback)
    return parsed if isinstance(parsed, dict) else fallback


def extract_system_architecture(user_input: str, rag_context: str = "") -> Dict[str, Any]:
    """Dynamically extract a complete, project-specific SRS blueprint using Gemini + RAG."""
    if not input_is_safe(user_input):
        raise ValueError("Input rejected by security guardrails.")

    masked_input, redacted_items = mask_sensitive_data(user_input)
    rag_section = (
        f"\n\nRetrieved Knowledge Base & Policy Context (use and cite when relevant):\n{rag_context[:4000]}"
        if rag_context
        else "\n\nNote: No external RAG documents matched; ground citations in well-established financial/security standards only where genuinely applicable."
    )

    system_prompt = (
        "You are a Principal Financial Software Requirements Architect. "
        "Analyze the stakeholder's request and retrieved RAG evidence to generate a tailored, "
        "accurate, and non-bloated Software Requirements Specification (SRS) strictly relevant to the user's requested financial system. "
        "Do NOT invent unrelated features or force-fit irrelevant regulations. "
        "Include functional requirements, non-functional requirements (security, privacy, compliance, performance, availability, auditability, integration), "
        "risk assessment, applicable regulations/standards, technical controls, and context-aware clarification questions for ambiguities. "
        "Output ONLY valid JSON."
    )

    user_prompt = f"""Stakeholder Project Request:
\"\"\"{masked_input[:5000]}\"\"\"
{rag_section}

Generate a complete, project-specific SRS specification in strict JSON matching this schema:
{{
  "project_title": "Concise specific title for this system",
  "domain": "Specific financial sub-domain (e.g., Digital Lending, Payment Processing, Fraud Detection, Core Banking, Insurance Claims, Financial Analytics, KYC Onboarding)",
  "system_summary": "2-3 sentence architectural summary tailored to this project",
  "stakeholders": ["Primary stakeholder roles"],
  "moscow_priority": "Must Have | Should Have | Could Have",
  "priority_justification": "Rationale for overall baseline priority",
  "risk_level": "CRITICAL | HIGH | MEDIUM | LOW",
  "risk_score_pct": 75,
  "risk_summary": "Concise risk assessment rationale based on financial impact, data sensitivity, and regulatory exposure",
  "risk_factors": ["Specific risk factor 1", "Specific risk factor 2"],
  "applicable_regulations": [
    {{
      "regulation": "Applicable regulation or standard name",
      "relevance": "Why it applies to this specific project",
      "statutory_citation": "Section/clause or retrieved KB document filename"
    }}
  ],
  "requirements": [
    {{
      "id": "REQ-01",
      "title": "Specific requirement title",
      "category": "Choose appropriate category (e.g. Functional Core, Security & Privacy, Compliance & Audit, Performance & Availability, Data & Integration, User Interface & Experience)",
      "description": "Clear, verifiable engineering specification tailored to this project",
      "priority": "Must Have | Should Have | Could Have",
      "regulation": "Specific standard/regulation or 'Internal Business Rule'",
      "acceptance_criteria": "Measurable, testable verification criteria",
      "confidence_score": 0.90
    }}
  ],
  "technical_controls": [
    {{
      "control_id": "CTRL-01",
      "title": "Control title",
      "specification": "Concrete technical control specification",
      "verification": "How this control is verified/tested"
    }}
  ],
  "clarification_questions": [
    "Specific question to resolve ambiguity, missing constraint, or stakeholder conflict"
  ]
}}

Generate 12 to 18 high-value, non-redundant requirements covering both Functional and Non-Functional dimensions relevant to this specific project. Ensure every item is grounded in the request and RAG context."""

    raw = generate_content(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        json_mode=True,
        temperature=0.2,
        max_output_tokens=8192,
        timeout=75,
    )

    allowed, cleaned, reason = filter_output(raw, preserve_json=True)
    if not allowed:
        raise ValueError(f"Output rejected by guardrails: {reason}")

    parsed = parse_json_safely(cleaned, fallback=None)
    if isinstance(parsed, dict) and parsed.get("requirements"):
        normalized = _normalize_extracted_data(parsed, masked_input)
        normalized["masked_pii"] = redacted_items
        return normalized

    raise RuntimeError("LLM returned invalid or incomplete SRS JSON structure.")


def _normalize_extracted_data(data: Dict[str, Any], user_input: str) -> Dict[str, Any]:
    """Normalize LLM-returned JSON into a consistent internal schema."""
    lowered = {str(k).lower(): v for k, v in data.items()}

    raw_reqs = (
        lowered.get("requirements")
        or lowered.get("functional_requirements")
        or lowered.get("features")
        or []
    )
    norm_reqs: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_reqs):
        if not isinstance(item, dict):
            continue
        i_low = {str(k).lower(): v for k, v in item.items()}
        title = str(i_low.get("title") or i_low.get("name") or f"Requirement {idx + 1}").strip()
        desc = str(i_low.get("description") or i_low.get("specification") or title).strip()
        cat = str(i_low.get("category") or "Functional").strip()
        prio = str(i_low.get("priority") or "Must Have").strip()
        reg = str(i_low.get("regulation") or "Standard Engineering Practice").strip()
        crit = str(i_low.get("acceptance_criteria") or "Passes automated verification suite.").strip()
        conf = i_low.get("confidence_score", 0.85)
        try:
            conf = round(max(0.0, min(1.0, float(conf))), 2)
        except (TypeError, ValueError):
            conf = 0.85

        norm_reqs.append(
            {
                "id": str(i_low.get("id") or f"REQ-{idx + 1:02d}"),
                "title": title,
                "category": cat,
                "description": desc,
                "priority": prio,
                "regulation": reg,
                "acceptance_criteria": crit,
                "confidence_score": conf,
            }
        )

    risk_score = lowered.get("risk_score_pct", 65)
    try:
        risk_score = max(0, min(100, int(float(risk_score))))
    except (TypeError, ValueError):
        risk_score = 65

    risk_level = str(lowered.get("risk_level") or "").upper().strip()
    if risk_level not in {"CRITICAL", "HIGH", "MEDIUM", "LOW"}:
        if risk_score >= 85:
            risk_level = "CRITICAL"
        elif risk_score >= 68:
            risk_level = "HIGH"
        elif risk_score >= 45:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

    regulations = lowered.get("applicable_regulations") or lowered.get("regulations") or []
    norm_regs: List[Dict[str, str]] = []
    for r in regulations:
        if isinstance(r, dict):
            norm_regs.append(
                {
                    "regulation": str(r.get("regulation", "Applicable Standard")),
                    "relevance": str(r.get("relevance", "Governance & compliance alignment")),
                    "statutory_citation": str(r.get("statutory_citation", "Referenced Policy / Standard")),
                }
            )
        elif isinstance(r, str) and r.strip():
            norm_regs.append(
                {
                    "regulation": r.strip(),
                    "relevance": "Applicable regulatory or policy standard",
                    "statutory_citation": r.strip(),
                }
            )

    controls = lowered.get("technical_controls") or []
    norm_controls: List[Dict[str, str]] = []
    for idx, c in enumerate(controls):
        if isinstance(c, dict):
            norm_controls.append(
                {
                    "control_id": str(c.get("control_id") or f"CTRL-{idx + 1:02d}"),
                    "title": str(c.get("title") or f"Control {idx + 1}"),
                    "specification": str(c.get("specification") or ""),
                    "verification": str(c.get("verification") or "Automated test & audit verification"),
                }
            )

    clarifications = lowered.get("clarification_questions") or []
    if not isinstance(clarifications, list):
        clarifications = []
    clarifications = [str(q).strip() for q in clarifications if str(q).strip()]

    risk_factors = lowered.get("risk_factors") or []
    if not isinstance(risk_factors, list):
        risk_factors = []

    stakeholders = lowered.get("stakeholders") or ["Product Owner", "Software Architect", "Compliance Officer", "Security Team"]
    if not isinstance(stakeholders, list):
        stakeholders = [str(stakeholders)]

    return {
        "project_title": str(lowered.get("project_title") or f"SRS: {user_input[:50].strip()}").strip(),
        "domain": str(lowered.get("domain") or "Financial Software Systems").strip(),
        "system_summary": str(lowered.get("system_summary") or f"Architecture specification for {user_input[:180]}").strip(),
        "stakeholders": [str(s) for s in stakeholders],
        "moscow_priority": str(lowered.get("moscow_priority") or "Must Have").strip(),
        "priority_justification": str(lowered.get("priority_justification") or "Derived from functional criticality and compliance scope.").strip(),
        "risk_level": risk_level,
        "risk_score_pct": risk_score,
        "risk_summary": str(lowered.get("risk_summary") or "Evaluated from operational, security, and regulatory risk dimensions.").strip(),
        "risk_factors": [str(f) for f in risk_factors],
        "applicable_regulations": norm_regs,
        "requirements": norm_reqs,
        "technical_controls": norm_controls,
        "clarification_questions": clarifications,
    }
