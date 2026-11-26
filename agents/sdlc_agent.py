# agents/sdlc_agent.py
"""SDLC Recommendation & Workflow Agent for Financial Software Systems.
Uses Google Gemini API + RAG governance context to evaluate project decision factors,
rank suitable SDLC methodologies, generate a tailored SDLC workflow with gates,
and critique human-selected SDLC overrides.
"""

from typing import Any, Dict, List
from agents.llm_client import generate_content, parse_json_safely
from guardrails.input_filter import is_safe as input_is_safe, mask_sensitive_data
from guardrails.output_filter import filter_output


def _retrieve_sdlc_rag(query_hint: str) -> str:
    try:
        from rag.retriever import retrieve
        hits = retrieve(f"SDLC lifecycle model governance compliance risk {query_hint[:150]}", top_k=3)
        return "\n\n".join(
            f"[{h.get('citation', 'Knowledge Base')}]:\n{h.get('text', '').strip()[:500]}"
            for h in hits
            if h.get("text")
        )
    except Exception:
        return ""


def recommend_sdlc_structured(requirement: str, rag_context: str = "") -> Dict[str, Any]:
    """Recommend and rank SDLC methodologies with a tailored workflow for the project."""
    if not input_is_safe(requirement):
        raise ValueError("Input rejected by security guardrails.")

    masked_req, _ = mask_sensitive_data(requirement)
    sdlc_rag = _retrieve_sdlc_rag(masked_req)
    combined_rag = f"{rag_context}\n\n{sdlc_rag}".strip()
    rag_section = f"\n\nGovernance & Policy Context:\n{combined_rag[:2500]}" if combined_rag else ""

    system_prompt = (
        "You are an SDLC Selection & Governance Architect for financial software systems. "
        "Analyze the project's requirement stability, regulatory criticality, security/financial risk, "
        "change frequency, and complexity. Recommend the best-fit SDLC methodology (e.g., Agile-DevSecOps Hybrid, "
        "V-Model with DevSecOps Gates, V-Model, Spiral Model, Waterfall, Iterative/Prototyping, Dual-Track Agile), "
        "provide ranked alternatives with suitability percentages, and outline tailored phases with security/compliance gates. "
        "Output ONLY valid JSON."
    )

    user_prompt = f"""Project Requirements / Scope:
\"\"\"{masked_req[:3500]}\"\"\"
{rag_section}

Output ONLY valid JSON matching this schema:
{{
  "recommended_model": "Best-fit SDLC model name",
  "confidence_pct": 88,
  "decision_factors": {{
    "requirement_stability": "Low | Medium | High",
    "regulatory_criticality": "Low | Medium | High | Critical",
    "security_and_financial_risk": "Low | Medium | High | Critical",
    "expected_change_frequency": "Low | Medium | High"
  }},
  "reasoning": "Clear project-specific rationale explaining why this model is optimal",
  "ranked_recommendations": [
    {{"model": "Top Model", "score_pct": 88, "summary": "Why it ranks #1"}},
    {{"model": "Runner-up Model", "score_pct": 74, "summary": "Where it fits and why it ranked #2"}},
    {{"model": "Third Option", "score_pct": 55, "summary": "Trade-offs for this project"}}
  ],
  "alternatives_considered": [
    {{"model": "Rejected Model 1", "why_not": "Specific reason it is suboptimal for this project"}},
    {{"model": "Rejected Model 2", "why_not": "Specific reason it is suboptimal for this project"}}
  ],
  "tailored_workflow": [
    {{
      "phase": "Phase 1: Requirements & Threat Modeling",
      "activities": "Key engineering activities",
      "validation_gates": "Security, compliance, and human-approval gates"
    }}
  ]
}}"""

    raw = generate_content(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        json_mode=True,
        temperature=0.2,
        max_output_tokens=4096,
        timeout=60,
    )
    allowed, cleaned, reason = filter_output(raw, preserve_json=True)
    if not allowed:
        raise ValueError(f"Output rejected by guardrails: {reason}")

    parsed = parse_json_safely(cleaned, fallback=None)
    if isinstance(parsed, dict) and parsed.get("recommended_model"):
        return parsed

    raise RuntimeError("LLM returned invalid SDLC recommendation JSON.")


def critique_human_sdlc_selection(
    requirement: str,
    chosen_model: str,
    human_feedback: str = "",
    rag_context: str = "",
) -> Dict[str, Any]:
    """Evaluate a human-chosen SDLC model and suggest required engineering & compliance gates."""
    combined_input = f"{requirement} {chosen_model} {human_feedback}"
    if not input_is_safe(combined_input):
        raise ValueError("Input rejected by security guardrails.")

    masked_req, _ = mask_sensitive_data(requirement)
    masked_notes, _ = mask_sensitive_data(human_feedback)
    sdlc_rag = _retrieve_sdlc_rag(f"{chosen_model} {masked_req}")
    combined_rag = f"{rag_context}\n\n{sdlc_rag}".strip()
    rag_section = f"\n\nGovernance & Policy Context:\n{combined_rag[:2000]}" if combined_rag else ""

    system_prompt = (
        "You are a Principal Software Architect & Compliance Governance Advisor. "
        "A human architect has selected an SDLC methodology for their financial software project. "
        "Provide an objective architectural evaluation: strengths, risks/gaps, and concrete engineering gates "
        "needed to deliver safely and compliantly. Output ONLY valid JSON."
    )

    user_prompt = f"""Project Requirement:
"{masked_req[:2500]}"
{rag_section}

Human Architect's Chosen SDLC Model: "{chosen_model}"
Architect Notes: "{masked_notes or 'Selected based on team delivery constraints.'}"

Output ONLY valid JSON:
{{
  "chosen_model": "{chosen_model}",
  "architectural_synthesis": "Technical evaluation of how this model fits the project",
  "strengths_of_choice": "Key advantages of adopting this model here",
  "statutory_risks_and_mitigations": "Specific risks and required engineering/compliance mitigations",
  "governance_cadence": "Recommended cadence, automated security gates, and human approval checkpoints"
}}"""

    raw = generate_content(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        json_mode=True,
        temperature=0.2,
        max_output_tokens=1536,
        timeout=60,
    )
    allowed, cleaned, reason = filter_output(raw, preserve_json=True)
    if not allowed:
        raise ValueError(f"Output rejected by guardrails: {reason}")

    parsed = parse_json_safely(cleaned, fallback=None)
    if isinstance(parsed, dict) and parsed.get("architectural_synthesis"):
        return parsed

    return {
        "chosen_model": chosen_model,
        "architectural_synthesis": f"Adopted {chosen_model} with human architectural oversight.",
        "strengths_of_choice": "Aligns with organizational delivery preferences.",
        "statutory_risks_and_mitigations": "Ensure automated SAST/DAST, traceability, and compliance sign-off gates before release.",
        "governance_cadence": "Regular milestone reviews with security and compliance approval gates.",
    }


def recommend_sdlc(requirement: str) -> str:
    """Return a markdown-formatted summary from structured SDLC recommendation."""
    result = recommend_sdlc_structured(requirement)
    lines = [
        f"**Recommended SDLC Model:** {result.get('recommended_model')}",
        "",
        f"**Reasoning:** {result.get('reasoning')}",
        "",
        "**Alternatives Considered:**",
    ]
    for alt in result.get("alternatives_considered", []):
        lines.append(f"  - **{alt.get('model')}**: {alt.get('why_not')}")
    return "\n".join(lines)


if __name__ == "__main__":
    import json
    sample_req = "Build a fraud detection microservice for card payments with real-time scoring and audit trails."
    print(json.dumps(recommend_sdlc_structured(sample_req), indent=2))
