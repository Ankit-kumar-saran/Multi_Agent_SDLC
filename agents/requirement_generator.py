# agents/requirement_generator.py
"""Utility to generate a requirement from a user query via Gemini API, RAG, and guardrails."""

from typing import Dict, List
from agents.llm_client import generate_content
from guardrails.input_filter import is_safe as input_is_safe, mask_sensitive_data
from guardrails.output_filter import filter_output
from rag.retriever import retrieve


def _build_prompt(query: str, contexts: List[Dict]) -> str:
    context_block = "\n---\n".join(
        f"[{c.get('citation', 'KB')}] {c.get('text', '')[:500]}"
        for c in contexts
        if c.get("text")
    )
    return (
        "Using the following knowledge base context (if relevant), draft a precise, testable "
        "software requirement addressing the user's financial system request.\n\n"
        f"Context:\n{context_block or 'None'}\n\n"
        f"User request: {query}\n\n"
        "Requirement:"
    )


def generate_requirement(user_query: str) -> str:
    """Generate a safe, grounded requirement string for the given user query."""
    if not input_is_safe(user_query):
        raise ValueError("Input rejected by security guardrails")

    masked_query, _ = mask_sensitive_data(user_query)
    retrieved = retrieve(masked_query, top_k=5)
    prompt = _build_prompt(masked_query, retrieved)

    generated = generate_content(
        user_prompt=prompt,
        system_prompt="You are a financial software requirements engineer. Never include sensitive personal data.",
        temperature=0.2,
        max_output_tokens=512,
    )
    allowed, cleaned, reason = filter_output(generated)
    if not allowed:
        raise ValueError(f"Output rejected by guardrails: {reason}")
    return cleaned
