# scripts/evaluate_system.py
"""End-to-End Evaluation Benchmark for the Gemini + Local RAG Agentic RE-SDLC System."""

import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.sdlc_agent import recommend_sdlc_structured
from guardrails.input_filter import check_domain_scope, is_safe, mask_sensitive_data
from pdf_generator import generate_pdf_report
from rag.retriever import get_rag_stats, retrieve
from requirements_analysis import analyze_requirement, is_conversational


def run_system_evaluation():
    print("=" * 70)
    print("STARTING AGENT RE-SDLC EVALUATION (GEMINI API + LOCAL RAG)")
    print("=" * 70)
    print("RAG Stats:", get_rag_stats())

    test_cases = [
        {
            "id": "TC-PAY-01",
            "name": "Real-Time Payment Gateway & Fraud Scoring",
            "prompt": "Design a payment gateway supporting card tokenization, real-time fraud risk scoring, 3DS authentication, and immutable audit logging.",
        },
        {
            "id": "TC-LOAN-02",
            "name": "Digital Loan Origination & Credit Underwriting",
            "prompt": "Build a digital loan origination platform with KYC verification, credit bureau integration, automated underwriting, and transparent APR disclosures.",
        },
        {
            "id": "TC-INS-03",
            "name": "Insurance Claims Adjudication Portal",
            "prompt": "Create an automated insurance claims processing and TPA settlement platform with fraud checks and policy validation.",
        },
    ]

    results = []
    for tc in test_cases:
        print(f"\nEvaluating {tc['id']}: {tc['name']}...")
        t0 = time.time()

        rag_hits = retrieve(tc["prompt"], top_k=3)
        analysis = analyze_requirement(tc["prompt"])
        elapsed_re = round(time.time() - t0, 2)

        reqs = analysis.get("requirements_list", [])
        categories = list(analysis.get("categories_breakdown", {}).keys())

        t1 = time.time()
        sdlc = recommend_sdlc_structured(tc["prompt"], rag_context=analysis.get("rag_context", ""))
        elapsed_sdlc = round(time.time() - t1, 2)

        pdf_path = f"output/eval_{tc['id']}.pdf"
        generate_pdf_report(analysis, pdf_path)
        pdf_ok = os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1000

        row = {
            "id": tc["id"],
            "domain_detected": analysis.get("domain"),
            "rag_hits": len(rag_hits),
            "requirements_count": len(reqs),
            "categories_count": len(categories),
            "risk_level": analysis.get("risk_level"),
            "risk_score_pct": analysis.get("risk_score_pct"),
            "clarifications_count": len(analysis.get("clarification_questions", [])),
            "recommended_sdlc": sdlc.get("recommended_model"),
            "pdf_ok": pdf_ok,
            "re_time_sec": elapsed_re,
            "sdlc_time_sec": elapsed_sdlc,
        }
        results.append(row)
        print(" ->", row)

    # Guardrail checks
    _, masked = mask_sensitive_data("Contact test@example.com or phone 9876543210 for account")
    in_dom_finance, _, _ = check_domain_scope("Build a digital banking app with KYC")
    in_dom_nonfin, _, _ = check_domain_scope("Build a recipe cooking blog")
    guardrails_ok = (
        len(masked) >= 2
        and in_dom_finance is True
        and in_dom_nonfin is False
        and is_safe("drop table users") is False
        and is_conversational("hello") is True
    )
    print("\nGuardrails & Intent Checks Passed:", guardrails_ok)

    os.makedirs("output", exist_ok=True)
    summary_path = "output/system_evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"scenarios": results, "guardrails_ok": guardrails_ok}, f, indent=2)
    print(f"Saved evaluation summary to {summary_path}")


if __name__ == "__main__":
    run_system_evaluation()
