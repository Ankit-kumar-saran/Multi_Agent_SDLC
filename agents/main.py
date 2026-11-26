# agents/main.py
"""CLI entry point for requirement generation, analysis, and PDF compilation."""

import sys
from guardrails.input_filter import is_safe as input_is_safe, check_domain_scope
from requirements_analysis import analyze_requirement
from pdf_generator import generate_pdf_report


def main():
    if len(sys.argv) < 2:
        print('Usage: python -m agents.main "<financial system request>"')
        sys.exit(1)

    user_query = " ".join(sys.argv[1:])

    if not input_is_safe(user_query):
        print("[ERROR] Input rejected by security guardrails.")
        sys.exit(1)

    in_domain, tag, clarification = check_domain_scope(user_query)
    if not in_domain:
        print(f"[ERROR] Out of financial scope ({tag}): {clarification}")
        sys.exit(1)

    analysis = analyze_requirement(user_query)
    pdf_path = "output/requirements_analysis.pdf"
    generate_pdf_report(analysis, pdf_path)

    print(f"[SUCCESS] Analysis complete for: {analysis.get('title')}")
    print(f"          - Domain: {analysis.get('domain')}")
    print(f"          - Requirements: {len(analysis.get('requirements_list', []))}")
    print(f"          - Risk: {analysis.get('risk_level')} ({analysis.get('risk_score_pct')}/100)")
    print(f"          - PDF Report: {pdf_path}")


if __name__ == "__main__":
    main()
