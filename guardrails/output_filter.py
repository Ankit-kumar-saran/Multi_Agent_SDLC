# guardrails/output_filter.py
"""Output guardrail – validates and sanitizes LLM responses before returning to users."""

import os
import re
from typing import Tuple

MAX_OUTPUT_LENGTH = int(os.environ.get("OUTPUT_MAX_LENGTH", "50000"))

DISALLOWED_PHRASES = {
    "expose private key",
    "run sudo rm",
    "drop table",
    "drop database",
    "malware",
    "ransomware",
    "root password",
}


def filter_output(llm_output: str, preserve_json: bool = False) -> Tuple[bool, str, str]:
    """Validate and sanitize *llm_output*.

    Returns ``(allowed, cleaned_output, reason)``.
    """
    if not llm_output or not llm_output.strip():
        return False, "", "Empty output"

    cleaned_output = llm_output.strip()
    if len(cleaned_output) > MAX_OUTPUT_LENGTH:
        cleaned_output = cleaned_output[:MAX_OUTPUT_LENGTH]

    for phrase in DISALLOWED_PHRASES:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        cleaned_output = pattern.sub("[REDACTED]", cleaned_output)

    is_json_like = preserve_json or cleaned_output.startswith(("{", "[", "```json"))
    if not is_json_like and not re.search(r"[.!?`\"'\)]$", cleaned_output):
        cleaned_output = cleaned_output.rstrip() + "."

    return True, cleaned_output, ""


if __name__ == "__main__":
    import sys
    out = " ".join(sys.argv[1:])
    ok, cleaned, msg = filter_output(out)
    print({"allowed": ok, "cleaned": cleaned, "reason": msg})
