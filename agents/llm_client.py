# agents/llm_client.py
"""Lightweight Google Gemini API Client for Agentic RE-SDLC.
Replaces local GPU model loading with configurable Gemini REST API calls.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional
import requests
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_FALLBACK_MODEL = "gemini-3.1-flash-lite-preview"


def get_api_key() -> str:
    load_dotenv(override=False)
    return os.getenv("GEMINI_API_KEY", "").strip()


def get_active_model_name() -> str:
    load_dotenv(override=False)
    return os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def get_fallback_model_name() -> str:
    load_dotenv(override=False)
    return os.getenv("GEMINI_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL).strip() or DEFAULT_FALLBACK_MODEL


def generate_content(
    user_prompt: str,
    system_prompt: str = "",
    *,
    json_mode: bool = False,
    temperature: float = 0.2,
    max_output_tokens: int = 4096,
    timeout: int = 60,
    model_override: Optional[str] = None,
) -> str:
    """Call Google Gemini generateContent REST API with automatic fallback model."""
    api_key = get_api_key()
    if not api_key or api_key == "your_gemini_api_key_here":
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY in .env."
        )

    primary = model_override or get_active_model_name()
    fallback = get_fallback_model_name()
    candidates: List[str] = [primary]
    if fallback and fallback not in candidates:
        candidates.append(fallback)

    payload: Dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}],
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max(max_output_tokens, 512),
        },
    }

    if system_prompt:
        payload["system_instruction"] = {"parts": [{"text": system_prompt}]}

    if json_mode:
        payload["generationConfig"]["responseMimeType"] = "application/json"

    last_error: Optional[Exception] = None
    for model_name in candidates:
        url = f"{GEMINI_API_BASE}/{model_name}:generateContent?key={api_key}"
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                cands = data.get("candidates") or []
                if not cands:
                    last_error = RuntimeError(f"Gemini ({model_name}) returned no candidates: {data}")
                    continue
                parts = (cands[0].get("content") or {}).get("parts") or []
                text = "".join(
                    p.get("text", "")
                    for p in parts
                    if isinstance(p, dict) and not p.get("thought", False)
                ).strip()
                if text:
                    return text
                last_error = RuntimeError(f"Gemini ({model_name}) returned empty text (finishReason={cands[0].get('finishReason')}).")
                continue

            # Retry on transient / unavailable errors with fallback model
            if resp.status_code in (404, 429, 500, 502, 503, 504):
                last_error = RuntimeError(f"Gemini ({model_name}) HTTP {resp.status_code}: {resp.text[:300]}")
                continue
            raise RuntimeError(f"Gemini ({model_name}) HTTP {resp.status_code}: {resp.text[:300]}")
        except requests.RequestException as exc:
            last_error = exc
            continue

    raise RuntimeError(f"All Gemini API attempts failed. Last error: {last_error}")


def parse_json_safely(text: str, fallback: Any = None) -> Any:
    """Extract and parse JSON from an LLM response (handling markdown fences or stray prose)."""
    if not text:
        return fallback

    cleaned = text.strip()
    # Strip markdown code fences if present
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned, re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            pass

    if start != -1:
        for suffix in ["]}]}", "}]}", "]}}", "}}", "}"]:
            try:
                return json.loads(cleaned[start:] + suffix)
            except Exception:
                continue

    return fallback
