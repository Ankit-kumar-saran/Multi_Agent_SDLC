#!/usr/bin/env python3
"""End-to-end smoke test for the FastAPI backend."""

import os
import subprocess
import sys
import time
import requests

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUN_BACKEND = os.path.join(PROJECT_ROOT, "scripts", "run_backend.sh")


def wait_for_health(url, timeout=30):
    for _ in range(timeout):
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200 and r.json().get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def main():
    backend_proc = subprocess.Popen(["bash", RUN_BACKEND], cwd=PROJECT_ROOT)
    try:
        if not wait_for_health("http://127.0.0.1:8000/health"):
            print("[ERROR] Backend health check failed.")
            sys.exit(1)
        print("[INFO] Backend is healthy.")

        sess_resp = requests.post("http://127.0.0.1:8000/session", timeout=10)
        sess_resp.raise_for_status()
        session_id = sess_resp.json()["session_id"]
        print(f"[INFO] Created session {session_id}")

        msg_resp = requests.post(
            f"http://127.0.0.1:8000/session/{session_id}/message",
            json={"content": "Design a digital banking payment service with fraud detection and KYC checks"},
            timeout=90,
        )
        msg_resp.raise_for_status()
        print("[INFO] Message endpoint:", msg_resp.json().get("response"))

        req_resp = requests.get(f"http://127.0.0.1:8000/requirements/{session_id}", timeout=10)
        req_resp.raise_for_status()
        print("[INFO] Requirements status:", req_resp.json().get("status"))

        sdlc_resp = requests.post(f"http://127.0.0.1:8000/sdlc/{session_id}/recommend", timeout=90)
        sdlc_resp.raise_for_status()
        print("[INFO] SDLC recommended:", sdlc_resp.json().get("recommended_model"))

        print("[SUCCESS] Smoke test completed.")
    finally:
        backend_proc.terminate()
        backend_proc.wait()
        print("[INFO] Backend process stopped.")


if __name__ == "__main__":
    main()
