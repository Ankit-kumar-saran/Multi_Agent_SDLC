# backend/main.py
"""FastAPI backend for the lightweight Gemini + RAG Agentic RE-SDLC system."""

import os
import uuid
from typing import Any, Dict, List, Optional
from fastapi import Body, FastAPI, File, HTTPException, Path, UploadFile
from pydantic import BaseModel

app = FastAPI(title="Agentic RE-SDLC Backend")

SESSIONS: Dict[str, Dict[str, Any]] = {}
REQUIREMENTS: Dict[str, Dict[str, Any]] = {}
SDLC_RECOMMENDATIONS: Dict[str, Dict[str, Any]] = {}
APPROVALS: List[Dict[str, Any]] = []


class CreateSessionResponse(BaseModel):
    session_id: str


class MessageRequest(BaseModel):
    content: str


class ApprovalRequest(BaseModel):
    action: str
    comment: Optional[str] = None


class CritiqueRequest(BaseModel):
    chosen_model: str
    human_feedback: Optional[str] = ""


@app.post("/session", response_model=CreateSessionResponse)
def create_session():
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {"messages": [], "uploaded_files": []}
    return {"session_id": session_id}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = os.path.basename(file.filename or "document.txt")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in (".docx", ".pdf", ".txt", ".md"):
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {ext}")

    save_dir = os.path.join("data", "uploads")
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, filename)

    content = await file.read()
    with open(file_path, "wb") as out:
        out.write(content)

    chunks_ingested = 0
    try:
        from rag.ingest import ingest_file
        chunks_ingested = ingest_file(file_path, category="uploads")
    except Exception as exc:
        print(f"[WARN] Auto-ingestion warning: {exc}")

    return {
        "status": "success",
        "filename": filename,
        "filepath": file_path,
        "chunks_ingested": chunks_ingested,
        "message": "Document saved to data/uploads and indexed in Qdrant.",
    }


@app.post("/session/{session_id}/message")
def post_message(
    session_id: str = Path(..., description="Session identifier"),
    payload: MessageRequest = Body(...),
):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    user_text = payload.content
    SESSIONS[session_id]["messages"].append({"role": "user", "content": user_text})

    pdf_path = "output/requirements_analysis.pdf"
    try:
        from pdf_generator import generate_pdf_report
        from requirements_analysis import analyze_requirement

        analysis_data = analyze_requirement(user_text)

        if analysis_data.get("is_conversational"):
            reply = analysis_data.get("conversational_response", "")
            SESSIONS[session_id]["messages"].append({"role": "assistant", "content": reply})
            return {"response": reply, "requirements": None}

        if analysis_data.get("is_security_violation"):
            reply = analysis_data.get("security_rejection_response", "Blocked by security guardrails.")
            SESSIONS[session_id]["messages"].append({"role": "assistant", "content": reply})
            return {"response": reply, "requirements": None}

        if analysis_data.get("is_out_of_domain"):
            reply = analysis_data.get("domain_rejection_response", "Outside financial domain scope.")
            SESSIONS[session_id]["messages"].append({"role": "assistant", "content": reply})
            return {"response": reply, "requirements": None}

        generate_pdf_report(analysis_data, pdf_path)
        REQUIREMENTS[session_id] = {
            "status": "pending_approval",
            "requirement": analysis_data.get("original_requirement", user_text),
            "analysis": analysis_data,
            "pdf_path": pdf_path,
        }
        req_count = len(analysis_data.get("requirements_list", []))
        response_text = (
            f"Extracted {req_count} requirements for '{analysis_data.get('title', 'Financial System')}' "
            f"(Domain: {analysis_data.get('domain', 'Financial')}, "
            f"Risk: {analysis_data.get('risk_level', 'MEDIUM')}). PDF generated at {pdf_path}."
        )
    except Exception as exc:
        print(f"[ERROR] Requirements analysis failed: {exc}")
        REQUIREMENTS[session_id] = {
            "status": "error",
            "requirement": user_text,
            "analysis": {"error": str(exc), "requirement_statement": user_text},
        }
        response_text = f"Requirements analysis failed: {exc}"

    SESSIONS[session_id]["messages"].append({"role": "assistant", "content": response_text})
    return {
        "response": response_text,
        "requirements": REQUIREMENTS[session_id],
    }


@app.get("/requirements/{session_id}")
def get_requirements(session_id: str = Path(...)):
    if session_id not in REQUIREMENTS:
        raise HTTPException(status_code=404, detail="No requirements found for session")
    return REQUIREMENTS[session_id]


@app.post("/sdlc/{session_id}/recommend")
def generate_sdlc_recommendation(session_id: str = Path(...)):
    if session_id not in REQUIREMENTS:
        raise HTTPException(status_code=400, detail="Requirements must be created before SDLC recommendation")

    req_data = REQUIREMENTS[session_id]
    statement = req_data.get("requirement", "")
    rag_ctx = (req_data.get("analysis") or {}).get("rag_context", "")

    try:
        from agents.sdlc_agent import recommend_sdlc_structured
        sdlc_data = recommend_sdlc_structured(statement, rag_context=rag_ctx)
        SDLC_RECOMMENDATIONS[session_id] = {
            "status": "pending_approval",
            **sdlc_data,
        }
    except Exception as exc:
        print(f"[ERROR] SDLC recommendation failed: {exc}")
        SDLC_RECOMMENDATIONS[session_id] = {
            "status": "error",
            "recommended_model": "",
            "reasoning": str(exc),
            "alternatives_considered": [],
        }

    return SDLC_RECOMMENDATIONS[session_id]


@app.post("/sdlc/{session_id}/critique")
def critique_sdlc_choice(session_id: str = Path(...), payload: CritiqueRequest = Body(...)):
    if session_id not in REQUIREMENTS:
        raise HTTPException(status_code=400, detail="Requirements not found for session")

    req_data = REQUIREMENTS[session_id]
    statement = req_data.get("requirement", "")
    rag_ctx = (req_data.get("analysis") or {}).get("rag_context", "")

    from agents.sdlc_agent import critique_human_sdlc_selection
    return critique_human_sdlc_selection(
        requirement=statement,
        chosen_model=payload.chosen_model,
        human_feedback=payload.human_feedback or "",
        rag_context=rag_ctx,
    )


@app.get("/sdlc/{session_id}")
def get_sdlc(session_id: str = Path(...)):
    if session_id not in SDLC_RECOMMENDATIONS:
        raise HTTPException(status_code=404, detail="No SDLC recommendation for this session")
    return SDLC_RECOMMENDATIONS[session_id]


@app.post("/approve/{session_id}/{item_type}")
def approve_item(
    session_id: str = Path(...),
    item_type: str = Path(..., description="'requirements' or 'sdlc'"),
    payload: ApprovalRequest = Body(...),
):
    record = {
        "session_id": session_id,
        "item_type": item_type,
        "action": payload.action,
        "comment": payload.comment,
    }
    APPROVALS.append(record)

    if item_type == "requirements" and session_id in REQUIREMENTS:
        REQUIREMENTS[session_id]["status"] = payload.action
    elif item_type == "sdlc" and session_id in SDLC_RECOMMENDATIONS:
        SDLC_RECOMMENDATIONS[session_id]["status"] = payload.action

    return {"status": "recorded", "approval": record}


@app.get("/health")
def health_check():
    return {"status": "ok"}
