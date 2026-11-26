# db/database.py
"""Database Layer with SQLAlchemy ORM.
Supports PostgreSQL when available and automatically falls back to local SQLite
so authentication and chat session persistence work cleanly out of the box.
"""

import hashlib
import os
import secrets
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    create_engine,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PG_USER = os.getenv("POSTGRES_USER", "postgres")
PG_PASS = os.getenv("POSTGRES_PASSWORD", "postgres")
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = int(os.getenv("POSTGRES_PORT", "5433"))
PG_DB = os.getenv("POSTGRES_DB", "agentic_db")
SQLITE_PATH = os.getenv("SQLITE_FALLBACK_PATH", str(PROJECT_ROOT / "agentic_re_sdlc.db"))


def _is_port_open(host: str, port: int, timeout: float = 0.4) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


if _is_port_open(PG_HOST, PG_PORT):
    DATABASE_URL = f"postgresql://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=10, max_overflow=20)
    DB_BACKEND = "postgresql"
else:
    DATABASE_URL = f"sqlite:///{SQLITE_PATH}"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    DB_BACKEND = "sqlite"

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

JSONType = JSON().with_variant(JSONB(), "postgresql")


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
    return f"{salt}${key.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, key_hex = stored_hash.split("$", 1)
        test_key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
        return secrets.compare_digest(test_key.hex(), key_hex)
    except Exception:
        return False


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False, default="")
    role = Column(String(100), nullable=False, default="Software Architect")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String(64), primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.created_at")
    state = relationship("SessionStateRecord", back_populates="session", uselist=False, cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    session = relationship("ChatSession", back_populates="messages")


class SessionStateRecord(Base):
    __tablename__ = "session_state_records"

    session_id = Column(String(64), ForeignKey("chat_sessions.id", ondelete="CASCADE"), primary_key=True)
    analysis_json = Column(JSONType, nullable=True)
    sdlc_json = Column(JSONType, nullable=True)
    requirements_approved = Column(Boolean, default=False)
    sdlc_approved = Column(Boolean, default=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    session = relationship("ChatSession", back_populates="state")


def init_db():
    """Create tables and seed a default demo user if not already present."""
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        default_email = "architect@fintech.local"
        existing = db.query(User).filter(User.email == default_email).first()
        if not existing:
            db.add(
                User(
                    email=default_email,
                    password_hash=hash_password("DemoPass123!"),
                    full_name="Lead Financial Architect",
                    role="Principal Architect",
                )
            )
            db.commit()


def register_user(email: str, password: str, full_name: str = "", role: str = "Software Architect") -> Dict[str, Any]:
    email_clean = email.strip().lower()
    if not email_clean or "@" not in email_clean:
        return {"success": False, "error": "Invalid email address format."}
    if len(password) < 6:
        return {"success": False, "error": "Password must be at least 6 characters."}

    with SessionLocal() as db:
        existing = db.query(User).filter(User.email == email_clean).first()
        if existing:
            return {"success": False, "error": "An account with this email already exists."}

        new_user = User(
            email=email_clean,
            password_hash=hash_password(password),
            full_name=full_name.strip() or email_clean.split("@")[0].title(),
            role=role,
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return {
            "success": True,
            "user": {
                "id": new_user.id,
                "email": new_user.email,
                "full_name": new_user.full_name,
                "role": new_user.role,
            },
        }


def authenticate_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    email_clean = email.strip().lower()
    with SessionLocal() as db:
        user = db.query(User).filter(User.email == email_clean).first()
        if not user or not verify_password(password, user.password_hash):
            return None
        return {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
        }


def get_user_chat_sessions(user_id: int) -> List[Dict[str, Any]]:
    with SessionLocal() as db:
        sessions = (
            db.query(ChatSession)
            .filter(ChatSession.user_id == user_id)
            .order_by(ChatSession.updated_at.desc())
            .limit(25)
            .all()
        )
        return [
            {
                "id": s.id,
                "title": s.title,
                "updated_at": s.updated_at.isoformat() if s.updated_at else "",
            }
            for s in sessions
        ]


def load_chat_session(session_id: str, user_id: int) -> Optional[Dict[str, Any]]:
    with SessionLocal() as db:
        session = (
            db.query(ChatSession)
            .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
            .first()
        )
        if not session:
            return None

        messages = [{"role": m.role, "content": m.content} for m in session.messages]
        state = session.state
        return {
            "session_id": session.id,
            "title": session.title,
            "messages": messages,
            "analysis": state.analysis_json if state else None,
            "sdlc": state.sdlc_json if state else None,
            "req_approved": state.requirements_approved if state else False,
            "sdlc_approved": state.sdlc_approved if state else False,
        }


def save_chat_session_state(
    user_id: int,
    session_id: str,
    title: str,
    messages: List[Dict[str, str]],
    analysis: Optional[Dict[str, Any]] = None,
    sdlc: Optional[Dict[str, Any]] = None,
    req_approved: bool = False,
    sdlc_approved: bool = False,
) -> bool:
    now = datetime.now(timezone.utc)
    with SessionLocal() as db:
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if not session:
            session = ChatSession(
                id=session_id,
                user_id=user_id,
                title=title[:250],
                created_at=now,
                updated_at=now,
            )
            db.add(session)
            db.flush()
        else:
            session.title = title[:250]
            session.updated_at = now

        db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
        for msg in messages:
            db.add(
                ChatMessage(
                    session_id=session_id,
                    role=msg.get("role", "user"),
                    content=msg.get("content", ""),
                    created_at=now,
                )
            )

        state = db.query(SessionStateRecord).filter(SessionStateRecord.session_id == session_id).first()
        if not state:
            state = SessionStateRecord(
                session_id=session_id,
                analysis_json=analysis,
                sdlc_json=sdlc,
                requirements_approved=req_approved,
                sdlc_approved=sdlc_approved,
                updated_at=now,
            )
            db.add(state)
        else:
            state.analysis_json = analysis
            state.sdlc_json = sdlc
            state.requirements_approved = req_approved
            state.sdlc_approved = sdlc_approved
            state.updated_at = now

        db.commit()
        return True


def delete_chat_session(session_id: str, user_id: int) -> bool:
    with SessionLocal() as db:
        session = (
            db.query(ChatSession)
            .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
            .first()
        )
        if session:
            db.delete(session)
            db.commit()
            return True
        return False
