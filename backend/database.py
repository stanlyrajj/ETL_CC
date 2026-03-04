"""
database.py — PostgreSQL async models + CRUD helpers.
All tables in one place. Import `db` singleton everywhere.

New vs previous version:
  - chat_sessions table (multi-session sidebar)
  - chat_messages table (per-message history with role)
  - Papers.processing_stage (fine-grained pipeline state)
  - Startup health check
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey, Index, Integer,
    String, Text, delete, func, select, update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from config import cfg

logger = logging.getLogger(__name__)


# ── Base ──────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── Models ────────────────────────────────────────────────────────────────────

class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    paper_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(20), nullable=False)
    title: Mapped[Optional[str]] = mapped_column(Text)
    authors: Mapped[Optional[List]] = mapped_column(JSONB, default=list)
    abstract: Mapped[Optional[str]] = mapped_column(Text)
    url: Mapped[Optional[str]] = mapped_column(Text)
    file_path: Mapped[Optional[str]] = mapped_column(Text)

    # Pipeline state machine
    # States: pending → downloading → downloaded → processing →
    #         processed → failed_download → failed_processing
    pipeline_stage: Mapped[str] = mapped_column(String(30), default="pending", index=True)
    download_error: Mapped[Optional[str]] = mapped_column(Text)
    processing_error: Mapped[Optional[str]] = mapped_column(Text)

    published_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    downloaded_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    section_count: Mapped[int] = mapped_column(Integer, default=0)
    extra_metadata: Mapped[Optional[Dict]] = mapped_column(JSONB, default=dict)

    # Relationships
    sections: Mapped[List["PaperSection"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )
    chunks: Mapped[List["PaperChunk"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )
    chat_sessions: Mapped[List["ChatSession"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )
    social_contents: Mapped[List["SocialContent"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )
    processing_logs: Mapped[List["ProcessingLog"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )


class PaperSection(Base):
    __tablename__ = "paper_sections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    paper_id: Mapped[str] = mapped_column(
        String(100), ForeignKey("papers.paper_id", ondelete="CASCADE"), nullable=False, index=True
    )
    section_type: Mapped[str] = mapped_column(String(50), nullable=False)
    section_title: Mapped[Optional[str]] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    section_order: Mapped[int] = mapped_column(Integer, default=0)
    word_count: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    paper: Mapped["Paper"] = relationship(back_populates="sections")

    __table_args__ = (
        Index("ix_paper_sections_paper_type", "paper_id", "section_type"),
    )


class PaperChunk(Base):
    __tablename__ = "paper_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    paper_id: Mapped[str] = mapped_column(
        String(100), ForeignKey("papers.paper_id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    section_type: Mapped[Optional[str]] = mapped_column(String(50))
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_id: Mapped[Optional[str]] = mapped_column(String(200))
    char_count: Mapped[Optional[int]] = mapped_column(Integer)
    chunk_metadata: Mapped[Optional[Dict]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    paper: Mapped["Paper"] = relationship(back_populates="chunks")


class ChatSession(Base):
    """One session per topic/paper — the multi-chat sidebar."""
    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    paper_id: Mapped[str] = mapped_column(
        String(100), ForeignKey("papers.paper_id", ondelete="CASCADE"), nullable=False, index=True
    )
    topic: Mapped[Optional[str]] = mapped_column(Text)         # user's original search query
    level: Mapped[str] = mapped_column(String(20), default="beginner")  # current teaching level
    title: Mapped[Optional[str]] = mapped_column(Text)         # auto-generated session title
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_active_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    paper: Mapped["Paper"] = relationship(back_populates="chat_sessions")
    messages: Mapped[List["ChatMessage"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.created_at"
    )


class ChatMessage(Base):
    """Individual message in a chat session."""
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(100), ForeignKey("chat_sessions.session_id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    level: Mapped[Optional[str]] = mapped_column(String(20))       # teaching level at time of message
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    session: Mapped["ChatSession"] = relationship(back_populates="messages")


class SocialContent(Base):
    __tablename__ = "social_content"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    paper_id: Mapped[str] = mapped_column(
        String(100), ForeignKey("papers.paper_id", ondelete="CASCADE"), nullable=False, index=True
    )
    platform: Mapped[str] = mapped_column(String(20), nullable=False)
    content_type: Mapped[str] = mapped_column(String(30), nullable=False)  # thread | post | carousel
    content: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="draft")
    style_config: Mapped[Optional[Dict]] = mapped_column(JSONB, default=dict)
    hashtags: Mapped[Optional[List]] = mapped_column(JSONB, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    paper: Mapped["Paper"] = relationship(back_populates="social_contents")


class ProcessingLog(Base):
    __tablename__ = "processing_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    paper_id: Mapped[str] = mapped_column(
        String(100), ForeignKey("papers.paper_id", ondelete="CASCADE"), nullable=False, index=True
    )
    stage: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    message: Mapped[Optional[str]] = mapped_column(Text)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    paper: Mapped["Paper"] = relationship(back_populates="processing_logs")


# ── Database class ────────────────────────────────────────────────────────────

class Database:
    def __init__(self):
        self._engine = None
        self._session_factory = None

    async def init(self):
        self._engine = create_async_engine(
            cfg.DATABASE_URL,
            echo=False,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
        self._session_factory = async_sessionmaker(
            self._engine, expire_on_commit=False, class_=AsyncSession
        )
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database ready.")

    async def close(self):
        if self._engine:
            await self._engine.dispose()

    @asynccontextmanager
    async def session(self):
        async with self._session_factory() as sess:
            try:
                yield sess
                await sess.commit()
            except Exception:
                await sess.rollback()
                raise

    # ── Paper CRUD ────────────────────────────────────────────────────────

    async def upsert_paper(self, session: AsyncSession, data: Dict[str, Any]) -> Paper:
        result = await session.execute(select(Paper).where(Paper.paper_id == data["paper_id"]))
        paper = result.scalar_one_or_none()
        if paper:
            for k, v in data.items():
                if hasattr(paper, k):
                    setattr(paper, k, v)
        else:
            paper = Paper(**data)
            session.add(paper)
        await session.flush()
        return paper

    async def get_paper(self, session: AsyncSession, paper_id: str) -> Optional[Paper]:
        result = await session.execute(select(Paper).where(Paper.paper_id == paper_id))
        return result.scalar_one_or_none()

    async def list_papers(
        self,
        session: AsyncSession,
        stage: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> List[Paper]:
        q = select(Paper)
        if stage:
            q = q.where(Paper.pipeline_stage == stage)
        if source:
            q = q.where(Paper.source == source)
        q = q.order_by(Paper.created_at.desc()).limit(limit).offset(offset)
        result = await session.execute(q)
        return list(result.scalars().all())

    async def set_stage(
        self,
        session: AsyncSession,
        paper_id: str,
        stage: str,
        error: Optional[str] = None,
    ):
        values: Dict[str, Any] = {"pipeline_stage": stage}
        if stage == "downloaded":
            values["downloaded_at"] = func.now()
        elif stage == "processed":
            values["processed_at"] = func.now()
        if error:
            if "download" in stage:
                values["download_error"] = error
            else:
                values["processing_error"] = error
        await session.execute(update(Paper).where(Paper.paper_id == paper_id).values(**values))

    async def delete_paper(self, session: AsyncSession, paper_id: str) -> bool:
        result = await session.execute(delete(Paper).where(Paper.paper_id == paper_id))
        return result.rowcount > 0

    # ── Sections & Chunks ─────────────────────────────────────────────────

    async def save_sections(self, session: AsyncSession, paper_id: str, sections: List[Dict]) -> int:
        await session.execute(delete(PaperSection).where(PaperSection.paper_id == paper_id))
        objs = [PaperSection(paper_id=paper_id, **s) for s in sections]
        session.add_all(objs)
        await session.flush()
        await session.execute(
            update(Paper).where(Paper.paper_id == paper_id).values(section_count=len(objs))
        )
        return len(objs)

    async def save_chunks(self, session: AsyncSession, paper_id: str, chunks: List[Dict]) -> int:
        await session.execute(delete(PaperChunk).where(PaperChunk.paper_id == paper_id))
        objs = [PaperChunk(paper_id=paper_id, **c) for c in chunks]
        session.add_all(objs)
        await session.flush()
        await session.execute(
            update(Paper).where(Paper.paper_id == paper_id).values(chunk_count=len(objs))
        )
        return len(objs)

    async def get_sections(self, session: AsyncSession, paper_id: str) -> List[PaperSection]:
        result = await session.execute(
            select(PaperSection)
            .where(PaperSection.paper_id == paper_id)
            .order_by(PaperSection.section_order)
        )
        return list(result.scalars().all())

    # ── Chat sessions ─────────────────────────────────────────────────────

    async def create_session(
        self,
        session: AsyncSession,
        session_id: str,
        paper_id: str,
        topic: str,
        level: str = "beginner",
    ) -> ChatSession:
        obj = ChatSession(
            session_id=session_id,
            paper_id=paper_id,
            topic=topic,
            level=level,
            title=topic[:80],
        )
        session.add(obj)
        await session.flush()
        return obj

    async def get_session(self, session: AsyncSession, session_id: str) -> Optional[ChatSession]:
        result = await session.execute(
            select(ChatSession).where(ChatSession.session_id == session_id)
        )
        return result.scalar_one_or_none()

    async def list_sessions(self, session: AsyncSession, limit: int = 50) -> List[ChatSession]:
        result = await session.execute(
            select(ChatSession)
            .order_by(ChatSession.last_active_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def add_message(
        self,
        session: AsyncSession,
        session_id: str,
        role: str,
        content: str,
        level: Optional[str] = None,
    ) -> ChatMessage:
        msg = ChatMessage(session_id=session_id, role=role, content=content, level=level)
        session.add(msg)
        # Update last_active_at on the session
        await session.execute(
            update(ChatSession)
            .where(ChatSession.session_id == session_id)
            .values(last_active_at=func.now())
        )
        await session.flush()
        return msg

    async def get_messages(self, session: AsyncSession, session_id: str) -> List[ChatMessage]:
        result = await session.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at)
        )
        return list(result.scalars().all())

    async def update_session_level(
        self, session: AsyncSession, session_id: str, level: str
    ):
        await session.execute(
            update(ChatSession)
            .where(ChatSession.session_id == session_id)
            .values(level=level)
        )

    # ── Social content ────────────────────────────────────────────────────

    async def save_social(
        self,
        session: AsyncSession,
        paper_id: str,
        platform: str,
        content_type: str,
        content: str,
        hashtags: Optional[List[str]] = None,
        style_config: Optional[Dict] = None,
    ) -> SocialContent:
        obj = SocialContent(
            paper_id=paper_id,
            platform=platform,
            content_type=content_type,
            content=content,
            hashtags=hashtags or [],
            style_config=style_config or {},
        )
        session.add(obj)
        await session.flush()
        return obj

    async def list_social(
        self,
        session: AsyncSession,
        paper_id: Optional[str] = None,
        platform: Optional[str] = None,
    ) -> List[SocialContent]:
        q = select(SocialContent)
        if paper_id:
            q = q.where(SocialContent.paper_id == paper_id)
        if platform:
            q = q.where(SocialContent.platform == platform)
        q = q.order_by(SocialContent.created_at.desc())
        result = await session.execute(q)
        return list(result.scalars().all())

    # ── Logs ──────────────────────────────────────────────────────────────

    async def log(
        self,
        session: AsyncSession,
        paper_id: str,
        stage: str,
        status: str,
        message: Optional[str] = None,
        duration: Optional[float] = None,
    ):
        session.add(ProcessingLog(
            paper_id=paper_id, stage=stage, status=status,
            message=message, duration_seconds=duration,
        ))
        await session.flush()


# ── Singleton ─────────────────────────────────────────────────────────────────

db = Database()