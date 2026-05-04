# database.py

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

from sqlalchemy import (
    JSON, DateTime, Float, ForeignKey, Integer, String, Text, delete, select,
    update,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from config import cfg


# ── Base ──────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── Models ────────────────────────────────────────────────────────────────────

class Paper(Base):
    __tablename__ = "papers"

    paper_id:       Mapped[str]           = mapped_column(String, primary_key=True)
    source:         Mapped[str]           = mapped_column(String)
    title:          Mapped[str | None]    = mapped_column(Text)
    abstract:       Mapped[str | None]    = mapped_column(Text)
    authors:        Mapped[list | None]   = mapped_column(JSON)
    url:            Mapped[str | None]    = mapped_column(Text)
    file_path:      Mapped[str | None]    = mapped_column(Text)
    pipeline_stage: Mapped[str]           = mapped_column(String, default="pending")
    chunk_count:    Mapped[int]           = mapped_column(Integer, default=0)
    error_message:  Mapped[str | None]    = mapped_column(Text)
    topic:          Mapped[str | None]    = mapped_column(Text)
    created_at:     Mapped[datetime]      = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    processed_at:   Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    chat_sessions: Mapped[list["ChatSession"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )
    social_content: Mapped[list["SocialContent"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )
    processing_logs: Mapped[list["ProcessingLog"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    session_id:     Mapped[str]           = mapped_column(String, primary_key=True)
    paper_id:       Mapped[str]           = mapped_column(ForeignKey("papers.paper_id", ondelete="CASCADE"))
    topic:          Mapped[str | None]    = mapped_column(Text)
    level:          Mapped[str | None]    = mapped_column(String)
    mode:           Mapped[str]           = mapped_column(String, default="standard")  # standard | study | technical
    title:          Mapped[str | None]    = mapped_column(Text)
    last_active_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at:     Mapped[datetime]      = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    paper:    Mapped["Paper"]              = relationship(back_populates="chat_sessions")
    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id:         Mapped[int]        = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str]        = mapped_column(ForeignKey("chat_sessions.session_id", ondelete="CASCADE"))
    role:       Mapped[str]        = mapped_column(String)
    content:    Mapped[str]        = mapped_column(Text)
    level:      Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime]   = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    session: Mapped["ChatSession"] = relationship(back_populates="messages")


class SocialContent(Base):
    __tablename__ = "social_content"

    id:           Mapped[int]           = mapped_column(Integer, primary_key=True, autoincrement=True)
    paper_id:     Mapped[str]           = mapped_column(ForeignKey("papers.paper_id", ondelete="CASCADE"))
    platform:     Mapped[str]           = mapped_column(String)
    content_type: Mapped[str | None]    = mapped_column(String)
    content:      Mapped[str]           = mapped_column(Text)
    hashtags:     Mapped[list | None]   = mapped_column(JSON)
    created_at:   Mapped[datetime]      = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    paper: Mapped["Paper"] = relationship(back_populates="social_content")


class ProcessingLog(Base):
    __tablename__ = "processing_logs"

    id:               Mapped[int]        = mapped_column(Integer, primary_key=True, autoincrement=True)
    paper_id:         Mapped[str]        = mapped_column(ForeignKey("papers.paper_id", ondelete="CASCADE"))
    stage:            Mapped[str]        = mapped_column(String)
    status:           Mapped[str]        = mapped_column(String)
    message:          Mapped[str | None] = mapped_column(Text)
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    created_at:       Mapped[datetime]   = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    paper: Mapped["Paper"] = relationship(back_populates="processing_logs")


class GeneratedCache(Base):
    """
    Persistent cache for study and technical generated content.

    cache_type : 'study_outline'  | 'study_section' | 'study_flashcards'
                 | 'technical_section'
    cache_key  : unique identifier within the type:
                   study_outline    → paper_id
                   study_section    → "{paper_id}::{section_title}"
                   study_flashcards → paper_id
                   technical_section→ "{paper_id}::{section_key}"
    level      : the teaching level used when content was generated (informational only)
    content    : markdown / JSON text — whatever the generator returned
    """

    __tablename__ = "generated_cache"

    id:         Mapped[int]        = mapped_column(Integer, primary_key=True, autoincrement=True)
    paper_id:   Mapped[str]        = mapped_column(ForeignKey("papers.paper_id", ondelete="CASCADE"))
    cache_type: Mapped[str]        = mapped_column(String, index=True)
    cache_key:  Mapped[str]        = mapped_column(String, index=True)
    level:      Mapped[str | None] = mapped_column(String)
    content:    Mapped[str]        = mapped_column(Text)
    created_at: Mapped[datetime]   = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime]   = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    paper: Mapped["Paper"] = relationship()


# ── Database class ────────────────────────────────────────────────────────────

class Database:
    def __init__(self):
        self._engine        = None
        self._session_maker = None

    async def init(self):
        """Create the async engine and all tables."""
        from sqlalchemy import event

        if self._engine is not None:
            await self._engine.dispose()

        self._engine = create_async_engine(cfg.DATABASE_URL, echo=False)

        @event.listens_for(self._engine.sync_engine, "connect")
        def _set_sqlite_pragma(dbapi_conn, _conn_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        self._session_maker = async_sessionmaker(
            self._engine, expire_on_commit=False
        )
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Migration-safe: add mode column to existing databases that predate it
        await self._migrate_add_mode_column()

    async def _migrate_add_mode_column(self) -> None:
        """
        Add the mode column to chat_sessions if it does not exist.
        SQLite does not support IF NOT EXISTS on ALTER TABLE, so we
        check the column list first.
        """
        async with self._engine.begin() as conn:
            result = await conn.execute(
                # SQLite PRAGMA returns one row per column
                __import__("sqlalchemy").text("PRAGMA table_info(chat_sessions)")
            )
            columns = [row[1] for row in result.fetchall()]
            if "mode" not in columns:
                await conn.execute(
                    __import__("sqlalchemy").text(
                        "ALTER TABLE chat_sessions ADD COLUMN mode VARCHAR DEFAULT 'standard'"
                    )
                )

    async def close(self):
        if self._engine:
            await self._engine.dispose()

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        async with self._session_maker() as sess:
            try:
                yield sess
                await sess.commit()
            except Exception:
                await sess.rollback()
                raise

    # ── Paper helpers ─────────────────────────────────────────────────────────

    async def upsert_paper(self, sess: AsyncSession, data: dict[str, Any]) -> Paper:
        paper = await self.get_paper(sess, data["paper_id"])
        if paper is None:
            paper = Paper(**data)
            sess.add(paper)
        else:
            for key, value in data.items():
                setattr(paper, key, value)
        return paper

    async def get_paper(self, sess: AsyncSession, paper_id: str) -> Paper | None:
        result = await sess.execute(select(Paper).where(Paper.paper_id == paper_id))
        return result.scalar_one_or_none()

    async def list_papers(
        self,
        sess:   AsyncSession,
        stage:  str | None = None,
        source: str | None = None,
        limit:  int = 200,
        offset: int = 0,
    ) -> list[Paper]:
        query = select(Paper)
        if stage:
            query = query.where(Paper.pipeline_stage == stage)
        if source:
            query = query.where(Paper.source == source)
        query = query.order_by(Paper.created_at.desc()).limit(limit).offset(offset)
        result = await sess.execute(query)
        return list(result.scalars().all())

    async def set_stage(
        self,
        sess:     AsyncSession,
        paper_id: str,
        stage:    str,
        error:    str | None = None,
    ) -> None:
        values: dict[str, Any] = {"pipeline_stage": stage, "error_message": error}
        if stage == "processed":
            values["processed_at"] = datetime.now(timezone.utc)
        else:
            values["processed_at"] = None
        await sess.execute(
            update(Paper).where(Paper.paper_id == paper_id).values(**values)
        )

    async def update_chunk_count(
        self, sess: AsyncSession, paper_id: str, count: int
    ) -> None:
        await sess.execute(
            update(Paper)
            .where(Paper.paper_id == paper_id)
            .values(chunk_count=count)
        )

    async def delete_paper(self, sess: AsyncSession, paper_id: str) -> bool:
        result = await sess.execute(
            delete(Paper).where(Paper.paper_id == paper_id)
        )
        return result.rowcount > 0

    # ── Chat session helpers ──────────────────────────────────────────────────

    async def create_session(
        self,
        sess:       AsyncSession,
        session_id: str,
        paper_id:   str,
        topic:      str,
        level:      str,
        mode:       str = "standard",
    ) -> "ChatSession":
        chat_session = ChatSession(
            session_id=session_id,
            paper_id=paper_id,
            topic=topic,
            level=level,
            mode=mode,
            last_active_at=datetime.now(timezone.utc),
        )
        sess.add(chat_session)
        return chat_session

    async def get_session(self, sess: AsyncSession, session_id: str) -> "ChatSession | None":
        result = await sess.execute(
            select(ChatSession).where(ChatSession.session_id == session_id)
        )
        return result.scalar_one_or_none()

    async def list_sessions(
        self,
        sess:     AsyncSession,
        paper_id: str | None = None,
        limit:    int = 50,
    ) -> list["ChatSession"]:
        query = select(ChatSession)
        if paper_id:
            query = query.where(ChatSession.paper_id == paper_id)
        query = query.order_by(ChatSession.last_active_at.desc()).limit(limit)
        result = await sess.execute(query)
        return list(result.scalars().all())

    async def add_message(
        self,
        sess:       AsyncSession,
        session_id: str,
        role:       str,
        content:    str,
        level:      str,
    ) -> "ChatMessage":
        session = await self.get_session(sess, session_id)
        if session is None:
            raise ValueError(f"ChatSession {session_id!r} does not exist.")
        msg = ChatMessage(session_id=session_id, role=role, content=content, level=level)
        sess.add(msg)
        await sess.execute(
            update(ChatSession)
            .where(ChatSession.session_id == session_id)
            .values(last_active_at=datetime.now(timezone.utc))
        )
        return msg

    async def get_messages(self, sess: AsyncSession, session_id: str) -> list["ChatMessage"]:
        result = await sess.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at)
        )
        return list(result.scalars().all())

    async def update_session_level(
        self, sess: AsyncSession, session_id: str, level: str
    ) -> None:
        await sess.execute(
            update(ChatSession)
            .where(ChatSession.session_id == session_id)
            .values(level=level)
        )

    async def update_session_mode(
        self, sess: AsyncSession, session_id: str, mode: str
    ) -> None:
        await sess.execute(
            update(ChatSession)
            .where(ChatSession.session_id == session_id)
            .values(mode=mode)
        )

    # ── Social content helpers ────────────────────────────────────────────────

    async def save_social(
        self,
        sess:         AsyncSession,
        paper_id:     str,
        platform:     str,
        content_type: str,
        content:      str,
        hashtags:     list[str],
    ) -> "SocialContent":
        item = SocialContent(
            paper_id=paper_id,
            platform=platform,
            content_type=content_type,
            content=content,
            hashtags=hashtags,
        )
        sess.add(item)
        return item

    async def list_social(
        self,
        sess:     AsyncSession,
        paper_id: str,
        platform: str | None = None,
    ) -> list["SocialContent"]:
        query = select(SocialContent).where(SocialContent.paper_id == paper_id)
        if platform:
            query = query.where(SocialContent.platform == platform)
        query = query.order_by(SocialContent.created_at.desc())
        result = await sess.execute(query)
        return list(result.scalars().all())

    async def get_social_by_id(
        self, sess: AsyncSession, content_id: int
    ) -> "SocialContent | None":
        result = await sess.execute(
            select(SocialContent).where(SocialContent.id == content_id)
        )
        return result.scalar_one_or_none()

    # ── Processing log helpers ────────────────────────────────────────────────

    async def log(
        self,
        sess:     AsyncSession,
        paper_id: str,
        stage:    str,
        status:   str,
        message:  str,
        duration: float,
    ) -> "ProcessingLog":
        entry = ProcessingLog(
            paper_id=paper_id,
            stage=stage,
            status=status,
            message=message,
            duration_seconds=duration,
        )
        sess.add(entry)
        return entry

    async def get_logs(
        self, sess: AsyncSession, paper_id: str
    ) -> list["ProcessingLog"]:
        result = await sess.execute(
            select(ProcessingLog)
            .where(ProcessingLog.paper_id == paper_id)
            .order_by(ProcessingLog.created_at)
        )
        return list(result.scalars().all())

    # ── Generated content cache helpers ───────────────────────────────────────

    async def get_cache(
        self,
        sess:       AsyncSession,
        cache_type: str,
        cache_key:  str,
    ) -> "GeneratedCache | None":
        result = await sess.execute(
            select(GeneratedCache)
            .where(
                GeneratedCache.cache_type == cache_type,
                GeneratedCache.cache_key  == cache_key,
            )
        )
        return result.scalar_one_or_none()

    async def set_cache(
        self,
        sess:       AsyncSession,
        paper_id:   str,
        cache_type: str,
        cache_key:  str,
        content:    str,
        level:      str | None = None,
    ) -> "GeneratedCache":
        existing = await self.get_cache(sess, cache_type, cache_key)
        now = datetime.now(timezone.utc)
        if existing is not None:
            existing.content    = content
            existing.level      = level
            existing.updated_at = now
            return existing
        entry = GeneratedCache(
            paper_id=paper_id,
            cache_type=cache_type,
            cache_key=cache_key,
            level=level,
            content=content,
            created_at=now,
            updated_at=now,
        )
        sess.add(entry)
        return entry

    async def delete_cache_for_paper(
        self,
        sess:     AsyncSession,
        paper_id: str,
        cache_type: str | None = None,
    ) -> int:
        """Delete cached entries for a paper. Pass cache_type to limit scope."""
        from sqlalchemy import delete as sql_delete
        stmt = sql_delete(GeneratedCache).where(GeneratedCache.paper_id == paper_id)
        if cache_type is not None:
            stmt = stmt.where(GeneratedCache.cache_type == cache_type)
        result = await sess.execute(stmt)
        return result.rowcount


# Single shared instance — import this everywhere
db = Database()