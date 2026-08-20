"""
backend.app.services.memory — conversation memory for the Clinical RAG backend.

Short-term: per-session turn history (in-memory, not persisted across restarts).
Long-term:  cited facts accumulated across sessions (persisted to JSON on disk).

MemoryManager lives here in the backend, NOT in rag_core.  rag_core only
receives pre-formatted context strings — it has no knowledge of sessions or
storage.
"""
from __future__ import annotations

import json
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ConversationTurn:
    question: str
    answer_summary: str
    status: str
    claims: list[dict]
    timestamp: float


@dataclass
class ConversationSession:
    session_id: str
    turns: deque[ConversationTurn] = field(default_factory=lambda: deque(maxlen=20))
    created_at: float = field(default_factory=time.time)

    def add_turn(self, turn: ConversationTurn) -> None:
        self.turns.append(turn)

    def get_history(self, n: int = 5) -> list[ConversationTurn]:
        return list(self.turns)[-n:]

    def format_history_for_prompt(self, n: int = 5) -> str:
        """Format the last *n* turns into a prompt-ready string."""
        history = self.get_history(n)
        if not history:
            return ""
        lines = ["Previous conversation:"]
        for t in history:
            lines.append(f"Q: {t.question}")
            summary = t.answer_summary[:300]
            if len(t.answer_summary) > 300:
                summary += "..."
            lines.append(f"A: {summary}")
        return "\n".join(lines)


@dataclass
class LongTermFact:
    fact_text: str
    source_chunk_id: str
    source_section: str
    source_page: int
    source_question: str
    timestamp: float


# ---------------------------------------------------------------------------
# Memory Manager
# ---------------------------------------------------------------------------
class MemoryManager:
    """Manages short-term session history and long-term fact retrieval."""

    def __init__(
        self,
        persist_path: Path | str | None = None,
        max_sessions: int = 50,
        session_ttl: float = 7200.0,
        max_long_term: int = 500,
    ):
        self._sessions: dict[str, ConversationSession] = {}
        self._max_sessions = max_sessions
        self._session_ttl = session_ttl
        self._facts: list[LongTermFact] = []
        self._max_long_term = max_long_term
        self._persist_path = Path(persist_path) if persist_path else None
        self._vectorizer: TfidfVectorizer | None = None
        self._fact_matrix = None

        if self._persist_path and self._persist_path.exists():
            self._load()

    # -- session management ---------------------------------------------------
    def get_or_create_session(self, session_id: str | None = None) -> ConversationSession:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        sid = session_id if session_id else str(uuid.uuid4())
        session = ConversationSession(session_id=sid)
        self._sessions[sid] = session
        self._evict_stale_sessions()
        return session

    def _evict_stale_sessions(self) -> None:
        now = time.time()
        stale = [
            sid
            for sid, s in self._sessions.items()
            if now - s.created_at > self._session_ttl
        ]
        for sid in stale:
            del self._sessions[sid]
        if len(self._sessions) > self._max_sessions:
            oldest = sorted(
                self._sessions, key=lambda k: self._sessions[k].created_at
            )[: len(self._sessions) - self._max_sessions]
            for sid in oldest:
                del self._sessions[sid]

    # -- short-term context ---------------------------------------------------
    def get_short_term_context(self, session_id: str, n: int = 5) -> str:
        session = self._sessions.get(session_id)
        if not session:
            return ""
        return session.format_history_for_prompt(n)

    # -- long-term context ----------------------------------------------------
    def retrieve_long_term_context(self, query: str, k: int = 3) -> str:
        """Retrieve the *k* most relevant past facts for *query*."""
        if not self._facts or self._vectorizer is None:
            return ""
        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._fact_matrix).ravel()
        top_idx = np.argsort(sims)[::-1][:k]
        relevant = [self._facts[i] for i in top_idx if sims[i] > 0.05]
        if not relevant:
            return ""
        lines = ["Prior relevant findings from past conversations:"]
        for f in relevant:
            lines.append(f"- {f.fact_text} (section: {f.source_section}, p.{f.source_page})")
        return "\n".join(lines)

    # -- recording turns ------------------------------------------------------
    def record_turn(
        self,
        session_id: str,
        question: str,
        answer_summary: str,
        status: str,
        claims: list[dict],
    ) -> None:
        session = self._sessions.get(session_id)
        if not session:
            return
        turn = ConversationTurn(
            question=question,
            answer_summary=answer_summary,
            status=status,
            claims=claims,
            timestamp=time.time(),
        )
        session.add_turn(turn)
        new_facts = self._extract_facts(question, claims)
        if new_facts:
            self._facts.extend(new_facts)
            if len(self._facts) > self._max_long_term:
                self._facts = self._facts[-self._max_long_term:]
            self._rebuild_fact_index()
            self._persist()

    def _extract_facts(self, question: str, claims: list[dict]) -> list[LongTermFact]:
        """Extract cited claims as long-term facts."""
        facts: list[LongTermFact] = []
        for c in claims:
            cit = c.get("citation")
            if not cit:
                continue
            facts.append(
                LongTermFact(
                    fact_text=c.get("claim_text", ""),
                    source_chunk_id=cit.get("chunk_id", ""),
                    source_section=cit.get("section", ""),
                    source_page=cit.get("page", -1),
                    source_question=question,
                    timestamp=time.time(),
                )
            )
        return facts

    # -- TF-IDF index over facts ----------------------------------------------
    def _rebuild_fact_index(self) -> None:
        if not self._facts:
            self._vectorizer = None
            self._fact_matrix = None
            return
        texts = [f.fact_text for f in self._facts]
        self._vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), max_features=5000
        )
        self._fact_matrix = self._vectorizer.fit_transform(texts)

    # -- persistence ----------------------------------------------------------
    def _persist(self) -> None:
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(f) for f in self._facts]
        with open(self._persist_path, "w") as fh:
            json.dump(data, fh, indent=2)

    def _load(self) -> None:
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            with open(self._persist_path) as fh:
                data = json.load(fh)
            self._facts = [LongTermFact(**item) for item in data]
            self._rebuild_fact_index()
        except Exception:
            self._facts = []


# ---------------------------------------------------------------------------
# Module-level singleton — one MemoryManager for the whole backend.
# ---------------------------------------------------------------------------
from pathlib import Path as _Path

_PERSIST = _Path(__file__).resolve().parent.parent.parent.parent / "index_store" / "conversation_memory.json"
memory = MemoryManager(persist_path=_PERSIST)
