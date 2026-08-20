"""
Thin wrapper around rag_core.answer_question() + memory.

Orchestrates: session management, short-term history retrieval, long-term
fact retrieval, the rag_core call, recording the turn, and attaching the
conversation_id to the result.
"""
from rag_core import answer_question

from app.schemas.chat import AskResponse
from app.services.memory import memory


def ask(question: str, conversation_id: str | None = None) -> AskResponse:
    # Get or create session
    session = memory.get_or_create_session(conversation_id)
    returned_id = session.session_id

    # Gather context strings for the prompt
    history = memory.get_short_term_context(returned_id)
    long_term = memory.retrieve_long_term_context(question)

    # Call rag_core — it receives plain strings, knows nothing about memory
    result = answer_question(
        question, history=history, long_term_context=long_term
    )

    # Record this turn in session memory
    memory.record_turn(
        session_id=returned_id,
        question=question,
        answer_summary=result["answer_summary"],
        status=result["status"],
        claims=result.get("claims", []),
    )

    # Attach conversation_id and build response
    result["conversation_id"] = returned_id
    return AskResponse(**result)
