"""
Thin wrapper around rag_core.answer_question(). If this file ever needs real
logic beyond "call rag_core and hand the dict to the schema", that logic
belongs in rag_core/ instead — this stays a pass-through on purpose.
"""
from rag_core import answer_question

from app.schemas.chat import AskResponse


def ask(question: str) -> AskResponse:
    result = answer_question(question)
    return AskResponse(**result)
