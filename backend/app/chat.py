from fastapi import APIRouter
from pydantic import BaseModel

from app.ingestion.embedder import embed_query
from app.retrieval import retrieve
from app.llm import call_claude
from app.prompts import get_system_prompt, get_escalation, build_context_block, build_user_message

router = APIRouter()

# session_id -> list of {"role": "user"|"assistant", "content": str}
_sessions: dict[str, list[dict]] = {}
_MAX_HISTORY_MESSAGES = 12  # 6 turns × 2 messages each


class ChatRequest(BaseModel):
    session_id: str
    message: str
    language: str = "en"


class Citation(BaseModel):
    file: str
    page: int


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    session_id: str


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    history = _sessions.setdefault(request.session_id, [])

    query_embedding = embed_query(request.message)
    chunks = retrieve(query_embedding)
    system = get_system_prompt(request.language)

    if not chunks:
        answer = get_escalation(request.language)
        citations: list[Citation] = []
    else:
        context_block = build_context_block(chunks)
        user_msg = build_user_message(context_block, request.message)

        # Cap history to last N messages, append current user turn
        messages = list(history[-_MAX_HISTORY_MESSAGES:])
        messages.append({"role": "user", "content": user_msg})

        answer = call_claude(messages, system)
        citations = [Citation(file=c.source_file, page=c.page) for c in chunks]

    # Store original (non-context-augmented) message in session
    history.append({"role": "user", "content": request.message})
    history.append({"role": "assistant", "content": answer})

    return ChatResponse(answer=answer, citations=citations, session_id=request.session_id)
