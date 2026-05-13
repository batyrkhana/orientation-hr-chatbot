from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def _fake_embed(text: str) -> list[float]:
    return [0.1] * 768


def _fake_retrieve_with_results(embedding: list[float]):
    from app.retrieval import RetrievedChunk
    return [RetrievedChunk(
        text="Employees accrue 3.077 hours biweekly for the first 5 years.",
        page=9,
        chunk_type="table",
        source_file="handbook.pdf",
        score=0.91,
    )]


def test_english_request_uses_english_system_prompt():
    with patch("app.chat.embed_query", side_effect=_fake_embed), \
         patch("app.chat.retrieve", side_effect=_fake_retrieve_with_results), \
         patch("app.chat.call_claude", return_value="You get 10 days.") as mock_llm:

        resp = client.post("/api/chat", json={
            "session_id": "test-en-001",
            "message": "How many vacation days?",
            "language": "en",
        })

    assert resp.status_code == 200
    _, system_arg = mock_llm.call_args[0]
    assert "English" in system_arg


def test_spanish_request_uses_spanish_system_prompt():
    with patch("app.chat.embed_query", side_effect=_fake_embed), \
         patch("app.chat.retrieve", side_effect=_fake_retrieve_with_results), \
         patch("app.chat.call_claude", return_value="Tienes 10 días.") as mock_llm:

        resp = client.post("/api/chat", json={
            "session_id": "test-es-001",
            "message": "¿Cuántos días de vacaciones?",
            "language": "es",
        })

    assert resp.status_code == 200
    _, system_arg = mock_llm.call_args[0]
    assert "español" in system_arg


def test_no_retrieval_returns_escalation_without_calling_llm():
    with patch("app.chat.embed_query", side_effect=_fake_embed), \
         patch("app.chat.retrieve", return_value=[]), \
         patch("app.chat.call_claude") as mock_llm:

        resp = client.post("/api/chat", json={
            "session_id": "test-offtopic-001",
            "message": "What is the capital of France?",
            "language": "en",
        })

    assert resp.status_code == 200
    data = resp.json()
    assert data["citations"] == []
    mock_llm.assert_not_called()
    assert "hr@abcwidgets.fake" in data["answer"]


def test_response_includes_citations():
    with patch("app.chat.embed_query", side_effect=_fake_embed), \
         patch("app.chat.retrieve", side_effect=_fake_retrieve_with_results), \
         patch("app.chat.call_claude", return_value="See handbook [handbook.pdf, p.9]."):

        resp = client.post("/api/chat", json={
            "session_id": "test-cite-001",
            "message": "Vacation accrual?",
            "language": "en",
        })

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["citations"]) == 1
    assert data["citations"][0]["file"] == "handbook.pdf"
    assert data["citations"][0]["page"] == 9


def test_session_history_preserved_across_calls():
    session_id = "test-history-001"
    with patch("app.chat.embed_query", side_effect=_fake_embed), \
         patch("app.chat.retrieve", side_effect=_fake_retrieve_with_results), \
         patch("app.chat.call_claude", return_value="First answer."):
        client.post("/api/chat", json={"session_id": session_id, "message": "First question", "language": "en"})

    with patch("app.chat.embed_query", side_effect=_fake_embed), \
         patch("app.chat.retrieve", side_effect=_fake_retrieve_with_results), \
         patch("app.chat.call_claude", return_value="Second answer.") as mock_llm:
        client.post("/api/chat", json={"session_id": session_id, "message": "Follow-up", "language": "en"})

    messages_arg = mock_llm.call_args[0][0]
    roles = [m["role"] for m in messages_arg]
    assert "user" in roles
    assert "assistant" in roles
