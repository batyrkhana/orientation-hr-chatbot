from app.prompts import (
    build_context_block,
    build_user_message,
    get_system_prompt,
    get_escalation,
)
from app.retrieval import RetrievedChunk


def _chunk(text: str = "Policy text.", page: int = 5, source: str = "handbook.pdf") -> RetrievedChunk:
    return RetrievedChunk(text=text, page=page, chunk_type="text", source_file=source, score=0.9)


def test_context_block_contains_source_and_page():
    block = build_context_block([_chunk(page=9)])
    assert "handbook.pdf" in block
    assert "p.9" in block


def test_context_block_numbers_each_chunk():
    block = build_context_block([_chunk(), _chunk(), _chunk()])
    assert "[1]" in block
    assert "[2]" in block
    assert "[3]" in block


def test_context_block_contains_chunk_text():
    block = build_context_block([_chunk(text="vacation accrual schedule")])
    assert "vacation accrual schedule" in block


def test_user_message_wraps_context_and_question():
    block = build_context_block([_chunk()])
    msg = build_user_message(block, "How many days?")
    assert "<context>" in msg
    assert "</context>" in msg
    assert "How many days?" in msg


def test_system_prompt_english_contains_english_instruction():
    prompt = get_system_prompt("en")
    assert "English" in prompt
    assert "ABC Widgets" in prompt


def test_system_prompt_spanish_contains_spanish_instruction():
    prompt = get_system_prompt("es")
    assert "español" in prompt
    assert "ABC Widgets" in prompt


def test_escalation_english_contains_hr_contact():
    msg = get_escalation("en")
    assert "hr@abcwidgets.fake" in msg
    assert "Jane Smith" in msg
    assert "555" in msg


def test_escalation_spanish_contains_hr_contact():
    msg = get_escalation("es")
    assert "hr@abcwidgets.fake" in msg
