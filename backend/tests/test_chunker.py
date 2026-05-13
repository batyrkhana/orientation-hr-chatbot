import tiktoken
from app.ingestion.chunker import chunk_raw, chunk_all, OVERLAP_TOKENS
from app.ingestion.parser import RawChunk


def _text(text: str, page: int = 1) -> RawChunk:
    return RawChunk(text=text, page=page, chunk_type="text", source_file="test.pdf")


def _table(text: str, page: int = 1) -> RawChunk:
    return RawChunk(text=text, page=page, chunk_type="table", source_file="test.pdf")


def test_short_text_stays_single_chunk():
    chunks = chunk_raw(_text("Short paragraph."))
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "text"
    assert chunks[0].text == "Short paragraph."


def test_long_text_splits_into_multiple_chunks():
    long_text = "The quick brown fox jumped over the lazy dog. " * 200
    chunks = chunk_raw(_text(long_text))
    assert len(chunks) > 1


def test_table_never_splits():
    big_table = "| Col1 | Col2 |\n|---|---|\n" + "| data | value |\n" * 100
    chunks = chunk_raw(_table(big_table))
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "table"


def test_chunk_metadata_preserved():
    chunks = chunk_raw(_text("Some policy text.", page=7))
    assert chunks[0].page == 7
    assert chunks[0].source_file == "test.pdf"


def test_consecutive_chunks_overlap():
    long_text = "word " * 700
    chunks = chunk_raw(_text(long_text))
    assert len(chunks) >= 2
    enc = tiktoken.get_encoding("cl100k_base")
    tokens_0 = enc.encode(chunks[0].text)
    tokens_1 = enc.encode(chunks[1].text)
    assert len(tokens_0) <= 600
    # The last OVERLAP_TOKENS of chunk 0 must equal the first OVERLAP_TOKENS of chunk 1
    assert tokens_0[-OVERLAP_TOKENS:] == tokens_1[:OVERLAP_TOKENS]


def test_chunk_all_processes_mixed_list():
    raw = [
        _text("Short text.", page=1),
        _table("| A | B |\n|---|---|\n| x | y |", page=2),
    ]
    chunks = chunk_all(raw)
    assert len(chunks) == 2
    assert chunks[0].page == 1
    assert chunks[1].chunk_type == "table"
