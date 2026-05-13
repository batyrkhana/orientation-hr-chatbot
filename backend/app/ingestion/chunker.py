from dataclasses import dataclass
from typing import Literal
import tiktoken
from app.ingestion.parser import RawChunk

_enc = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 600
OVERLAP_TOKENS = 100


@dataclass
class Chunk:
    text: str
    page: int
    chunk_type: Literal["text", "table"]
    source_file: str


def chunk_raw(raw: RawChunk) -> list[Chunk]:
    if raw.chunk_type == "table":
        return [Chunk(text=raw.text, page=raw.page, chunk_type="table", source_file=raw.source_file)]

    tokens = _enc.encode(raw.text)
    if len(tokens) <= MAX_TOKENS:
        return [Chunk(text=raw.text, page=raw.page, chunk_type="text", source_file=raw.source_file)]

    chunks: list[Chunk] = []
    start = 0
    while start < len(tokens):
        end = min(start + MAX_TOKENS, len(tokens))
        chunk_text = _enc.decode(tokens[start:end])
        chunks.append(Chunk(text=chunk_text, page=raw.page, chunk_type="text", source_file=raw.source_file))
        if end == len(tokens):
            break
        start = end - OVERLAP_TOKENS

    return chunks


def chunk_all(raw_chunks: list[RawChunk]) -> list[Chunk]:
    result: list[Chunk] = []
    for raw in raw_chunks:
        result.extend(chunk_raw(raw))
    return result
