from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import pdfplumber


@dataclass
class RawChunk:
    text: str
    page: int
    chunk_type: Literal["text", "table"]
    source_file: str


def parse_pdf(pdf_path: Path) -> list[RawChunk]:
    chunks: list[RawChunk] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.find_tables()
            table_bboxes = [t.bbox for t in tables]

            # Extract text excluding table bounding boxes
            words = page.extract_words()
            non_table_words = [
                w for w in words
                if not any(_word_in_bbox(w, bbox) for bbox in table_bboxes)
            ]
            page_text = " ".join(w["text"] for w in non_table_words).strip()
            if page_text:
                chunks.append(RawChunk(
                    text=page_text,
                    page=page_num,
                    chunk_type="text",
                    source_file=pdf_path.name,
                ))

            # Extract each table as Markdown
            for table_obj in tables:
                rows = table_obj.extract()
                if rows and len(rows) > 1:
                    md = _table_to_markdown(rows)
                    chunks.append(RawChunk(
                        text=md,
                        page=page_num,
                        chunk_type="table",
                        source_file=pdf_path.name,
                    ))

    return chunks


def _word_in_bbox(word: dict, bbox: tuple[float, float, float, float]) -> bool:
    x0, top, x1, bottom = bbox
    return (
        word["x0"] >= x0
        and word["x1"] <= x1
        and word["top"] >= top
        and word["bottom"] <= bottom
    )


def _table_to_markdown(rows: list[list]) -> str:
    def cell(v: object) -> str:
        return str(v).strip() if v is not None else ""

    headers = [cell(c) for c in rows[0]]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows[1:]:
        lines.append("| " + " | ".join(cell(c) for c in row) + " |")
    return "\n".join(lines)
