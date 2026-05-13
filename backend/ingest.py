#!/usr/bin/env python3
import argparse
from pathlib import Path

from app.ingestion.parser import parse_pdf
from app.ingestion.chunker import chunk_all
from app.ingestion.embedder import embed_passages
from app.retrieval import get_client, drop_collection, ensure_collection, upsert_chunks

DOCUMENTS_DIR = Path("documents")


def ingest(full: bool = False) -> None:
    client = get_client()

    if full:
        print("Dropping existing collection...")
        drop_collection(client)

    ensure_collection(client)

    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {DOCUMENTS_DIR}/")
        return

    for pdf_path in pdf_files:
        print(f"Processing {pdf_path.name}...")
        raw_chunks = parse_pdf(pdf_path)
        chunks = chunk_all(raw_chunks)
        texts = [c.text for c in chunks]
        embeddings = embed_passages(texts)
        upsert_chunks(chunks, embeddings)
        print(f"  {len(chunks)} chunks stored.")

    print("Ingest complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Drop and recreate collection before ingesting")
    args = parser.parse_args()
    ingest(full=args.full)
