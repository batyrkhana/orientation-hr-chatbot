# HR Chatbot Design Spec
**Date:** 2026-05-13  
**Project:** ABC Widgets Multilingual HR Chatbot

---

## Overview

A multilingual, RAG-based HR chatbot deployed on the ABC Widgets intranet. Answers employee questions strictly from company-provided HR PDF documents, cites source and page on every answer, responds in English or Spanish per employee selection, and supports multi-turn conversation. Escalates to a human HR contact when the answer is not in the documents.

---

## Decisions from Clarification

- Language selector defaults to **English**.
- Conversation history persists across page reloads via **in-memory server sessions** keyed by a UUID stored in browser `localStorage`. Sessions are cleared on container restart; this is acceptable for short-lived intranet Q&A sessions.

---

## Project Layout

```
orientation-hr-chatbot/
├── docker-compose.yml
├── .env.example
├── .gitignore
├── documents/                      # HR PDFs — bind-mounted into backend
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── ingest.py                   # CLI: parse → chunk → embed → store in Qdrant
│   ├── app/
│   │   ├── main.py                 # FastAPI app factory
│   │   ├── chat.py                 # POST /api/chat — session mgmt, retrieval, LLM
│   │   ├── retrieval.py            # Qdrant query, threshold guard
│   │   ├── llm.py                  # Claude API adapter
│   │   ├── ingestion/
│   │   │   ├── parser.py           # pdfplumber → structured page/table chunks
│   │   │   ├── chunker.py          # split + metadata tagging
│   │   │   └── embedder.py         # multilingual-e5-base wrapper
│   │   └── prompts.py              # system prompt templates (EN / ES)
│   └── tests/
│       ├── test_chunker.py
│       ├── test_retrieval.py
│       ├── test_prompt_assembly.py
│       └── test_language.py
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   └── src/
│       ├── App.tsx
│       ├── components/
│       │   ├── ChatWindow.tsx
│       │   ├── MessageBubble.tsx
│       │   ├── CitationBadge.tsx
│       │   └── LanguageSelector.tsx
│       └── hooks/
│           └── useSession.ts       # UUID generation + localStorage persistence
└── qdrant/                         # Qdrant storage volume (gitignored)
```

---

## Containers

| Service    | Image / Build      | Ports      | Notes |
|------------|--------------------|------------|-------|
| `qdrant`   | `qdrant/qdrant`    | internal   | Data volume at `./qdrant` |
| `backend`  | `./backend`        | 8000 (internal) | `documents/` bind-mounted read-only |
| `frontend` | `./frontend`       | 3000:80    | Nginx; proxies `/api/*` to backend |

Re-indexing: `docker compose exec backend python ingest.py --full` — no container rebuild required.

---

## Ingestion Pipeline

### PDF Parsing (`parser.py`)
- `pdfplumber` opens each PDF page by page.
- Text blocks extracted as plain text strings.
- Tables detected via `pdfplumber.Page.extract_tables()` and converted to Markdown table format (header row + data rows). Mixed pages interleave text and table chunks preserving reading order.
- Output: list of `PageChunk(text: str, page: int, chunk_type: "text"|"table", source_file: str)`.

### Chunking (`chunker.py`)
- **Text chunks:** ~600 tokens, 100-token overlap. Preserves sentence boundaries.
- **Table chunks:** kept whole regardless of size — splitting a benefits matrix mid-row would corrupt the data.
- Each chunk carries metadata: `{source_file, page_number, chunk_type}`.

### Embedding (`embedder.py`)
- Model: `intfloat/multilingual-e5-base` (~550MB, runs fully locally via `sentence-transformers`).
- Chosen over `paraphrase-multilingual-MiniLM-L12-v2` because the e5 family was trained specifically for retrieval tasks (query/passage prefix scheme), giving better precision on factual Q&A. Handles cross-lingual retrieval: a Spanish query can retrieve English source chunks.
- Input prefix: `"passage: "` for document chunks at ingest; `"query: "` for user queries at runtime.

### Storage
- Single Qdrant collection: `hr_documents`.
- `ingest.py --full` deletes and recreates the collection, ensuring stale content from removed PDFs does not survive a re-index.
- Vector dimension: 768 (multilingual-e5-base output size).

---

## Chat Request Flow

**Endpoint:** `POST /api/chat`

**Request body:**
```json
{
  "session_id": "<uuid>",
  "message": "How many vacation days do I get after 7 years?",
  "language": "en"
}
```

**Response body:**
```json
{
  "answer": "After 6–10 years of service, you accrue...",
  "citations": [{"file": "ABC_Widgets_Employee_Handbook_Full.pdf", "page": 9}],
  "session_id": "<uuid>"
}
```

**Processing steps:**
1. Load session history from in-memory dict (keyed by `session_id`; create new if absent).
2. Embed user message with `"query: "` prefix using multilingual-e5-base.
3. Query Qdrant: retrieve top-5 chunks by cosine similarity.
4. **Threshold guard:** if no chunk scores ≥ 0.45, skip LLM and return canned escalation message in the selected language: *"I don't have that information in the HR documents. Please contact HR: Jane Smith at hr@abcwidgets.fake or (555) 867-5309."*
5. Build prompt (see Prompts section).
6. Call Claude API (`claude-sonnet-4-6`), non-streaming for simplicity in v1.
7. Append `{role: user, content: message}` and `{role: assistant, content: answer}` to session history (capped at last 6 turns to bound context size).
8. Return answer + citations.

**On-topic guard:** The system prompt instructs Claude to refuse non-HR questions and redirect to HR topics. The retrieval threshold provides a second independent guard — off-topic queries typically score below 0.45 against an HR document corpus.

---

## Prompts (`prompts.py`)

**System prompt (English):**
```
You are an HR assistant for ABC Widgets, Inc. You answer questions strictly based on the provided HR document excerpts. 

Rules:
- Answer only from the provided excerpts. Do not use outside knowledge.
- Cite the source document and page number for every factual claim, e.g. [ABC_Widgets_Employee_Handbook_Full.pdf, p.9].
- If the excerpts do not contain enough information to answer confidently, say so and direct the employee to contact HR: Jane Smith at hr@abcwidgets.fake or (555) 867-5309.
- If the question is not about ABC Widgets HR policies, politely decline and redirect: "I can only answer questions about ABC Widgets HR policies."
- Respond in English.
```

**System prompt (Spanish):** Same rules, last line changed to *"Responde en español."*

Retrieved chunks are injected into the user turn as a `<context>` block with source labels, before the actual employee question.

---

## Frontend

**Stack:** React 18 + TypeScript, Vite, plain CSS (no UI library — keeps the build simple).

**Components:**
- `LanguageSelector` — EN/ES toggle, defaults to EN, stored in component state.
- `ChatWindow` — scrollable list of `MessageBubble` components.
- `MessageBubble` — user vs. assistant styling; assistant messages render `CitationBadge` components below the text.
- `CitationBadge` — pill showing `filename, p.N`.

**`useSession` hook:** generates UUID v4 on first render, persists to `localStorage` under key `abc_widgets_session_id`. Retrieved on subsequent renders so history survives page reload.

**API:** all calls to `/api/*` (relative URL); Nginx proxies to backend. No hardcoded hostnames.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | yes | Claude API key |
| `QDRANT_HOST` | yes (default: `qdrant`) | Qdrant service hostname |
| `QDRANT_PORT` | no (default: `6333`) | Qdrant port |
| `EMBEDDING_MODEL` | no (default: `intfloat/multilingual-e5-base`) | Local embedding model |
| `SIMILARITY_THRESHOLD` | no (default: `0.45`) | Minimum score for retrieval to proceed to LLM |
| `TOP_K` | no (default: `5`) | Number of chunks to retrieve |

---

## Tests

Tests cover non-trivial logic only:

| File | What it tests |
|---|---|
| `test_chunker.py` | Text chunking respects token limits; tables kept whole |
| `test_retrieval.py` | Below-threshold queries return escalation, not LLM answer |
| `test_prompt_assembly.py` | Context block correctly injects chunks + citations into prompt |
| `test_language.py` | EN/ES system prompt selection; Spanish query returns Spanish answer |

---

## Out of Scope

Per PROMPT.md: authentication, admin UI for document management, analytics dashboards, real-time document updates.

---

## Implementation Phases

1. Project skeleton, Docker setup, environment config.
2. PDF ingestion pipeline (parse → chunk → embed → store).
3. Backend chat API (retrieval + LLM + session management).
4. Frontend chat UI (language selector, citations, session hook).
5. End-to-end smoke test against the example PDF.
