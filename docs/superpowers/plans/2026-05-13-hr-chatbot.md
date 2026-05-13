# ABC Widgets HR Chatbot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dockerized multilingual RAG chatbot that answers HR policy questions from PDFs with source citations, delivered as React frontend + FastAPI backend + Qdrant vector DB.

**Architecture:** PDFs are parsed with pdfplumber (tables extracted as Markdown), chunked at ≤600 tokens (tables kept whole), and embedded with `intfloat/multilingual-e5-base` into Qdrant at ingest time. At query time the user message is embedded and matched against Qdrant; chunks above the 0.45 similarity threshold are injected into a Claude API call. In-memory sessions keyed by a browser-generated UUID provide multi-turn context.

**Tech Stack:** Python 3.11, FastAPI, uvicorn, pdfplumber, sentence-transformers (multilingual-e5-base), tiktoken, qdrant-client, anthropic SDK, React 18 + TypeScript + Vite, Nginx, Docker Compose.

**Note:** Per project constraints (`PROMPT.md`), NO git operations are performed at any step. Steps that would normally commit instead end with a verification check.

---

## File Map

```
orientation-hr-chatbot/
├── docker-compose.yml
├── .env.example
├── .gitignore
├── documents/                          # HR PDFs (bind-mounted read-only into backend)
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── pytest.ini
│   ├── ingest.py
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── chat.py
│   │   ├── retrieval.py
│   │   ├── llm.py
│   │   ├── prompts.py
│   │   └── ingestion/
│   │       ├── __init__.py
│   │       ├── parser.py
│   │       ├── chunker.py
│   │       └── embedder.py
│   └── tests/
│       ├── __init__.py
│       ├── test_chunker.py
│       ├── test_retrieval.py
│       ├── test_prompt_assembly.py
│       └── test_language.py
├── frontend/
│   ├── Dockerfile
│   ├── nginx.conf
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── App.css
│       ├── components/
│       │   ├── ChatWindow.tsx
│       │   ├── MessageBubble.tsx
│       │   ├── CitationBadge.tsx
│       │   └── LanguageSelector.tsx
│       └── hooks/
│           └── useSession.ts
└── qdrant/                             # Qdrant storage volume (gitignored)
```

---

## Phase 1 — Project Skeleton, Docker, Environment Config

### Task 1: Root config files

**Files:**
- Create: `docker-compose.yml`
- Create: `.env.example`
- Create: `.gitignore`

- [ ] **Step 1: Create `docker-compose.yml`**

```yaml
version: '3.9'

services:
  qdrant:
    image: qdrant/qdrant:latest
    volumes:
      - ./qdrant:/qdrant/storage
    networks:
      - chatbot

  backend:
    build: ./backend
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
      - SIMILARITY_THRESHOLD=${SIMILARITY_THRESHOLD:-0.45}
      - TOP_K=${TOP_K:-5}
    volumes:
      - ./documents:/app/documents:ro
    depends_on:
      - qdrant
    networks:
      - chatbot

  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    depends_on:
      - backend
    networks:
      - chatbot

networks:
  chatbot:
    driver: bridge
```

- [ ] **Step 2: Create `.env.example`**

```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
SIMILARITY_THRESHOLD=0.45
TOP_K=5
```

- [ ] **Step 3: Create `.gitignore`**

```
.env
.venv/
__pycache__/
*.pyc
*.pyo
qdrant/
node_modules/
dist/
.pytest_cache/
*.egg-info/
```

- [ ] **Step 4: Create `.env` from example and fill in your API key**

Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` to your actual key.

- [ ] **Step 5: Verify docker compose config parses**

```bash
docker compose config
```

Expected: full merged config printed with no errors.

---

### Task 2: Backend Dockerfile and requirements

**Files:**
- Create: `backend/Dockerfile`
- Create: `backend/requirements.txt`
- Create: `backend/pytest.ini`
- Create: `backend/app/__init__.py`
- Create: `backend/app/ingestion/__init__.py`
- Create: `backend/tests/__init__.py`

- [ ] **Step 1: Create `backend/requirements.txt`**

```
fastapi==0.111.0
uvicorn[standard]==0.30.0
pdfplumber==0.11.0
sentence-transformers==3.0.0
qdrant-client==1.10.0
anthropic==0.30.0
python-dotenv==1.0.0
tiktoken==0.7.0
pytest==8.2.0
httpx==0.27.0
```

- [ ] **Step 2: Create `backend/Dockerfile`**

```dockerfile
FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embedding model so container start doesn't stall on first query
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-base')"

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Create `backend/pytest.ini`**

```ini
[pytest]
pythonpath = .
testpaths = tests
```

- [ ] **Step 4: Create empty `__init__.py` files**

Create these three files, each empty:
- `backend/app/__init__.py`
- `backend/app/ingestion/__init__.py`
- `backend/tests/__init__.py`

---

### Task 3: FastAPI stub

**Files:**
- Create: `backend/app/main.py`

- [ ] **Step 1: Create `backend/app/main.py`**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ABC Widgets HR Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}
```

- [ ] **Step 2: Verify FastAPI stub builds and starts**

```bash
docker compose build backend
docker compose up backend -d
docker compose exec backend curl http://localhost:8000/health
```

Expected: `{"status":"ok"}`

```bash
docker compose down
```

---

### Task 4: Frontend Dockerfile and Vite stub

**Files:**
- Create: `frontend/Dockerfile`
- Create: `frontend/nginx.conf`
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/App.css`

- [ ] **Step 1: Create `frontend/package.json`**

```json
{
  "name": "hr-chatbot-frontend",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "uuid": "^9.0.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "@types/uuid": "^9.0.0",
    "@vitejs/plugin-react": "^4.0.0",
    "typescript": "^5.0.0",
    "vite": "^5.0.0"
  }
}
```

- [ ] **Step 2: Create `frontend/tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src"]
}
```

- [ ] **Step 3: Create `frontend/vite.config.ts`**

```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
});
```

- [ ] **Step 4: Create `frontend/nginx.conf`**

```nginx
server {
    listen 80;

    root /usr/share/nginx/html;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://backend:8000/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

- [ ] **Step 5: Create `frontend/Dockerfile`**

```dockerfile
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
```

- [ ] **Step 6: Create `frontend/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>ABC Widgets HR Assistant</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 7: Create `frontend/src/main.tsx`**

```typescript
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
```

- [ ] **Step 8: Create `frontend/src/App.tsx` (stub)**

```typescript
function App() {
  return <div><h1>ABC Widgets HR Assistant</h1><p>Coming soon.</p></div>;
}

export default App;
```

- [ ] **Step 9: Create `frontend/src/App.css`** (empty for now)

```css
/* filled in Task 20 */
```

- [ ] **Step 10: Verify full stack builds**

```bash
docker compose build
```

Expected: all three services build without error. Frontend build will be fast (stub). Backend build will take several minutes the first time due to the embedding model download.

---

## Phase 2 — PDF Ingestion Pipeline

### Task 5: PDF parser

**Files:**
- Create: `backend/app/ingestion/parser.py`

- [ ] **Step 1: Create `backend/app/ingestion/parser.py`**

```python
from dataclasses import dataclass
from pathlib import Path
import pdfplumber


@dataclass
class RawChunk:
    text: str
    page: int
    chunk_type: str  # "text" | "table"
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
```

- [ ] **Step 2: Smoke-test the parser against the example PDF**

```bash
docker compose run --rm backend python - <<'EOF'
from pathlib import Path
from app.ingestion.parser import parse_pdf
chunks = parse_pdf(Path("documents/ABC_Widgets_Employee_Handbook_Full.pdf"))
tables = [c for c in chunks if c.chunk_type == "table"]
print(f"Total chunks: {len(chunks)}")
print(f"Table chunks: {len(tables)}")
if tables:
    print("\nFirst table chunk (p." + str(tables[0].page) + "):")
    print(tables[0].text[:400])
EOF
```

Expected: multiple table chunks; the vacation accrual table on page 9 should appear as a readable Markdown table with "Years of Continuous Service" and "Rate of Accrual" columns intact.

---

### Task 6: Chunker

**Files:**
- Create: `backend/app/ingestion/chunker.py`
- Create: `backend/tests/test_chunker.py`

- [ ] **Step 1: Write the failing tests in `backend/tests/test_chunker.py`**

```python
from app.ingestion.chunker import chunk_raw, chunk_all
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
    # The second chunk should start before the first chunk ends
    # (overlap = 100 tokens means second chunk starts at token ~500, first ends at ~600)
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    tokens_0 = enc.encode(chunks[0].text)
    tokens_1 = enc.encode(chunks[1].text)
    assert len(tokens_0) <= 600
    assert len(tokens_1) > 0


def test_chunk_all_processes_mixed_list():
    raw = [
        _text("Short text.", page=1),
        _table("| A | B |\n|---|---|\n| x | y |", page=2),
    ]
    chunks = chunk_all(raw)
    assert len(chunks) == 2
    assert chunks[0].page == 1
    assert chunks[1].chunk_type == "table"
```

- [ ] **Step 2: Run tests — expect failures (module not found)**

```bash
docker compose run --rm backend python -m pytest tests/test_chunker.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.ingestion.chunker'`

- [ ] **Step 3: Create `backend/app/ingestion/chunker.py`**

```python
from dataclasses import dataclass
import tiktoken
from app.ingestion.parser import RawChunk

_enc = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 600
OVERLAP_TOKENS = 100


@dataclass
class Chunk:
    text: str
    page: int
    chunk_type: str  # "text" | "table"
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
```

- [ ] **Step 4: Run tests — expect all pass**

```bash
docker compose run --rm backend python -m pytest tests/test_chunker.py -v
```

Expected:
```
test_short_text_stays_single_chunk PASSED
test_long_text_splits_into_multiple_chunks PASSED
test_table_never_splits PASSED
test_chunk_metadata_preserved PASSED
test_consecutive_chunks_overlap PASSED
test_chunk_all_processes_mixed_list PASSED
```

---

### Task 7: Embedder

**Files:**
- Create: `backend/app/ingestion/embedder.py`

- [ ] **Step 1: Create `backend/app/ingestion/embedder.py`**

```python
import os
from functools import lru_cache
from sentence_transformers import SentenceTransformer

_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(_MODEL_NAME)


def embed_passages(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    prefixed = [f"passage: {t}" for t in texts]
    return model.encode(prefixed, normalize_embeddings=True).tolist()


def embed_query(text: str) -> list[float]:
    model = _get_model()
    return model.encode(f"query: {text}", normalize_embeddings=True).tolist()
```

The `"passage: "` / `"query: "` prefixes are required by the multilingual-e5 training regime — omitting them degrades retrieval quality.

- [ ] **Step 2: Smoke-test the embedder**

```bash
docker compose run --rm backend python - <<'EOF'
from app.ingestion.embedder import embed_query, embed_passages
v = embed_query("How many vacation days?")
print(f"Vector dimension: {len(v)}")
print(f"First 5 values: {v[:5]}")
vv = embed_passages(["vacation policy text", "texto en español"])
print(f"Passages embedded: {len(vv)}")
EOF
```

Expected: `Vector dimension: 768`, two passage embeddings returned.

---

### Task 8: Retrieval module

**Files:**
- Create: `backend/app/retrieval.py`
- Create: `backend/tests/test_retrieval.py`

- [ ] **Step 1: Write failing tests in `backend/tests/test_retrieval.py`**

```python
from unittest.mock import MagicMock, patch
from app.retrieval import retrieve, RetrievedChunk


def _mock_result(text: str, page: int, score: float) -> MagicMock:
    r = MagicMock()
    r.payload = {"text": text, "page": page, "chunk_type": "text", "source_file": "handbook.pdf"}
    r.score = score
    return r


def test_retrieve_maps_qdrant_results_to_dataclass():
    with patch("app.retrieval.get_client") as mock_gc:
        mock_client = MagicMock()
        mock_gc.return_value = mock_client
        mock_client.search.return_value = [_mock_result("vacation policy", 9, 0.85)]

        results = retrieve([0.1] * 768)

    assert len(results) == 1
    assert isinstance(results[0], RetrievedChunk)
    assert results[0].source_file == "handbook.pdf"
    assert results[0].page == 9
    assert results[0].score == 0.85


def test_retrieve_returns_empty_list_on_no_matches():
    with patch("app.retrieval.get_client") as mock_gc:
        mock_client = MagicMock()
        mock_gc.return_value = mock_client
        mock_client.search.return_value = []

        results = retrieve([0.1] * 768)

    assert results == []


def test_retrieve_passes_threshold_to_qdrant():
    with patch("app.retrieval.get_client") as mock_gc:
        mock_client = MagicMock()
        mock_gc.return_value = mock_client
        mock_client.search.return_value = []

        retrieve([0.1] * 768)

        call_kwargs = mock_client.search.call_args[1]
        assert "score_threshold" in call_kwargs
        assert call_kwargs["score_threshold"] == 0.45
```

- [ ] **Step 2: Run tests — expect failures**

```bash
docker compose run --rm backend python -m pytest tests/test_retrieval.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.retrieval'`

- [ ] **Step 3: Create `backend/app/retrieval.py`**

```python
import os
import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

COLLECTION = "hr_documents"
VECTOR_DIM = 768
THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.45"))
TOP_K = int(os.getenv("TOP_K", "5"))


@dataclass
class RetrievedChunk:
    text: str
    page: int
    chunk_type: str
    source_file: str
    score: float


def get_client() -> QdrantClient:
    return QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
    )


def ensure_collection(client: QdrantClient) -> None:
    names = [c.name for c in client.get_collections().collections]
    if COLLECTION not in names:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )


def drop_collection(client: QdrantClient) -> None:
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass


def upsert_chunks(chunks: list, embeddings: list[list[float]]) -> None:
    client = get_client()
    ensure_collection(client)
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb,
            payload={
                "text": chunk.text,
                "page": chunk.page,
                "chunk_type": chunk.chunk_type,
                "source_file": chunk.source_file,
            },
        )
        for chunk, emb in zip(chunks, embeddings)
    ]
    client.upsert(collection_name=COLLECTION, points=points)


def retrieve(query_embedding: list[float]) -> list[RetrievedChunk]:
    client = get_client()
    results = client.search(
        collection_name=COLLECTION,
        query_vector=query_embedding,
        limit=TOP_K,
        score_threshold=THRESHOLD,
    )
    return [
        RetrievedChunk(
            text=r.payload["text"],
            page=r.payload["page"],
            chunk_type=r.payload["chunk_type"],
            source_file=r.payload["source_file"],
            score=r.score,
        )
        for r in results
    ]
```

- [ ] **Step 4: Run tests — expect all pass**

```bash
docker compose run --rm backend python -m pytest tests/test_retrieval.py -v
```

Expected: all 3 tests PASS.

---

### Task 9: Ingest script

**Files:**
- Create: `backend/ingest.py`

- [ ] **Step 1: Create `backend/ingest.py`**

```python
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
```

- [ ] **Step 2: Start Qdrant and run full ingest**

```bash
docker compose up qdrant -d
docker compose run --rm backend python ingest.py --full
```

Expected output:
```
Dropping existing collection...
Processing ABC_Widgets_Employee_Handbook_Full.pdf...
  NNN chunks stored.
Ingest complete.
```

- [ ] **Step 3: Confirm chunks in Qdrant**

```bash
docker compose run --rm backend python - <<'EOF'
from app.retrieval import get_client, COLLECTION
client = get_client()
info = client.get_collection(COLLECTION)
print(f"Vectors stored: {info.vectors_count}")
EOF
```

Expected: a number > 0 (expect 50–150 chunks from the example PDF).

---

## Phase 3 — Backend Chat API

### Task 10: LLM adapter

**Files:**
- Create: `backend/app/llm.py`

- [ ] **Step 1: Create `backend/app/llm.py`**

```python
import os
import anthropic

_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 1024


def call_claude(messages: list[dict], system: str) -> str:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=_MODEL,
        max_tokens=_MAX_TOKENS,
        system=system,
        messages=messages,
    )
    return response.content[0].text
```

---

### Task 11: Prompts module

**Files:**
- Create: `backend/app/prompts.py`
- Create: `backend/tests/test_prompt_assembly.py`

- [ ] **Step 1: Write failing tests in `backend/tests/test_prompt_assembly.py`**

```python
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
```

- [ ] **Step 2: Run tests — expect failures**

```bash
docker compose run --rm backend python -m pytest tests/test_prompt_assembly.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.prompts'`

- [ ] **Step 3: Create `backend/app/prompts.py`**

```python
from app.retrieval import RetrievedChunk

_SYSTEM_EN = """You are an HR assistant for ABC Widgets, Inc. You answer questions strictly from the provided HR document excerpts.

Rules:
1. Answer ONLY from the provided <context> excerpts. Do not use outside knowledge.
2. Cite the source and page for every factual claim using the format [filename, p.N].
3. If the excerpts do not contain enough information to answer confidently, say: "I don't have that information in the HR documents. Please contact HR: Jane Smith at hr@abcwidgets.fake or (555) 867-5309."
4. If the question is not about ABC Widgets HR policies, say: "I can only answer questions about ABC Widgets HR policies. Please contact HR: Jane Smith at hr@abcwidgets.fake or (555) 867-5309."
5. Respond in English."""

_SYSTEM_ES = """Eres un asistente de Recursos Humanos para ABC Widgets, Inc. Respondes preguntas estrictamente a partir de los fragmentos de documentos proporcionados.

Reglas:
1. Responde ÚNICAMENTE a partir de los fragmentos <context> proporcionados. No uses conocimiento externo.
2. Cita la fuente y página de cada afirmación con el formato [archivo, p.N].
3. Si los fragmentos no contienen suficiente información, di: "No tengo esa información en los documentos de RRHH. Por favor contacta a RRHH: Jane Smith en hr@abcwidgets.fake o al (555) 867-5309."
4. Si la pregunta no es sobre las políticas de RRHH de ABC Widgets, di: "Solo puedo responder preguntas sobre las políticas de RRHH de ABC Widgets. Por favor contacta a RRHH: Jane Smith en hr@abcwidgets.fake o al (555) 867-5309."
5. Responde en español."""

_ESCALATION_EN = (
    "I don't have that information in the HR documents. "
    "Please contact HR: Jane Smith at hr@abcwidgets.fake or (555) 867-5309."
)
_ESCALATION_ES = (
    "No tengo esa información en los documentos de RRHH. "
    "Por favor contacta a RRHH: Jane Smith en hr@abcwidgets.fake o al (555) 867-5309."
)


def get_system_prompt(language: str) -> str:
    return _SYSTEM_ES if language == "es" else _SYSTEM_EN


def get_escalation(language: str) -> str:
    return _ESCALATION_ES if language == "es" else _ESCALATION_EN


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    parts = [
        f"[{i}] Source: {c.source_file}, p.{c.page}\n---\n{c.text}"
        for i, c in enumerate(chunks, 1)
    ]
    return "<context>\n" + "\n\n".join(parts) + "\n</context>"


def build_user_message(context_block: str, question: str) -> str:
    return f"{context_block}\n\nEmployee question: {question}"
```

- [ ] **Step 4: Run tests — expect all pass**

```bash
docker compose run --rm backend python -m pytest tests/test_prompt_assembly.py -v
```

Expected: all 8 tests PASS.

---

### Task 12: Chat endpoint

**Files:**
- Modify: `backend/app/main.py`
- Create: `backend/app/chat.py`
- Create: `backend/tests/test_language.py`

- [ ] **Step 1: Write failing tests in `backend/tests/test_language.py`**

```python
from unittest.mock import patch, MagicMock
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
            "session_id": "test-offtopc-001",
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
    # History should include the first turn (user + assistant)
    roles = [m["role"] for m in messages_arg]
    assert "user" in roles
    assert "assistant" in roles
```

- [ ] **Step 2: Run tests — expect failures**

```bash
docker compose run --rm backend python -m pytest tests/test_language.py -v
```

Expected: `404 Not Found` for `/api/chat` (route not registered yet).

- [ ] **Step 3: Create `backend/app/chat.py`**

```python
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

    # Store original (non-context-augmented) message for readability
    history.append({"role": "user", "content": request.message})
    history.append({"role": "assistant", "content": answer})

    return ChatResponse(answer=answer, citations=citations, session_id=request.session_id)
```

- [ ] **Step 4: Register chat router in `backend/app/main.py`**

Replace the contents of `backend/app/main.py` with:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.chat import router as chat_router

app = FastAPI(title="ABC Widgets HR Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
```

- [ ] **Step 5: Run all tests — expect all pass**

```bash
docker compose run --rm backend python -m pytest tests/ -v
```

Expected: all tests (chunker + retrieval + prompt_assembly + language) PASS.

- [ ] **Step 6: Smoke-test the live chat endpoint**

Start the full stack (Qdrant must already have data from Task 9):

```bash
docker compose up backend qdrant -d
```

Test with a grounded question:

```bash
docker compose exec backend curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"smoke1","message":"How many vacation days do I get after 7 years?","language":"en"}' \
  | python -m json.tool
```

Expected: answer about 15 days/year for years 6–10, with a citation to the handbook on page 9.

Test escalation on off-topic question:

```bash
docker compose exec backend curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"smoke2","message":"What is the capital of France?","language":"en"}' \
  | python -m json.tool
```

Expected: answer contains "hr@abcwidgets.fake", `citations` is empty `[]`.

```bash
docker compose down
```

---

## Phase 4 — Frontend Chat UI

### Task 13: Session hook

**Files:**
- Create: `frontend/src/hooks/useSession.ts`

- [ ] **Step 1: Create `frontend/src/hooks/useSession.ts`**

```typescript
import { useState } from 'react';
import { v4 as uuidv4 } from 'uuid';

const SESSION_KEY = 'abc_widgets_session_id';

export function useSession(): string {
  const [sessionId] = useState<string>(() => {
    const existing = localStorage.getItem(SESSION_KEY);
    if (existing) return existing;
    const id = uuidv4();
    localStorage.setItem(SESSION_KEY, id);
    return id;
  });
  return sessionId;
}
```

---

### Task 14: LanguageSelector component

**Files:**
- Create: `frontend/src/components/LanguageSelector.tsx`

- [ ] **Step 1: Create `frontend/src/components/LanguageSelector.tsx`**

```typescript
interface Props {
  language: 'en' | 'es';
  onChange: (lang: 'en' | 'es') => void;
}

export function LanguageSelector({ language, onChange }: Props) {
  return (
    <div className="language-selector">
      <button
        className={language === 'en' ? 'active' : ''}
        onClick={() => onChange('en')}
      >
        EN
      </button>
      <button
        className={language === 'es' ? 'active' : ''}
        onClick={() => onChange('es')}
      >
        ES
      </button>
    </div>
  );
}
```

---

### Task 15: CitationBadge component

**Files:**
- Create: `frontend/src/components/CitationBadge.tsx`

- [ ] **Step 1: Create `frontend/src/components/CitationBadge.tsx`**

```typescript
export interface Citation {
  file: string;
  page: number;
}

interface Props {
  citations: Citation[];
}

export function CitationBadge({ citations }: Props) {
  if (citations.length === 0) return null;
  return (
    <div className="citations">
      {citations.map((c, i) => (
        <span key={i} className="citation-badge">
          {c.file}, p.{c.page}
        </span>
      ))}
    </div>
  );
}
```

---

### Task 16: MessageBubble component

**Files:**
- Create: `frontend/src/components/MessageBubble.tsx`

- [ ] **Step 1: Create `frontend/src/components/MessageBubble.tsx`**

```typescript
import { CitationBadge, Citation } from './CitationBadge';

interface Props {
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
}

export function MessageBubble({ role, content, citations = [] }: Props) {
  return (
    <div className={`message-bubble ${role}`}>
      <p>{content}</p>
      {role === 'assistant' && <CitationBadge citations={citations} />}
    </div>
  );
}
```

---

### Task 17: ChatWindow component

**Files:**
- Create: `frontend/src/components/ChatWindow.tsx`

- [ ] **Step 1: Create `frontend/src/components/ChatWindow.tsx`**

```typescript
import { useState, useRef, useEffect } from 'react';
import { MessageBubble } from './MessageBubble';
import { Citation } from './CitationBadge';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
}

interface Props {
  sessionId: string;
  language: 'en' | 'es';
}

export function ChatWindow({ sessionId, language }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async () => {
    const text = input.trim();
    if (!text || loading) return;

    setMessages(prev => [...prev, { role: 'user', content: text }]);
    setInput('');
    setLoading(true);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, message: text, language }),
      });
      const data = await res.json();
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.answer,
        citations: data.citations,
      }]);
    } catch {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: language === 'es'
          ? 'Error al conectar con el servidor. Intenta de nuevo.'
          : 'Error connecting to server. Please try again.',
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="chat-window">
      <div className="messages">
        {messages.map((msg, i) => (
          <MessageBubble key={i} role={msg.role} content={msg.content} citations={msg.citations} />
        ))}
        {loading && <div className="loading-indicator">…</div>}
        <div ref={bottomRef} />
      </div>
      <div className="input-row">
        <textarea
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={language === 'es' ? 'Escribe tu pregunta…' : 'Type your question…'}
          rows={2}
          disabled={loading}
        />
        <button onClick={sendMessage} disabled={loading || !input.trim()}>
          {language === 'es' ? 'Enviar' : 'Send'}
        </button>
      </div>
    </div>
  );
}
```

---

### Task 18: App root and styles

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/App.css`

- [ ] **Step 1: Replace `frontend/src/App.tsx`**

```typescript
import { useState } from 'react';
import { LanguageSelector } from './components/LanguageSelector';
import { ChatWindow } from './components/ChatWindow';
import { useSession } from './hooks/useSession';
import './App.css';

function App() {
  const [language, setLanguage] = useState<'en' | 'es'>('en');
  const sessionId = useSession();

  return (
    <div className="app">
      <header className="app-header">
        <h1>ABC Widgets HR Assistant</h1>
        <LanguageSelector language={language} onChange={setLanguage} />
      </header>
      <main className="app-main">
        <ChatWindow sessionId={sessionId} language={language} />
      </main>
    </div>
  );
}

export default App;
```

- [ ] **Step 2: Replace `frontend/src/App.css`**

```css
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body { font-family: system-ui, -apple-system, sans-serif; background: #f4f5f7; }

.app { display: flex; flex-direction: column; height: 100vh; max-width: 800px; margin: 0 auto; background: #fff; box-shadow: 0 0 20px rgba(0,0,0,0.08); }

.app-header { display: flex; justify-content: space-between; align-items: center; padding: 1rem 1.5rem; border-bottom: 1px solid #e2e4e8; }
.app-header h1 { font-size: 1.1rem; font-weight: 600; color: #1a2332; }

.language-selector { display: flex; gap: 0.4rem; }
.language-selector button { padding: 0.3rem 0.8rem; font-size: 0.85rem; border: 1px solid #ccc; border-radius: 4px; background: #fff; cursor: pointer; color: #555; transition: background 0.15s; }
.language-selector button.active { background: #1a2332; color: #fff; border-color: #1a2332; }

.app-main { flex: 1; overflow: hidden; }

.chat-window { display: flex; flex-direction: column; height: 100%; }

.messages { flex: 1; overflow-y: auto; padding: 1.25rem; display: flex; flex-direction: column; gap: 0.75rem; }

.message-bubble { max-width: 78%; padding: 0.75rem 1rem; border-radius: 12px; line-height: 1.55; font-size: 0.95rem; }
.message-bubble.user { align-self: flex-end; background: #1a2332; color: #fff; border-bottom-right-radius: 3px; }
.message-bubble.assistant { align-self: flex-start; background: #f0f2f5; color: #1a2332; border-bottom-left-radius: 3px; }
.message-bubble p { white-space: pre-wrap; }

.citations { margin-top: 0.6rem; display: flex; flex-wrap: wrap; gap: 0.3rem; }
.citation-badge { font-size: 0.72rem; padding: 0.15rem 0.55rem; background: #dde1e8; border-radius: 10px; color: #444; }

.loading-indicator { align-self: flex-start; padding: 0.5rem 1rem; color: #999; font-size: 1.2rem; }

.input-row { display: flex; gap: 0.6rem; padding: 1rem 1.25rem; border-top: 1px solid #e2e4e8; }
.input-row textarea { flex: 1; padding: 0.6rem 0.8rem; border: 1px solid #ccc; border-radius: 8px; resize: none; font-family: inherit; font-size: 0.95rem; line-height: 1.4; }
.input-row textarea:focus { outline: none; border-color: #1a2332; }
.input-row button { padding: 0.5rem 1.2rem; background: #1a2332; color: #fff; border: none; border-radius: 8px; font-size: 0.9rem; cursor: pointer; white-space: nowrap; }
.input-row button:disabled { background: #bcc0c8; cursor: not-allowed; }
```

---

### Task 19: Build and verify the frontend

- [ ] **Step 1: Build the full stack**

```bash
docker compose build
```

- [ ] **Step 2: Start the full stack**

```bash
docker compose up -d
```

- [ ] **Step 3: Open in browser and verify the UI renders**

Open `http://localhost:3000` in a browser.

Expected:
- Header with "ABC Widgets HR Assistant" and EN/ES toggle.
- Empty chat area with a text input at the bottom.
- EN button is highlighted (active).

- [ ] **Step 4: Send a test question in English**

Type: `How many vacation days do I get after 7 years?`

Expected:
- User message appears on the right.
- Assistant replies with a grounded answer mentioning 15 days and a citation badge showing the handbook filename and page 9.

- [ ] **Step 5: Switch to Spanish and send the same question**

Click ES, then type: `¿Cuántos días de vacaciones tengo después de 7 años?`

Expected:
- Assistant replies in Spanish.
- Citation badge still appears.

- [ ] **Step 6: Test off-topic refusal**

Type (in English): `What is the capital of France?`

Expected: polite refusal mentioning only HR topics and the HR contact. No citation badges.

- [ ] **Step 7: Test multi-turn context**

Send: `How many sick days do I get?`
Then send: `What about vacation?`

Expected: the second answer is about vacation — the chatbot uses conversation context to know what "vacation" refers to.

- [ ] **Step 8: Reload the page and verify session persists**

After reloading, send a follow-up question like `Can you repeat your last answer?`

Expected: the chatbot does NOT have memory of the previous reload session (in-memory sessions reset on container restart) — but the session UUID in `localStorage` persists, so the session is identified consistently within the same server uptime window. Verify the UUID in browser DevTools → Application → Local Storage.

```bash
docker compose down
```

---

## Phase 5 — End-to-End Smoke Test

### Task 20: Full requirements verification

- [ ] **Step 1: Start the full stack**

```bash
docker compose up -d
```

Wait ~10 seconds for all services to be ready.

- [ ] **Step 2: Verify all five PROMPT.md test cases via the UI**

Open `http://localhost:3000` and run each case:

| Test | Input | Expected |
|---|---|---|
| Grounded answer + citation | "How many vacation days after 7 years?" | Answer with 15 days/year, citation badge `handbook.pdf, p.9` |
| Escalation on unknown | "What is the retirement 401k matching percentage?" (if not in doc) | "I don't have that information… contact HR" |
| Off-topic refusal | "What is the capital of France?" | Polite refusal, redirects to HR topics |
| Spanish answer | Switch to ES, ask "¿Qué días festivos tiene la empresa?" | Spanish answer listing holidays |
| Multi-turn context | "What is the sick leave policy?" then "How much do I accrue per month?" | Second answer uses first answer's context |

- [ ] **Step 3: Test re-index clears stale content**

Copy the example PDF to a second name to simulate a new document:

```bash
copy documents\ABC_Widgets_Employee_Handbook_Full.pdf documents\test_extra.pdf
docker compose exec backend python ingest.py --full
```

Verify the new file appears in a retrieval result. Then remove it and re-index:

```bash
del documents\test_extra.pdf
docker compose exec backend python ingest.py --full
```

Ask a question and verify only the original handbook is cited — no `test_extra.pdf` citations.

- [ ] **Step 4: Confirm secrets are not in source files**

```bash
findstr /r /s "ANTHROPIC_API_KEY" backend\app\*.py backend\*.py frontend\src\*.ts frontend\src\*.tsx
```

Expected: no matches (key comes from environment variable only).

- [ ] **Step 5: Confirm `docker compose up` is the only command needed from a clean state**

```bash
docker compose down -v
docker compose up -d
```

Wait for all containers to be healthy, then re-run the ingest:

```bash
docker compose exec backend python ingest.py --full
```

Open `http://localhost:3000` and confirm the chatbot answers correctly.

- [ ] **Step 6: Run the full test suite one final time**

```bash
docker compose exec backend python -m pytest tests/ -v
```

Expected: all tests PASS.

---

## Summary of Run Commands

| Action | Command |
|---|---|
| Build everything | `docker compose build` |
| Start full stack | `docker compose up -d` |
| Index documents | `docker compose exec backend python ingest.py --full` |
| Run tests | `docker compose exec backend python -m pytest tests/ -v` |
| Stop everything | `docker compose down` |
| View backend logs | `docker compose logs backend -f` |
