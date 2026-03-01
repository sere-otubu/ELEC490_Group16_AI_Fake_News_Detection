<p align="center">
  <img src="chrome-extension/icons/icon128.png" alt="MedCheck AI Logo" width="80" />
</p>

<h1 align="center">EvidenceMD</h1>

<p align="center">
  <strong>AI-Powered Medical Misinformation Detection</strong><br/>
  A Retrieval-Augmented Generation (RAG) system that fact-checks medical claims using peer-reviewed sources.
</p>

<p align="center">
  <a href="../../actions/workflows/ci.yml"><img src="../../actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <img src="https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white" alt="Python 3.12" />
  <img src="https://img.shields.io/badge/TypeScript-React-blue?logo=react&logoColor=white" alt="React + TypeScript" />
  <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white" alt="Docker" />
</p>

---

## Overview

MedCheck AI is a full-stack application built for the **ELEC 490 Capstone** at Queen's University. It uses a **Retrieval-Augmented Generation (RAG)** pipeline to evaluate medical claims against a curated knowledge base of peer-reviewed research from PubMed, WHO, arXiv, PLOS, and more.

Users submit a claim (as text, a URL, or an image), and the system:

1. **Retrieves** the most relevant research documents using vector similarity search (pgvector)
2. **Generates** an evidence-based verdict (e.g., **Accurate**, **Inaccurate**, **Partially Accurate**, **Misleading**, **Unverifiable**, **Outdated**, and more) using an LLM
3. **Cites** the exact source documents and similarity scores so the user can verify the answer

---

## Features

| Feature | Description |
| :--- | :--- |
| **RAG Pipeline** | Retrieves context from 15,000+ indexed medical documents before generating a response |
| **Multi-Input Support** | Accepts plain text, URLs (auto-extracts article content), and images (OCR via Tesseract) |
| **Evidence-Based Verdicts** | Every response includes source citations with document names and similarity scores |
| **Chrome Extension** | Highlight any text on the web → right-click → instant fact-check inline |
| **Query History** | Stores past queries and source documents in PostgreSQL for reference |
| **Off-Topic Guardrails** | Rejects non-medical queries to stay focused on its domain |
| **CI/CD Pipeline** | GitHub Actions → automated tests → deploy to Render (backend) + Vercel (frontend) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interfaces                       │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────┐  │
│  │  Web App      │  │ Chrome Ext.   │  │  API (Swagger)   │  │
│  │  (React/TS)   │  │ (Manifest V3) │  │  /docs           │  │
│  └──────┬───────┘  └──────┬────────┘  └────────┬─────────┘  │
└─────────┼─────────────────┼────────────────────┼────────────┘
          │                 │                    │
          ▼                 ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                            │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────┐  │
│  │  RAG Service    │  │  History       │  │  Input        │  │
│  │  (LlamaIndex)   │  │  Service       │  │  Processing   │  │
│  │                 │  │  (SQLModel)    │  │  (OCR/URL)    │  │
│  └───────┬────────┘  └───────┬────────┘  └───────────────┘  │
└──────────┼───────────────────┼──────────────────────────────┘
           │                   │
           ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│              Supabase (PostgreSQL + pgvector)                 │
│  ┌─────────────────────┐  ┌──────────────────────────────┐  │
│  │  knowledge_base      │  │  queryhistory                │  │
│  │  (vector embeddings) │  │  sourcedocumenthistory       │  │
│  └─────────────────────┘  └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Backend** | Python 3.12, FastAPI, LlamaIndex, SQLModel, Pydantic |
| **Frontend** | React 19, TypeScript, Vite, Tailwind CSS, Radix UI |
| **Chrome Extension** | Manifest V3, vanilla JS |
| **Database** | Supabase (PostgreSQL + pgvector) |
| **LLM & Embeddings** | OpenRouter API (OpenAI GPT-4o-mini + text-embedding-3-small) |
| **OCR** | Tesseract |
| **CI/CD** | GitHub Actions → Render (backend) + Vercel (frontend) |
| **Containerization** | Docker (multi-stage build) |

---

## Quick Start

### Option 1: Docker (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/sere-otubu/ELEC490_Group16_AI_Fake_News_Detection.git
cd ELEC490_Group16_AI_Fake_News_Detection

# 2. Copy and configure environment variables
cp .env.example .env
# Edit .env with your Supabase credentials and OpenRouter API key

# 3. Build and start the application
docker-compose build --no-cache backend
docker-compose up -d

# 4. (First time only) Initialize the database tables
docker-compose exec backend uv run python -m src.vector_db.run_init_db

# 5. (First time only) Load document embeddings
docker-compose exec backend uv run python src/vector_db/run_load_embeddings.py
```

The application will be available at **http://localhost:8000**.

### Option 2: Local Development

<details>
<summary><strong>Backend Setup</strong></summary>

```bash
# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r src/requirements.txt -r src/requirements-dev.txt

# Configure environment
cp .env.example .env
# Edit .env with your credentials

# Initialize DB (first time only)
python -m src.vector_db.run_init_db

# Start the backend
uvicorn src.main:app --reload --port 8000
```

</details>

<details>
<summary><strong>Frontend Setup</strong></summary>

```bash
cd frontend

# Install dependencies
pnpm install

# Start the dev server (proxies API requests to localhost:8000)
pnpm dev
```

The frontend dev server runs at **http://localhost:5173** and proxies `/rag` and `/history` routes to the backend.

</details>

<details>
<summary><strong>Chrome Extension Setup</strong></summary>

1. Open `chrome://extensions/` in Chrome
2. Enable **Developer mode** (top-right toggle)
3. Click **Load unpacked** and select the `chrome-extension/` directory
4. The MedCheck AI icon will appear in your toolbar

See [chrome-extension/README.md](chrome-extension/README.md) for detailed usage instructions.

</details>

---

## Project Structure

```
.
├── src/                        # Backend (FastAPI)
│   ├── main.py                 # Application entry point & SPA routing
│   ├── config.py               # Pydantic settings (env vars)
│   ├── schemas.py              # Shared Pydantic models
│   ├── rag/                    # RAG pipeline
│   │   ├── routes.py           #   POST /rag/query
│   │   ├── services.py         #   Query orchestration
│   │   ├── repositories.py     #   Vector store & LLM integration
│   │   ├── input_processing.py #   URL/image/OCR extraction
│   │   └── prompt.py           #   System prompt template
│   ├── history/                # Conversation history
│   │   ├── routes.py           #   GET/DELETE /history/*
│   │   ├── services.py         #   History CRUD logic
│   │   └── repositories.py     #   PostgreSQL queries
│   └── vector_db/              # Knowledge base management
│       ├── run_init_db.py      #   Create DB tables
│       ├── run_load_embeddings.py  # Index documents into pgvector
│       ├── fetch_pubmed.py     #   PubMed article fetcher
│       ├── fetch_arxiv.py      #   arXiv paper fetcher
│       ├── fetch_who.py        #   WHO document fetcher
│       └── fetch_plos.py       #   PLOS article fetcher
├── frontend/                   # Frontend (React + TypeScript + Vite)
│   ├── src/
│   │   ├── App.tsx             #   Main application component
│   │   ├── components/ui/      #   Radix UI components
│   │   ├── hooks/              #   Custom React hooks
│   │   └── lib/                #   API client & utilities
│   └── vite.config.ts          #   Vite configuration with API proxy
├── chrome-extension/           # Chrome Extension (Manifest V3)
│   ├── manifest.json           #   Extension configuration
│   ├── content.js              #   Page content script (text selection)
│   ├── background.js           #   Service worker (context menu)
│   └── popup.html/js/css       #   Extension popup UI
├── tests/                      # Test suite
│   ├── unit/                   #   Offline unit tests (pytest)
│   ├── quality/                #   RAG accuracy benchmarks
│   ├── differentiators/        #   Adversarial & robustness tests
│   ├── e2e/                    #   End-to-end latency benchmarks
│   └── load/                   #   Load testing (Locust)
├── docs/                       # Project documentation & reports
├── Dockerfile                  # Multi-stage build (frontend + backend)
├── docker-compose.yaml         # Container orchestration
└── .github/workflows/ci.yml   # CI/CD pipeline
```

---

## Environment Variables

Copy `.env.example` to `.env` and configure the following:

| Variable | Required | Description |
| :--- | :---: | :--- |
| `APP_DATABASE_URL` | ✅ | PostgreSQL connection string (Supabase recommended) |
| `APP_OPENROUTER_API_KEY` | ✅ | API key from [OpenRouter](https://openrouter.ai) |
| `APP_OPENROUTER_LLM_MODEL` | | LLM model (default: `openai/gpt-4o-mini`) |
| `APP_OPENROUTER_EMBEDDING_MODEL` | | Embedding model (default: `openai/text-embedding-3-small`) |
| `APP_EMBED_DIM` | | Embedding dimensions (default: `1536`) |
| `APP_SUPABASE_URL` | | Supabase project URL (for document storage) |
| `APP_SUPABASE_KEY` | | Supabase service role key |
| `APP_SUPABASE_BUCKET_NAME` | | Supabase storage bucket name |
| `APP_VECTOR_TABLE_NAME` | | Vector table name (default: `knowledge_base`) |
| `VITE_API_BASE_URL` | | Frontend API URL (default: same-origin) |

---

## API Reference

The backend exposes a Swagger UI at **`/docs`** and ReDoc at **`/redoc`** when running.

### Key Endpoints

| Method | Endpoint | Description |
| :---: | :--- | :--- |
| `GET` | `/health` | Health check |
| `POST` | `/rag/query` | Submit a medical claim for fact-checking |
| `GET` | `/history/sessions` | List all conversation sessions |
| `GET` | `/history/sessions/{id}` | Get a specific session with messages |
| `DELETE` | `/history/sessions/{id}` | Delete a session |

---

## Testing

The test suite includes **66 unit tests** and multiple live evaluation suites. See [tests/README.md](tests/README.md) for detailed instructions.

```bash
# Run offline unit tests (no backend required)
python -m pytest tests/unit/ -v --tb=short

# Run the full live test suite against a running backend
python tests/run_all_live.py --url http://localhost:8000
```

| Test Category | Description |
| :--- | :--- |
| **Unit Tests** | Schemas, config parsing, input processing, API contracts |
| **RAG Quality** | 25 curated medical claims evaluated for verdict accuracy |
| **Adversarial** | Prompt injection resistance and paraphrase stability |
| **Hallucination** | Citation validity — checks if cited documents actually exist |
| **Consistency** | Identical queries return identical verdicts |
| **Off-Topic** | Guardrail rejection for non-medical queries |
| **Latency** | End-to-end response time benchmarks |
| **Load** | Concurrent user simulation with Locust |

---

## CI/CD Pipeline

The project uses **GitHub Actions** for continuous integration and deployment:

```
Push/PR to main or dev
        │
        ▼
┌───────────────────┐
│  Backend Tests     │──► Python 3.12 + pytest
│  Frontend Lint     │──► pnpm lint + build
│  Docker Build      │──► Multi-stage image
└────────┬──────────┘
         │ (all pass)
         ▼
┌───────────────────┐
│  Deploy to Render  │──► Backend API
│  Deploy to Vercel  │──► Frontend SPA
└───────────────────┘
         │
         ▼
  Discord notification
```

---

## Team

**ELEC 490 — Group 16** | Queen's University

| Member | GitHub |
| :--- | :--- |
| Sere Otubu | [@sere-otubu](https://github.com/sere-otubu) |
| Ivan Samardzic | [@ivansamardzic](https://github.com/ivansamardzic) |
| Mihran Asadullah | [@Mihran03](https://github.com/Mihran03) |

---

<p align="center">
  Built with ❤️ at Queen's University · ELEC 490 Capstone 2025–2026
</p>
