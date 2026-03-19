# Using Supabase Cloud Database

Supabase gives you **one hosted PostgreSQL** with **pgvector** support. This app keeps using the same stack (PostgreSQL + pgvector for vectors, SQLModel for query history).

---

## Why Supabase for this project

- **One service** for both vector store (RAG) and query history.
- **No code changes** to RAG or history logic—only config.
- **pgvector** is supported; enable it in the dashboard and your existing `PGVectorStore` works.

---

## 1. Create a Supabase project

1. Sign up at [supabase.com](https://supabase.com) and create a new project.
2. Wait for the project to finish provisioning (database is ready when the dashboard is fully loaded).

---

## 2. Enable pgvector

1. In the Supabase dashboard, go to **Database** → **Extensions**.
2. Search for **vector** and enable the **vector** extension (pgvector).

The app runs `CREATE EXTENSION IF NOT EXISTS vector` on connect, but enabling it in the dashboard avoids permission issues and confirms it’s available.

---

## 3. Get your connection details

1. Go to **Project Settings** (gear) → **Database**.
2. Under **Connection string**, choose:
   - **URI** – use this for `APP_DATABASE_URL` (e.g. Session or Transaction pooler).
   - Or use **Connection parameters** (Host, Port, User, Password, Database) and set `APP_PG_*` instead.

**Recommended:** Use the **Session pooler** (port **5432**) or **Transaction pooler** (port **6543**) URI for serverless/long-running apps. Copy the URI and replace `[YOUR-PASSWORD]` with your database password.

Example format:

```text
postgresql://postgres.[project-ref]:[YOUR-PASSWORD]@aws-0-[region].pooler.supabase.com:6543/postgres
```

---

## 4. Configure your app

### Option A: Single connection string (recommended for Supabase)

In `.env`:

```env
# Supabase – single URL (replace [YOUR-PASSWORD] with your DB password)
APP_DATABASE_URL=postgresql://postgres.[project-ref]:[YOUR-PASSWORD]@aws-0-[region].pooler.supabase.com:6543/postgres

# If you use the URI above, you can leave these blank (they’re overridden by APP_DATABASE_URL)
APP_PG_HOST=localhost
APP_PG_PORT=5432
APP_PG_USER=
APP_PG_PASSWORD=
APP_PG_DATABASE=postgres
```

The app uses `APP_DATABASE_URL` for both the main connection and for building the pgvector store params, so one variable is enough.

### Option B: Individual parameters

Alternatively, fill in from the **Connection parameters** section in the dashboard:

```env
APP_PG_HOST=db.[project-ref].supabase.co
APP_PG_PORT=5432
APP_PG_USER=postgres
APP_PG_PASSWORD=your_password
APP_PG_DATABASE=postgres
```

---

## 5. Create history tables (first time only)

The app expects `queryhistory` and `sourcedocumenthistory` tables. Either:

- **Run the init script** against Supabase (with `APP_DATABASE_URL` or `APP_PG_*` pointing at Supabase):

  ```bash
  python -m src.vector_db.run_init_db
  ```

- Or run the same SQLModel metadata create in your own migration; the script is the simplest.

The RAG vector table is created automatically by LlamaIndex when you index documents.

---

## 6. Load documents into the vector store

Same as before; the app will use Supabase as the backend:

```bash
python -m src.vector_db.run_load_embeddings
```

---

## 7. Docker (optional)

If you run the backend in Docker and no longer use a local Postgres container:

  ```yaml
  environment:
    - APP_DATABASE_URL=${APP_DATABASE_URL}
    # ... other APP_* vars (Ollama, etc.)
  ```

Keep your `.env` with `APP_DATABASE_URL` so the container can connect to Supabase.

---

## Summary checklist

- [ ] Create Supabase project.
- [ ] Enable **vector** (pgvector) in Database → Extensions.
- [ ] Copy connection URI (or connection parameters) and set `APP_DATABASE_URL` (or `APP_PG_*`) in `.env`.
- [ ] Run `python -m src.vector_db.run_init_db` once to create history tables.
- [ ] Run `python -m src.vector_db.run_load_embeddings` to index documents.
- [ ] Update Docker to point at Supabase and remove local `db` service if desired.

No changes to RAG or history code are required; only configuration and environment variables.