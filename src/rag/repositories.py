"""Repositories module for RAG database operations.

This module provides a communication layer for database operations in the RAG system,
including vector storage, document indexing, and query operations.
"""

import logging
import traceback
from contextlib import suppress
from typing import Any, List

import httpx
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from src.config import settings
from src.schemas import DocumentMetadata, QueryRequest, QueryResponse, SourceDocument
from src.rag.prompt import RAG_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

class OpenRouterEmbedding(BaseEmbedding):
    """Custom embedding class for OpenRouter's embedding API."""

    _api_key: str = PrivateAttr()
    _api_base: str = PrivateAttr()
    _model: str = PrivateAttr()
    _client: httpx.Client = PrivateAttr()

    def __init__(
        self,
        api_key: str,
        model: str = "google/gemini-embedding-001",
        api_base: str = "https://openrouter.ai/api/v1",
        **kwargs: Any,
    ) -> None:
        """Initialize OpenRouter embedding client."""
        super().__init__(**kwargs)
        self._api_key = api_key
        self._api_base = api_base.rstrip("/")
        self._model = model
        self._client = httpx.Client(timeout=60.0)

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text with retries."""
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                response = self._client.post(
                    f"{self._api_base}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/capstone-project",
                        "X-Title": "Capstone RAG System",
                    },
                    json={
                        "model": self._model,
                        "input": text,
                    },
                )
                
                # Check for HTTP errors before parsing
                if response.status_code != 200:
                    logger.warning(f"Embedding API attempt {attempt + 1} failed (Status {response.status_code}): {response.text}")
                    if response.status_code in [429, 502, 503, 504]:
                        import time
                        time.sleep(retry_delay * (2 ** attempt))
                        continue
                    response.raise_for_status()

                data = response.json()
                
                # Robust key checking to avoid crashes
                if "data" in data and len(data["data"]) > 0:
                    return data["data"][0]["embedding"]
                
                # If we got 200 OK but no data, it's an API-level error
                logger.error(f"Embedding API returned 200 OK but missing 'data': {data}")
                
                # Check for explicitly reported errors in the body
                if "error" in data:
                    err_msg = data["error"].get("message", "Unknown error")
                    logger.error(f"OpenRouter Error: {err_msg}")
                    
                if attempt < max_retries - 1:
                    import time
                    time.sleep(retry_delay)
                    continue
                
                raise KeyError(f"Embedding data missing from response: {data.keys()}")

            except httpx.RequestError as e:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(retry_delay)
                    continue
                raise
            except Exception as e:
                logger.error(f"Unexpected error in _get_embedding: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(retry_delay)
                    continue
                raise

        raise Exception("Failed to get embedding after multiple retries")

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a query."""
        return self._get_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a text."""
        return self._get_embedding(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async get embedding for a query."""
        return self._get_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Async get embedding for a text."""
        return self._get_embedding(text)


class RAGRepository:
    """Repository class for RAG database operations."""

    def __init__(self) -> None:
        """Initialize the RAG repository with database connection and models."""
        self.engine: None | Engine = None
        self.vector_store: None | PGVectorStore = None
        self.storage_context: None | StorageContext = None
        self.index: None | VectorStoreIndex = None
        self._actual_embed_dim: int | None = None
        self._setup_models()
        self._setup_database()

    def _setup_models(self) -> None:
        """Setup the LLM and embedding models using OpenRouter."""
        try:
            if not settings.OPENROUTER_API_KEY:
                raise ValueError(
                    "OPENROUTER_API_KEY is required. Get one from https://openrouter.ai"
                )

            # Setup LLM using OpenRouter (OpenAI-compatible API)
            Settings.llm = OpenAILike(
                model=settings.OPENROUTER_LLM_MODEL,
                api_key=settings.OPENROUTER_API_KEY,
                api_base=settings.OPENROUTER_BASE_URL,
                is_chat_model=True,
                timeout=120.0,
                default_headers={
                    "HTTP-Referer": "https://github.com/capstone-project",
                    "X-Title": "Capstone RAG System",
                },
            )
            logger.info("LLM configured: OpenRouter/%s", settings.OPENROUTER_LLM_MODEL)

            # Setup Embeddings using custom OpenRouter embedding class
            Settings.embed_model = OpenRouterEmbedding(
                api_key=settings.OPENROUTER_API_KEY,
                model=settings.OPENROUTER_EMBEDDING_MODEL,
                api_base=settings.OPENROUTER_BASE_URL,
            )
            logger.info(
                "Embedding model configured: OpenRouter/%s",
                settings.OPENROUTER_EMBEDDING_MODEL,
            )

            # Probe embedding dimension
            try:
                test_vector = Settings.embed_model.get_text_embedding("__dim_probe__")
                self._actual_embed_dim = len(test_vector)
                if self._actual_embed_dim != settings.EMBED_DIM:
                    logger.warning(
                        "Configured EMBED_DIM (%s) does not match model output dimension (%s). Using model output dimension.",
                        settings.EMBED_DIM,
                        self._actual_embed_dim,
                    )
                else:
                    logger.info(
                        "Embedding dimension confirmed: %s", self._actual_embed_dim
                    )
            except Exception as e:
                logger.error("Failed to probe embedding dimension: %s", e)
                logger.warning(
                    "Could not determine embedding dimension during setup; proceeding with configured EMBED_DIM=%s",
                    settings.EMBED_DIM,
                )
                self._actual_embed_dim = None

        except Exception as e:
            logger.error("Failed to setup models: %s", e)
            raise

    def _setup_database(self) -> None:
        """Setup the database connection, extension and vector store."""
        try:
            self.engine = create_engine(settings.database_url, echo=False)
            if self.engine:
                with self.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                    logger.info("Database connection established")
                    with suppress(Exception):
                        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                        conn.commit()
                        logger.info("pgvector extension ensured")

            embed_dim = self._actual_embed_dim or settings.EMBED_DIM
            if self._actual_embed_dim is None:
                logger.warning(
                    "Using configured EMBED_DIM=%s (model dimension probe failed earlier)",
                    settings.EMBED_DIM,
                )

            self.vector_store = PGVectorStore.from_params(
                database=settings.effective_pg_database,
                host=settings.effective_pg_host,
                password=settings.effective_pg_password,
                port=str(settings.effective_pg_port),
                user=settings.effective_pg_user,
                table_name=settings.VECTOR_TABLE_NAME,
                embed_dim=embed_dim,
            )
            logger.info(
                "Vector store configured with table '%s' (embed_dim=%s)",
                settings.VECTOR_TABLE_NAME,
                embed_dim,
            )
        except Exception as e:
            logger.error("Failed to setup database: %s", e)
            raise

    def index_documents(self, documents: list) -> bool:
        """Index documents into the vector store.

        Optimized for small document collections (single document with ~17 pages).
        Uses smaller chunk sizes and reduced overlap for better granularity.
        """
        try:
            logger.info("Creating index from documents...")
            logger.info(f"Number of documents to index: {len(documents)}")

            text_splitter = SentenceSplitter(
                chunk_size=256,
                chunk_overlap=20,
                separator=".\n",
                paragraph_separator="\n\n\n",
            )

            Settings.text_splitter = text_splitter

            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )

            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=self.storage_context,
                embed_model=Settings.embed_model,
                show_progress=True,
                transformations=[text_splitter],
            )

            if self.index:
                try:
                    docstore = self.index.docstore
                    node_count = (
                        len(docstore.docs) if hasattr(docstore, "docs") else "unknown"
                    )
                    logger.info(
                        f"Documents indexed successfully - Created {node_count} text chunks"
                    )
                except Exception as e:
                    logger.info("Documents indexed successfully")
                    logger.debug(f"Could not retrieve node count: {e}")

            return True

        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            traceback.print_exc()
            return False

    def query(self, query_request: QueryRequest) -> QueryResponse:
        """Query the RAG system.

        Args:
            query_text: The query text
            similarity_top_k: Number of similar documents to retrieve

        Returns:
            Optional[dict]: Dictionary containing 'response' and 'source_documents' or None if query failed
        """
        try:
            health = self.health_check(require_index=False)
            basic_health = {k: v for k, v in health.items() if k != "index"}
            if not all(basic_health.values()):
                logger.error("System not ready for queries - basic components failed")
                logger.error(f"Health status: {health}")
                raise ValueError("System not healthy for queries")

            doc_count = self.get_document_count()
            if doc_count == 0:
                logger.error("No documents in vector store - cannot perform queries")
                raise ValueError("No documents in vector store")

            logger.info(f"Vector store contains {doc_count} documents")

            if not self.index:
                logger.info("Index not initialized, creating from vector store...")
                if self.vector_store:
                    self.index = VectorStoreIndex.from_vector_store(self.vector_store)
                    logger.info("✓ Index successfully created from vector store")
                else:
                    logger.error("Vector store not initialized")
                    raise ValueError("Vector store not initialized")

            logger.info(f"Executing query: '{query_request.query[:50]}...'")

            # Adjust top_k to be at least 3 and at most 15
            optimized_top_k = min(query_request.top_k * 2 + 1, 15)

            query_engine = self.index.as_query_engine(
                text_qa_template=RAG_PROMPT_TEMPLATE,
                similarity_top_k=optimized_top_k,
                response_mode="compact",
                similarity_cutoff=0.6,
            )
            
            response = query_engine.query(query_request.query)

            source_documents: list[SourceDocument] = []
            if hasattr(response, "source_nodes") and response.source_nodes:
                top_nodes = response.source_nodes[: query_request.top_k]
                for node in top_nodes:
                    node_metadata = node.metadata if hasattr(node, "metadata") else {}
                    
                    # 1. Extract raw filename
                    raw_file_name = (
                        node_metadata.get("file_name")
                        or node_metadata.get("filename")
                        or node_metadata.get("file_path", "").split("/")[-1]
                        or "Unknown Document"
                    )

                    # 2. Initialize display variables
                    display_name = raw_file_name
                    source_link = node_metadata.get("file_path")
                    
                    # 3. INTELLIGENT PARSING LOGIC
                    # Check for PubMed Files (e.g., pubmed_12345.txt)
                    if "pubmed_" in raw_file_name:
                        # Extract ID
                        pmid = raw_file_name.split("pubmed_")[-1].replace(".txt", "")
                        if pmid.isdigit():
                            display_name = f"PubMed Article (ID: {pmid})"
                            source_link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

                    # Check for Scraped Websites (e.g., www.who.int_topic.txt)
                    elif "www." in raw_file_name or "http" in raw_file_name:
                        # Remove extension
                        clean = raw_file_name.replace(".txt", "")
                        # Reconstruct URL (assuming your scraper replaced / with _)
                        # "www.cdc.gov_flu" -> "www.cdc.gov/flu"
                        reconstructed_url = clean.replace("_", "/")
                        
                        # Add protocol if missing
                        if not reconstructed_url.startswith("http"):
                            reconstructed_url = f"https://{reconstructed_url}"
                            
                        display_name = clean.replace("_", "/") # Show cleaner path as title
                        source_link = reconstructed_url

                    # 4. Extract page number
                    page = (
                        node_metadata.get("page")
                        or node_metadata.get("page_number")
                        or node_metadata.get("page_label")
                    )
                    
                    if isinstance(page, str) and page.isdigit():
                        page = int(page)
                    elif not isinstance(page, int):
                        page = None

                    # 5. Create Metadata Object with refined names and links
                    structured_metadata = DocumentMetadata(
                        file_name=display_name,
                        page=page,
                        source=source_link, 
                    )

                    source_doc = SourceDocument(
                        content=node.text,
                        score=getattr(node, "score", 0.0),
                        metadata=structured_metadata,
                    )
                    source_documents.append(source_doc)

            logger.info("Query executed successfully")
            return QueryResponse(
                chat_response=str(response), source_documents=source_documents
            )

        except Exception as e:
            logger.error(f"Failed to execute query: {e}")

            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def get_document_count(self) -> int:
        """Get the total number of documents in the vector store.

        Returns:
            int: Number of documents
        """
        if not self.engine:
            logger.warning("get_document_count called before engine initialization")
            return 0
        try:
            with self.engine.connect() as conn:
                exists_result = conn.execute(
                    text(
                        "SELECT 1 FROM information_schema.tables WHERE table_name = :tbl"
                    ),
                    {"tbl": f"data_{settings.VECTOR_TABLE_NAME}"},
                ).fetchone()
                if not exists_result:
                    logger.info(
                        "Vector table '%s' not found (no rows to count yet)",
                        settings.VECTOR_TABLE_NAME,
                    )
                    return 0
                result = conn.execute(
                    text(f"SELECT COUNT(*) FROM data_{settings.VECTOR_TABLE_NAME}")
                )
                row = result.fetchone()
                count = int(row[0]) if row and row[0] is not None else 0
                return count
        except Exception as e:
            logger.error(
                "Failed to get document count from table '%s': %s",
                settings.VECTOR_TABLE_NAME,
                e,
            )
            return 0

    def clear_index(self) -> bool:
        """Clear all documents from the index.

        Returns:
            bool: True if clearing was successful
        """
        try:
            if not self.vector_store or not self.engine:
                return False

            with self.engine.connect() as conn:
                conn.execute(
                    text(f"DROP TABLE IF EXISTS data_{settings.VECTOR_TABLE_NAME}")
                )
                conn.commit()

            self._setup_database()
            self.index = None

            logger.info("Index cleared successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to clear index: {e}")
            return False

    def force_recreate_index(self) -> bool:
        """Force drop and recreate the vector table (dimension mismatch recovery)."""
        if not self.engine:
            logger.error("Cannot recreate index: engine not initialized")
            return False
        try:
            with self.engine.connect() as conn:
                conn.execute(
                    text(f"DROP TABLE IF EXISTS data_{settings.VECTOR_TABLE_NAME}")
                )
                conn.commit()
            logger.info(
                "Dropped existing vector table '%s'", settings.VECTOR_TABLE_NAME
            )
            self.vector_store = None
            self.index = None
            self._setup_database()
            return True
        except Exception as e:
            logger.error("Failed to force recreate index: %s", e)
            return False

    def health_check(self, require_index: bool = False) -> dict:
        """Perform a health check on the repository.

        Args:
            require_index: Whether to require an existing index for the check

        Returns:
            dict: Health check results
        """
        health = {
            "database": False,
            "vector_store": False,
            "models": False,
            "index": False,
        }

        try:
            if self.engine:
                with self.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                health["database"] = True

            health["vector_store"] = self.vector_store is not None

            health["models"] = (
                Settings.llm is not None and Settings.embed_model is not None
            )

            if require_index:
                health["index"] = self.index is not None
            else:
                health["index"] = True

        except Exception as e:
            logger.error(f"Health check failed: {e}")

        return health


rag_repository = RAGRepository()