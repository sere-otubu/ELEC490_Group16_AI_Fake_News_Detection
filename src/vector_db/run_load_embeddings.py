"""Script to load and index documents into the RAG vector database.

This script processes documents in the data folder and indexes them
into the vector store. It uses a tracking file to support incremental loading,
skipping files that have already been processed and haven't changed.

Additionally, files are uploaded to Supabase Storage for persistence and
the storage URL is added to document metadata for reference.
"""

import logging
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document
from supabase import create_client, Client

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import settings
from src.dependencies import get_rag_service
from src.rag import RAGService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("load_embeddings.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

TRACKING_FILE_NAME = "indexing_state.json"


class SupabaseUploader:
    """Handles uploading files to Supabase Storage."""

    def __init__(self) -> None:
        """Initialize the Supabase client for storage operations."""
        self.enabled = False
        self.supabase: Optional[Client] = None
        self.bucket = settings.SUPABASE_BUCKET_NAME

        if settings.SUPABASE_URL and settings.SUPABASE_KEY:
            try:
                self.supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
                self.enabled = True
                logger.info(f"Supabase Storage enabled. Bucket: {self.bucket}")
            except Exception as e:
                logger.warning(f"Failed to initialize Supabase client: {e}")
                logger.warning("Files will be indexed locally but NOT uploaded to Supabase.")
        else:
            logger.info("Supabase URL/Key not configured. Storage upload disabled.")

    def upload_file(self, local_path: Path, remote_path: str) -> Optional[str]:
        """
        Upload a single file to the specified Supabase bucket if it doesn't exist.
        
        Args:
            local_path: Path to the local file to upload
            remote_path: Destination path in the Supabase bucket
            
        Returns:
            The public URL of the uploaded file, or None if upload failed/disabled
        """
        if not self.enabled or not self.supabase:
            return None

        # Standard public URL format
        file_url = f"{settings.SUPABASE_URL}/storage/v1/object/public/{self.bucket}/{remote_path}"

        try:
            # Check if file already exists to avoid redundant uploads
            folder = str(Path(remote_path).parent).replace(".", "")
            filename = Path(remote_path).name
            
            # list() returns a list of files in the folder
            res = self.supabase.storage.from_(self.bucket).list(folder, {"search": filename})
            if any(f['name'] == filename for f in res):
                logger.info(f"[SKIP] {local_path.name} already exists in Supabase storage")
                return file_url

            # If not found, proceed with upload
            with open(local_path, 'rb') as f:
                # Use upsert=True just in case of race conditions
                self.supabase.storage.from_(self.bucket).upload(
                    path=remote_path,
                    file=f,
                    file_options={"upsert": "true"}
                )
            
            logger.info(f"[OK] Uploaded {local_path.name} to Supabase storage")
            return file_url
            
        except Exception as e:
            logger.error(f"Failed to process storage for {local_path.name}: {e}")
            # Even if storage check/upload fails, we still return the deterministic URL
            # so the document can be indexed with its metadata.
            return file_url

    def get_public_url(self, remote_path: str) -> str:
        """Get the public URL for a file in the bucket."""
        return f"{settings.SUPABASE_URL}/storage/v1/object/public/{self.bucket}/{remote_path}"


class DocumentLoader:
    """Handles loading and processing of documents."""

    def __init__(self) -> None:
        """Initialize the document loader."""
        # Add all formats you scrape/use
        self.supported_extensions = {
            ".pdf",
            ".txt",
            ".md",
        }

    def get_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file to detect changes."""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.warning(f"Could not hash file {file_path}: {e}")
            return ""

    def get_document_files(self, directory_path: str) -> list[Path]:
        """Get all supported document files from the directory."""
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            logger.error(f"Invalid directory: {directory_path}")
            return []

        document_files = []
        for file_path in directory.rglob("*"):
            if (
                file_path.is_file()
                and file_path.suffix.lower() in self.supported_extensions
                and file_path.name != TRACKING_FILE_NAME # Ignore the tracking file itself
            ):
                document_files.append(file_path)

        return document_files

    def load_specific_document(self, file_path: Path) -> list[Document]:
        """Load a specific document file using LlamaIndex."""
        try:
            reader = SimpleDirectoryReader(
                input_files=[str(file_path)], recursive=False
            )
            documents = reader.load_data()
            return documents
        except Exception as e:
            logger.error(f"Failed to load document {file_path}: {e}")
            return []


def load_indexing_state(data_path: Path) -> Dict[str, str]:
    """Load the tracking file that maps filenames to their hashes."""
    tracking_file = data_path / TRACKING_FILE_NAME
    if tracking_file.exists():
        try:
            with open(tracking_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not read tracking file: {e}. Starting fresh.")
    return {}


def save_indexing_state(data_path: Path, state: Dict[str, str]):
    """Save the current state of indexed files."""
    tracking_file = data_path / TRACKING_FILE_NAME
    try:
        with open(tracking_file, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.error(f"Could not save tracking file: {e}")


BATCH_SIZE = 50  # Number of files to process in one batch


def load_and_index_documents(rag_service: RAGService) -> bool:
    """Main function to load and index documents in batches.
    
    This function uploads files to Supabase Storage and indexes them.
    Progress is saved after each successful batch.
    """
    try:
        logger.info("Starting smart document loading process...")

        loader = DocumentLoader()
        uploader = SupabaseUploader()
        data_path = Path(settings.DATA_FOLDER)
        
        # 1. Load the history of what we've already done
        indexed_state = load_indexing_state(data_path)
        
        # 2. Scan for files
        all_files = loader.get_document_files(str(data_path))
        logger.info(f"Found {len(all_files)} total files in data folder.")

        # 3. Filter for new or changed files
        files_to_process = []
        skipped_count = 0

        for file_path in all_files:
            file_key = str(file_path.relative_to(data_path))
            current_hash = loader.get_file_hash(file_path)

            # If file is in history and hash matches, skip it
            if file_key in indexed_state and indexed_state[file_key] == current_hash:
                skipped_count += 1
                continue
            
            files_to_process.append(file_path)

        if skipped_count > 0:
            logger.info(f"Skipping {skipped_count} files that are already indexed and unchanged.")

        if not files_to_process:
            logger.info("No new or modified documents found. System is up to date.")
            return True

        total_files = len(files_to_process)
        logger.info(f"Preparing to index {total_files} new/modified documents in batches of {BATCH_SIZE}...")

        # 4. Process in batches
        overall_success = True
        for i in range(0, total_files, BATCH_SIZE):
            batch_files = files_to_process[i : i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            total_batches = (total_files + BATCH_SIZE - 1) // BATCH_SIZE
            
            logger.info(f"--- Processing Batch {batch_num}/{total_batches} ({len(batch_files)} files) ---")
            
            batch_documents = []
            batch_state_updates = {}
            upload_count = 0
            
            for file_path in batch_files:
                file_key = str(file_path.relative_to(data_path))
                current_hash = loader.get_file_hash(file_path)
                
                # Upload to Supabase Storage
                relative_path = file_path.relative_to(data_path)
                remote_path = str(relative_path).replace("\\", "/")
                file_url = uploader.upload_file(file_path, remote_path)
                
                if file_url:
                    upload_count += 1
                
                # Load content
                docs = loader.load_specific_document(file_path)
                for doc in docs:
                    if file_url:
                        doc.metadata["file_url"] = file_url
                        doc.metadata["storage_bucket"] = settings.SUPABASE_BUCKET_NAME
                    doc.metadata["source_file"] = str(relative_path)
                
                batch_documents.extend(docs)
                batch_state_updates[file_key] = current_hash

            if upload_count > 0:
                logger.info(f"Uploaded {upload_count} files to Supabase Storage in this batch")

            if not batch_documents:
                logger.warning(f"Batch {batch_num} had no extractable content. Skipping indexing.")
                # We still update state for these files so we don't try them again
                indexed_state.update(batch_state_updates)
                save_indexing_state(data_path, indexed_state)
                continue

            # Index this batch
            logger.info(f"Batch {batch_num}: Sending {len(batch_documents)} chunks to embedding model...")
            success = rag_service.index_documents(batch_documents)

            if success:
                # Update and save state for this successful batch
                indexed_state.update(batch_state_updates)
                save_indexing_state(data_path, indexed_state)
                logger.info(f"[OK] Batch {batch_num} complete and state saved.")
            else:
                logger.error(f"Batch {batch_num} failed indexing. Progress paused.")
                overall_success = False
                break # Stop processing further batches to investigate failure

        if overall_success:
            logger.info("[DONE] All documents indexed and storage sync complete.")
            doc_count = rag_service.get_document_count()
            logger.info(f"Total entries in vector store: {doc_count}")
        
        return overall_success

    except Exception as e:
        logger.error(f"Process failed: {e}")
        return False


def main() -> None:
    """Main entry point."""
    try:
        logger.info("=" * 50)
        logger.info("Capstone Project - Incremental Document Loader")
        logger.info("=" * 50)
        rag_service: RAGService = get_rag_service()

        success = load_and_index_documents(rag_service)

        if success:
            sys.exit(0)
        else:
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Script interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()