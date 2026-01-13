"""Script to load and index documents into the RAG vector database.

This script processes documents in the data folder and indexes them
into the vector store. It uses a tracking file to support incremental loading,
skipping files that have already been processed and haven't changed.
"""

import logging
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document

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


def load_and_index_documents(rag_service: RAGService) -> bool:
    """Main function to load and index ONLY new or changed documents."""
    try:
        logger.info("Starting smart document loading process...")

        loader = DocumentLoader()
        data_path = Path(settings.DATA_FOLDER)
        
        # 1. Load the history of what we've already done
        indexed_state = load_indexing_state(data_path)
        
        # 2. Scan for files
        all_files = loader.get_document_files(str(data_path))
        logger.info(f"Found {len(all_files)} total files in data folder.")

        # 3. Filter for new or changed files
        files_to_process = []
        skipped_count = 0
        new_state = indexed_state.copy()

        # Temporary list to update state only if successful
        pending_state_updates = {}

        for file_path in all_files:
            file_key = str(file_path.relative_to(data_path))
            current_hash = loader.get_file_hash(file_path)

            # If file is in history and hash matches, skip it
            if file_key in indexed_state and indexed_state[file_key] == current_hash:
                skipped_count += 1
                continue
            
            # Otherwise, add to process list
            files_to_process.append(file_path)
            pending_state_updates[file_key] = current_hash

        if skipped_count > 0:
            logger.info(f"Skipping {skipped_count} files that are already indexed and unchanged.")

        if not files_to_process:
            logger.info("No new or modified documents found. System is up to date.")
            return True

        logger.info(f"Preparing to index {len(files_to_process)} new/modified documents...")

        # 4. Load content only for the new files
        new_documents = []
        for file_path in files_to_process:
            logger.info(f"Loading content: {file_path.name}")
            docs = loader.load_specific_document(file_path)
            new_documents.extend(docs)

        if not new_documents:
            logger.warning("Files were detected but no content could be extracted.")
            return True

        # 5. Index the new content
        logger.info(f"Sending {len(new_documents)} document chunks to embedding model...")
        success = rag_service.index_documents(new_documents)

        # 6. Update the tracking file if successful
        if success:
            new_state.update(pending_state_updates)
            
            # Optional: Clean up state for files that were deleted from disk
            # current_keys = {str(p.relative_to(data_path)) for p in all_files}
            # keys_to_remove = [k for k in new_state.keys() if k not in current_keys]
            # for k in keys_to_remove:
            #     del new_state[k]

            save_indexing_state(data_path, new_state)
            
            logger.info("✓ Indexing complete and tracking file updated.")
            doc_count = rag_service.get_document_count()
            logger.info(f"Total documents in vector store: {doc_count}")
        else:
            logger.error("Indexing failed. Tracking file was NOT updated (will retry next time).")

        return success

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