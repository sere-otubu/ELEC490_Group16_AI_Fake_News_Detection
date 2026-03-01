"""
Module for fetching and processing medical articles from the PubMed database.

This module searches the NCBI PubMed database for medical articles related to
specified topics using the Entrez E-utilities API (via Biopython), and saves the
article metadata (title, abstract, publication metadata) to local text files. It's
designed to populate the knowledge base for the RAG (Retrieval-Augmented Generation)
system with trusted medical research.

PubMed is the primary repository of biomedical literature, providing access to
millions of peer-reviewed articles from MEDLINE and life sciences journals.

Key features:
- Configurable topic list and results per topic
- Rate-limiting between API requests (respects PubMed API rate limits)
- Idempotency: Skips articles already saved to avoid duplicates
- Comprehensive logging of the extraction process
- Structured metadata preservation (PMID, title, abstract, URL)
- Minimum abstract length filtering to ensure data quality

Key functions:
- load_topics: Reads topics from a configuration file
- search_med_articles: Queries PubMed for article IDs matching a topic
- fetch_details: Retrieves full article metadata and abstracts from PubMed
- process_paper: Extracts and formats article data from XML records
- main: Orchestrates the entire fetch workflow
"""

from Bio import Entrez
import os
import time
import logging
from pathlib import Path

# --- CONFIGURATION ---
Entrez.email = "dummy@queensu.ca"  # Use a dummy or real email
DATA_DIR = Path("data/pubmed")
TOPICS_FILE = Path("src/vector_db/data/topics.txt")
MAX_RESULTS_PER_TOPIC = 20  # Keep this reasonable (20 * 500 = 10,000 docs)S
DELAY_BETWEEN_TOPICS = 1.0  # Seconds

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_topics():
    if not TOPICS_FILE.exists():
        logger.error(f"Topics file not found at {TOPICS_FILE}")
        return []
    with open(TOPICS_FILE, "r") as f:
        return [line.strip() for line in f if line.strip()]

def search_med_articles(query, max_results=20):
    try:
        # Search for IDs
        handle = Entrez.esearch(
            db="pubmed", 
            term=query, 
            retmax=max_results, 
            sort="relevance"
        )
        record = Entrez.read(handle)
        handle.close()
        return record["IdList"]
    except Exception as e:
        logger.warning(f"Search failed for topic '{query}': {e}")
        return []

def fetch_details(id_list):
    if not id_list:
        return None
    try:
        # Fetch actual content
        ids = ",".join(id_list)
        handle = Entrez.efetch(
            db="pubmed", 
            id=ids, 
            retmode="xml"
        )
        results = Entrez.read(handle)
        handle.close()
        return results
    except Exception as e:
        logger.warning(f"Fetch details failed: {e}")
        return None

def process_paper(paper, topic):
    """Extract useful text from a PubMed XML record."""
    try:
        medline = paper.get('MedlineCitation', {})
        article = medline.get('Article', {})
        pmid = medline.get('PMID', '0')
        
        title = article.get('ArticleTitle', 'No Title')
        
        # Extract Abstract
        abstract_obj = article.get('Abstract', {}).get('AbstractText', [])
        if isinstance(abstract_obj, list):
            abstract_text = " ".join([str(x) for x in abstract_obj])
        else:
            abstract_text = str(abstract_obj)

        if not abstract_text or len(abstract_text) < 50:
            return None

        # Build content
        content = (
            f"SEARCH_TOPIC: {topic}\n"
            f"TITLE: {title}\n"
            f"PMID: {pmid}\n"
            f"SOURCE_URL: https://pubmed.ncbi.nlm.nih.gov/{pmid}/\n"
            f"TYPE: PubMed Medical Abstract\n\n"
            f"ABSTRACT:\n{abstract_text}"
        )
        return pmid, content
    except Exception:
        return None

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    topics = load_topics()
    if not topics:
        logger.error("No topics loaded. Run generate_topics.py first.")
        return

    logger.info(f"Starting bulk extraction for {len(topics)} topics.")
    logger.info(f"Targeting max {MAX_RESULTS_PER_TOPIC} articles per topic.")

    total_saved = 0
    total_skipped = 0

    for i, topic in enumerate(topics):
        progress = (i + 1) / len(topics) * 100
        logger.info(f"[{progress:.1f}%] Processing topic: {topic}")

        ids = search_med_articles(topic, MAX_RESULTS_PER_TOPIC)
        if not ids:
            logger.info(f"  - No results found.")
            continue

        papers = fetch_details(ids)
        if not papers or 'PubmedArticle' not in papers:
            continue

        saved_for_topic = 0
        for paper in papers['PubmedArticle']:
            result = process_paper(paper, topic)
            if not result:
                continue
            
            pmid, content = result
            filename = DATA_DIR / f"pubmed_{pmid}.txt"

            # Skip if already exists (Idempotency)
            if filename.exists():
                total_skipped += 1
                continue

            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
            
            saved_for_topic += 1
            total_saved += 1

        logger.info(f"  - Saved {saved_for_topic} new articles.")
        
        # Respect API Rate Limits
        time.sleep(DELAY_BETWEEN_TOPICS)

    logger.info("=" * 40)
    logger.info(f"JOB COMPLETE")
    logger.info(f"Total New Documents Saved: {total_saved}")
    logger.info(f"Total Documents Skipped (Already existed): {total_skipped}")
    logger.info("=" * 40)

if __name__ == "__main__":
    main()