"""
Module for fetching and processing articles from the PLOS (Public Library of Science) API.

This module searches the PLOS API for peer-reviewed open-access articles related to
specified topics, and saves the article metadata (title, abstract, publication date, URL)
to local text files. It's designed to populate the knowledge base for the RAG 
system with high-quality, open-access research.

PLOS publishes multidisciplinary open-access research across medicine, biology, and
related fields, making it an excellent source for trustworthy scientific information.

Key features:
- Configurable topic list and results per topic
- Rate-limiting between API requests
- Duplicate detection to avoid saving articles twice
- Comprehensive logging of the extraction process
- Structured metadata preservation (title, abstract, publication date, DOI, URLs)

Key functions:
- load_topics: Reads topics from a configuration file
- search_plos: Queries the PLOS API for articles matching a topic
- process_and_save_paper: Processes and saves article metadata to files
- main: Orchestrates the entire fetch workflow
"""

import requests
import os
import time
import logging
from pathlib import Path

# --- CONFIGURATION ---
DATA_DIR = Path("data/plos")
TOPICS_FILE = Path("src/vector_db/data/topics.txt")
MAX_RESULTS_PER_TOPIC = 10
DELAY_BETWEEN_TOPICS = 2.0  # Seconds

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

def search_plos(query):
    """
    Search PLOS API for papers.
    Docs: http://api.plos.org/solr/search
    """
    base_url = "http://api.plos.org/search"
    params = {
        "q": f'title:"{query}" OR abstract:"{query}"',
        "fl": "id,title,abstract,publication_date,score",
        "wt": "json",
        "rows": MAX_RESULTS_PER_TOPIC
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get('response', {}).get('docs', [])
    except Exception as e:
        logger.error(f"Error fetching PLOS for topic '{query}': {e}")
        return []

def process_and_save_paper(doc, topic):
    try:
        # PLOS IDs are DOIs usually
        paper_id = doc.get('id', '')
        # Create a safe filename from ID (remove slashes/dots)
        safe_id = paper_id.replace('/', '_').replace('.', '_')
        
        title = doc.get('title_display', doc.get('title', 'No Title'))
        abstract = doc.get('abstract', ['No Abstract'])[0] if isinstance(doc.get('abstract'), list) else doc.get('abstract', 'No Abstract')
        published = doc.get('publication_date', 'Unknown')
        
        # Build content
        content = (
            f"SEARCH_TOPIC: {topic}\n"
            f"TITLE: {title}\n"
            f"PLOS_ID: {paper_id}\n"
            f"PUBLISHED: {published}\n"
            f"SOURCE_URL: https://journals.plos.org/plosone/article?id={paper_id}\n"
            f"TYPE: PLOS Article\n\n"
            f"ABSTRACT:\n{abstract}"
        )
        
        filename = DATA_DIR / f"plos_{safe_id}.txt"
        
        # Skip if already exists
        if filename.exists():
            return False

        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        
        return True
    
    except Exception as e:
        logger.error(f"Error processing PLOS paper {doc.get('id')}: {e}")
        return False

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    topics = load_topics()
    if not topics:
        logger.error("No topics loaded.")
        return

    logger.info(f"Starting PLOS extraction for {len(topics)} topics.")
    
    total_new = 0
    
    for i, topic in enumerate(topics):
        logger.info(f"[{i+1}/{len(topics)}] Processing topic: {topic}")
        
        # Retrieve papers
        papers = search_plos(topic)
        
        saved_count = 0
        for paper in papers:
            if process_and_save_paper(paper, topic):
                saved_count += 1
        
        if saved_count > 0:
            logger.info(f"  - Saved {saved_count} new papers.")
            total_new += saved_count
        else:
            logger.info(f"  - No new papers saved.")
            
        time.sleep(DELAY_BETWEEN_TOPICS)

    logger.info(f"Job Complete. Total new documents: {total_new}")

if __name__ == "__main__":
    main()
