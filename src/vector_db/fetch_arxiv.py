"""
Module for fetching and processing ArXiv papers related to specified topics.

This module loads medical and AI-related topics from a topics file, searches the
ArXiv API for relevant papers in each topic, and saves the paper metadata
(title, abstract, URL, etc.) to local text files. It's designed to populate the
knowledge base for the RAG system with peer-reviewed research papers.

Key features:
- Configurable topic list and results per topic
- Rate-limiting between API requests
- Duplicate detection to avoid saving papers twice
- Comprehensive logging of the extraction process
- Structured metadata preservation (title, abstract, published date, URLs)

Key functions:
- load_topics: Reads topics from a configuration file
- fetch_arxiv_papers: Queries ArXiv API for papers on a given topic
- process_and_save_paper: Processes and saves paper metadata to files
- main: Orchestrates the entire fetch workflow
"""

import arxiv
import os
import time
import logging
from pathlib import Path

# --- CONFIGURATION ---
DATA_DIR = Path("data/arxiv")
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

def fetch_arxiv_papers(topic):
    """
    Search ArXiv for papers related to the topic.
    Prioritize CS (Computer Science) and Stats categories for AI/Fake News context,
    but ArXiv search is broad enough.
    """
    client = arxiv.Client()
    
    # Construct a query. 
    # specific categories: cat:cs.AI OR cat:cs.CL OR cat:cs.LG
    # We combine topic with general health/misinformation context or just search the topic?
    # Given the topics are medical (e.g. "Diabetes"), searching ArXiv might return bio-physics or quantitative bio.
    # Let's try to keep it relevant. For general medical topics, ArXiv might be less dense than PubMed,
    # but for "Artificial Intelligence" or specific technical topics it's great.
    # If the user wants "Fake News Detection" context, we might filter.
    # HOWEVER, the user asked for "trusted peer reviewed sources" for knowledge base.
    # Let's search the topic directly but sort by relevance.
    
    search = arxiv.Search(
        query=f'all:"{topic}"',
        max_results=MAX_RESULTS_PER_TOPIC,
        sort_by=arxiv.SortCriterion.Relevance
    )

    results = []
    try:
        for result in client.results(search):
            results.append(result)
    except Exception as e:
        logger.error(f"Error fetching ArXiv for topic '{topic}': {e}")
    
    return results

def process_and_save_paper(result, topic):
    try:
        paper_id = result.entry_id.split('/')[-1]
        title = result.title
        summary = result.summary.replace("\n", " ")
        published = result.published.strftime("%Y-%m-%d")
        source_url = result.entry_id # Use abstract URL for better viewing
        pdf_url = result.pdf_url
        
        # Build content
        content = (
            f"SEARCH_TOPIC: {topic}\n"
            f"TITLE: {title}\n"
            f"ARXIV_ID: {paper_id}\n"
            f"PUBLISHED: {published}\n"
            f"SOURCE_URL: {source_url}\n"
            f"PDF_URL: {pdf_url}\n"
            f"TYPE: ArXiv Preprint\n\n"
            f"ABSTRACT:\n{summary}"
        )
        
        filename = DATA_DIR / f"arxiv_{paper_id}.txt"
        
        # Skip if already exists
        if filename.exists():
            return False

        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        
        return True
    except Exception as e:
        logger.error(f"Error processing paper {result.title}: {e}")
        return False

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    topics = load_topics()
    if not topics:
        logger.error("No topics loaded.")
        return

    logger.info(f"Starting ArXiv extraction for {len(topics)} topics.")
    
    total_new = 0
    
    for i, topic in enumerate(topics):
        logger.info(f"[{i+1}/{len(topics)}] Processing topic: {topic}")
        
        # Retrieve papers
        papers = fetch_arxiv_papers(topic)
        
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
