"""
Module for fetching and saving articles from trusted web sources.

This module scrapes HTML content from specified URLs, extracts paragraph text
using BeautifulSoup, and saves the extracted content to local files in the data
directory. It's designed to collect articles from trusted health and medical
sources (e.g., WHO, CDC) for use in the RAG system's knowledge base.

Key functions:
- sanitize_filename: Converts URLs into safe filenames
- fetch_and_save_article: Fetches content from a URL and saves it locally
"""

import requests
from bs4 import BeautifulSoup
import os
from pathlib import Path
from urllib.parse import urlparse

# Configuration
DATA_DIR = Path("data")
# Example trusted sources (you can add more)
URLS_TO_SCRAPE = [
    "https://www.who.int/news-room/questions-and-answers/item/stress",
    "https://www.cdc.gov/flu/about/index.html"
]

def sanitize_filename(url):
    """Create a safe filename from a URL."""
    parsed = urlparse(url)
    # Use the path part of the URL, replace slashes with underscores
    name = parsed.path.strip("/").replace("/", "_")
    if not name:
        name = "index"
    return f"{parsed.netloc}_{name}.txt"

def fetch_and_save_article(url):
    try:
        print(f"Fetching: {url}")
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Basic extraction strategy: Get all paragraph text
        # For specific sites, you might target specific divs (e.g., soup.find('div', class_='content'))
        paragraphs = soup.find_all('p')
        text_content = "\n\n".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])

        if not text_content:
            print(f"Warning: No text found for {url}")
            return

        # Save to file
        filename = sanitize_filename(url)
        file_path = DATA_DIR / filename
        
        # Ensure data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Source URL: {url}\n\n")
            f.write(text_content)
        
        print(f"Saved to: {file_path}")

    except Exception as e:
        print(f"Error processing {url}: {e}")

if __name__ == "__main__":
    print("Starting article fetch...")
    for url in URLS_TO_SCRAPE:
        fetch_and_save_article(url)
    print("Done.")