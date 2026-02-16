import requests
from bs4 import BeautifulSoup
import os
import time
import logging
from pathlib import Path

# --- CONFIGURATION ---
DATA_DIR = Path("data/who")
TOPICS_FILE = Path("src/vector_db/data/topics.txt")
DELAY_BETWEEN_TOPICS = 0.5

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

def build_combined_index():
    index = {}
    headers = {"User-Agent": "Mozilla/5.0"}
    
    # 1. Fact Sheets
    try:
        url = "https://www.who.int/news-room/fact-sheets"
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.content, 'html.parser')
        for a in soup.find_all('a', href=True):
            if '/news-room/fact-sheets/detail/' in a['href']:
               href = a['href']
               full = href if href.startswith('http') else f"https://www.who.int{href}"
               text = a.get_text(strip=True).lower()
               index[text] = full
               slug = href.split('/')[-1].replace('-', ' ').lower()
               index[slug] = full
    except Exception as e:
        logger.error(f"Fact sheet index error: {e}")

    # 2. Health Topics
    try:
        url = "https://www.who.int/health-topics/"
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.content, 'html.parser')
        for a in soup.find_all('a', href=True):
            if '/health-topics/' in a['href']:
               href = a['href']
               # Filter out # anchors if they are just page jumps
               if href.endswith('#'): continue
               
               full = href if href.startswith('http') else f"https://www.who.int{href}"
               text = a.get_text(strip=True).lower()
               if text:
                   index[text] = full
                   slug = href.split('/')[-1].replace('-', ' ').lower()
                   index[slug] = full
    except Exception as e:
        logger.error(f"Health topics index error: {e}")
        
    logger.info(f"Built combined index with {len(index)} keys.")
    return index

def fetch_content(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200: return None, None
        
        soup = BeautifulSoup(resp.content, 'html.parser')
        
        # Try finding content
        content = (
            soup.find('div', class_='sf-detail-body-wrapper') or 
            soup.find('article') or 
            soup.find('main')
        )
        
        if content:
            for s in content(["script", "style", "nav", "header", "footer", "button", "iframe"]):
                s.decompose()
            for d in content.find_all("div", class_=["refine-slide", "sidebar", "related-content", "read-more"]):
                d.decompose()
            text = content.get_text(separator='\n', strip=True)
            if "All topics" in text[:50] and "Z" in text[:100]: text = ""
        else:
            text = ""
            
        title = soup.title.string.strip() if soup.title else "WHO Article"
        return title, text
    except Exception:
        return None, None

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    topics = load_topics()
    index = build_combined_index()
    if not index: return

    logger.info(f"Starting WHO extraction for {len(topics)} topics.")
    stats = {"found": 0}
    
    for i, topic in enumerate(topics):
        if i % 20 == 0: logger.info(f"Progress: {i}/{len(topics)}...")
        
        topic_lower = topic.lower()
        matched_url = None
        
        # Logic: 
        # 1. Exact match
        if topic_lower in index:
            matched_url = index[topic_lower]
        else:
            # 2. Key contained in topic (e.g. index="diabetes" in topic="type 2 diabetes")
            # Sort keys by length desc to match longest first
            for key in sorted(index.keys(), key=len, reverse=True):
                if len(key) > 3 and key in topic_lower:
                    matched_url = index[key]
                    break
        
        if matched_url:
            slug = matched_url.split('/')[-1]
            filename = DATA_DIR / f"who_{slug}.txt"
            if not filename.exists():
                title, text = fetch_content(matched_url)
                if text and len(text) > 200:
                    content = (
                        f"SEARCH_TOPIC: {topic}\n"
                        f"TITLE: {title}\n"
                        f"SOURCE_URL: {matched_url}\n"
                        f"TYPE: WHO Content\n\n"
                        f"CONTENT:\n{text}"
                    )
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(content)
                    stats["found"] += 1
                    logger.info(f"Saved: {title} for '{topic}'")
        
        time.sleep(DELAY_BETWEEN_TOPICS)

    logger.info(f"Job Complete. Saved {stats['found']} documents.")

if __name__ == "__main__":
    main()
