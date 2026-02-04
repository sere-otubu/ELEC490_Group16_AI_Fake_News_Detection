from datasets import load_dataset
import os
from pathlib import Path

DATA_DIR = Path("data/pubmed")
os.makedirs(DATA_DIR, exist_ok=True)

# Load a subset of PubMed (medical abstracts)
print("Downloading dataset...")
dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")

print(f"Processing {len(dataset)} records...")

# Save top 1000 records as text files
for i, record in enumerate(dataset):
    if i >= 1000: break
    
    # Create a unique filename
    filename = f"pubmed_{record['pubid']}.txt"
    
    # Format the content: Question + Context + Long Answer
    content = f"""
    SOURCE: PubMed QA (ID: {record['pubid']})
    QUESTION: {record['question']}
    CONTEXT: {record['context']['contexts'][0]}
    ANSWER: {record['long_answer']}
    """
    
    with open(DATA_DIR / filename, "w", encoding="utf-8") as f:
        f.write(content)

print("Done! Files saved to data/pubmed/")