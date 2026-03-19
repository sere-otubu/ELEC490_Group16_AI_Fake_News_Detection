"""
Module for fetching and processing PubMed QA dataset from Hugging Face.

This module downloads the PubMed QA dataset (pqa_labeled split) from Hugging Face,
which contains medical question-answer pairs with context from PubMed abstracts.
It processes the dataset and saves the top 1000 records as individual text files
containing the question, context, and answer for use in the RAG (Retrieval-Augmented
Generation) system's knowledge base.

The PubMed QA dataset is a valuable resource for medical knowledge, covering
evidence-based question-answer pairs extracted from PubMed literature.

Key operations:
- Downloads the PubMed QA dataset from Hugging Face Hub
- Extracts and formats question-context-answer triplets
- Saves records as structured text files with metadata
"""

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