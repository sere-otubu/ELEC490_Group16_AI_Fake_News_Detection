import pandas as pd
from datasets import load_dataset
import re
import json
import os

# --- CONFIGURATION ---
OUTPUT_DIR = "training_data"
MIN_EXPLANATION_WORDS = 5
MAX_EXPLANATION_WORDS = 300  # Cap extremely long ones to save context window

# PubHealth Mapping (Same as your benchmark)
LABEL_MAP = {
    0: "Harmful",
    1: "Misleading",
    2: "Accurate",
    3: "Unverifiable"
}

def clean_text(text):
    """
    Basic cleaner to remove HTML tags, extra spaces, and common PubHealth noise.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags (e.g., <b>, <a href...>)
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def format_llama3_chat(row):
    """
    Formats a single row into Llama 3 Chat JSON structure.
    """
    claim = clean_text(row['claim'])
    explanation = clean_text(row['explanation'])
    label_idx = row['label']
    
    # Skip invalid labels or empty text
    if label_idx not in LABEL_MAP or not claim or not explanation:
        return None

    ground_truth_label = LABEL_MAP[label_idx]

    # --- THE SYSTEM PROMPT ---
    # We teach the model to ALWAYS explain first, then verdict.
    system_prompt = (
    "You are an expert medical fact-checker. "
    "Your task is to analyze the following claim to determine its accuracy. "
    "First, provide a step-by-step 'Analysis' based on medical evidence. "
    "Second, state the final 'Verdict' as one of: Accurate, Misleading, Harmful, or Unverifiable."
    )

    # --- THE EXPECTED OUTPUT ---
    # We explicitly format the assistant's reply to match the PubHealth style
    assistant_response = f"Analysis: {explanation}\n\nVerdict: {ground_truth_label}"

    # Llama 3 / OpenAI style standard format
    message = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Claim: {claim}"},
            {"role": "assistant", "content": assistant_response}
        ]
    }
    
    return message

# --- MAIN EXECUTION ---
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("Loading PubHealth dataset...")
dataset = load_dataset("ImperialCollegeLondon/health_fact", trust_remote_code=True)

# Define the splits we want to process
splits_to_process = {
    "train": "train_finetune.jsonl",
    "validation": "val_finetune.jsonl"
}

for split_name, filename in splits_to_process.items():
    print(f"\n--- Processing Split: {split_name.upper()} ---")
    
    # Check if the split exists (just in case)
    if split_name not in dataset:
        print(f"WARNING: Split '{split_name}' not found. Skipping.")
        continue
        
    raw_data = dataset[split_name]
    print(f"Original size: {len(raw_data)} rows")

    processed_rows = []
    skipped_count = 0

    for row in raw_data:
        # 1. Check lengths
        # Ensure explanation exists and is a string
        exp = row.get('explanation', '')
        if not isinstance(exp, str):
            skipped_count += 1
            continue
            
        exp_len = len(exp.split())
        
        if exp_len < MIN_EXPLANATION_WORDS:
            skipped_count += 1
            continue
            
        # 2. Format using the same function
        formatted_entry = format_llama3_chat(row)
        
        if formatted_entry:
            processed_rows.append(formatted_entry)
        else:
            skipped_count += 1

    print(f"Skipped {skipped_count} rows (empty/short/invalid).")
    print(f"Final {split_name} Size: {len(processed_rows)} rows.")

    # Save to file
    out_path = os.path.join(OUTPUT_DIR, filename)
    print(f"Saving to {out_path}...")
    
    with open(out_path, 'w', encoding='utf-8') as f:
        for entry in processed_rows:
            json.dump(entry, f)
            f.write('\n')

print("\nDONE! Train and Validation files are ready.")