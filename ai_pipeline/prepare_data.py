import json
import random
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

# --- CONFIGURATION ---
INPUT_FILE = "synthetic_data_balanced.json" 
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_DIR = "processed_data"

# --- 1. LOAD DATA ---
print(f"Loading {INPUT_FILE}...")
try:
    with open(INPUT_FILE, 'r') as f:
        raw_data = json.load(f)
except FileNotFoundError:
    print(f"Error: {INPUT_FILE} not found. Did you run the generator script (generate_synthetic_data.py)?")
    exit(1)

print(f"Found {len(raw_data)} examples.")

# --- 2. FORMAT FOR LLAMA 3 ---
print(f"Formatting data with tokenizer for {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

formatted_data = []

for item in raw_data:
    # Create the conversation structure
    conversation = [
        {"role": "system", "content": "You are a medical fact-checking assistant. Analyze the following text for misinformation."},
        {"role": "user", "content": item['input']},
        {"role": "assistant", "content": item['output']}
    ]
    
    # Apply the tokenizer's template to make it a single string
    # This ensures the model learns exactly when to stop generating
    try:
        text = tokenizer.apply_chat_template(conversation, tokenize=False)
        formatted_data.append({"text": text})
    except Exception as e:
        print(f"Skipping invalid item: {e}")

# --- 3. SPLIT TRAIN / VAL / TEST (80/10/10) ---
# Shuffle first to ensure randomness
random.seed(42) # Fixed seed for reproducibility
random.shuffle(formatted_data)

total_count = len(formatted_data)
train_count = int(total_count * 0.8)
val_count = int(total_count * 0.1)
# The rest goes to test (handling any rounding errors)

train_data = formatted_data[:train_count]
val_data = formatted_data[train_count : train_count + val_count]
test_data = formatted_data[train_count + val_count:]

print("-" * 30)
print(f"Total Samples:      {total_count}")
print(f"Training (80%):     {len(train_data)}")
print(f"Validation (10%):   {len(val_data)}")
print(f"Test (10%):         {len(test_data)}")
print("-" * 30)

# --- 4. SAVE TO DISK ---
dataset = DatasetDict({
    "train": Dataset.from_list(train_data),
    "validation": Dataset.from_list(val_data),
    "test": Dataset.from_list(test_data)
})

dataset.save_to_disk(OUTPUT_DIR)
print(f"Data processed and saved to '{OUTPUT_DIR}/'")