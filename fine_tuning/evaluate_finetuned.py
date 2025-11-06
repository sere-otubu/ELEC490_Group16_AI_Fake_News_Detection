import torch
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import classification_report
from tqdm import tqdm
import os
import logging
from datetime import datetime

# --- 1. CHANGE THIS ---
# You must change this to the name of the model you
# will create AFTER you fine-tune it.
# Example: "sereotubu/biobert-fakehealth-v1"
HUB_MODEL_NAME = "sereotubu/biobert-finetune-v1"
# ----------------------

# --- 2. DEFINE LABELS ---
# These are the 3 classes for our project
LABEL_LIST = ["False", "Uncertain", "True"]
# ------------------------

# --- 3. SETUP LOGGING ---
LOG_FILE = f"finetuned_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
# ------------------------

def evaluate_finetuned():
    logging.info(f"Loading fine-tuned model from Hub: {HUB_MODEL_NAME}...")
    device = 0 if torch.cuda.is_available() else -1

    if HUB_MODEL_NAME == "YOUR_USERNAME/YOUR_NEW_FINETUNED_MODEL_NAME":
        logging.error("="*50)
        logging.error("ERROR: Please update HUB_MODEL_NAME in the script!")
        logging.error("This script cannot run until you provide the name of your fine-tuned model.")
        logging.error("="*50)
        return

    try:
        # --- 4. MANUALLY LOAD TOKENIZER (to force truncation) ---
        # We load the tokenizer from our fine-tuned model repo.
        # This is the same trick we used in the baseline script.
        logging.info(f"Explicitly loading tokenizer from {HUB_MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(
            HUB_MODEL_NAME,
            model_max_length=512  # Force max length
        )
        logging.info("Tokenizer loaded successfully.")

        # --- 5. LOAD PIPELINE (with our tokenizer) ---
        fine_tuned_classifier = pipeline(
            "text-classification",
            model=HUB_MODEL_NAME,
            tokenizer=tokenizer,  # Pass our configured tokenizer
            device=device,
            truncation=True       # Tell the pipeline to use truncation
        )
        logging.info("Fine-tuned model pipeline loaded successfully!")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        logging.error("Make sure your HUB_MODEL_NAME is correct and the model is public (or you are logged in).")
        return

    TEST_FILE = 'test.csv'
    if not os.path.exists(TEST_FILE):
        logging.error(f"Error: {TEST_FILE} not found. Run 'python split_dataset.py' first.")
        return

    try:
        dataset = load_dataset('csv', data_files={'test': TEST_FILE})['test']
        
        # --- 6. FIX LABEL MAPPING (for 3 classes) ---
        def map_labels_to_strings(example):
            if example['label'] == 0:
                return {"label_str": "False"}
            elif example['label'] == 1:
                return {"label_str": "Uncertain"}
            else: # label == 2
                return {"label_str": "True"}
        
        dataset = dataset.map(map_labels_to_strings)
        logging.info(f"Loaded {len(dataset)} test samples from {TEST_FILE}.")
    except Exception as e:
        logging.error(f"Error loading {TEST_FILE}: {e}")
        return

    # --- 7. FIX INPUT TEXT (Combine Title and Text) ---
    logging.info("Preparing input texts (combining title + text)...")
    input_texts = []
    y_true = []
    
    for item in dataset:
        title = item.get('title', '')
        text = item.get('text', '')
        
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            continue
            
        if title and isinstance(title, str) and len(title.strip()) > 0:
            input_text = title + " </s></s> " + text
        else:
            input_text = text
            
        input_texts.append(input_text)
        y_true.append(item['label_str'])
    
    logging.info(f"Total items to evaluate: {len(input_texts)}")
    
    # --- 8. RUN EVALUATION ---
    logging.info("\nStarting fine-tuned evaluation...")
    y_pred = []
    # Use batch_size for faster inference. Pass our combined text list.
    for result in tqdm(fine_tuned_classifier(input_texts, batch_size=16), total=len(input_texts)):
        # The pipeline automatically returns the label string (e.g., "True", "False")
        # based on the fine-tuned model's config.
        y_pred.append(result['label'])

    # --- 9. FIX REPORT (for 3 classes) ---
    logging.info("\n" + "="*50)
    logging.info(f"  Fine-Tuned Model Performance Report ({HUB_MODEL_NAME})")
    logging.info("="*50)
    if y_true:
        report = classification_report(y_true, y_pred, labels=LABEL_LIST, zero_division=0)
        logging.info("\n" + report)
    else:
        logging.warning("No predictions were made.")
    logging.info("="*50)

if __name__ == "__main__":
    evaluate_finetuned()