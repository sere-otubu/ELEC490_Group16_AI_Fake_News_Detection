import torch
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import classification_report
from tqdm import tqdm
import os
import logging # Import logging
from datetime import datetime

# --- Setup Logging ---
# This will log to both a file and the console
LOG_FILE = f"baseline_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler() # This will also print to console
    ]
)
# ---------------------

# --- CHANGE 1: Use explicit, canonical model names ---
MODELS_TO_TEST = [
    "dmis-lab/biobert-large-cased-v1.1-mnli"
]

# --- CHANGE 2: Define mapping for cleaner prediction logic ---
LABEL_MAP = {
    "False or Misleading": "False",
    "Uncertain or Incomplete": "Uncertain",
    "True or Accurate": "True"
}
# These are the labels the classifier will see
CANDIDATE_LABELS = list(LABEL_MAP.keys())


def evaluate_baseline():
    TEST_FILE = 'test.csv'
    if not os.path.exists(TEST_FILE):
        logging.error(f"Error: {TEST_FILE} not found. Run the data splitting script first.")
        return

    try:
        dataset = load_dataset('csv', data_files={'test': TEST_FILE})['test']
        
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

    for model_name in MODELS_TO_TEST:
            logging.info("\n" + "="*50)
            logging.info(f"  Loading Baseline Model: {model_name}")
            logging.info("="*50)
            
            device = 0 if torch.cuda.is_available() else -1

            # --- START: NEW CODE BLOCK ---
            # Manually load tokenizer to force truncation settings
            try:
                logging.info(f"Explicitly loading tokenizer for {model_name}...")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    model_max_length=512  # <-- Force the max length on the tokenizer itself
                )
                logging.info("Tokenizer loaded successfully with model_max_length=512.")
            except Exception as e:
                logging.error(f"Error loading tokenizer {model_name}: {e}")
                continue # Skip to the next model
            # --- END: NEW CODE BLOCK ---
            
            try:
                # --- MODIFIED PIPELINE CALL ---
                logging.info("Loading pipeline with pre-loaded tokenizer...")
                classifier = pipeline(
                    "zero-shot-classification",
                    model=model_name,
                    tokenizer=tokenizer,  # <-- Pass the CONFIGURED tokenizer object
                    device=device,
                    truncation=True     # <-- Still tell the pipeline to use truncation
                )
                # --- END MODIFICATION ---
                logging.info("Model loaded successfully with pre-loaded tokenizer!")
            except Exception as e:
                logging.error(f"Error loading model {model_name}: {e}")
                continue # Skip to the next model

            y_true = []
            y_pred = []

            logging.info(f"\nStarting evaluation for {model_name}...")
            for item in tqdm(dataset):
                
                # --- CHANGE 3: Combine Title and Text ---
                # Use .get() for safety, in case columns are missing or
                # this is run on an old CSV without the 'title' column.
                title = item.get('title', '')
                text = item.get('text', '')

                if not text or not isinstance(text, str) or len(text.strip()) == 0:
                    continue

                # Combine title and text for the model
                if title and isinstance(title, str) and len(title.strip()) > 0:
                    # </s></s> is the separator RoBERTa was trained on
                    input_text = title + " </s></s> " + text
                else:
                    input_text = text
                # --------------------------------------------
                
                true_label = item['label_str']
                
                try:
                    result = classifier(
                        input_text,
                        candidate_labels=CANDIDATE_LABELS,
                        hypothesis_template="This text is {}."
                    )

                    # --- CHANGE 4: Use dictionary for cleaner logic ---
                    top_label = result['labels'][0]
                    predicted_label = LABEL_MAP.get(top_label) # Safe lookup
                    # -----------------------------------------------
                    
                    if predicted_label:
                        y_true.append(true_label)
                        y_pred.append(predicted_label)
                    else:
                        logging.warning(f"Model predicted an unknown label: {top_label}")

                except Exception as e:
                    # Log the error and the text that caused it (truncated)
                    logging.warning(f"\n[Warning] Error during prediction: {e}. Skipping item that starts with: {input_text[:50]}...")
                    continue

            logging.info("\n" + "="*50)
            logging.info(f"  Baseline Performance Report ({model_name})")
            logging.info("="*50)
            
            if y_true:
                target_names = ["False", "Uncertain", "True"]
                # The report is a multi-line string, so log it as such
                report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
                logging.info("\n" + report)
            else:
                logging.warning("No predictions were made. Output is empty.")
            logging.info("="*50)

if __name__ == "__main__":
    evaluate_baseline()