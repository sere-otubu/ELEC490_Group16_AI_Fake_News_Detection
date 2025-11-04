import torch
from transformers import pipeline
from datasets import load_dataset
from sklearn.metrics import classification_report
from tqdm import tqdm
import os
from datetime import datetime

# --- CHANGE 1: Define a list of models to test ---
# We use the two models that are compatible with the zero-shot pipeline
MODELS_TO_TEST = [
    "roberta-large-mnli", 
    "dmis-lab/biobert-v1.1-mnli"
]

def evaluate_baseline():
    TEST_FILE = 'test.csv'
    if not os.path.exists(TEST_FILE):
        print(f"Error: {TEST_FILE} not found. Run 'python split_dataset.py' first.")
        return

    try:
        dataset = load_dataset('csv', data_files={'test': TEST_FILE})['test']
        
        # This is the 3-class mapping you need
        def map_labels_to_strings(example):
            if example['label'] == 0:
                return {"label_str": "False"}
            elif example['label'] == 1:
                return {"label_str": "Uncertain"}
            else: # label == 2
                return {"label_str": "True"}
                
        dataset = dataset.map(map_labels_to_strings)
        print(f"Loaded {len(dataset)} test samples from {TEST_FILE}.")
    except Exception as e:
        print(f"Error loading {TEST_FILE}: {e}")
        return

    # --- CHANGE 2: Create a loop to run for each model ---
    for model_name in MODELS_TO_TEST:
        print("\n" + "="*50)
        print(f"  Loading Baseline Model: {model_name}")
        print("="*50)
        
        device = 0 if torch.cuda.is_available() else -1
        
        try:
            classifier = pipeline(
                "zero-shot-classification",
                model=model_name,  # <-- Use the model_name from the loop
                device=device
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            continue # Skip to the next model

        y_true = []
        y_pred = []

        print(f"\nStarting evaluation for {model_name}...")
        for item in tqdm(dataset):
            text = item['text']
            true_label = item['label_str']

            if not text or not isinstance(text, str) or len(text.strip()) == 0:
                continue
            
            try:
                # --- CHANGE 3: Update Candidate Labels (Corrected for 3 classes) ---
                candidate_labels = ["False or Misleading", "Uncertain or Incomplete", "True or Accurate"]
                result = classifier(
                    text,
                    candidate_labels=candidate_labels,
                    hypothesis_template="This text is {}."
                )
                # ------------------------------------------

                # --- CHANGE 4: Update Prediction Logic (Corrected for 3 classes) ---
                top_label = result['labels'][0]
                
                if top_label == "False or Misleading":
                    predicted_label = "False"
                elif top_label == "Uncertain or Incomplete":
                    predicted_label = "Uncertain"
                else: # "True or Accurate"
                    predicted_label = "True"
                # -----------------------------------------

                y_true.append(true_label)
                y_pred.append(predicted_label)
            except Exception as e:
                print(f"\n[Warning] Error during prediction: {e}. Skipping item.")
                continue

        print("\n" + "="*50)
        print(f"  Baseline Performance Report ({model_name}) [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
        print("="*50)
        if y_true:
            target_names = ["False", "Uncertain", "True"]
            print(classification_report(y_true, y_pred, target_names=target_names))
        else:
            print("No predictions were made. Output is empty.")
        print("="*50)

if __name__ == "__main__":
    evaluate_baseline()