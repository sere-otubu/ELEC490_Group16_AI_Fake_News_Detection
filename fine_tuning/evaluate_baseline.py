import torch
from transformers import pipeline
from datasets import load_dataset
from sklearn.metrics import classification_report
from tqdm import tqdm
import os

def evaluate_baseline():
    print("Loading RoBERTa model (zero-shot)...")
    device = 0 if torch.cuda.is_available() else -1
    
    try:
        classifier = pipeline(
            "zero-shot-classification",
            model="roberta-large-mnli",
            device=device
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    TEST_FILE = 'test.csv'
    if not os.path.exists(TEST_FILE):
        print(f"Error: {TEST_FILE} not found. Run 'python split_dataset.py' first.")
        return

    try:
        dataset = load_dataset('csv', data_files={'test': TEST_FILE})['test']
        
        # --- CHANGE 1: Update the label mapping ---
        # This new code maps all 3 of your labels (0, 1, 2) to strings.
        def map_labels_to_strings(example):
            if example['label'] == 0:
                return {"label_str": "False"}
            elif example['label'] == 1:
                return {"label_str": "Uncertain"}
            else: # label == 2
                return {"label_str": "True"}
                
        dataset = dataset.map(map_labels_to_strings)
        # -------------------------------------------
        
        print(f"Loaded {len(dataset)} test samples from {TEST_FILE}.")
    except Exception as e:
        print(f"Error loading {TEST_FILE}: {e}")
        return

    y_true = []
    y_pred = []

    print("\nStarting baseline evaluation...")
    for item in tqdm(dataset):
        text = item['text']
        true_label = item['label_str']

        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            continue
        
        try:
            # --- CHANGE 2: Update Candidate Labels ---
            # Give the model all 3 options to choose from.
            # Using descriptive labels helps the zero-shot model.
            candidate_labels = ["False or Misleading", "Uncertain or Incomplete", "True or Accurate"]
            result = classifier(
                text,
                candidate_labels=candidate_labels,
                hypothesis_template="This text is {}."
            )
            # ------------------------------------------

            # --- CHANGE 3: Update Prediction Logic ---
            # The old logic was for binary.
            # The new logic just takes the label with the highest score.
            # We then map it back to our simple class name.
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
    print("  Baseline Model Performance Report (roberta-large-mnli)")
    print("="*50)
    if y_true:
        # --- CHANGE 4: Update Target Names in Report ---
        # Add all 3 class names to the final report.
        target_names = ["False", "Uncertain", "True"]
        print(classification_report(y_true, y_pred, target_names=target_names))
        # -----------------------------------------------
    else:
        print("No predictions were made. Output is empty.")
    print("="*50)

if __name__ == "__main__":
    evaluate_baseline()