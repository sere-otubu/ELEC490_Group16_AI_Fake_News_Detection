from datasets import load_dataset
import torch
import wandb
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
import evaluate
import os
from huggingface_hub import login
from getpass import getpass
from dotenv import load_dotenv

# --- 1. Configuration (CHANGED FOR 3-CLASS BIOBERT) ---
MODEL_NAME = "dmis-lab/biobert-large-cased-v1.1-mnli"
OUTPUT_DIR = "biobert-fakehealth-finetuned"

# --- CHANGE 1: Define your 3 labels ---
LABEL_LIST = ["False", "Uncertain", "True"]
id2label = {i: label for i, label in enumerate(LABEL_LIST)}
label2id = {label: i for i, label in enumerate(LABEL_LIST)}

# --- CHANGE 2: Set your W&B Project Name ---
WANDB_PROJECT_NAME = "fake-health-biobert" # Or whatever you want

# --- CHANGE 3: Set your NEW Hub Model Name ---
# This should be a NEW repository name on your Hugging Face account
# Example: "sereotubu/biobert-fakehealth-v1"
HUB_MODEL_NAME = "sereotubu/biobert-finetune-v1" 
# -----------------------------------------------------------------

def login_to_huggingface():
    """Logs into Hugging Face Hub"""
    if HUB_MODEL_NAME == "YOUR_USERNAME/YOUR_NEW_MODEL_NAME":
        print("="*50)
        print("ERROR: Please update HUB_MODEL_NAME in the script!")
        print("This script cannot run until you provide a new model name.")
        print("="*50)
        exit()
        
    token = os.environ.get('HF_TOKEN')
    if token:
        print("Logging in with HF_TOKEN environment variable...")
        login(token=token)
    else:
        print("Please enter your Hugging Face token:")
        token = getpass()
        login(token=token)

def login_to_wandb():
    """Logs into Weights & Biases"""
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT_NAME # Set project name
    
    token = os.environ.get('WANDB_API_KEY')
    if token:
        print("Logging in to Weights & Biases with WANDB_API_KEY environment variable...")
        wandb.login(key=token)
    else:
        print("WANDB_API_KEY token is not found. Please add it to your environment.")
        exit()

def train_model():
    # --- 2. Load Data ---
    if not os.path.exists('train.csv') or not os.path.exists('validation.csv'):
        print("Error: 'train.csv' or 'validation.csv' not found.")
        print("Please run 'python split_dataset.py' first.")
        return

    dataset = load_dataset('csv', data_files={'train': 'train.csv', 'validation': 'validation.csv'})

    # --- 3. Preprocessing (CHANGED to combine Title + Text) ---
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        # Handle cases where title might be missing or None
        titles = [t if t else "" for t in examples['title']]
        texts = examples['text']
        
        # --- THIS IS THE CRITICAL FIX ---
        # Combine title and text using the separator token
        inputs = [f"{t} </s></s> {x}" for t, x in zip(titles, texts)]
        # --------------------------------
        
        # Tokenize the combined string
        return tokenizer(
            inputs, 
            padding="max_length", 
            truncation=True, 
            max_length=512 # Explicitly set max length
        )

    print("Tokenizing dataset (this may take a while)...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # --- 4. Load Model (CHANGED for 3 labels) ---
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(LABEL_LIST), # --- CHANGE ---
        id2label=id2label,         # --- CHANGE ---
        label2id=label2id          # --- CHANGE ---
    )

    # --- 5. Evaluation Metric (CHANGED for multi-class) ---
    f1_metric = evaluate.load("f1")
    acc_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        # --- CHANGE ---
        # Use 'weighted' F1 for multi-class, especially if imbalanced
        f1 = f1_metric.compute(
            predictions=predictions, 
            references=labels, 
            average="weighted"
        )['f1']
        
        acc = acc_metric.compute(
            predictions=predictions, 
            references=labels
        )['accuracy']
        
        # Return a dict, as required by the Trainer
        return {"f1": f1, "accuracy": acc}

    # --- 6. Set Training Arguments (CHANGED) ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=3,               
        per_device_train_batch_size=4,    # Keep low for a 'large' model
        per_device_eval_batch_size=4,     # Keep low for a 'large' model
        learning_rate=2e-5,               # Common default for fine-tuning
        load_best_model_at_end=True,      
        metric_for_best_model="f1",       # --- CHANGE ---
        logging_dir='./logs',
        logging_steps=100,                # Log progress every 100 steps
        push_to_hub=True,                 
        hub_model_id=HUB_MODEL_NAME,
        report_to="wandb",                # --- CHANGE: Enable W&B
    )

    # --- 7. Create Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    # --- 8. Train! ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} ---")
    print("Starting fine-tuning...")
    trainer.train()

    # --- 9. Save and Upload ---
    print("Training complete. Saving and uploading model...")
    trainer.save_model(OUTPUT_DIR)
    trainer.push_to_hub()

    print("="*50)
    print(f"✅ All done! Your fine-tuned model is saved locally in '{OUTPUT_DIR}'")
    print(f"and has been pushed to your Hugging Face Hub at: https://huggingface.co/{HUB_MODEL_NAME}")

if __name__ == "__main__":
    load_dotenv()
    try:
        login_to_huggingface()
        login_to_wandb()
        train_model()
    except Exception as e:
        print(f"\nAn error occurred: {e}")