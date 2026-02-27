import os
import time
import datetime
import torch
import pandas as pd
import csv
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from sklearn.metrics import classification_report, confusion_matrix

# --- HELPER FUNCTIONS FOR PLOTTING ---
def plot_confusion_matrix(df, save_path=None):
    labels = ["Accurate", "Misleading", "Harmful", "Unverifiable"]
    cm = confusion_matrix(df["truth"], df["pred"], labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Fine-Tuned Medical Classification — Confusion Matrix")
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_precision_recall(df, save_path=None):
    labels = ["Accurate", "Misleading", "Harmful", "Unverifiable"]
    report = classification_report(df["truth"], df["pred"], labels=labels, output_dict=True, zero_division=0)
    precision = [report[l]["precision"] for l in labels]
    recall = [report[l]["recall"] for l in labels]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, precision, width, label="Precision")
    plt.bar(x + width/2, recall, width, label="Recall")
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Per-Class Precision & Recall")
    plt.legend()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# --- 1. SETUP ---
load_dotenv()
token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError("HF_TOKEN not found in .env file.")
login(token=token)

# --- CONFIGURATION ---
BASE_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_PATH = "sereotubu/Llama-3.2-3B-Medical-Fact-Checker" # Folder where train.py saved results
MASTER_LOG_FILE = "experiment_history_log_v2.csv"
RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

if not os.path.exists("run_results"):
    os.makedirs("run_results")

# --- 2. MODEL LOADING (Base + Adapter) ---
print(f"Loading Base Model: {BASE_MODEL_ID}...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load Base
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# Load Adapter
print(f"Loading Adapter from: {ADAPTER_PATH}...")
try:
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print("Success! Adapter loaded.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load adapter. Did you run train.py? Error: {e}")
    exit()

# --- 3. DATA LOADING (Parquet Fix) ---
print("Loading PubHealth Test Set (Parquet Revision)...")
try:
    dataset = load_dataset(
        "ImperialCollegeLondon/health_fact", 
        revision="refs/convert/parquet", 
        trust_remote_code=False
    )
    # Filter for standard labels only
    test_data = dataset['test']
    # OPTIONAL: Limit rows for speed testing? Remove [.select(range(100))] to run full test
    # test_data = test_data.select(range(50)) 
    print(f"Loaded {len(test_data)} samples for benchmarking.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

def map_pubhealth_label(label):
    try: label = int(label)
    except: return "Unknown"
    
    if label == 0: return "Harmful"
    if label == 1: return "Misleading"
    if label == 2: return "Accurate"
    if label == 3: return "Unverifiable"
    return "Unknown"

# --- 4. PARSING LOGIC ---
def extract_verdict(raw_text):
    """
    Parses 'Analysis: ... Verdict: Harmful' to extract just 'Harmful'
    """
    # Look for the last occurrence of "Verdict:" to avoid false positives in the analysis text
    if "Verdict:" in raw_text:
        parts = raw_text.rsplit("Verdict:", 1)
        potential_verdict = parts[-1].strip()
        # Grab first word, remove punctuation like '.'
        clean_verdict = re.split(r'\s|\.|,', potential_verdict)[0].lower()
        return clean_verdict
    return "unsure"

# --- 5. INFERENCE LOOP ---
results = []
print(f"Running inference on {len(test_data)} samples...")
start_time = time.time()

for i, row in enumerate(test_data):
    claim = row['claim']
    ground_truth = map_pubhealth_label(row['label'])

    # THE REASONING PROMPT (Matches Training)
    system_prompt = (
        "You are an expert medical fact-checker. "
        "Your task is to analyze the following claim to determine its accuracy. "
        "First, provide a step-by-step 'Analysis' based on medical evidence. "
        "Second, state the final 'Verdict' as one of: Accurate, Misleading, Harmful, or Unverifiable."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Claim: {claim}"}
    ]
    
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs, 
            max_new_tokens=300, # Enough space for Analysis
            do_sample=False,    # Deterministic
            temperature=0.0
        )
    
    # Decode only the response
    response_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    
    # Extract Verdict
    pred_raw = extract_verdict(response_text)
    
    # Map to final classes
    if "accurate" in pred_raw: prediction = "Accurate"
    elif "misleading" in pred_raw: prediction = "Misleading"
    elif "harmful" in pred_raw: prediction = "Harmful"
    elif "unverifiable" in pred_raw: prediction = "Unverifiable"
    else: prediction = "Unsure"

    results.append({
        "claim": claim[:50],
        "truth": ground_truth,
        "pred": prediction,
        "full_response": response_text
    })

    if i % 10 == 0: print(f"Processed {i}/{len(test_data)}")

end_time = time.time()
elapsed_time = round(end_time - start_time, 2)

# --- 6. REPORTING & SAVING ---
df = pd.DataFrame(results)
filename = f"run_results/finetuned_benchmark_{RUN_TIMESTAMP}.csv"
df.to_csv(filename, index=False)

# Filter for metrics
df_valid = df[(df["pred"] != "Unsure") & (df["truth"] != "Unknown")]

print("\n" + "="*30)
print(f"REPORT FOR: Fine-Tuned Adapter ({ADAPTER_PATH})")
print("="*30)

if len(df_valid) > 0:
    print(classification_report(
        df_valid['truth'], 
        df_valid['pred'], 
        labels=["Accurate", "Misleading", "Harmful", "Unverifiable"],
        zero_division=0
    ))
    
    # --- METRIC EXTRACTION FOR LOGGING ---
    report_dict = classification_report(
        df_valid["truth"], df_valid["pred"],
        labels=["Accurate", "Misleading", "Harmful", "Unverifiable"],
        output_dict=True, zero_division=0
    )

    def get_metrics(label):
        if label in report_dict:
            return (
                round(report_dict[label]['precision'], 4),
                round(report_dict[label]['recall'], 4),
                round(report_dict[label]['f1-score'], 4),
                report_dict[label]['support']
            )
        return (0.0, 0.0, 0.0, 0)

    harm_p, harm_r, harm_f1, harm_s = get_metrics("Harmful")
    mis_p, mis_r, mis_f1, mis_s = get_metrics("Misleading")
    unv_p, unv_r, unv_f1, unv_s = get_metrics("Unverifiable")
    acc_p, acc_r, acc_f1, acc_s = get_metrics("Accurate")
    accuracy = round((df_valid["truth"] == df_valid["pred"]).mean(), 4)

    # --- 7. APPEND TO MASTER LOG ---
    file_exists = os.path.isfile(MASTER_LOG_FILE)
    with open(MASTER_LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                "Timestamp", "Model_Name", "Samples", "Time_Seconds", "Accuracy", 
                "Harmful_Precision", "Harmful_Recall", "Harmful_F1", "Harmful_Support",
                "Misleading_Precision", "Misleading_Recall", "Misleading_F1", "Misleading_Support",
                "Unverifiable_Precision", "Unverifiable_Recall", "Unverifiable_F1", "Unverifiable_Support",
                "Accurate_Precision", "Accurate_Recall", "Accurate_F1", "Accurate_Support",
                "Result_File_Path"
            ])
        
        writer.writerow([
            RUN_TIMESTAMP,
            f"{BASE_MODEL_ID} + {ADAPTER_PATH}",
            len(test_data),
            elapsed_time,
            accuracy,
            harm_p, harm_r, harm_f1, harm_s,
            mis_p, mis_r, mis_f1, mis_s,
            unv_p, unv_r, unv_f1, unv_s,
            acc_p, acc_r, acc_f1, acc_s,
            filename,
        ])

    print(f"Detailed results logged to {MASTER_LOG_FILE}")
    
    # Generate Plots
    plot_confusion_matrix(df_valid, save_path=f"run_results/ft_confusion_{RUN_TIMESTAMP}.png")
    plot_precision_recall(df_valid, save_path=f"run_results/ft_prec_recall_{RUN_TIMESTAMP}.png")

else:
    print("WARNING: No valid predictions parsed. Check CSV for format errors.")