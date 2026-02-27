import os
import time
import datetime
import torch
import pandas as pd
import csv
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def extract_clean_answer(raw_text, model_name):
    """
    Universal extractor that handles different model formats
    """
    answer = raw_text
    
    # CASE A: Llama 3 format
    if "assistant<|end_header_id|>" in raw_text:
        answer = raw_text.split("assistant<|end_header_id|>")[-1]
        
    # CASE B: Qwen / ChatML format
    elif "<|im_start|>assistant" in raw_text:
        answer = raw_text.split("<|im_start|>assistant")[-1]

    # CASE C: Mistral Format
    elif "[/INST]" in raw_text:
        answer = raw_text.split("[/INST]")[-1]
        
    # Clean up all possible end-of-turn tokens
    cleanup_tokens = ["<|im_end|>", "<|eot_id|>", "<end_of_turn>", "</s>"]
    for token in cleanup_tokens:
        answer = answer.replace(token, "")
    
    return answer.strip()

def map_pubhealth_label(label):
    """
    Maps PubHealth original dataset labels to 'Accurate', 'Harmful', 'Misleading', 'Unverifiable'.
    """
    # PubHealth original labels:
    # 0 = 'false'
    # 1 = 'mixture'
    # 2 = 'true' 
    # 3 = 'unverified'

    # Ensure label is an integer to handle cases where it might be loaded as string "2"
    try:
        label = int(label)
    except (ValueError, TypeError):
        return "Unknown"

    if label == 0:
        return "False"
    if label == 1:
        return "Mixture"
    if label == 2:
        return "True"
    if label == 3:
        return "Unverified"
    
    return "Unknown"

# --- 1. SETUP ---
load_dotenv()
token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError("HF_TOKEN not found in .env file.")
login(token=token)

# --- CONFIGURATION ---
MASTER_LOG_FILE = "experiment_history_log_v3.csv"

if not os.path.exists("run_results"):
    os.makedirs("run_results")

# --- 2. MODEL SELECTION ---
# Toggle these to compare!
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct" 
# MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
# MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct" 
# MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

print(f"Loading {MODEL_ID}...")

# Optimized 4-bit config 
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto", 
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# --- 3. DATA ---
print("Loading PubHealth...")
dataset = load_dataset("ImperialCollegeLondon/health_fact", trust_remote_code=True)

test_data = dataset['test']

df = pd.DataFrame(test_data)[["claim", "explanation", "label", "sources"]]
df.to_csv("pubhealth_claim_explanation_label_sources.csv", index=False)
print(df.head())

# --- 4. INFERENCE ---
results = []
print(f"Running inference on {len(test_data)} samples...")

run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
start_time = time.time()

for i, row in enumerate(test_data):
    claim = row['claim']
    ground_truth = map_pubhealth_label(row['label'])

    system_prompt = """You are a medical fact-checker.
    Analyze the following claim and classify it into ONE of these categories based on these strict definitions:

    1. True: The claim is factually true and supported by scientific consensus.
    2. Mixture: The claim may contain some truth but is cherry-picked, exaggerates findings, or presents correlation as causation.
    3. False: The claim is factually false, dangerous, or creates fear/panic without evidence.
    4. Unverified: There is insufficient evidence to prove or disprove the claim.

    Return ONLY the category name. Do not explain.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Claim: {claim}"}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    outputs = pipe(
        prompt, 
        max_new_tokens=20,
        do_sample=False
    )
    
    raw = outputs[0]['generated_text']
    
    # Use the new universal extractor
    answer = extract_clean_answer(raw, MODEL_ID)

    ans = answer.lower().strip()

    if "true" in ans:
        prediction = "True"
    elif "mixture" in ans:
        prediction = "Mixture"
    elif "false" in ans:
        prediction = "False"
    elif "unverified" in ans:
        prediction = "Unverified"
    else:
        prediction = "Unsure"

    results.append({
        "claim": claim[:50],
        "truth": ground_truth,
        "pred": prediction,
        "raw": answer
    })
    
    if i % 10 == 0: print(f"Processed {i}...")

end_time = time.time()
elapsed_time = round(end_time - start_time, 2)

# --- 5. RESULTS SAVING & REPORTING ---
df = pd.DataFrame(results)
model_short_name = MODEL_ID.split('/')[-1]
unique_filename = f"run_results/results_{model_short_name}_{run_timestamp}.csv"
df.to_csv(unique_filename, index=False)

# 1. Drop rows where truth or pred is totally missing (NaN)
df = df.dropna(subset=["truth", "pred"])

# Keep "Unsure" predictions in evaluation so abstentions are penalized.
# Only remove rows with unknown ground-truth labels.
df_valid = df[
    (df["truth"] != "Unknown")
]

# Check if we have any valid predictions
if len(df_valid) == 0:
    print("WARNING: No valid rows remain after filtering unknown ground truth labels.")
    # Create empty df just to prevent crash in subsequent code
    df_valid = pd.DataFrame(columns=["truth", "pred"])
else:
    # Safe to generate report
    eval_labels = ["True", "Mixture", "False", "Unverified"]
    unsure_rate = round((df_valid["pred"] == "Unsure").mean(), 4)

    print("\n" + "="*30)
    print(f"REPORT FOR: {MODEL_ID}")
    print("="*30)
    print(f"Unsure prediction rate: {unsure_rate}")
    print(classification_report(
        df_valid['truth'], 
        df_valid['pred'], 
        labels=eval_labels,
        zero_division=0
    ))

# Get the dictionary version of the report to access specific numbers
eval_labels = ["True", "Mixture", "False", "Unverified"]
report_dict = classification_report(
    df_valid["truth"],
    df_valid["pred"],
    labels=eval_labels,
    output_dict=True,
    zero_division=0,
)

def bootstrap_macro_f1_std(df_eval, labels, n_bootstrap=300, random_state=42):
    if len(df_eval) == 0:
        return 0.0

    rng = np.random.default_rng(random_state)
    y_true = df_eval["truth"].to_numpy()
    y_pred = df_eval["pred"].to_numpy()
    size = len(df_eval)
    macro_f1_scores = []

    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, size, size=size)
        boot_report = classification_report(
            y_true[sample_idx],
            y_pred[sample_idx],
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        macro_f1_scores.append(boot_report.get("macro avg", {}).get("f1-score", 0.0))

    return round(float(np.std(macro_f1_scores)), 4)

# Helper to safely get metrics even if a class (e.g. "Unsure") is missing
def get_class_metrics(report, label):
    if label in report:
        return (
            round(report[label]['precision'], 4),
            round(report[label]['recall'], 4),
            round(report[label]['f1-score'], 4),
            report[label]['support']
        )
    return (0.0, 0.0, 0.0, 0)

# Extract "False" (Misinformation) metrics
false_p, false_r, false_f1, false_s = get_class_metrics(report_dict, "False")

# Extract "Mixture" metrics
mislead_p, mislead_r, mislead_f1, mislead_s = get_class_metrics(report_dict, "Mixture")

# Extract "Unverified" metrics
unverify_p, unverify_r, unverify_f1, unverify_s = get_class_metrics(report_dict, "Unverified")

# Extract "True" (Accurate) metrics
accurate_p, accurate_r, accurate_f1, accurate_s = get_class_metrics(report_dict, "True")

# Aggregate F1 metrics
macro_f1 = round(report_dict.get("macro avg", {}).get("f1-score", 0.0), 4)
weighted_f1 = round(report_dict.get("weighted avg", {}).get("f1-score", 0.0), 4)

# Class_F1_StdDev: spread across per-class F1 scores
class_f1_values = [
    report_dict.get(label, {}).get("f1-score", 0.0)
    for label in ["True", "Mixture", "False", "Unverified"]
]
class_f1_std = round(float(np.std(class_f1_values)), 4)

# F1_StdDev as uncertainty estimate: bootstrap std of macro-F1
f1_std = bootstrap_macro_f1_std(df_valid, eval_labels)

accuracy = round((df_valid["truth"] == df_valid["pred"]).mean(), 4)

# --- APPEND TO MASTER LOG ---
file_exists = os.path.isfile(MASTER_LOG_FILE)

with open(MASTER_LOG_FILE, mode='a', newline='') as file:
    writer = csv.writer(file)
    
    # Write NEW header with detailed columns
    if not file_exists:
        writer.writerow([
            "Timestamp", "Model_Name", "Samples", "Time_Seconds", "Accuracy", 
            "Macro_F1", "Weighted_F1", "F1_StdDev", "Class_F1_StdDev",
            "False_Precision", "False_Recall", "False_F1", "False_Support",
            "Mixture_Precision", "Mixture_Recall", "Mixture_F1", "Mixture_Support",
            "Unverified_Precision", "Unverified_Recall", "Unverified_F1", "Unverified_Support",
            "True_Precision", "True_Recall", "True_F1", "True_Support",
            "Result_File_Path"
        ])
    
    writer.writerow([
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        MODEL_ID,
        len(test_data),
        elapsed_time,
        accuracy,
        macro_f1,
        weighted_f1,
        f1_std,
        class_f1_std,
        # False (Misinformation)
        false_p, false_r, false_f1, false_s,
        # Mixture (Misleading)
        mislead_p, mislead_r, mislead_f1, mislead_s,
        # Unverified (Unverifiable)
        unverify_p, unverify_r, unverify_f1, unverify_s,
        # Accurate
        accurate_p, accurate_r, accurate_f1, accurate_s,
        unique_filename,
    ])

print(f"Detailed results logged to: {MASTER_LOG_FILE}")