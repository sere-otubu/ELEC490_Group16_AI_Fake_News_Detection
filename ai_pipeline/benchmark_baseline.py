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

def plot_confusion_matrix(df, save_path=None):
    labels = ["Accurate", "Misleading", "Harmful", "Unverifiable"]

    cm = confusion_matrix(df["truth"], df["pred"], labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Medical Misinformation Classification — Confusion Matrix")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_precision_recall(df, save_path=None):
    labels = ["Accurate", "Misleading", "Harmful", "Unverifiable"]

    report = classification_report(df["truth"], df["pred"], 
                                   labels=labels, output_dict=True)

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

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

# --- 1. SETUP ---
load_dotenv()
token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError("HF_TOKEN not found in .env file.")
login(token=token)

# --- CONFIGURATION ---
MASTER_LOG_FILE = "experiment_history_log_v2.csv"

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

# Uncomment to conduct quick data analysis
# # --- Quick Data Analysis ---
# print("\n--- ANALYZING CLAIM LENGTHS ---")

# # Convert the test split to a Pandas DataFrame for easier analysis
# df_analysis = pd.DataFrame(dataset['test'])

# # Calculate word counts for every claim
# df_analysis['word_count'] = df_analysis['claim'].apply(lambda x: len(str(x).split()))

# # 1. Get Statistics
# print("Word Count Statistics:")
# print(df_analysis['word_count'].describe())

# # 2. Print Random Examples to eyeball them
# print("\n--- 5 Random Examples ---")
# for i, row in df_analysis.sample(5).iterrows():
#     print(f"Length: {row['word_count']} words | Claim: {row['claim']}")

# # 3. Check for extremely short claims (potential noise)
# short_claims = df_analysis[df_analysis['word_count'] < 5]
# print(f"\nNumber of claims under 5 words: {len(short_claims)}")
# if not short_claims.empty:
#     print("Examples of very short claims:", short_claims['claim'].head().tolist())

# print("-" * 30)

test_data = dataset['test']

# --- 4. INFERENCE ---
results = []
print(f"Running inference on {len(test_data)} samples...")

run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
start_time = time.time()

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
        return "Harmful"
    if label == 1:
        return "Misleading"
    if label == 2:
        return "Accurate"
    if label == 3:
        return "Unverifiable"
    
    return "Unknown"

for i, row in enumerate(test_data):
    claim = row['claim']
    ground_truth = map_pubhealth_label(row['label'])

    system_prompt = """You are a medical fact-checker.
    Analyze the following claim and classify it into ONE of these categories based on these strict definitions:

    1. Accurate: The claim is factually true and supported by scientific consensus.
    2. Misleading: The claim may contain some truth but is cherry-picked, exaggerates findings, or presents correlation as causation.
    3. Harmful: The claim is factually FALSE, dangerous, or creates fear/panic without evidence (e.g., anti-vaccine myths, fake cures).
    4. Unverifiable: There is insufficient evidence to prove or disprove the claim.

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

    if "accurate" in ans:
        prediction = "Accurate"
    elif "misleading" in ans:
        prediction = "Misleading"
    elif "harmful" in ans:
        prediction = "Harmful"
    elif "unverifiable" in ans:
        prediction = "Unverifiable"
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

# Filter out "Unsure" predictions for cleaner classification report
df_valid = df[
    (df["pred"] != "Unsure") & 
    (df["truth"] != "Unknown")
]

# Check if we have any valid predictions
if len(df_valid) == 0:
    print("WARNING: All predictions are 'Unsure'. Using full dataset for metrics.")
    # Create empty df just to prevent crash in subsequent code
    df_valid = pd.DataFrame(columns=["truth", "pred"])
else:
    # Safe to generate report
    print("\n" + "="*30)
    print(f"REPORT FOR: {MODEL_ID}")
    print("="*30)
    print(classification_report(
        df_valid['truth'], 
        df_valid['pred'], 
        labels=["Accurate", "Misleading", "Harmful", "Unverifiable"],
        zero_division=0
    ))

# Get the dictionary version of the report to access specific numbers
report_dict = classification_report(
    df_valid["truth"],
    df_valid["pred"],
    labels=["Accurate", "Misleading", "Harmful", "Unverifiable"],
    output_dict=True,
    zero_division=0,
)

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

# Extract "Harmful" (Misinformation) metrics
harmful_p, harmful_r, harmful_f1, harmful_s = get_class_metrics(report_dict, "Harmful")

# Extract "Misleading" metrics
mislead_p, mislead_r, mislead_f1, mislead_s = get_class_metrics(report_dict, "Misleading")

# Extract "Unveriable" metrics
unverify_p, unverify_r, unverify_f1, unverify_s = get_class_metrics(report_dict, "Unverifiable")

# Extract "Accurate" (Accurate) metrics
accurate_p, accurate_r, accurate_f1, accurate_s = get_class_metrics(report_dict, "Accurate")

accuracy = round((df_valid["truth"] == df_valid["pred"]).mean(), 4)

# --- APPEND TO MASTER LOG ---
file_exists = os.path.isfile(MASTER_LOG_FILE)

with open(MASTER_LOG_FILE, mode='a', newline='') as file:
    writer = csv.writer(file)
    
    # Write NEW header with detailed columns
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
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        MODEL_ID,
        len(test_data),
        elapsed_time,
        accuracy,
        # Harmful
        harmful_p, harmful_r, harmful_f1, harmful_s,
        # Misleading
        mislead_p, mislead_r, mislead_f1, mislead_s,
        # Unverifiable
        unverify_p, unverify_r, unverify_f1, unverify_s,
        # Accurate
        accurate_p, accurate_r, accurate_f1, accurate_s,
        unique_filename,
    ])

print(f"Detailed results logged to: {MASTER_LOG_FILE}")
plot_confusion_matrix(df_valid, save_path=f"run_results/llama_3B_confusion_matrix.png")
plot_precision_recall(df_valid, save_path=f"run_results/llama_3B_precision_recall.png")