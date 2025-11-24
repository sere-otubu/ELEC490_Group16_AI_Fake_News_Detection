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
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

# --- 1. SETUP ---
load_dotenv()
token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError("HF_TOKEN not found in .env file.")
login(token=token)

# --- CONFIGURATION ---
MASTER_LOG_FILE = "experiment_history_log.csv"

if not os.path.exists("run_results"):
    os.makedirs("run_results")

# --- 2. MODEL SELECTION ---
# Toggle these to compare!
# MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct" 
# MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
# MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct" 
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

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
test_data = dataset['test'].filter(lambda x: x['label'] in [0, 2]).select(range(50))

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

for i, row in enumerate(test_data):
    claim = row['claim']
    ground_truth = "True" if row['label'] == 2 else "False"

    messages = [
        {"role": "system", "content": "You are a medical fact-checker. Classify the following claim as 'True' or 'False' only."},
        {"role": "user", "content": f"Claim: {claim}\n\nVerdict:"}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    outputs = pipe(
        prompt, 
        max_new_tokens=5,
        do_sample=False
    )
    
    raw = outputs[0]['generated_text']
    
    # Use the new universal extractor
    answer = extract_clean_answer(raw, MODEL_ID)

    prediction = "Unsure"
    if "true" in answer.lower(): prediction = "True"
    elif "false" in answer.lower(): prediction = "False"

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

# Get the dictionary version of the report to access specific numbers
report_dict = classification_report(df['truth'], df['pred'], output_dict=True, zero_division=0)

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

# Extract "True" (Accurate) metrics
true_p, true_r, true_f1, true_s = get_class_metrics(report_dict, "True")

accuracy = round(report_dict['accuracy'], 4)

print("\n" + "="*30)
print(f"REPORT FOR: {MODEL_ID}")
print("="*30)
print(classification_report(df['truth'], df['pred'], zero_division=0))

# --- APPEND TO MASTER LOG ---
file_exists = os.path.isfile(MASTER_LOG_FILE)

with open(MASTER_LOG_FILE, mode='a', newline='') as file:
    writer = csv.writer(file)
    
    # Write NEW header with detailed columns
    if not file_exists:
        writer.writerow([
            "Timestamp", "Model_Name", "Samples", "Time_Seconds", "Accuracy", 
            "False_Precision", "False_Recall", "False_F1", "False_Support",
            "True_Precision", "True_Recall", "True_F1", "True_Support",
            "Result_File_Path"
        ])
    
    writer.writerow([
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        MODEL_ID,
        len(test_data),
        elapsed_time,
        accuracy,
        false_p, false_r, false_f1, false_s,
        true_p, true_r, true_f1, true_s,
        unique_filename
    ])

print(f"Detailed results logged to: {MASTER_LOG_FILE}")