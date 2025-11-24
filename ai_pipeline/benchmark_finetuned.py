import os
import time
import datetime
import torch
import pandas as pd
import csv
import re
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
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
BASE_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_PATH = "medical_llama_3b_finetuned"

if not os.path.exists("run_results"):
    os.makedirs("run_results")

print(f"Loading Base Model: {BASE_MODEL_ID}...")

# Optimized 4-bit config with safe dtype fallback
try:
    bf16_supported = False
    if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported"):
        bf16_supported = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16_supported else torch.float16
except Exception:
    compute_dtype = torch.float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# 1. Load the Base Model (Frozen)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto", 
)

# 2. Load the Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

# 3. Load YOUR Fine-Tuned Adapter
print(f"Loading Fine-Tuned Adapter from: {ADAPTER_PATH}...")
try:
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print("✅ Adapter loaded successfully!")
    try:
        model.eval()
    except Exception:
        pass
except Exception as e:
    print(f"❌ Error loading adapter: {e}")
    print("Did you complete the training? Check if the folder exists.")
    exit()

# 4. Create Pipeline
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

def extract_clean_answer(raw_text):
    if "assistant<|end_header_id|>" in raw_text:
        answer = raw_text.split("assistant<|end_header_id|>")[-1]
    elif "<|im_start|>assistant" in raw_text:
        answer = raw_text.split("<|im_start|>assistant")[-1]
    else:
        answer = raw_text
        
    return answer.replace("<|im_end|>", "").replace("<|eot_id|>", "").strip()

for i, row in enumerate(test_data):
    claim = row['claim']
    ground_truth = "True" if row['label'] == 2 else "False"

    messages = [
        {"role": "system", "content": "You are a medical fact-checking assistant. Analyze the following text for misinformation."},
        {"role": "user", "content": claim}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    outputs = pipe(
        prompt, 
        max_new_tokens=10,
        do_sample=False
    )
    
    raw = outputs[0]['generated_text']
    answer = extract_clean_answer(raw)

    # Stricter verdict parsing using regex; prefer explicit True/False tokens.
    prediction = "Unsure"

    # Priority 1: explicit fine-tuned signal (case-insensitive)
    if "MISINFORMATION DETECTED" in answer.upper():
        prediction = "False"
    else:
        # look for explicit True/False tokens first
        m = re.search(r"\b(True|False)\b", answer, flags=re.IGNORECASE)
        if m:
            prediction = "True" if m.group(1).lower() == "true" else "False"
        else:
            # fallback to keyword matching
            if re.search(r"\b(incorrect|false|wrong|inaccurate)\b", answer, flags=re.IGNORECASE):
                prediction = "False"
            elif re.search(r"\b(true|correct|accurate)\b", answer, flags=re.IGNORECASE):
                prediction = "True"
            else:
                prediction = "Unsure"

    results.append({
        "claim": claim[:50],
        "truth": ground_truth,
        "pred": prediction,
        "raw": answer
    })
    
    if (i+1) % 10 == 0: print(f"Processed {i+1}...")

end_time = time.time()
elapsed_time = round(end_time - start_time, 2)

# --- 5. SAVE RESULTS ---
df = pd.DataFrame(results)
unique_filename = f"run_results/results_FINETUNED_3B_{run_timestamp}.csv"
df.to_csv(unique_filename, index=False)

# Metrics
precision, recall, f1, _ = precision_recall_fscore_support(df['truth'], df['pred'], average='weighted', zero_division=0)
accuracy = accuracy_score(df['truth'], df['pred'])

print("\n" + "="*30)
print(f"REPORT FOR: Fine-Tuned Llama 3.2 3B")
print("="*30)
print(classification_report(df['truth'], df['pred'], zero_division=0))

# Log
file_exists = os.path.isfile(MASTER_LOG_FILE)
with open(MASTER_LOG_FILE, mode='a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(["Timestamp", "Model_Name", "Samples", "Time_Seconds", "Accuracy", "Precision", "Recall", "F1_Score", "Result_File_Path"])
    
    writer.writerow([
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Llama-3.2-3B-FineTuned",
        len(test_data),
        elapsed_time,
        round(accuracy, 4),
        round(precision, 4),
        round(recall, 4),
        round(f1, 4),
        unique_filename
    ])

print(f"Results logged to: {MASTER_LOG_FILE}")