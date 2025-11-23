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
    claim = row['main_text']
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

# --- 5. SAVE RESULTS ---
df = pd.DataFrame(results)
model_short_name = MODEL_ID.split('/')[-1]
unique_filename = f"run_results/results_{model_short_name}_{run_timestamp}.csv"
df.to_csv(unique_filename, index=False)

# Calculate Metrics
precision, recall, f1, _ = precision_recall_fscore_support(df['truth'], df['pred'], average='weighted', zero_division=0)
accuracy = accuracy_score(df['truth'], df['pred'])

print("\n" + "="*30)
print(f"REPORT FOR: {MODEL_ID}")
print("="*30)
print(classification_report(df['truth'], df['pred'], zero_division=0))

# Append to master log
file_exists = os.path.isfile(MASTER_LOG_FILE)

with open(MASTER_LOG_FILE, mode='a', newline='') as file:
    writer = csv.writer(file)
    
    if not file_exists:
        writer.writerow(["Timestamp", "Model_Name", "Samples", "Time_Seconds", "Accuracy", "Precision", "Recall", "F1_Score", "Result_File_Path"])
    
    writer.writerow([
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        MODEL_ID,
        len(test_data),
        elapsed_time,
        round(accuracy, 4),
        round(precision, 4),
        round(recall, 4),
        round(f1, 4),
        unique_filename
    ])

print(f"Corrected log saved to: {MASTER_LOG_FILE}")