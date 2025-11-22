import os
import torch
import pandas as pd
from transformers import pipeline, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from sklearn.metrics import classification_report

# --- 1. SETUP & AUTH ---
load_dotenv()
token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError("HF_TOKEN not found in .env file.")
login(token=token)

# --- 2. MODEL CONFIGURATION (Optimized for RTX 3050 4GB) ---
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

# Aggressive compression to fit in 4GB VRAM
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Initialize the pipeline
pipe = pipeline(
    "text-generation",
    model=MODEL_ID,
    model_kwargs={"quantization_config": bnb_config},
    tokenizer=tokenizer,
    device_map="auto",  # vital for handling memory spillover
)

# --- 3. DATA PREPARATION ---
print("Loading and filtering PubHealth dataset...")
# We only want True (Label=2) or False (Label=0) claims for a binary baseline
dataset = load_dataset("ImperialCollegeLondon/health_fact", trust_remote_code=True)

def filter_binary(example):
    # 0=False, 2=True. We skip 'Mixture'(1) and 'Unproven'(3)
    return example['label'] in [0, 2]

# Take just 50 samples for a quick initial test (increase this later!)
test_data = dataset['test'].filter(filter_binary).select(range(50))

# --- 4. INFERENCE LOOP ---
results = []
print(f"Starting inference on {len(test_data)} samples...")

for i, row in enumerate(test_data):
    claim = row['main_text']
    # Map dataset labels to text: 2->True, 0->False
    ground_truth = "True" if row['label'] == 2 else "False"

    # Llama 3.2 Specific Prompt Format
    messages = [
        {"role": "system", "content": "You are a medical fact-checking assistant. Your task is to classify the following medical claim as either 'True' or 'False'. Do not explain. Just answer with one word."},
        {"role": "user", "content": f"Claim: {claim}\n\nVerdict:"}
    ]
    
    # Apply the chat template safely
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Generate
    outputs = pipe(
        prompt, 
        max_new_tokens=5, 
        do_sample=False, # Deterministic (Greedy) decoding is better for benchmarks
        temperature=0.1
    )
    
    raw_output = outputs[0]['generated_text']
    # Extract just the new text (Llama returns the whole prompt + answer)
    answer = raw_output.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()

    # Simple cleanup
    prediction = "Unsure"
    if "true" in answer.lower(): prediction = "True"
    elif "false" in answer.lower(): prediction = "False"

    results.append({
        "claim_preview": claim[:60] + "...",
        "ground_truth": ground_truth,
        "prediction": prediction,
        "raw_model_output": answer
    })
    
    if (i + 1) % 5 == 0:
        print(f"Processed {i + 1}/{len(test_data)}...")

# --- 5. REPORTING ---
df = pd.DataFrame(results)

# Save to CSV for your supervisor
output_filename = "baseline_llama_3b_results.csv"
df.to_csv(output_filename, index=False)

print("\n" + "="*30)
print("CONFUSION MATRIX BASELINE")
print("="*30)
# Check if we have enough data to run the report (needs at least one of each class ideally)
try:
    print(classification_report(df['ground_truth'], df['prediction'], zero_division=0))
except Exception as e:
    print(f"Could not generate report: {e}")

print(f"Results saved to {output_filename}")