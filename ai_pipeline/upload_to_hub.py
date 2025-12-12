import torch
import os
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login, HfApi

# --- CONFIGURATION ---
# 1. Your Hugging Face Username
HF_USERNAME = "sereotubu" 

# 2. The folder where train.py saved your model
ADAPTER_PATH = "medical_llama_standard" 

# 3. The Base Model you fine-tuned
BASE_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

# 4. Name for your new repo
NEW_MODEL_NAME = "Llama-3.2-3B-Medical-Fact-Checker"

# 5. MERGE OPTION (Important!)
# Set True to upload a full, standalone model (easier for edge devices).
# Set False to upload only the adapter (faster upload, requires base model to run).
MERGE_AND_UPLOAD = False 

# --- AUTH ---
from dotenv import load_dotenv
load_dotenv()
token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError("HF_TOKEN not found in .env file.")
login(token=token)

print(f"Loading Base Model: {BASE_MODEL_ID}...")
# Note: For merging, we generally need 16-bit, not 4-bit
if MERGE_AND_UPLOAD:
    print("Loading in 16-bit for merging...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
else:
    # 4-bit is fine if just verifying adapter
    print("Loading in 4-bit for adapter verification...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

print(f"Loading Adapter from {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

repo_id = f"{HF_USERNAME}/{NEW_MODEL_NAME}"
print(f"Target Repo: {repo_id}")

if MERGE_AND_UPLOAD:
    print("Merging adapter into base model (This takes RAM)...")
    model = model.merge_and_unload()
    
    print("Pushing FULL MODEL to Hub...")
    model.push_to_hub(repo_id, safe_serialization=True)
    tokenizer.push_to_hub(repo_id)
else:
    print("Pushing ADAPTER ONLY to Hub...")
    model.push_to_hub(repo_id, safe_serialization=True)
    # We don't necessarily need to push the tokenizer for just an adapter, 
    # but it's good practice to keep them linked.
    tokenizer.push_to_hub(repo_id)

print("\n---------------------------------------")
print(f"Upload Complete! View your model here:")
print(f"https://huggingface.co/{repo_id}")
print("---------------------------------------")