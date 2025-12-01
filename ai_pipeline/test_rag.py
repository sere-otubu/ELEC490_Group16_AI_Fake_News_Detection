import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel
from duckduckgo_search import DDGS
from dotenv import load_dotenv
from huggingface_hub import login
import os

# --- CONFIGURATION ---
TEST_CLAIM = "Drinking boiled garlic water cures cancer instantly." 
BASE_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
# Point this to your Hugging Face Adapter or local folder
ADAPTER_PATH = "sereotubu/medical-llama-3b-small-claims-v1" 

# --- 1. SETUP ---
load_dotenv()
token = os.getenv("HF_TOKEN")
if token:
    login(token=token)

print("Step 1: Loading Model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

print(f"Loading Adapter: {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# --- 2. SEARCH FUNCTION ---
def get_medical_evidence(query, max_results=3):
    print(f"\nStep 2: Searching the web for: '{query}'...")
    try:
        results = DDGS().text(query, max_results=max_results)
        if not results:
            return "No relevant search results found."
        
        evidence_text = ""
        for i, res in enumerate(results):
            evidence_text += f"Source {i+1}: {res['body']}\n"
        
        print(f"   -> Found {len(results)} sources.")
        return evidence_text
    except Exception as e:
        return f"Search failed: {e}"

# --- 3. RUN RAG ---
evidence = get_medical_evidence(TEST_CLAIM)

print("\nStep 3: Asking AI...")
rag_prompt = f"""
You are a medical fact-checker. Use the provided evidence to verify the claim.

EVIDENCE:
{evidence}

CLAIM:
{TEST_CLAIM}

INSTRUCTIONS:
Based ONLY on the evidence above, determine if the claim is True or False.
"""

messages = [
    {"role": "system", "content": "You are a medical AI."},
    {"role": "user", "content": rag_prompt}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

outputs = pipe(prompt, max_new_tokens=150, do_sample=False)

# --- 4. RESULT ---
raw = outputs[0]['generated_text']
if "assistant<|end_header_id|>" in raw:
    answer = raw.split("assistant<|end_header_id|>")[-1].strip()
else:
    answer = raw

print("\n" + "="*40)
print(f"Evidence: {evidence}")
print("\n" + "="*40)
print(f"CLAIM: {TEST_CLAIM}")
print("-" * 40)
print(f"VERDICT:\n{answer}")
print("="*40)