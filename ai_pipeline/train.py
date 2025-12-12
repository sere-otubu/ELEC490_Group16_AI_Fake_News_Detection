import torch
import os
import sys
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from dotenv import load_dotenv

print(f"PyTorch Version: {torch.__version__}")
if not torch.cuda.is_available():
    print("CRITICAL ERROR: PyTorch is running in CPU mode.")
    print("You must reinstall PyTorch with CUDA support to train.")
    sys.exit(1)
else:
    print(f"Success! Training on GPU: {torch.cuda.get_device_name(0)}")

# --- SETUP & LOGIN ---
load_dotenv()
wb_token = os.getenv("WANDB_API_KEY")

if wb_token:
    wandb.login(key=wb_token)
    project_name = "medical-llama-capstone"
else:
    project_name = None

# --- CONFIGURATION ---
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct" 
OUTPUT_DIR = "medical_llama_standard"

# 1. LOAD MODEL
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    bnb_4bit_use_double_quant=True,
)

print(f"Loading {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    use_cache=False
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 2. PREPARE FOR LORA
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0.05, bias="none", 
    task_type="CAUSAL_LM", 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, peft_config)

# 3. LOAD DATA
dataset = load_dataset("json", data_files={
    "train": "training_data/train_finetune.jsonl", 
    "test": "training_data/val_finetune.jsonl"
})

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

# 4. CONFIGURATION (TRL v0.26+ Style)
# In newer TRL versions, arguments must go into SFTConfig, 
# and 'dataset_text_field' is strictly handled there.
args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    optim="paged_adamw_32bit",
    save_strategy="epoch",
    report_to="wandb" if wb_token else "none",
    run_name="llama-3b-standard-finetune",
    
    # NEW TRL REQUIREMENTS
    max_length=2048,      # Must be here
    dataset_text_field="text", # Must be here
    packing=False              # Explicitly disable packing to avoid conflicts
)

# 5. TRAINER
print("Initializing Trainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer, # <-- NAME CHANGE: 'tokenizer' was renamed to 'processing_class' in v0.25+
    args=args,
)

print("Starting Standard Training...")
trainer.train()

print("Saving adapter...")
trainer.save_model(OUTPUT_DIR)
print(f"Done! Model saved to {OUTPUT_DIR}")
if wb_token:
    wandb.finish()