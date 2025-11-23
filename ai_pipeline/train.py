import os
import torch
import wandb
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from dotenv import load_dotenv
from huggingface_hub import login

# --- SETUP ---
load_dotenv()

# 1. Login to HuggingFace
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN not found in .env")
login(token=hf_token)

# 2. Login to W&B
wb_token = os.getenv("WANDB_API_KEY")
if wb_token:
    wandb.login(key=wb_token)
    os.environ["WANDB_PROJECT"] = "medical-fact-checker"
else:
    print("WANDB_API_KEY not found in .env. Proceeding without W&B logging.")

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
DATA_DIR = "processed_data"
OUTPUT_DIR = "medical_llama_3b_finetuned"

# --- 1. LOAD MODEL ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    use_cache=False
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- 2. LOAD DATA ---
print(f"Loading data from {DATA_DIR}...")
dataset = load_from_disk(DATA_DIR)

if "validation" not in dataset:
    raise ValueError("Validation split not found. Please re-run prepare_data.py!")

print(f"Train size: {len(dataset['train'])}")
print(f"Valid size: {len(dataset['validation'])}")

# --- 3. CONFIGURE LoRA ---
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# Training configuration
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field="text",
    max_length=512,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,

    # Evaluation: Check progress every 10 steps (approx 5 times per epoch)
    eval_strategy="steps",
    eval_steps=10,

    # Logging: Track loss frequently
    logging_steps=5,

    # Saving: Save a checkpoint every time we evaluate
    save_strategy="steps",
    save_steps=10,

    # Best Model: Always keep the winner
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",

    # Typical learning rate for SFT
    learning_rate=2e-4,

    # Hardware settings
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",

    # Reporting
    report_to="wandb" if wb_token else "none",
    run_name="llama-3b-medical-fact-checker"
)

# --- 5. TRAINER ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=peft_config,
    processing_class=tokenizer,
    args=sft_config,
)

print("Starting Training...")
trainer.train()

print("Training Complete! Saving best model...")
trainer.save_model(OUTPUT_DIR)
wandb.finish()
print(f"Model saved to {OUTPUT_DIR}")