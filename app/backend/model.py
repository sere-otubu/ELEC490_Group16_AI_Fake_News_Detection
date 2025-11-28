import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel
import os
from dotenv import load_dotenv
from huggingface_hub import login

class MedicalFactChecker:
    def __init__(self):
        self.pipe = None
        self.tokenizer = None
        
        # Load environment and authenticate immediately
        load_dotenv()
        token = os.getenv("HF_TOKEN")
        if token:
            login(token=token)

    def load_model(self):
        """Loads the Base Model + Fine-Tuned Adapter from Hugging Face."""
        print("⏳ Loading Llama 3.2 Model... (This may take a minute)")
        
        # --- CONFIGURATION ---
        # 1. The Base Model (Meta's original)
        base_model_id = "meta-llama/Llama-3.2-3B-Instruct"
        
        # 2. Fine-Tuned Adapter from Hugging Face
        adapter_id = "sereotubu/medical-llama-3b-small-claims-v1"

        # 3. Configure 4-bit Quantization (Crucial for 8GB VRAM)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # 4. Load Base Model
        print(f"   - Downloading Base: {base_model_id}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
        
        # 5. Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)

        # 6. Load & Attach Your Adapter
        print(f"   - Downloading Adapter: {adapter_id}...")
        try:
            model = PeftModel.from_pretrained(base_model, adapter_id)
            print("✅ Adapter loaded and merged successfully!")
        except Exception as e:
            print(f"❌ Error loading adapter: {e}")
            raise e

        # 7. Create Pipeline
        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            max_new_tokens=200,
            do_sample=False,
        )
        print("AI Engine Ready!")

    def predict(self, claim: str):
        """Runs the model on a claim and parses the result."""
        if not self.pipe:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        # Use the EXACT prompt format from Phase 4 Training
        messages = [
            {"role": "system", "content": "You are a medical fact-checking assistant. Analyze the following text for misinformation."},
            {"role": "user", "content": claim}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate
        outputs = self.pipe(prompt)
        raw_response = outputs[0]['generated_text']
        
        # Extract the assistant's answer
        if "assistant<|end_header_id|>" in raw_response:
            answer = raw_response.split("assistant<|end_header_id|>")[-1].strip()
        else:
            answer = raw_response

        # Clean up
        answer = answer.replace("<|eot_id|>", "").strip()

        # Parse Verdict
        verdict = "Uncertain"
        if "MISINFORMATION DETECTED" in answer:
            verdict = "False"
        elif "FACTUALLY ACCURATE" in answer:
            verdict = "True"
        
        return {
            "claim": claim,
            "verdict": verdict,
            "explanation": answer
        }

# Singleton instance
ai_engine = MedicalFactChecker()