import os
import json
import time
import google.generativeai as genai
from dotenv import load_dotenv

# 1. SETUP
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

genai.configure(api_key=api_key)


MODEL_NAME = "gemini-2.5-flash" 

generation_config = {
  "temperature": 0.7,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "application/json",
}

model = genai.GenerativeModel(
  model_name=MODEL_NAME,
  generation_config=generation_config,
  system_instruction="""You are an expert medical content generator creating a dataset for 'Adversarial Misinformation Detection'. 
  Your goal is to write highly professional, academic-sounding medical abstracts that contain subtle factual errors.
  These are for research purposes to train an AI model to detect lies.
  """
)

# False data generation prompt
# PROMPT = """
# Generate 5 unique training examples in strict JSON format.

# **Constraints for each example:**
# 1.  **Topic:** Choose distinct medical topics (COVID-19, Cardiology, Oncology, Pediatrics, etc.).
# 2.  **Input:** Write a 150-word "Medical Abstract". It must use high-level terminology and sound authoritative.
# 3.  **The Lie:** Buried inside, include ONE specific factual error (e.g., incorrect dosage, reversed mechanism, wrong contraindication). It must NOT be obvious. It must sound plausible to a layman.
# 4.  **Output:** A critique explaining exactly why it is wrong.

# **Output JSON Structure:**
# [
#   {
#     "instruction": "Analyze the text for medical accuracy.",
#     "input": "...",
#     "output": "MISINFORMATION DETECTED... [Explanation]"
#   }
# ]
# """

# True data generation prompt
PROMPT = """
Generate 5 unique training examples in strict JSON format.

**Constraints for each example:**
1.  **Topic:** Choose distinct medical topics (Cardiology, Oncology, Pediatrics, etc.).
2.  **Input:** Write a 150-word "Medical Abstract". It must use high-level terminology and be **100% medically accurate** and aligned with standard clinical consensus.
3.  **Output:** A validation confirming its accuracy.

**Output JSON Structure:**
[
  {
    "instruction": "Analyze the following medical text for misinformation.",
    "input": "...",
    "output": "FACTUALLY ACCURATE.\n\nAnalysis: The text correctly describes [Mechanism/Treatment]. This aligns with current clinical guidelines."
  }
]
"""

# 3. THE LOOP
# OUTPUT_FILE = "synthetic_fake_data_gemini.json"
OUTPUT_FILE = "synthetic_true_data_gemini.json"
TARGET_COUNT = 500  # Adjust as needed

# Load existing data if file exists
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'r') as f:
        try:
            existing_data = json.load(f)
        except:
            existing_data = []
else:
    existing_data = []

print(f"Starting with {len(existing_data)} examples. Target: {TARGET_COUNT}")

while len(existing_data) < TARGET_COUNT:
    try:
        print(f"Requesting batch... (Current: {len(existing_data)})")
        
        # Call Gemini
        response = model.generate_content(PROMPT)
        
        # Parse JSON
        new_batch = json.loads(response.text)
        
        # Append and Save
        existing_data.extend(new_batch)
        
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(existing_data, f, indent=2)
            
        print(f"Saved! Total count: {len(existing_data)}")
        
        time.sleep(2)
        
    except Exception as e:
        print(f"Error/Safety Block: {e}")
        time.sleep(5)

print("Generation Complete!")