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
  "temperature": 0.8,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "application/json",
}

model = genai.GenerativeModel(
  model_name=MODEL_NAME,
  generation_config=generation_config,
  system_instruction="""You are an expert medical content generator. 
  Your goal is to generate a dataset of 'Short Medical Claims' for fact-checking training.
  """
)

# False data generation prompt
# PROMPT = """
# Generate 10 unique training examples in strict JSON format.

# **Constraints:**
# 1.  **Style:** Write **Short, Punchy Claims** (1-2 sentences max).
#     * *Tone:* Viral social media posts, news headlines, or "forwarded email" warnings.
#     * *Example:* "Expired boxes of cake and pancake mix are dangerously toxic."
#     * *Example:* "Drinking boiled garlic water clears blocked arteries instantly."
# 2.  **Content:** ALL examples must be **FALSE (Misinformation)**.
#     * Include varied topics: Diet, Vaccines, Cures, household toxins, etc.
#     * Make them sound confident and alarming (the way misinformation usually spreads).

# **Output JSON Structure:**
# [
#   {
#     "instruction": "Analyze this medical claim for misinformation.",
#     "input": "[Insert Short False Claim]",
#     "output": "MISINFORMATION DETECTED.\n\nAnalysis: The claim that [Restate Claim] is false. [Brief scientific correction]."
#   }
# ]
# """

# True data generation prompt
PROMPT = """
Generate 10 unique training examples in strict JSON format.

**Constraints:**
1.  **Style:** Write **Short, Punchy Claims** (1-2 sentences max).
    * *Tone:* Viral social media posts, news headlines, or "Did you know?" facts.
    * *Example:* "The CDC recommends annual flu shots for everyone over the age of 6 months."
    * *Example:* "Regular exercise has been proven to lower the risk of heart disease and stroke."
2.  **Content:** ALL examples must be **TRUE (Factually Accurate)**.
    * Use standard, consensus-based medical knowledge (Guidelines, Anatomy, Treatments, Public Health).
    * Cover varied topics: Nutrition, Epidemiology, Pharmacology, Cardiology, etc.

**Output JSON Structure:**
[
  {
    "instruction": "Analyze this medical claim for misinformation.",
    "input": "[Insert Short True Claim]",
    "output": "FACTUALLY ACCURATE.\n\nAnalysis: The claim is correct. [Brief sentence citing why/consensus]."
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