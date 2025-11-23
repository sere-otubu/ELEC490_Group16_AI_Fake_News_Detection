import os
from huggingface_hub import login, create_repo, upload_folder
from dotenv import load_dotenv

# 1. SETUP
load_dotenv()
token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError("HF_TOKEN not found in .env file.")
login(token=token)

# 2. CONFIGURATION
# The folder on your computer where train.py saved the model
LOCAL_DIR = "medical_llama_3b_finetuned" 

# The name you want on Hugging Face (username/model-name)
# Based on your error message, this is your username:
REPO_NAME = "sereotubu/medical-llama-3b-adapter-v2" 

# 3. CREATE REPO
print(f"Creating repository: {REPO_NAME}...")
try:
    # This creates the empty repo on Hugging Face if it doesn't exist
    create_repo(REPO_NAME, repo_type="model", exist_ok=True)
except Exception as e:
    print(f"Note on repo creation: {e}")

# 4. UPLOAD
print(f"🚀 Uploading '{LOCAL_DIR}' to Hugging Face...")
print("This might take a minute...")

upload_folder(
    folder_path=LOCAL_DIR,
    repo_id=REPO_NAME,
    repo_type="model"
)

print(f"✅ Success! Your model is live at: https://huggingface.co/{REPO_NAME}")