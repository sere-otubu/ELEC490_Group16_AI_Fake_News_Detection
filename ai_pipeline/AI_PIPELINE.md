# Medical Fact-Checking AI Pipeline

## 📌 Project Overview
This folder contains the benchmarking evaluation for the capstone project.

**Goal:** To empirically prove that "Edge-tier" LLMs (3B parameters) and even standard 7B-9B models fail to detect subtle medical misinformation without specific fine-tuning.

We benchmark models against the **PUBHEALTH** dataset (https://huggingface.co/datasets/ImperialCollegeLondon/health_fact).

## 🏗️ Architecture
* **Hardware:** Optimized for local NVIDIA GPUs (RTX 3060 Ti / 3050).
* **Quantization:** 4-bit NF4 (Normal Float 4) via `bitsandbytes` to minimize VRAM usage.
* **Training Method:** QLoRA (Quantized Low-Rank Adaptation) via Hugging Face `trl`.
* **Teacher Model:** Google `gemini-2.5-flash` (used for synthetic data generation).
* **Student Model:** `meta-llama/Llama-3.2-3B-Instruct`.

* **Models Tested:**
    * `meta-llama/Llama-3.2-3B-Instruct` - https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
    * `meta-llama/Llama-3.1-8B-Instruct` - https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
    * `Qwen/Qwen2.5-7B-Instruct` - https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
    * `mistralai/Mistral-7B-Instruct-v0.3` - https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3

---

## ⚙️ First-Time Setup Guide (Windows & NVIDIA)

**Prerequisite:** Python 3.10+ and an NVIDIA GPU.

### 1. Environment Creation
Create a clean virtual environment to avoid conflicts.
```powershell
# Open terminal project folder
python -m venv venv
# In case you are using Bain Lab computers (Queen's University)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\activate
```

### 2. Basic Dependencies
Install the core libraries for data processing and Hugging Face interaction.
```powershell
pip install -r requirements.txt
```

### 3. CUDA & GPU Setup
Standard pip install torch often installs the CPU version by default, and standard bitsandbytes does not work on Windows. You must run these specific commands to force the GPU versions.

#### Step A: Uninstall any existing CPU versions
```powershell
pip uninstall torch torchvision torchaudio bitsandbytes -y
```

#### Step B: Install PyTorch with CUDA 12.4 support This uses the direct index URL to bypass the default CPU package.
```powershell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

#### Step C: Install Bitsandbytes
```powershell
pip install bitsandbytes accelerate
```

### 4. Security & Authentication
Models like Llama 3 and Gemma 2 are "Gated." You need a Hugging Face token.
1. Create a file named .env in the ```ai_pipline``` folder.
2. Add your token inside: ```HF_TOKEN=hf_YourTokenHere```
3. **Note**: Ensure you have accepted the license terms for Llama on the Hugging Face website.

---
## 📊 Phase 1: The Problem
We benchmarked standard models against the **PUBHEALTH** dataset. We discovered a critical failure mode:
* Models fail catastrophically on short, news-style headlines (poor context), defaulting to paranoia (flagging everything as false).

```
Baseline Performance on Short Claims:
|  Model    |  Parameters | Accuracy |
| Llama 3.2 |  3 Billion  | 38%      |
| Llama 3.1 |  8 Billion  | 57%      |
```

**Conclusion:** Scaling up to 8B does not solve the problem. *Fine-tuning* is required.

---
## 🧪 Phase 2: Synthetic Data Strategy

To fix the domain mismatch, we generated a custom dataset using Google's **Gemini 2.5 Flash**.
* **Objective:** Teach the model to recognize "News Style" medical truths, similar to the claims found in PUBHEALTH.
* **Dataset Size:** 1,000 Examples (500 True / 500 False).
* **Format:** Short, punchy claims (1-2 sentences) mimicking social media/headlines.
* **Balance:** Strictly 50/50 to prevent bias.

**Scripts Used:**
* `generate_synthetic_data.py`: Generates batches of "True" or "False" claims via an API call.
* `merge_datasets.py`: Combines and shuffles them into `synthetic_claims_balanced.json`.

---
## ⚙️ Phase 3: Fine-Tuning (QLoRA)
We trained the 3B model using **Supervised Fine-Tuning (SFT)**.

**Configuration (`train.py`):**
* **Epochs:** 3
* **Effective Batch Size:** 8 (2 (Per Device) with 4 Gradient Accumulation Steps)
* **Learning Rate:** 2e-4
* **Target Modules:** All Linear Layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, etc.)
* **Hardware:** Fits on 8GB VRAM (RTX 3060 Ti) using 4-bit loading.
---
## 🏆 Phase 4: Final Results (The Victory)
After fine-tuning on the custom "Short Claims" dataset, the 3B model significantly outperformed its baseline and the larger 8B model.
```
|       Metric       | Base 3B (Before) | Base 8B | **Fine-Tuned 3B (After)** |
| Accuracy           | 0.38             | 0.56    |            0.68           |
| Recall (True)      | 0.11             | 0.44    |            0.65           |
| Recall (False)     | 0.70             | 0.70    |            0.70           |
```

**Key Finding:**
We successfully fixed the "Paranoia" problem. The fine-tuned model recovered **+59% Recall on Truth**, proving that a small, specialized model can beat a large, generalist model at specific tasks.

---
## 📂 Scripts & Tools

### Data Pipeline
* **`generate_synthetic_data.py`**: Connects to the Gemini API to generate batches of synthetic medical claims.
    * *Usage:* Update the `PROMPT` variable to switch between "True" and "False" generation.
* **`merge_datasets.py`**: Combines the "True" and "False" JSON files and shuffles them to create a balanced training set.
* **`prepare_data.py`**: Converts the raw JSON into a Hugging Face `Arrow` dataset.
    * **Splitting:** Automatically creates an **80/10/10** split (Train/Validation/Test).
    * **Output:** Saves to the `processed_data/` folder, which is the direct input for training.

### Benchmarking
* **`benchmark_baseline.py`**: Tests off-the-shelf models (Llama, Qwen, Mistral).
    * *Features:* Includes a "Universal Extractor" to handle different chat templates and logs detailed metrics (Precision/Recall per class) to `experiment_history_log.csv`.
* **`benchmark_finetuned.py`**: Tests your specific local adapter.
    * *Features:* Uses the custom "Analyze..." prompt used during training to trigger the correct model behavior.

### Deployment / Backup
* **`upload_to_hub.py`**: Backs up your fine-tuned adapter to Hugging Face.
    * *Why:* GitHub cannot host large model weights.
    * *Usage:* Authenticates with your HF Write Token and pushes the `medical_llama_3b_finetuned` folder to your private hub repository.
---

## 🚀 How to Reproduce

Set up environment

**2. Generate Data**
```powershell
# Edit PROMPT in script to generate True/False batches
python generate_synthetic_data.py
# Merge them
python merge_datasets.py
```

**3. Train Model**
```powershell
# Prepare data for Hugging Face
python prepare_data.py
# Run QLoRA Training
python train.py
```

**4. Run Benchmarks**

Open ```benchmark_baseline.py``` and edit the Section 2 variable:
```python
# --- 2. MODEL SELECTION ---
# Toggle these to compare!
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct" 
# MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
# MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct" 
# MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
```

```powershell
# Test the Base Model
python benchmark_baseline.py
```

For ```benchmark_finetuned.py```, make sure to use **sereotubu/medical-llama-3b-small-claims-v1** (https://huggingface.co/sereotubu/medical-llama-3b-small-claims-v1)
```powershell
# Test Your Fine-Tuned Adapter
python benchmark_finetuned.py
```
---

## 📊 Output & Logging
The pipeline generates two types of data:

**1. The Master Log (**```experiment_history_log.csv```**)**

- A single cumulative file tracking every run.
- Columns: Timestamp, Model Name, Time_Seconds, Accuracy, False_Precision, False_Recall, False_F1, False_Support, True_Precision, True_Recall, True_F1, True_Support, Result_File_Path.
- This data can be used later for performance comparison graphs

**2. Detailed Run Files (**```run_results/```**)

- A unique CSV generated for every single run (e.g., ```results_Llama-3.1-8B_20251122_1930.csv```).
- Contains row-by-row predictions.
- Use this to find specific "Failure Cases" (where the model said 'True' but the fact was 'False').