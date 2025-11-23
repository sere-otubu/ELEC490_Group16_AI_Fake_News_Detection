# Medical Fact-Checking AI Pipeline

## 📌 Project Overview
This folder contains the benchmarking evaluation for the capstone project.

**Goal:** To empirically prove that "Edge-tier" LLMs (3B parameters) and even standard 7B-9B models fail to detect subtle medical misinformation without specific fine-tuning.

We benchmark models against the **PubHealth** dataset (Truth vs. False binary classification).

## 🏗️ Architecture
* **Hardware:** Optimized for local NVIDIA GPUs (RTX 3060 Ti / 3050).
* **Quantization:** 4-bit NF4 (Normal Float 4) via `bitsandbytes` to minimize VRAM usage.

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

## 🚀 How to Run the Benchmark
The main script is ```benchmark_baseline.py```. It automatically handles:

- Loading the model in 4-bit mode.
- Formatting prompts for different models (Llama, Qwen, Mistral).
- Logging results

**1. Select Your Model**

Open ```benchmark_baseline.py``` and edit the Section 2 variable:
```python
# --- 2. MODEL SELECTION ---
# Toggle these to compare!
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct" 
# MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
# MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct" 
# MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
```

**2. Run the Script**
```powershell
python benchmark_baseline.py
```

---

## 📊 Output & Logging
The pipeline generates two types of data:

**1. The Master Log (**```experiment_history_log.csv```**)**

- A single cumulative file tracking every run.
- Columns: Timestamp, Model Name, Time Taken, Accuracy, Precision, Recall, F1.
- Will be used later for "Performance Comparison" graphs

**2. Detailed Run Files (**```run_results/```**)

- A unique CSV generated for every single run (e.g., ```results_Llama-3.1-8B_20251122_1930.csv```).
- Contains row-by-row predictions.
- Use this to find specific "Failure Cases" (where the model said 'True' but the fact was 'False').