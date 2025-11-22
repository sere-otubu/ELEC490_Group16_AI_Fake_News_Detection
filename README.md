# Medical Fact-Checker: Edge-Optimized LLM Approach

## 📌 Project Overview
This project aims to develop a lightweight Medical Fact-Checking model capable of running on edge devices (smartphones/watches). Moving away from traditional classification (BioBERT), we are implementing a **Generative AI Alignment** strategy.

The core hypothesis is that while small Language Models (SLMs) like Llama 3.2 are linguistically fluent, they are "gullible" to professional-sounding medical misinformation. We aim to fix this via **Instruction Tuning** on a synthetic adversarial dataset.

---

## 🚀 Current Phase: Phase 1 (The Baseline)
**Goal:** Establish a performance baseline to prove that off-the-shelf models fail to detect subtle medical misinformation.

### Technical Architecture
* **Model:** `meta-llama/Llama-3.2-3B-Instruct` (Chosen for edge compatibility).
* **Dataset:** `PubHealth` (ImperialCollegeLondon), filtered for binary True/False classification.
* **Quantization:** 4-bit NF4 (Normal Float 4) via `bitsandbytes`.
* **Hardware Target:** Local Laptop GPU (NVIDIA RTX 3050, 4GB VRAM).

### ⚙️ Environment Setup
This project requires specific optimization libraries to fit the model into 4GB VRAM on Windows.

1.  **Prerequisites**
    * Python 3.10+
    * Hugging Face Account (with Llama 3.2 license accepted)

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Windows & NVIDIA Setup (Crucial)**
    Standard `bitsandbytes` does not support Windows natively. You must install the Windows-specific wheel to enable 4-bit quantization:
    ```bash
    # Remove any conflicting versions
    pip uninstall bitsandbytes -y
    
    # Install Windows-compatible version
    pip install [https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_main/bitsandbytes-0.48.2-py3-none-win_amd64.whl](https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_main/bitsandbytes-0.48.2-py3-none-win_amd64.whl)
    ```

4.  **Security**
    Create a `.env` file in the root directory to store your Hugging Face token:
    ```env
    HF_TOKEN=hf_YourActualTokenHere
    ```

---

## 🏃‍♂️ How to Run the Baseline
We have created a benchmark script that handles authentication, quantization, and inference.

**Run the script:**
```bash
python benchmark_baseline.py