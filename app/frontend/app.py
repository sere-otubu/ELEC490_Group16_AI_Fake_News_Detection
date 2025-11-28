import streamlit as st
import requests
import json

# --- CONFIGURATION ---
# This points to your running FastAPI backend
API_URL = "http://127.0.0.1:8001/predict"

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Medical Fact-Checker",
    page_icon="🩺",
    layout="centered"
)

# --- HEADER ---
st.title("🩺 Medical Fact-Checker")
st.markdown(
    """
    **AI-Powered Misinformation Detection**
    
    Paste a medical headline, tweet, or claim below. Our fine-tuned AI (Llama 3.2 3B) 
    will analyze it against known medical consensus.
    """
)

# --- INPUT SECTION ---
claim_text = st.text_area(
    "Enter Medical Claim:", 
    height=100,
    placeholder="e.g. Drinking boiled garlic water cures cancer instantly..."
)

# --- ACTION ---
if st.button("🔍 Verify Claim", type="primary"):
    if not claim_text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Thinking 🤔..."):
            try:
                # Send request to FastAPI Backend
                payload = {"text": claim_text}
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    verdict = data["verdict"]
                    explanation = data["explanation"]
                    
                    # --- DISPLAY RESULTS ---
                    st.divider()
                    
                    if verdict == "True":
                        st.success(f"### ✅ Verdict: {verdict}")
                    elif verdict == "False":
                        st.error(f"### 🚨 Verdict: {verdict} (Misinformation)")
                    else:
                        st.warning(f"### ⚠️ Verdict: {verdict}")
                    
                    st.markdown("### 🧠 AI Analysis:")
                    st.info(explanation)
                    
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to the backend. Is `main.py` running?")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

# --- FOOTER ---
st.markdown("---")
st.caption("Powered by Llama 3.2 (Fine-Tuned on Synthetic Medical Claims) | Running locally on RTX 3060 Ti")