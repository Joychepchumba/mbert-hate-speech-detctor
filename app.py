import streamlit as st
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import uvicorn
import threading

# ✅ Set Streamlit page config first
st.set_page_config(page_title="🌍 Multilingual Hate Speech Detector", page_icon="🛑", layout="centered")

# ✅ Initialize FastAPI
app = FastAPI()

# ✅ Enable CORS for Power BI access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load Model & Tokenizer (Cache to prevent reloading)
@st.cache_resource()
def load_model():
    model_name = "JCKipkemboi/hate_speech_detector_mbert"  # mBERT for multilingual support
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# ✅ FastAPI Request Schema
class TextInput(BaseModel):
    text: str

# ✅ API Endpoint for Predictions
@app.post("/predict")
def predict(input: TextInput):
    """Classifies text as Hate Speech or Not Hate Speech (Multilingual)"""
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1).item()
    
    label = "Hate Speech" if prediction == 1 else "Not Hate Speech"
    confidence = probabilities[0][prediction].item() * 100

    return {
        "text": input.text,
        "prediction": label,
        "confidence": f"{confidence:.2f}%",
        "message": "This model supports multiple languages using mBERT."
    }

# ✅ Streamlit UI
def streamlit_ui():
    st.title("🌍 Multilingual Hate Speech Detector")
    st.write("This tool detects hate speech across multiple languages using mBERT (Multilingual BERT).")

    st.write("Enter a sentence below to check if it's **Hate Speech** or **Not Hate Speech**.")

    # User Input
    text = st.text_area("📝 Enter your text:", height=100)

    # Streamlit Prediction Function
    def streamlit_predict(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1).item()
        
        label = "Hate Speech" if prediction == 1 else "Not Hate Speech"
        confidence = probabilities[0][prediction].item() * 100
        return label, confidence

    # Example Inputs
    with st.expander("💡 Example Inputs"):
        st.write("🔹 *You are stupid!* → Hate Speech (English)")
        st.write("🔹 *Tu es horrible!* → Hate Speech (French)")
        st.write("🔹 *Hola amigo, cómo estás?* → Not Hate Speech (Spanish)")

    # Predict Button
    if st.button("🚀 Predict"):
        if text.strip():
            with st.spinner("🔄 Processing... Please wait"):
                label, confidence = streamlit_predict(text)
            st.success(f"✅ **Prediction:** {label}")
            st.write(f"📊 **Confidence:** {confidence:.2f}%")
        else:
            st.warning("⚠️ Please enter some text before predicting.")

    # Footer
    st.markdown(
        """
        ---
        **Note:** This model uses **mBERT** (Multilingual BERT) to detect hate speech across different languages.
        """,
        unsafe_allow_html=True,
    )

# ✅ Run FastAPI & Streamlit Simultaneously
def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    # Run FastAPI in a separate thread (optional for serving both APIs)
    threading.Thread(target=run_fastapi, daemon=True).start()
    # Directly call Streamlit UI
    streamlit_ui()
