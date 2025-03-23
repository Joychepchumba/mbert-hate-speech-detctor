import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ✅ Load model & tokenizer (with caching to prevent reloading every run)
@st.cache_resource()
def load_model():
    model_name = "JCKipkemboi/hate_speech_detector_mbert"  # Update with correct Hugging Face repo
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# ✅ Streamlit UI
st.set_page_config(page_title="Multilingual Hate Speech Detector", page_icon="🌍", layout="centered")

st.title("🌍 Multilingual Hate Speech Detector (mBERT)")
st.write("Enter a sentence below to check if it's **Hate Speech** or **Not Hate Speech**.")

# Input Box
text = st.text_area("🔤 Enter your text:", height=100)

# ✅ Prediction Function
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1).item()
    
    label = "Hate Speech" if prediction == 1 else "Not Hate Speech"
    confidence = probabilities[0][prediction].item() * 100
    return label, confidence

# ✅ Predict Button with Loading Indicator
if st.button("🚀 Predict"):
    if text.strip():
        with st.spinner("🔄 Processing... Please wait"):
            label, confidence = predict(text)
        st.success(f"✅ **Prediction:** {label}")
        st.write(f"📊 **Confidence:** {confidence:.2f}%")
    else:
        st.warning("⚠️ Please enter some text before predicting.")

# ✅ Footer
st.markdown(
    """
    ---
    **Note:** This tool is a prototype and may not always be accurate.
    """,
    unsafe_allow_html=True,
)
