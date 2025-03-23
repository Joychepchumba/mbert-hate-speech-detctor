import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# ✅ Load model & tokenizer
model_name = "JCKipkemboi/hate_speech_detector_mbert"  # Replace with your actual model repo
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ✅ Streamlit UI
# Streamlit UI
st.title("Multilingual Hate Speech Detector (mBERT)")
st.write("Enter a sentence to check if it's hate speech or not.")

# Input box
user_input = st.text_input("Enter text:")

if st.button("Predict"):
    if user_input.strip():
        # Tokenization
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()

        # ✅ Display results
        label = "Hate Speech" if prediction == 1 else "Not Hate Speech"
        st.subheader(f"Prediction: {label}")

    else:
        st.warning("Please enter some text before predicting.")

# ✅ Run Streamlit only when executed directly
if __name__ == "__main__":
    st.write("Ready to classify text.")
    # Get model prediction
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    
    # Display prediction result
    result = "Hate Speech" if prediction == 1 else "Not Hate Speech"
    st.write(f"**Prediction:** {result}")
