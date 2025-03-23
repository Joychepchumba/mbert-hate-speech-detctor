
import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load the mBERT model & tokenizer
model_name = "JCKipkemboi/hate_speech_detector_mbert"  # Make sure this matches your Hugging Face model!
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Streamlit UI
st.title("Multilingual Hate Speech Detector (mBERT)")
st.write("Enter a sentence to check if it's hate speech or not.")

# Input box
user_input = st.text_input("Enter text:")

if user_input:
    # Tokenize input
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    
    # Get model prediction
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    
    # Display prediction result
    result = "Hate Speech" if prediction == 1 else "Not Hate Speech"
    st.write(f"**Prediction:** {result}")
