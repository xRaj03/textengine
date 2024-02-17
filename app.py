import streamlit as st
from transformers import  BartTokenizer, BartForConditionalGeneration

def bart_summarize(input_text):
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    inputs = tokenizer(input_text, max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit App
st.title("Text Summarization with mc Streamlit")

input_text = st.text_area("Input Text:", "Enter your text here...")

if st.button("Summarize"):
    summarized_text = bart_summarize(input_text)
    st.subheader("Summarized Text:")
    st.write(summarized_text)
