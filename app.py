import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline, AutoTokenizer, AutoModel
import faiss
import numpy as np
import torch

@st.cache_resource(show_spinner = False)
# Function to read and extract text from a PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to summarize the extracted text
def summarize_text(text):
    summarizer = pipeline('summarization', model='Snowflake/snowflake-arctic-instruct')
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    summary = ''
    for chunk in chunks:
        summary += summarizer(chunk, max_length=125, min_length=30, do_sample=False)[0]['summary_text'] + ' '
    return summary.strip()

# Function to embed text using a Transformer model
def embed_text(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.detach().numpy()

# Function to answer questions based on the extracted text using FAISS
def answer_question(text_chunks, question, model, tokenizer, index):
    question_embedding = embed_text(question, model, tokenizer)
    _, I = index.search(question_embedding, k=1)
    answer_context = text_chunks[I[0][0]]
    return answer_context

# Streamlit app
st.title("PDF Summarizer and QA Bot")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from the uploaded PDF
    text = extract_text_from_pdf(uploaded_file)

    # Summarize the extracted text
    st.subheader("Summary")
    summary = summarize_text(text)
    st.write(summary)

    # Prepare text chunks and embeddings for QA
    text_chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    embeddings = np.vstack([embed_text(chunk, model, tokenizer) for chunk in text_chunks])
    
    # Use FAISS to build the index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # QA Bot
    st.subheader("Ask a Question")
    question = st.text_input("Enter your question about the PDF:")
    if question:
        answer = answer_question(text_chunks, question, model, tokenizer, index)
        st.write(f"Answer: {answer}")

# Run the app with: streamlit run app.py
