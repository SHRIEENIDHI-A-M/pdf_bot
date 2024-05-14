import streamlit as st
import pdfplumber
from transformers import pipeline
from transformers.pipelines import PipelineException

# Function to extract text from a PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to summarize text
def summarize_text(text):
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    except PipelineException as e:
        st.error(f"Error loading summarization model: {e}")
        return ""

# Function to answer questions based on the text
def answer_question(context, question):
    try:
        qa_model = pipeline("question-answering")
        result = qa_model(question=question, context=context)
        return result['answer']
    except PipelineException as e:
        st.error(f"Error loading question-answering model: {e}")
        return ""

# Streamlit app layout
st.title('PDF Summarizer and Q&A Bot')
uploaded_file = st.file_uploader('Upload a PDF', type='pdf')

if uploaded_file is not None:
    # Extract text from uploaded PDF
    text = extract_text_from_pdf(uploaded_file)
    
    # Display extracted text
    st.subheader("Extracted Text")
    st.write(text)
    
    # Summarize the text
    summary = summarize_text(text)
    st.subheader("Summary")
    st.write(summary)
    
    # Question and Answer section
    st.subheader("Ask a Question about the PDF")
    question = st.text_input("Enter your question")
    if question:
        answer = answer_question(text, question)
        st.write("Answer:")
        st.write(answer)
