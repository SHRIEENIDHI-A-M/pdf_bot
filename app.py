import streamlit as st
import pdfplumber
from transformers import pipeline, PipelineException

# Function to extract text from a PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# Cached function to load summarization model
@st.cache_resource
def load_summarizer():
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        return summarizer
    except Exception as e:
        st.error(f"Error loading summarization model: {e}")
        return None

# Cached function to load question-answering model
@st.cache_resource
def load_qa_model():
    try:
        qa_model = pipeline("question-answering")
        return qa_model
    except Exception as e:
        st.error(f"Error loading question-answering model: {e}")
        return None

# Function to summarize text
def summarize_text(text, summarizer):
    try:
        summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    except PipelineException as e:
        st.error(f"Error during summarization: {e}")
        return ""

# Function to answer questions based on the text
def answer_question(context, question, qa_model):
    try:
        result = qa_model(question=question, context=context)
        return result['answer']
    except PipelineException as e:
        st.error(f"Error during question answering: {e}")
        return ""

# Streamlit app layout
st.title('PDF Summarizer and Q&A Bot')
uploaded_file = st.file_uploader('Upload a PDF', type='pdf')

if uploaded_file is not None:
    # Extract text from uploaded PDF
    text = extract_text_from_pdf(uploaded_file)
    
    if text:
        # Display extracted text
        st.subheader("Extracted Text")
        st.write(text)

        # Load models
        summarizer = load_summarizer()
        qa_model = load_qa_model()

        if summarizer:
            # Summarize the text
            summary = summarize_text(text, summarizer)
            st.subheader("Summary")
            st.write(summary)
        
        # Question and Answer section
        if qa_model:
            st.subheader("Ask a Question about the PDF")
            question = st.text_input("Enter your question")
            if question:
                answer = answer_question(text, question, qa_model)
                st.write("Answer:")
                st.write(answer)
    else:
        st.error("Failed to extract text from the PDF.")
