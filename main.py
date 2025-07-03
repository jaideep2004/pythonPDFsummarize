import streamlit as st
import PyPDF2
from transformers import pipeline
import nltk
import os

# Set NLTK data path to a local directory in the repo
nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

# Download NLTK data locally (run this locally once, not on Streamlit Cloud)
try:
    nltk.download('punkt', download_dir=nltk_data_dir)
    nltk.download('punkt_tab', download_dir=nltk_data_dir)
except Exception as e:
    st.warning(f"Error downloading NLTK data: {e}")

print(nltk.data.path)

# Initialize the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def extract_text_from_pdf(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
        return text if text else None
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None 

def summarize_text(text, max_length=150, min_length=50):
    try:
        # Split text into sentences
        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return "No valid sentences to summarize."

        # Create chunks of text within the model's input limit
        chunk_size = 1000  # Max input length for BART
        chunks = []
        current_chunk = []
        current_length = 0

        # Group sentences into chunks without exceeding chunk_size
        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length <= chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        summaries = []
        for chunk in chunks:
            if len(chunk.strip()) > 50:
                try:
                    summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
                    if summary and isinstance(summary, list) and len(summary) > 0:
                        summaries.append(summary[0].get('summary_text', ''))
                    else:
                        st.warning(f"Skipping chunk: No summary generated for chunk of length {len(chunk)}")
                except Exception as e:
                    st.warning(f"Error summarizing chunk: {e}")
            else:
                st.warning(f"Skipping chunk: Text too short ({len(chunk)} characters)")

        return ' '.join(summaries) if summaries else "No valid text to summarize."
    except Exception as e:
        st.error(f"Error summarizing text: {e}")
        return None

def main(): 
    st.title("Free PDF Summarizer")
    st.write("Upload a PDF file to get a concise summary of its content.")
     
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        with st.spinner("Extracting text..."):
            text = extract_text_from_pdf(uploaded_file)
        
        if text:
            with st.spinner("Generating summary..."):
                summary = summarize_text(text)
            
            if summary:
                st.subheader("Summary")
                st.write(summary)

if __name__ == "__main__":
    main()