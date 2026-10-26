import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load API Key from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    os.environ["GEMINI_API_KEY"] = api_key


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks


def build_and_save_vector_store(text_chunks, index_path="faiss_lab_index"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(index_path)
    return vector_store


def process_lab_manual(pdf_docs):
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing from the .env file.")

    raw_text = get_pdf_text(pdf_docs)
    if not raw_text.strip():
        raise ValueError("Could not extract any readable text from the provided PDF(s).")

    text_chunks = get_text_chunks(raw_text)
    build_and_save_vector_store(text_chunks)
    return len(text_chunks)


#Streamlit interface
import streamlit as st

# Streamlit Page Config
st.set_page_config(page_title="Lab Manual Assistant", layout="wide")
st.title(" Lab Manual Assistant ")
st.write("Upload your lab manual PDF to extract text and generate a FAISS vector index.")

# API Key Warning Notice
if not api_key:
    st.warning(" `GEMINI_API_KEY` missing from `.env`. Please add it to process PDFs.")

# Sidebar File Uploader
with st.sidebar:
    st.header(" Document Management")
    pdf_docs = st.file_uploader(
        "Upload Lab Manual PDFs", 
        accept_multiple_files=True, 
        type=["pdf"]
    )
    process_button = st.button("Process & Build Index")

# Processing Action
if process_button:
    if not pdf_docs:
        st.error("Please upload at least one PDF file.")
    else:
        with st.spinner("Extracting text and generating FAISS vector index..."):
            try:
                total_chunks = process_lab_manual(pdf_docs)
                st.success(f" Processing complete! Created {total_chunks} chunks and saved index to 'faiss_lab_index'.")
            except Exception as e:
                st.error(f" Error during processing: {str(e)}")