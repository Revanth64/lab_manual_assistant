import os
import streamlit as st
from dotenv import load_dotenv

# PDF & Text Processing
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector Store & Google Gemini LLM
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# LangChain LCEL Core Imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables (.env)
load_dotenv()

# Streamlit Page Configuration
st.set_page_config(page_title="Lab Manual Assistant", page_icon="📚")
st.title("Lab Manual Assistant")

# 1. BACKEND PROCESSING LOGIC

def get_pdf_text(pdf_docs):
    """Extracts raw text content from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text

def get_text_chunks(text):
    """Splits raw text into overlapping chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

def build_and_save_vector_store(text_chunks, index_path="faiss_lab_index"):
    """Generates embeddings using Gemini and saves local FAISS vector store."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(index_path)
    return vector_store

# 2. RAG RETRIEVAL CHAIN LOGIC (LCEL)

def format_docs(docs):
    """Formats retrieved document chunks into a single text string."""
    return "\n\n".join(doc.page_content for doc in docs)

def get_conversational_chain(vector_store):
    """Creates an LCEL RAG chain connecting FAISS retriever with Gemini LLM."""
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # Initialize Gemini Model
    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.3)
    
    system_prompt = (
        "You are an expert AI lab manual assistant. Use the following pieces of retrieved "
        "context to answer the user's question accurately. If you do not know the answer based "
        "on the context, clearly state that the information is not present in the manual.\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Updated LCEL Pipeline: extracts the string from {"input": user_input} before passing to retriever
    rag_chain = (
        {
            "context": (lambda x: x["input"]) | retriever | format_docs,
            "input": lambda x: x["input"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain
# 3. STREAMLIT INTERFACE & WORKFLOW
# Document Upload & Vector Indexing Sidebar
with st.sidebar:
    st.header("Upload Lab Manuals")
    pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])
    if st.button("Process & Build Index"):
        if pdf_docs:
            with st.spinner("Extracting text and building vector index..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_store = build_and_save_vector_store(text_chunks)
                st.session_state["vector_store"] = vector_store
                st.success("Vector store index successfully created!")
        else:
            st.warning("Please upload at least one PDF file.")

# Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load existing local FAISS index on launch if available
if "vector_store" not in st.session_state:
    if os.path.exists("faiss_lab_index"):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        st.session_state["vector_store"] = FAISS.load_local(
            "faiss_lab_index", 
            embeddings, 
            allow_dangerous_deserialization=True
        )

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Query Handler
if user_input := st.chat_input("Ask a question about your lab manual..."):
    # Display user query
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response via RAG Chain
    if "vector_store" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Searching lab manual..."):
                rag_chain = get_conversational_chain(st.session_state["vector_store"])
                answer = rag_chain.invoke({"input":user_input})
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        with st.chat_message("assistant"):
            warning_msg = "Please upload and process a lab manual PDF first."
            st.markdown(warning_msg)
            st.session_state.messages.append({"role": "assistant", "content": warning_msg})