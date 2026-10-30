import os
import numexpr as ne
import numpy as np
import sympy as sp
import streamlit as st
from dotenv import load_dotenv

# PDF & Text Processing
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector Store & Google Gemini Integrations
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Agent & Tool Imports
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.tools import Tool, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables (.env)
load_dotenv()

st.set_page_config(page_title="Agentic CS Lab Assistant", page_icon="🤖", layout="wide")
st.title("🤖 Agentic CS & Data Science Lab Assistant")

# ==========================================
# 1. DEFINE AGENT TOOLS
# ==========================================

@tool
def scientific_calculator(expression: str) -> str:
    """
    Evaluates scientific mathematical expressions or symbolic math equations.
    Use this tool whenever exact mathematical calculations, matrix operations, 
    derivatives, integrals, or complex arithmetic are needed.

    Examples:
    - Symbolic Calculus: 'diff(x**3 + 2*x, x)' or 'integrate(sin(x), x)'
    - Matrix Operations: 'Matrix([[1, 2], [3, 4]]).inv()'
    - Numerical Math: '3.5 * log(100) + sqrt(16)'
    """
    try:
        symbolic_keywords = ["x", "y", "z", "diff", "integrate", "Matrix", "symbols", "solve"]
        if any(keyword in expression for keyword in symbolic_keywords):
            x, y, z = sp.symbols('x y z')
            result = sp.sympify(expression)
            return str(result)
        else:
            result = ne.evaluate(expression)
            return str(result)
    except Exception as e:
        return f"Calculation Error: {str(e)}"


def get_lab_manual_retriever_tool(vector_store):
    """Wraps the FAISS vector store into a callable tool for the Agent."""
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    def search_manual(query: str) -> str:
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant information found in the lab manual."
        return "\n\n".join(doc.page_content for doc in docs)

    return Tool(
        name="lab_manual_search",
        func=search_manual,
        description=(
            "Useful for searching the uploaded Computer Science lab manual for "
            "experiment steps, code samples, theoretical backgrounds, or lab instructions. "
            "Pass specific keywords or queries about the experiment."
        )
    )

# ==========================================
# 2. PDF EXTRACTION & FAISS VECTOR STORE
# ==========================================

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

# ==========================================
# 3. AGENTIC WORKFLOW INITIALIZATION
# ==========================================

def build_agent_executor(vector_store):
    """Constructs a tool-calling AgentExecutor with Gemini models and fallbacks."""
    
    # Define primary model and fallbacks
    primary_llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.2)
    fallback_models = [
        ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.2),
        ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.2),
        ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.2),
	ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.2),
        ChatGoogleGenerativeAI(model="gemini-3-flash", temperature=0.2),
        ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2),
    ]
    llm = primary_llm.with_fallbacks(fallback_models)

    # Tool belt
    manual_tool = get_lab_manual_retriever_tool(vector_store)
    tools = [manual_tool, scientific_calculator]

    # Agent Prompt Template with Tool Calling Placeholder
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an autonomous AI CS & Data Science Lab Assistant.\n"
            "You have access to tools: 'lab_manual_search' and 'scientific_calculator'.\n"
            "Reason step-by-step to answer the user request:\n"
            "1. If information from the lab manual is needed, call 'lab_manual_search'.\n"
            "2. If calculations, formula derivations, or matrix math are needed, call 'scientific_calculator'.\n"
            "3. If both are required, execute them in sequence before forming your final response."
        ),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Construct Agent & Executor
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True
    )
    
    return agent_executor

# ==========================================
# 4. STREAMLIT INTERFACE
# ==========================================

# Sidebar
with st.sidebar:
    st.header("1. Upload Lab Manual")
    pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])
    if st.button("Process & Build Index"):
        if pdf_docs:
            with st.spinner("Extracting text and building vector index..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_store = build_and_save_vector_store(text_chunks)
                st.session_state["vector_store"] = vector_store
                st.success("Lab manual index successfully built!")
        else:
            st.warning("Please upload a lab manual PDF first.")

# Initialize Session Messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load local FAISS vector store on launch if available
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

# User Chat Input Handler
if user_input := st.chat_input("Ask a question, request an experiment, or ask for math calculations..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if "vector_store" in st.session_state:
        with st.chat_message("assistant"):
            # Streamlit Callback Handler renders agent thoughts and tool calls live in the UI
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
            
            agent_executor = build_agent_executor(st.session_state["vector_store"])
            
            response = agent_executor.invoke(
                {"input": user_input},
                {"callbacks": [st_cb]}
            )
            
            answer = response["output"]
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        with st.chat_message("assistant"):
            warning_msg = "Please upload and process your CS lab manual PDF first."
            st.markdown(warning_msg)
            st.session_state.messages.append({"role": "assistant", "content": warning_msg})