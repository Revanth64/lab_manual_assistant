# 1. DEPENDENCIES & ENVIRONMENT SETUP
import os
import numexpr as ne
import numpy as np
import sympy as sp
import streamlit as st
from dotenv import load_dotenv

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.tools import Tool, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

st.set_page_config(page_title="Agentic CS Lab Assistant", page_icon="🤖", layout="wide")
st.title("🤖 Agentic CS & Data Science Lab Assistant")

# 2. AGENT TOOLS DEFINITION

@tool
def scientific_calculator(expression: str) -> str:
    """
    Evaluates scientific mathematical expressions or symbolic math equations.
    Use this tool whenever exact mathematical calculations, matrix operations, 
    derivatives, integrals, or complex arithmetic are needed.
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

# 3. PDF PROCESSING & FAISS VECTOR STORE

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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)


def build_and_save_vector_store(text_chunks, index_path="faiss_lab_index"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(index_path)
    return vector_store

# 4. AGENT INITIALIZATION & FALLBACK ENGINE

def build_agent_executor(vector_store):
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

    manual_tool = get_lab_manual_retriever_tool(vector_store)
    tools = [manual_tool, scientific_calculator]

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        "You are an autonomous AI CS & Data Science Lab Assistant specializing in "
        "Indian University Lab Curricula (Autonomous, VTU, JNTU, Anna University, IITs/NITs).\n"
        "You have access to tools: 'lab_manual_search' and 'scientific_calculator'.\n\n"
        "When answering queries:\n"
        "- Format lab experiments into standard record layouts (Aim, Theory, Algorithm, Code, Expected Output, Viva Questions).\n"
        "- If information from the lab manual is needed, call 'lab_manual_search'.\n"
        "- If exact calculations, asymptotic complexity proofs, or matrix math are needed, call 'scientific_calculator'.\n"
        "- Execute tools in sequence when multi-step reasoning is required."
        ),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True
    )

# 5. STREAMLIT UI & INTERACTION LOOP

import shutil

# 5.1 Initialize Session State Variables
if "messages" not in st.session_state:
    st.session_state.messages = []

# Auto-load existing index on app launch/refresh
if "vector_store" not in st.session_state and os.path.exists("faiss_lab_index"):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        st.session_state["vector_store"] = FAISS.load_local(
            "faiss_lab_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"Failed to load cached index: {e}")

# 5.2 Sidebar Setup & File Upload
with st.sidebar:
    st.header("1. Upload Lab Manual")
    pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Process & Build Index", use_container_width=True):
            if pdf_docs:
                with st.spinner("Extracting text and building vector index..."):
                    if os.path.exists("faiss_lab_index"):
                        shutil.rmtree("faiss_lab_index")
                    
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = build_and_save_vector_store(text_chunks)
                    st.session_state["vector_store"] = vector_store
                    st.success("Fresh index successfully built!")
                    st.rerun()
            else:
                st.warning("Please upload a PDF first.")

    with col2:
        if st.button("Clear Index", use_container_width=True):
            if os.path.exists("faiss_lab_index"):
                shutil.rmtree("faiss_lab_index")
            if "vector_store" in st.session_state:
                del st.session_state["vector_store"]
            st.success("Vector memory cleared!")
            st.rerun()

# 5.3 Render Historical Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5.4 User Query Handler & Agent Execution
if user_input := st.chat_input("Ask a question, request an experiment, or ask for math calculations..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if "vector_store" in st.session_state:
        with st.chat_message("assistant"):
            with st.status("Agent thinking...", expanded=True) as status:
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
                agent_executor = build_agent_executor(st.session_state["vector_store"])
                
                # Format prior conversation history for prompt MessagesPlaceholder
                formatted_history = [
                    (msg["role"], msg["content"]) for msg in st.session_state.messages[:-1]
                ]
                
                try:
                    response = agent_executor.invoke(
                        {
                            "input": user_input,
                            "chat_history": formatted_history
                        },
                        {"callbacks": [st_cb]}
                    )
                except Exception:
                    response = agent_executor.invoke({
                        "input": user_input,
                        "chat_history": formatted_history
                    })
                
                status.update(label="Complete!", state="complete", expanded=False)

            answer_raw = response.get("output", "")
            
            if isinstance(answer_raw, list) and len(answer_raw) > 0 and isinstance(answer_raw[0], dict):
                answer = answer_raw[0].get("text", str(answer_raw))
            elif isinstance(answer_raw, dict):
                answer = answer_raw.get("text", str(answer_raw))
            else:
                answer = str(answer_raw)

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        with st.chat_message("assistant"):
            warning_msg = "Please upload and process your CS lab manual PDF first."
            st.markdown(warning_msg)
            st.session_state.messages.append({"role": "assistant", "content": warning_msg})