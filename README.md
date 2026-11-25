# lab_manual_assistant
Developed as part of my internship at Capabl, the Lab Manual Assistant is designed to streamline academic lab workflows by assisting students and instructors with navigating, generating, and referencing lab manuals efficiently.


# 🤖 Agentic CS & Data Science Lab Assistant

An autonomous, domain-specialized AI Lab Assistant tailored for Indian University Computer Science, Machine Learning, and Data Science laboratory curricula. 

Built using **LangChain**, **Google Gemini**, **FAISS**, **SymPy**, and **Streamlit**, this application combines Retrieval-Augmented Generation (RAG) with a ReAct tool-calling agent to guide students through lab experiments, generate formatted lab records, compute exact mathematical/complexity proofs, and prepare for Viva Voce examinations.

---

## ✨ Key Features

* **Autonomous Tool-Calling Agent**: Dynamically evaluates user queries to decide whether to search the uploaded lab manual, run exact mathematical calculations, or execute multi-step workflows combining both.
* **Domain-Specialized Output Formatting**: Structures responses into standard university lab record layouts (**Aim, Theory, Algorithm/Steps, Code Implementation, Expected Output, and Viva Voce Q&A**).
* **Zero-Hallucination Scientific Calculator**: Integrates **SymPy** and **NumExpr** as an agent tool to evaluate symbolic calculus, matrix inversions, statistical metrics, and asymptotic complexity derivations ($O(N \log N)$, $O(N^2)$).
* **Multi-Model Fallback Engine**: Configured with automatic failover across multiple Gemini models (`gemini-3.8-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, `gemini-2.5-flash-lite`, `gemini-3.6-flash`, `gemini-3-flash`, `gemini-2.5-flash`) for uninterrupted execution.
* **Local FAISS Vector Storage**: Extracts and chunks PDF lab manuals via `PyPDF2` and `RecursiveCharacterTextSplitter`, indexing them locally with one-click index wiping to prevent stale document bleed across uploads.
* **Live Thought Streaming**: Renders the agent's step-by-step reasoning and tool invocations directly in the Streamlit UI.

---

## 🛠️ Tech Stack

* **Frontend / UI**: Streamlit
* **LLM & Embeddings**: Google Gemini (`langchain-google-genai`, `models/gemini-embedding-001`)
* **Agent Framework**: LangChain (`langchain-classic`, `langchain-core`, `langchain-community`)
* **Vector Database**: FAISS (`faiss-cpu`)
* **Document Processing**: PyPDF2
* **Scientific & Symbolic Math**: SymPy, NumExpr, NumPy

---

## 📂 Project Structure

```text
lab_manual_assistant/
├── app.py               # Main Streamlit application & Agentic RAG workflow
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (excluded from Git)
├── .gitignore           # Git ignore rules
└── faiss_lab_index/     # Local FAISS vector store (auto-generated on upload)
```

---

## 🚀 Setup & Local Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/](https://github.com/)<your-username>/lab_manual_assistant.git
cd lab_manual_assistant
```

### 2. Create & Activate a Virtual Environment
```bash
python -m venv venv
# Windows (Git Bash)
source venv/Scripts/activate
# Linux / macOS
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory and add your Google Gemini API key:
```env
GOOGLE_API_KEY="your_google_api_key_here"
```

### 5. Run the Application
```bash
streamlit run app.py
```

---

## 💡 Usage Guide

1. **Upload Lab Manual**: Open the sidebar, upload one or more CS/Data Science lab manual PDFs, and click **Process & Build Index**.
2. **Ask Questions**:
   * *Experiment Walkthrough*: `"Give me the complete lab record format for Experiment 3."`
   * *Viva Prep*: `"Generate 5 external viva questions and answers based on the clustering experiment in the manual."`
   * *Mathematical & Complexity Proofs*: `"Find the derivative of x**3 + 4*x**2 and compute the inverse of Matrix([[1, 2], [3, 4]])."`
   * *Multi-Step Reasoning*: `"Search the manual for the Linear Regression experiment and calculate the MSE for predictions [2.5, 4.1] against targets [3.0, 4.0]."`
3. **Manage Vector Memory**: Click **Clear Index** in the sidebar anytime you want to wipe the current PDF embeddings before uploading a different subject manual.
