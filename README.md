# LangGraph Workflow Architectures & RAG Agents

This repository demonstrates various advanced AI agent architectures using **LangGraph**, **LangChain**, and the **Groq API**. It includes a full implementation of different node-graph patterns—ranging from simple sequential pipelines to complex Human-in-the-Loop (HITL) and Retrieval-Augmented Generation (RAG) systems.

## 🚀 Projects Included

### 1. Conditional Workflow (College Chatbot RAG)
**File:** `conditional_workflows.py`
A smart college assistant that routes user queries based on intent. 
* **Knowledge Bases:** Uses `academics_handbook.pdf` and `fee_structure.pdf`.
* **Flow:** Classifies the query -> Routes to Academic RAG, Fee RAG, or General Knowledge -> Generates a customized response based on the student's program (BCA, BBA, B.Com(H)).

### 2. Sequential Agents (Content Pipeline)
**File:** `sequential_agents.py`
A multi-stage AI content creation pipeline for YouTube scripts.
* **Flow:** Raw Text -> Editor Node (Grammar/Tone) -> Scriptwriter Node (Formatting/Engagement) -> Translator Node (Natural Hinglish Output).

### 3. Parallel Execution with Reducers (Safety Analyzer)
**File:** `parallel_reducer.py`
Demonstrates a Fan-Out/Fan-In architecture using custom state reducers.
* **Flow:** Takes a raw text input -> Analyzes it simultaneously for Toxicity, Copyright Risk, and Cultural Sensitivity -> Merges (reduces) all scores into a single state dictionary.

### 4. Human-in-the-Loop (HITL) Post Generator
**File:** `humanInTheLoop.py`
An interactive LinkedIn post generator that pauses execution to ask for human approval.
* **Flow:** AI drafts post -> Execution pauses (Interrupt) -> Human reviews -> If rejected, loops back to writer with feedback -> If approved, graph ends.

### 5. Iterative Feedback Loop (Self-Correcting AI)
**File:** `iterative_Feedback.py`
An autonomous workflow where an AI writer and an AI reviewer collaborate.
* **Flow:** Writer drafts a post (using Tavily search tool if needed) -> Reviewer evaluates strictly against guidelines -> If rejected, feedback is sent back to the writer -> Loops until approved (max 3 attempts).

### 6. State Management Concepts
**File:** `states.py`
A reference file demonstrating the 4 primary ways to handle State in LangGraph:
* `TypedDict`
* `Pydantic BaseModel` (with runtime validation)
* `Python Dataclasses`
* `MessagesState` (built-in LangGraph state)

## 🛠️ Tech Stack
* **Framework:** [LangGraph](https://python.langchain.com/docs/langgraph) / LangChain
* **LLM Provider:** [Groq API](https://groq.com/) (`openai/gpt-oss-120b`, `qwen/qwen3.8-27b`)
* **Embeddings:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
* **Vector Store:** FAISS
* **Tools:** Tavily Search API

## ⚙️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```

2. **Install Dependencies:**
   Ensure you have Python installed, then run:
   ```bash
   pip install langchain langgraph langchain-groq langchain-huggingface langchain-community faiss-cpu pypdf python-dotenv langchain-tavily
   ```

3. **Environment Variables:**
   Create a `.env` file in the root directory and add your API keys:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

## ▶️ Usage
You can run any of the standalone Python scripts to test the different workflows. For example:

To run the College RAG Chatbot:
```bash
python conditional_workflows.py
```

To run the Human-in-the-Loop generator:
```bash
python humanInTheLoop.py
```

To test your Groq API connection and list models:
```bash
python test.py
```
