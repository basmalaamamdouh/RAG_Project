#  NeuralLens — AI/ML RAG Visual Tutor

NeuralLens is a **Generative AI educational system** designed to transform traditional Machine Learning learning into an **interactive, explainable, and visual experience**.

It combines **Retrieval-Augmented Generation (RAG)**, **Large Language Models (Groq LLaMA 3.1)**, and **LLM-driven visualization generation (Plotly.js)** to help users understand complex AI/ML concepts intuitively.

---

##  Key Highlights

-  **RAG-powered AI system grounded in ML textbooks**
-  Semantic knowledge retrieval using FAISS vector database
-  Fast LLM inference using Groq (LLaMA 3.1)
-  Automatic ML visualizations using LLM-generated Plotly.js specs
-  Streaming real-time AI responses
-  Auto-generated quizzes for self-assessment
-  Adaptive explanations (Beginner / Intermediate / Expert modes)

---

##  AI System Design

### 1. Retrieval-Augmented Generation (RAG)
The system is built on a domain-specific knowledge base derived from ML textbooks and educational notes.

- Text is split into semantic chunks
- Each chunk is embedded using Sentence Transformers
- Stored in FAISS vector database for similarity search
- Relevant chunks are retrieved based on user query

---

### 2. LLM Layer (Groq - LLaMA 3.1)
- Retrieved context is injected into carefully engineered prompts
- Model generates:
  - Explanations
  - Step-by-step reasoning
  - Educational breakdowns adapted to user level

---

### 3. Generative Visualization Engine
Instead of static charts, the LLM generates structured **Plotly.js JSON specifications** that represent ML concepts such as:

- Decision boundaries (SVM, KNN)
- Optimization curves (Gradient Descent)
- Clustering distributions (K-Means)
- Neural network structures
- Attention heatmaps

This enables **visual reasoning of abstract ML concepts**.

---

### 4. Streaming Inference Architecture
- Real-time token streaming from Groq API
- Event-driven response pipeline:
  - Status updates
  - Token streaming
  - Visualization rendering
  - Completion signals

---

##  Knowledge Base (RAG Data)

The system is grounded in a curated dataset built from ML educational materials.

### Data Pipeline:
- Extract ML textbook content (PDF/text)
- Clean and split into semantic chunks
- Generate embeddings using Sentence Transformers
- Store in FAISS vector database

### Retrieval Flow:
1. User asks a question
2. Query is embedded into vector space
3. FAISS retrieves most relevant chunks
4. Context is passed to LLM
5. Final grounded answer is generated

---

## 🏗️ System Architecture

- **RAG Pipeline:** FAISS + Sentence Transformers
- **LLM Layer:** Groq API (LLaMA 3.1)
- **Visualization Engine:** LLM-generated Plotly.js JSON specs
- **Streaming Layer:** Server-Sent Events (SSE)
- **Frontend:** HTML, CSS, JavaScript (Plotly.js visualization layer)
- **Backend:** Flask API




---

##  Features

###  Visualization Mode
Automatically generates interactive charts for ML concepts.

###  AI Chat Tutor
Ask any ML/AI question and receive:
- Explanation
- Intuition
- Visual representation

### Quiz System
Auto-generated MCQs based on current topic for self-assessment.

###  Adaptive Learning
Responses adjust based on selected difficulty level:
- Beginner → simple intuition
- Intermediate → technical explanation
- Expert → deeper reasoning

---

##  Installation

```bash
git clone https://github.com/basmalaamamdouh/RAG_Project.git
cd RAG_Project
pip install -r requirements.txt
