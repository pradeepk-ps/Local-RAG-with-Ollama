# Local RAG with Ollama

A complete Local Retrieval-Augmented Generation (RAG) project built using:

- Python
- Ollama
- LangChain
- ChromaDB
- Streamlit

This project allows users to chat with their own custom knowledge base locally without using cloud LLM APIs.

---

# Project Overview

This project demonstrates how to build a complete Local RAG (Retrieval-Augmented Generation) system from scratch.

The application:

1. Collects data from external sources
2. Converts the data into embeddings
3. Stores embeddings in a vector database (ChromaDB)
4. Accepts user questions
5. Searches relevant information
6. Sends context to the LLM
7. Generates intelligent answers locally using Ollama

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Backend Programming |
| Ollama | Running local LLMs |
| LangChain | RAG pipeline framework |
| ChromaDB | Vector database |
| Streamlit | Frontend UI |
| Embedding Models | Convert text into vectors |

---

# Project Structure

```bash
Local-RAG-with-Ollama/
│
├── chroma_db/
│   ├── chroma.sqlite3
│   └── snapshots/
│
├── dataset/
│   └── wikipedia_data.txt
│
├── 1_scraping_wikipedia.py
├── 2_chunking_embedding_ingestion.py
├── 3_chatbot.py
│
├── requirements.txt
├── .env.example
├── README.md
└── thumbnail_small.png
```

---

# Understanding Each File

## 1. 1_scraping_wikipedia.py

This file is responsible for collecting data.

### What happens here?

- The script fetches information from Wikipedia
- The content is cleaned
- The text is saved inside the dataset folder

### Output

```bash
dataset/wikipedia_data.txt
```

This becomes the knowledge source for the chatbot.

---

## 2. 2_chunking_embedding_ingestion.py

This is the most important step in the RAG pipeline.

### What happens here?

### Step 1 — Read Dataset

The script reads the text file created earlier.

---

### Step 2 — Chunking

Large text cannot be directly processed efficiently.

So the text is split into smaller chunks.

Example:

```text
Chunk 1 → Introduction to Python
Chunk 2 → Features of Python
Chunk 3 → Python Applications
```

Why chunking?

- Better search accuracy
- Faster retrieval
- Efficient embeddings

---

### Step 3 — Create Embeddings

Each chunk is converted into vector embeddings.

Embeddings are numerical representations of text.

Example:

```text
"Python is a programming language"

→ [0.245, -0.782, 0.128, ...]
```

These vectors help AI understand semantic meaning.

---

### Step 4 — Store in ChromaDB

The embeddings are stored inside ChromaDB.

This creates:

```bash
chroma_db/
```

Inside this folder:

| File/Folder | Purpose |
|---|---|
| chroma.sqlite3 | Stores vector database metadata |
| snapshots/ | Stores vector index snapshots |

---

# Why ChromaDB is Needed?

Traditional databases search exact words.

Vector databases search meaning.

Example:

User asks:

```text
"What is Python used for?"
```

Even if dataset contains:

```text
"Python is widely used in AI and web development"
```

The vector database understands semantic similarity.

This is why RAG systems use vector databases.

---

# How snapshots Folder is Created?

When embeddings are stored:

```python
Chroma.from_documents(...)
```

ChromaDB automatically creates:

```bash
chroma_db/
└── snapshots/
```

Snapshots store vector indexes for fast retrieval.

Without snapshots:

- Search becomes slow
- Embeddings need regeneration every time

---

# Why requirements.txt is Important?

This file stores all project dependencies.

Example:

```txt
streamlit
langchain
chromadb
ollama
python-dotenv
```

Why needed?

Instead of installing libraries manually one by one:

```bash
pip install -r requirements.txt
```

installs everything automatically.

---

# 3. 3_chatbot.py

This file runs the chatbot application.

### What happens here?

- Loads ChromaDB
- Loads embedding model
- Accepts user query
- Retrieves relevant chunks
- Sends context + question to Ollama
- Displays final response

---

# Complete RAG Workflow

## Step-by-Step Flow

---

## Step 1 — Data Collection

Wikipedia data is collected using:

```bash
python 1_scraping_wikipedia.py
```

Output:

```bash
dataset/wikipedia_data.txt
```

---

## Step 2 — Chunking + Embeddings

Run:

```bash
python 2_chunking_embedding_ingestion.py
```

This:

- Reads dataset
- Splits into chunks
- Creates embeddings
- Stores vectors in ChromaDB

Output:

```bash
chroma_db/
```

---

## Step 3 — Run Chatbot

Run:

```bash
streamlit run 3_chatbot.py
```

Streamlit launches a local web application.

---

# How User Query Becomes Final Answer

Suppose user asks:

```text
"What is Python?"
```

---

## Internal Flow

### Step 1 — User Query

User enters question in Streamlit UI.

---

### Step 2 — Query Embedding

The question is converted into embeddings.

Example:

```text
"What is Python?"

→ [0.182, -0.734, 0.921, ...]
```

---

### Step 3 — Similarity Search

ChromaDB compares:

- User query embeddings
with
- Stored document embeddings

Then retrieves the most relevant chunks.

Example retrieved chunk:

```text
"Python is a high-level programming language..."
```

---

### Step 4 — Context Injection

The retrieved chunks are added to the prompt.

Example:

```text
Context:
Python is a high-level programming language...

Question:
What is Python?
```

---

### Step 5 — Send to Ollama

The prompt is sent to the local Ollama model.

Example models:

- llama3
- qwen3
- mistral

---

### Step 6 — LLM Generates Answer

Ollama generates the final response using:

- User question
- Retrieved context

---

### Final Output

```text
Python is a high-level programming language used for web development, AI, automation, and more.
```

---

# Installation Guide

## Step 1 — Clone Repository

```bash
git clone https://github.com/pradeepk-ps/Local-RAG-with-Ollama.git
cd Local-RAG-with-Ollama
```

---

## Step 2 — Create Virtual Environment

```bash
python -m venv venv
```

---

## Step 3 — Activate Environment

### Windows

```bash
venv\Scripts\Activate
```

### Mac/Linux

```bash
source venv/bin/activate
```

---

## Step 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Step 1 — Collect Data

```bash
python 1_scraping_wikipedia.py
```

---

## Step 2 — Generate Embeddings

```bash
python 2_chunking_embedding_ingestion.py
```

---

## Step 3 — Launch Chatbot

```bash
streamlit run 3_chatbot.py
```

---

# Features

- Fully Local RAG
- No cloud dependency
- Vector similarity search
- Local LLM integration
- Beginner-friendly architecture
- Fast semantic retrieval
- Streamlit UI

---

# Conclusion

This project demonstrates how Retrieval-Augmented Generation (RAG) systems work internally.

It explains:

- Data ingestion
- Chunking
- Embeddings
- Vector databases
- Semantic search
- Local LLM inference

The project can be extended into:

- AI assistants
- PDF chatbots
- Enterprise search systems
- Knowledge base systems
- Research assistants

---

# Author

Pradeep Kamma

GitHub:
https://github.com/pradeepk-ps
