# LEGAL REACH RAG - Project Documentation

**A Professional Guide to Understanding the Legal Document RAG System**

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [High-Level Design (HLD)](#high-level-design-hld)
3. [Low-Level Design (LLD)](#low-level-design-lld)
4. [Architecture & Workflows](#architecture--workflows)
5. [Technology Stack](#technology-stack)
6. [Core Components](#core-components)
7. [Code Explanations](#code-explanations)
8. [Data Flow Diagram](#data-flow-diagram)
9. [Setup & Execution](#setup--execution)
10. [Important Implementation Details](#important-implementation-details)

---

## Project Overview

### What is LEGAL REACH RAG?

**Legal Reach RAG** is an AI-powered legal assistant that retrieves information from legal documents and generates accurate answers using a local large language model (LLM). It follows the **Retrieval Augmented Generation (RAG)** pattern, which combines:

1. **Retrieval**: Search through a vector database to find relevant legal documents
2. **Augmentation**: Use retrieved documents as context
3. **Generation**: Generate accurate answers using an LLM

### Problem Statement

Traditional legal research requires:
- Manual document searching (time-consuming)
- Multiple document readings
- High risk of missing relevant information
- No automated source citation

**Our Solution**: An AI assistant that automatically searches, retrieves, and answers legal questions with proper source citations.

### Key Features

✓ Local LLM execution (no cloud dependency)
✓ Fast similarity-based document retrieval
✓ Source citations with page numbers
✓ Python 3.14 compatible
✓ Minimal dependencies (production-ready)
✓ Custom vector store (no complex databases)
✓ Support for PDF, DOCX, and TXT documents

---

## High-Level Design (HLD)

### System Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   USER INTERACTION LAYER                │
│  (CLI Chat Interface - main.py)                         │
└────────────────────┬────────────────────────────────────┘
                     │ Question Input
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  RAG PIPELINE LAYER                      │
│  (Orchestration & LLM Integration)                      │
│                                                          │
│  1. Question Processing                                │
│  2. Context Retrieval (vector.py)                      │
│  3. LLM Answer Generation (OllamaLLM)                  │
│  4. Source Citation Formatting                         │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
┌──────────────────┐    ┌──────────────────┐
│   VECTOR STORE   │    │   LLM SERVICE    │
│   (vector.py)    │    │   (Ollama)       │
│                  │    │                  │
│ • In-Memory      │    │ • llama3.2       │
│   Embeddings     │    │ • Temperature=0.1│
│ • Cosine         │    │ • Port: 11434    │
│   Similarity     │    │                  │
│ • k=4 retrieval  │    │                  │
└────────┬─────────┘    └──────────────────┘
         │
         ▼
┌──────────────────────────┐
│  DOCUMENT MANAGEMENT     │
│ (document_processor.py)  │
│                          │
│ • PDF/DOCX/TXT Loading   │
│ • Chunking (512 tokens)  │
│ • Metadata Tracking      │
└──────────────────────────┘
         │
         ▼
┌──────────────────────────┐
│   PERSISTENT STORAGE     │
│  (vectorstore/ folder)   │
│                          │
│ • vectors.npy (embeddings│
│ • documents.pkl (metadata
└──────────────────────────┘
```

### System Components at 10,000 ft

| Layer | Component | Purpose |
|-------|-----------|---------|
| **UI Layer** | main.py | CLI interface, user interaction |
| **RAG Orchestration** | main.py (functions) | Pipeline management |
| **Vector Store** | vector.py (SimpleVectorStore) | Document embeddings & retrieval |
| **LLM Service** | OllamaLLM | Local language model inference |
| **Document Processing** | document_processor.py | Load & chunk documents |
| **Configuration** | config.py | System parameters & settings |
| **Storage** | vectorstore/ | Persistent embeddings & metadata |

---

## Low-Level Design (LLD)

### 1. SimpleVectorStore Class (Core Engine)

Located in `vector.py`, this is our custom vector database replacement.

**Why Custom Vector Store?**
- Chromadb uses Pydantic V1 (incompatible with Python 3.14)
- Lightweight, dependency-free alternative
- Direct control over persistence and retrieval

**Key Methods:**

```python
class SimpleVectorStore:
    
    def add_documents(documents, ids):
        """
        Add documents to the store with their embeddings
        
        Process:
        1. Get embeddings for each document
        2. Store embeddings in numpy array
        3. Map IDs to documents
        4. Save to disk (vectors.npy + documents.pkl)
        """
    
    def similarity_search(query, k=4):
        """
        Find k most similar documents to the query
        
        Algorithm:
        1. Get embedding for query
        2. Compute cosine similarity between query and all documents
        3. Sort by similarity score (highest first)
        4. Return top-k documents
        """
    
    def as_retriever(search_kwargs):
        """
        Convert to LangChain retriever interface
        - Allows use in RAG chains
        - search_kwargs={'k': 4} means return top-4 documents
        """
```

**Data Structure:**
```python
vectors.npy:              # Numpy array, shape: (2431, 1024)
    2431 documents       # Each document has 1024-dimensional embedding
    Each row = 1 document embedding
    
documents.pkl:           # Pickled list of Document objects
    [Document1, Document2, ..., Document2431]
    Each Document has:
        - page_content: the actual text
        - metadata: {'source': 'file.pdf', 'page': 1}
```

### 2. OllamaEmbeddings & OllamaLLM Integration

**Embeddings Flow:**
```
Document Text
    ▼
OllamaEmbeddings.embed_documents()
    ▼ (calls http://localhost:11434/api/embed)
    ▼
mxbai-embed-large model
    ▼
1024-dimensional vector
    ▼
Stored in vectors.npy
```

**LLM Generation Flow:**
```
[Question + Retrieved Documents Context]
    ▼
Legal Template Prompt (with 6 responsibilities)
    ▼
OllamaLLM.invoke() (temperature=0.1)
    ▼ (calls http://127.0.0.1:11434/api/generate)
    ▼
llama3.2 model
    ▼
Generates answer with legal reasoning
```

### 3. Document Processing Pipeline

**Flow in document_processor.py:**

```
legal_documents/ (folder)
    ├── pdfs/
    │   └── Constitution of India.pdf
    ├── docx/
    │   └── ...
    └── txt/
        └── ...

        ▼

process_legal_documents()
    ├── Load all files (PDF, DOCX, TXT)
    ├── Extract text from each file
    │
    └── chunk_documents()
        ├── Split text into chunks (512 tokens each)
        ├── 50-token overlap between chunks
        ├── Add metadata (source filename, page #)
        │
        └── Return List[Document]

Result: 2431 document chunks ready for embedding
```

### 4. Python 3.14 Compatibility Patch

**Problem:** Pydantic V1 incompatible with Python 3.14

**Location:** main.py, lines 18-32

**Solution:**
```python
if sys.version_info >= (3, 14):
    patch_pydantic_v1()  # Custom validator patch
    # Allows Pydantic V1 to work with Python 3.14
    # Fallback for unknown field types
```

---

## Architecture & Workflows

### Data Flow Diagram

```
INPUT: User Question
    │
    ├─► [1] Question Processing
    │        • Normalize input
    │        • Validate query
    │
    ├─► [2] Vector Store Lookup
    │   ┌────────────────────────────────┐
    │   │ SimpleVectorStore.similarity   │
    │   │      _search(query, k=4)       │
    │   └────────────────────────────────┘
    │        • Generate question embedding
    │        • Compute cosine similarity
    │        • Return top-4 documents
    │
    ├─► [3] Context Preparation
    │        • Format retrieved documents
    │        • Add source information
    │        • Create prompt context
    │
    ├─► [4] LLM Generation
    │   ┌────────────────────────────────┐
    │   │    OllamaLLM.invoke()          │
    │   │  (llama3.2, temp=0.1)          │
    │   └────────────────────────────────┘
    │        • Call Ollama API
    │        • Generate answer
    │
    └─► [5] Response Formatting
             • Add source citations
             • Format for display
             
OUTPUT: Answer + Sources
```

### Query to Answer Workflow

```
User: "When was the Constitution written?"
    │
    ▼ (main.py: answer_legal_question())
    │
    ├─ Retrieval Phase (vector.py):
    │  │
    │  ├─ Embed query: "When was..." → [0.12, 0.45, ..., -0.33] (1024D)
    │  │
    │  ├─ Search vector store:
    │  │  ├─ Similarity with Doc1: 0.78
    │  │  ├─ Similarity with Doc2: 0.85 ✓ Top-1
    │  │  ├─ Similarity with Doc3: 0.82 ✓ Top-2
    │  │  └─ Similarity with Doc4: 0.75
    │  │
    │  └─ Return 4 documents with scores
    │
    ├─ Augmentation Phase (main.py: format_documents()):
    │  │
    │  └─ Create context:
    │     "[Document 1] Constitution of India.pdf (Page 32):
    │      The Constitution of India was adopted on..."
    │
    ├─ Generation Phase (OllamaLLM):
    │  │
    │  └─ Prompt:
    │     "You are a legal assistant...
    │      Context: [4 documents above]
    │      Question: When was Constitution written?
    │      Answer:"
    │
    └─ Result:
       "Based on the provided legal documents, specifically
        Document 1 'Constitution of India.pdf' (Page 32),
        the Constitution of India was adopted by the Constituent
        Assembly on November 26, 1949."
```

---

## Technology Stack

### Core Libraries (7 Packages)

| Package | Version | Purpose |
|---------|---------|---------|
| **langchain** | ≥0.0.350 | RAG framework, chains, prompts |
| **langchain-community** | ≥0.0.50 | Document loaders, utilities |
| **langchain-ollama** | latest | OllamaLLM & OllamaEmbeddings integration |
| **pypdf** | ≥4.0.0 | PDF document loading |
| **python-docx** | ≥1.0.0 | DOCX document loading |
| **python-dotenv** | ≥1.0.0 | Environment variable management |
| **numpy** | ≥1.21.0 | Vector math, cosine similarity |

### External Services

- **Ollama**: Local LLM inference engine
  - Models: `llama3.2` (LLM), `mxbai-embed-large` (embeddings)
  - URL: http://localhost:11434
  - No GPU required (CPU sufficient)

### Python Version
- **3.14** (with Pydantic V1 compatibility patch)

---

## Core Components

### 1. main.py - User Interface & RAG Pipeline

**Responsibility**: CLI chat interface and answer generation

**Key Functions:**

#### a) `answer_legal_question(question: str) → str`

```python
def answer_legal_question(question):
    """
    Core RAG pipeline function
    
    Steps:
    1. Retrieve relevant documents via vector search
    2. Format documents as context
    3. Generate answer using LLM
    4. Add source citations
    """
    
    # Step 1: Retrieve
    retrieved_docs = retriever.invoke(question)  # Get top-4 documents
    
    # Step 2: Augment
    context = format_documents(retrieved_docs)   # Format for prompt
    
    # Step 3: Generate
    result = chain.invoke({
        "context": context,
        "question": question
    })  # Call LLM with context + question
    
    # Step 4: Format response
    return result + source_citations
```

#### b) `format_documents(docs) → str`

```python
def format_documents(docs):
    """
    Convert retrieved documents to readable context
    
    Input: [Document1, Document2, Document3, Document4]
    
    Output:
    "[Document 1] Constitution.pdf (Page 32):
     The Constitution defines the fundamental rules..."
    
    [Document 2] Legal_Guide.pdf (Page 15):
     Section 2 states that..."
    """
```

**Legal Template Prompt:**
```python
legal_template = """You are an expert legal assistant...

Key Responsibilities:
1. Answer based ONLY on provided documents
2. Always cite document name & page number
3. State if information unavailable
4. Use professional legal language
5. Explain complex terms
6. Provide information only (not legal advice)

Context: {context}
Question: {question}
"""
```

### 2. vector.py - Vector Store & Retrieval Engine

**Responsibility**: Document embeddings, similarity search, retrieval

**Architecture:**

```python
class SimpleVectorStore:
    """
    Custom vector database (replacement for Chromadb)
    
    Why Custom?
    - Chromadb incompatible with Python 3.14
    - Simple, lightweight, zero complexity
    - Full control over persistence
    - All code is visible and understandable
    """
    
    def __init__(self, embeddings, k=4):
        self.embeddings = embeddings      # OllamaEmbeddings instance
        self.documents = []               # List of Document objects
        self.vectors = None               # Numpy array of embeddings
        self.k = k                        # Default retrieval count
    
    def add_documents(documents, ids):
        """
        Process:
        1. Get embeddings for each document (1024-dim vectors)
        2. Store vectors in numpy array (shape: n_docs × 1024)
        3. Keep mapping: doc_id → document_object
        4. Persist to disk
        """
    
    def similarity_search(query, k=4):
        """
        Algorithm (Cosine Similarity):
        
        1. Get embedding for query: query_vec (1024D)
        2. For each document vector:
            similarity = cosine(query_vec, doc_vec)
                       = dot(query_vec, doc_vec) / (||query_vec|| × ||doc_vec||)
        3. Sort by similarity (descending)
        4. Return top-k documents with highest similarity
        
        Example:
        Query: "Constitution written?"
        Similarities: [0.92, 0.85, 0.78, 0.71, 0.68, ...]
        Return: documents at indices [0, 1, 2, 3] (top-4)
        """
    
    def _save():
        """
        Persist vector store to disk
        - vectors.npy: Numpy array of embeddings
        - documents.pkl: Pickled metadata + content
        """
    
    def _load():
        """
        Load vector store from disk
        - Restores vectors and documents
        - Ready for similarity search
        """
```

**Embedding Process:**

```python
def create_embeddings():
    """
    Initialize OllamaEmbeddings
    
    Config:
    - Model: mxbai-embed-large (1024 dimensions)
    - Server: http://localhost:11434 (local Ollama)
    - Why this model?
      * 1024-dimensional (good balance between size & quality)
      * Optimized for document search
      * Fast inference
    """
    return OllamaEmbeddings(
        model="mxbai-embed-large",
        base_url="http://localhost:11434"
    )
```

**Retriever Creation:**

```python
def build_vectorstore():
    """
    Main orchestrator: Build complete vector store
    
    Workflow:
    1. Check if vectorstore exists (vectors.npy)
    2. If exists: Load existing vectors + documents
    3. If not exists:
       a. Load documents from legal_documents/ folder
       b. Chunk documents (512 tokens, 50 overlap)
       c. Generate embeddings for each chunk
       d. Create SimpleVectorStore
       e. Save to disk
    4. Create retriever interface
    5. Return (vector_store, retriever)
    """
```

### 3. document_processor.py - Document Loading & Chunking

**Responsibility**: Load legal documents, extract text, chunk for embeddings

**Key Function: `process_legal_documents()`**

```python
def process_legal_documents():
    """
    Main entry point for document loading
    
    Process:
    1. Scan legal_documents/ folder
    2. Load all files:
       - PDFs: pypdf library
       - DOCX: python-docx library
       - TXT: plain text read
    3. Extract text from each file
    4. Chunk documents:
       - Chunk size: 512 tokens (approx 2000 chars)
       - Overlap: 50 tokens (for context continuity)
       - Add metadata: source filename, page number
    5. Return List of Document objects
    
    Current Data:
    - Constitution of India (330+ pages)
    - Test document (smaller)
    - Total: 2431 chunks
    """
    
    # Example: Processing PDF
    documents = []
    for pdf_file in pdf_folder:
        text = extract_text_from_pdf(pdf_file)  # pypdf
        chunks = split_into_chunks(text)        # 512 tokens each
        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata={
                    'source': pdf_file.name,
                    'page': page_number
                }
            )
            documents.append(doc)
```

### 4. config.py - Configuration Management

**Responsibility**: Centralized system settings

```python
class LegalReachConfig:
    """
    Master configuration
    
    Key Parameters:
    - Embedding model: mxbai-embed-large
    - LLM: llama3.2
    - Chunk size: 512 tokens
    - Chunk overlap: 50 tokens
    - Retrieval k: 4 documents
    - LLM temperature: 0.1 (deterministic)
    - Device: CPU (Ollama handles GPU)
    """
```

---

## Code Explanations

### Most Important Code Sections

#### 1. Cosine Similarity Search (vector.py)

**This is the HEART of retrieval:**

```python
def similarity_search(self, query: str, k: int = 4):
    """
    Find most similar documents to query
    """
    # Get embedding for query (1024 dimensions)
    query_embedding = self.embeddings.embed_query(query)
    
    # Compute cosine similarity with all documents
    # Formula: cos(θ) = (A · B) / (||A|| × ||B||)
    
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    # Calculate similarity between query and all vectors
    query_embedding = np.array(query_embedding).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, self.vectors)[0]
    
    # Get indices of top-k highest similarities
    top_indices = np.argsort(similarities)[-k:][::-1]
    
    # Return documents in order of similarity (best first)
    return [self.documents[i] for i in top_indices]
```

**Why Cosine Similarity?**
- Measures angle between vectors (0 to 1, where 1 = identical)
- Ignores magnitude, focuses on direction
- Perfect for comparing semantic embeddings
- Fast computation with numpy

#### 2. LangChain Chain Construction (main.py)

```python
# Create prompt template
prompt = ChatPromptTemplate.from_template(legal_template)

# Create LLM instance
model = OllamaLLM(model="llama3.2", temperature=0.1)

# Chain them: Prompt → Format → LLM → Output
chain = prompt | model

# Usage:
result = chain.invoke({
    "context": "Constitution defines...",
    "question": "What is the Constitution?"
})
# The | (pipe) operator creates a sequence:
# 1. prompt.format(context=..., question=...)
# 2. model.invoke(formatted_prompt)
# 3. Return result
```

**Why temperature=0.1?**
- Range: 0 (deterministic) to 1 (random)
- Legal answers need consistency
- 0.1 ensures predictable, factual responses
- Not too robotic (0.1 vs 0.0 allows minor variations)

#### 3. Document Chunking Strategy (document_processor.py)

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,           # Split into 512-token chunks
    chunk_overlap=50,         # 50-token overlap between chunks
    separators=["\n\n", "\n", ".", " ", ""]  # Split hierarchy
)

# Example:
text = """
Constitution of India - Preamble
We, the people of India...
[long document]
"""

chunks = splitter.split_text(text)

# Result:
# Chunk 1: "Constitution of India - Preamble\nWe, the people..."
# Chunk 2: "...people of India, having solemnly resolved..." (overlaps with Chunk 1)
# Chunk 3: "...resolved to constitute India..." (overlaps with Chunk 2)
```

**Why Chunking?**
- Embeddings work on reasonable-sized text (not whole books)
- Chunks capture specific concepts
- Overlap ensures context continuity
- 512 tokens ≈ 2000 characters (good balance)

#### 4. Python 3.14 Compatibility Patch (main.py)

```python
if sys.version_info >= (3, 14):
    def patch_pydantic_v1():
        """
        Problem: Pydantic V1 expects to infer field types
        Python 3.14: Type inference broken for some classes
        
        Solution: Provide fallback validator
        """
        import pydantic.v1.validators as validators
        original_find = validators.find_validators
        
        def patched_find_validators(type_, config):
            try:
                yield from original_find(type_, config)
            except RuntimeError as e:
                if "no validator found" in str(e):
                    # Fallback: accept any value as-is
                    yield lambda v: v
                else:
                    raise
        
        validators.find_validators = patched_find_validators
    
    patch_pydantic_v1()
```

---

## Data Flow Diagram

### Complete System Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INPUT                                    │
│              "When was Constitution written?"                        │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
         ┌───────────────────────────────────────┐
         │ main.py: answer_legal_question()      │
         │ - Log query                           │
         │ - Call retriever.invoke(question)     │
         └───────────────────┬───────────────────┘
                             │
                             ▼
         ┌───────────────────────────────────────────────────┐
         │ vector.py: SimpleVectorStore.similarity_search()  │
         │                                                   │
         │ 1. Create query embedding                        │
         │    "When was Constitution written?"              │
         │    ↓                                              │
         │    OllamaEmbeddings.embed_query()                │
         │    ↓                                              │
         │    [0.12, 0.45, -0.33, ..., 0.67] (1024-dim)    │
         │                                                   │
         │ 2. Load pre-computed document embeddings         │
         │    (from vectors.npy)                            │
         │    Doc1: [0.15, 0.48, -0.31, ...]               │
         │    Doc2: [0.11, 0.44, -0.34, ...]               │
         │    ...                                            │
         │    Doc2431: [0.05, 0.22, -0.50, ...]            │
         │                                                   │
         │ 3. Compute cosine similarity                     │
         │    Similarity(query, Doc1) = 0.92                │
         │    Similarity(query, Doc2) = 0.78                │
         │    Similarity(query, Doc3) = 0.85                │
         │    ...                                            │
         │                                                   │
         │ 4. Return top-4 documents by similarity          │
         │    [Doc3(0.92), Doc1(0.85), Doc2(0.82), Doc5(...)] │
         └───────────────────┬───────────────────────────────┘
                             │
                             ▼
         ┌───────────────────────────────────────┐
         │ main.py: format_documents()           │
         │                                       │
         │ Convert to readable context:          │
         │ "[Document 1] Constitution.pdf        │
         │  (Page 32):                           │
         │  The Constitution of India...        │
         │                                       │
         │ [Document 2] Constitution.pdf        │
         │  (Page 105):                         │
         │  Part III of the Constitution..."    │
         └───────────────────┬───────────────────┘
                             │
                             ▼
         ┌───────────────────────────────────────────────┐
         │ main.py: chain.invoke()                       │
         │                                               │
         │ Create final prompt:                         │
         │ "You are expert legal assistant...            │
         │  Context: [4 documents above]                 │
         │  Question: When was Constitution written?     │
         │  Answer:"                                     │
         │                                               │
         │ Call OllamaLLM (llama3.2)                    │
         │ ↓                                              │
         │ HTTP POST http://127.0.0.1:11434/api/generate│
         │ ↓                                              │
         │ llama3.2 generates response                   │
         │ (temperature=0.1 for accuracy)                │
         └───────────────────┬───────────────────────────┘
                             │
                             ▼
         ┌───────────────────────────────────────┐
         │ main.py: Format Response              │
         │                                       │
         │ Add source citations:                 │
         │ "Based on provided legal documents,   │
         │  the Constitution of India was        │
         │  adopted on November 26, 1949.        │
         │                                       │
         │  --- SOURCES ---                      │
         │  • Constitution of India (Page 32)    │
         │  • Constitution of India (Page 105)   │
         │  • Constitution of India (Page 45)"   │
         └───────────────────┬───────────────────┘
                             │
                             ▼
         ┌───────────────────────────────────────┐
         │         DISPLAY TO USER                │
         │   (Answer with sources)                │
         └───────────────────────────────────────┘
```

---

## Setup & Execution

### Prerequisites

1. **Python 3.14** installed
2. **Ollama** running on localhost:11434 with models:
   - `llama3.2` (LLM)
   - `mxbai-embed-large` (embeddings)
3. **Legal documents** in `legal_documents/` folder

### Installation

```bash
# 1. Navigate to project directory
cd d:\ragbot\venv\LocalAIAgentWithRAG-main

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Create .env file (if needed)
# OLLAMA_BASE_URL=http://localhost:11434
```

### Running the Chatbot

```bash
# Start the interactive chat interface
python main.py

# You'll see:
# [LLM] Initializing Ollama LLM...
# [OK] LLM initialized
# 
# LEGAL REACH RAG - AI LEGAL ASSISTANT
# Ask your legal questions. Type 'q' to quit.
#
# [QUERY] Ask your legal question: When was Constitution written?
```

### How to Add New Documents

```
Place files in legal_documents/
├── pdfs/
│   ├── Constitution.pdf
│   └── Legal_Guide.pdf
├── docx/
│   └── Document.docx
└── txt/
    └── Notes.txt

When you run chatbot next time:
1. Documents are automatically detected
2. Chunked and embedded
3. Added to vector store
4. Available for queries
```

---

## Important Implementation Details

### 1. Embedding Persistence

**Problem**: Generating embeddings takes time (7+ seconds for 2431 documents)

**Solution**: Cache embeddings to disk

```
First run:
1. Load documents
2. Generate embeddings (calls Ollama API 2431 times)
3. Save to vectors.npy + documents.pkl
Time: ~2-3 minutes

Subsequent runs:
1. Load from vectors.npy + documents.pkl
2. Ready to use
Time: <1 second
```

### 2. Metadata Preservation

Each document chunk carries metadata:

```python
Document(
    page_content="Constitution defines...",
    metadata={
        'source': 'Constitution of India.pdf',
        'page': 32
    }
)
```

This enables:
- Source citations in answers
- User can find original document
- Audit trail for answers

### 3. Error Handling

```python
# Vector Store Initialization
try:
    vector_store, retriever = build_vectorstore()
    logger.info("[OK] Vector store initialized successfully!")
except Exception as e:
    logger.error(f"[ERROR] Failed to initialize: {e}")
    logger.error("Make sure Ollama server running on http://localhost:11434")
    raise
```

### 4. Logging Strategy

**Before (with emoji - Windows encoding errors):**
```python
logger.info("✅ Vector store loaded")     # UnicodeEncodeError!
```

**After (ASCII-safe):**
```python
logger.info("[OK] Vector store loaded")   # Works everywhere
```

---

## Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| Load existing embeddings | <1s | From disk (vectors.npy) |
| Generate query embedding | 0.5-1s | Single call to Ollama |
| Cosine similarity search | <0.01s | NumPy vectorized operation |
| LLM generation | 5-10s | Depends on answer length |
| **Total end-to-end** | **6-12s** | Including all steps |

### Memory Usage

| Component | Size |
|-----------|------|
| Vectors (2431 × 1024 float32) | ~10 MB |
| Documents metadata | ~5 MB |
| LLM model (llama3.2 quantized) | ~2 GB (Ollama manages) |
| Python runtime | ~200 MB |
| **Total** | **~2.2 GB** |

---

## Troubleshooting

### Issue: "Connection refused" on Ollama

**Cause**: Ollama server not running

**Solution**:
```bash
# Start Ollama
ollama serve

# In another terminal, pull models if needed
ollama pull llama3.2
ollama pull mxbai-embed-large
```

### Issue: "No documents found"

**Cause**: legal_documents/ folder empty or no valid files

**Solution**:
```bash
# Create folder structure
mkdir legal_documents
mkdir legal_documents\pdfs
mkdir legal_documents\docx
mkdir legal_documents\txt

# Add documents (PDFs, DOCX, TXT files)
# Documents are auto-detected next run
```

### Issue: Slow first run

**Cause**: Generating 2431 embeddings takes time

**Solution**:
- First run: 2-3 minutes (generating embeddings)
- Subsequent runs: <1 second (loaded from disk)
- This is normal and expected

---

## Key Takeaways for Junior Developers

1. **RAG = Retrieval + Augmentation + Generation**
   - Don't ask LLM to know everything
   - Retrieve relevant context first
   - Use that context to generate accurate answers

2. **Vector Search Magic**
   - Embeddings convert text → numbers
   - Similar text → similar vectors
   - Cosine similarity finds semantic matches

3. **Why Custom Vector Store?**
   - Sometimes off-the-shelf tools have issues
   - Understanding custom implementation helps debugging
   - SimpleVectorStore shows the core concept clearly

4. **Local LLM Advantage**
   - Ollama provides private, fast inference
   - No cloud dependency
   - Full control over model and parameters

5. **Metadata Matters**
   - Always track source of answers
   - Users need to verify information
   - Citations build trust and accountability

---

## References

- **LangChain Documentation**: https://python.langchain.com
- **Ollama**: https://ollama.ai
- **RAG Pattern**: https://en.wikipedia.org/wiki/Semantic_search
- **Cosine Similarity**: https://en.wikipedia.org/wiki/Cosine_similarity

---

**Document Version**: 1.0  
**Last Updated**: February 12, 2026  
**Author**: Senior Development Team  
**Target Audience**: Junior Developers & Project Maintainers

