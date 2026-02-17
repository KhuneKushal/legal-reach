# Legal Reach RAG - Conversion Plan for Your Codebase

## 🎯 PROJECT STATUS
**Current Setup:**
- ✅ Virtual Environment: Active at `D:\ragbot\venv`
- ✅ Base Code: LocalAIAgentWithRAG (Ollama + Chroma + CSV)
- ✅ Hardware: RTX 3050 (4GB VRAM) + 16GB RAM
- ❌ Legal Documents: Not yet configured
- ❌ PDF/DOCX Processing: Not implemented

---

## 📊 ANALYSIS OF YOUR CURRENT CODE

### What You Have:
```
main.py          → Uses Ollama LLM (llama3.2), ChatPromptTemplate
vector.py        → Uses Chroma DB + OllamaEmbeddings + CSV data
requirements.txt → Minimal deps (langchain, ollama, chroma, pandas)
```

### Key Architecture:
1. **Data Source**: CSV file (realistic_restaurant_reviews.csv)
2. **Vector DB**: Chroma (chrome_langchain_db)
3. **Embeddings**: OllamaEmbeddings (mxbai-embed-large)
4. **LLM**: OllamaLLM (llama3.2)
5. **Retrieval**: Top-5 similar reviews

### What Needs to Change:
| Component | Current | New |
|-----------|---------|-----|
| Data Input | CSV file | PDF + DOCX + TXT files |
| Processing | Pandas read_csv | pypdf + python-docx + text extraction |
| Chunking | None (whole reviews) | RecursiveCharacterTextSplitter (512 tokens) |
| Embeddings | OllamaEmbeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | Chroma DB | Chroma DB (same, optimized) |
| Prompt | Restaurant reviews | Legal document Q&A |
| Retrieval | k=5 | k=4 (legal context) |

---

## 🛠️ CONVERSION STEPS

### STEP 1: Create Project Structure (5 mins)
```
D:\ragbot\venv\LocalAIAgentWithRAG-main\
├── main.py (MODIFY)
├── vector.py (MODIFY)
├── requirements.txt (UPDATE)
├── config.py (CREATE NEW)
├── document_processor.py (CREATE NEW)
├── utils.py (CREATE NEW)
├── .env (CREATE NEW)
├── test_setup.py (CREATE NEW)
├── legal_documents/
│   ├── pdfs/                    ← DROP YOUR LEGAL PDFs HERE
│   ├── docx/                    ← DROP YOUR DOCX FILES HERE
│   └── txt/                     ← DROP TXT FILES HERE
├── vectorstore/                 ← Auto-created (vector embeddings)
└── logs/                        ← Auto-created (processing logs)
```

**Your Action:**
```powershell
cd D:\ragbot\venv\LocalAIAgentWithRAG-main
mkdir legal_documents\pdfs, legal_documents\docx, legal_documents\txt, vectorstore, logs
```

---

### STEP 2: Update requirements.txt (2 mins)

**Current:**
```
langchain
langchain-ollama
langchain-chroma
pandas
```

**New (adds PDF/DOCX support + GPU optimization):**
```
# Core LangChain
langchain==0.1.0
langchain-community==0.0.20
langchain-ollama

# Vector Store
langchain-chroma==0.4.22

# Document Processing
pypdf==4.0.1
python-docx==1.1.0

# Embeddings (LOCAL - no Ollama needed for embeddings)
sentence-transformers==2.3.1

# GPU Optimization for RTX 3050
torch==2.1.2
transformers==4.37.2
accelerate==0.26.1
bitsandbytes==0.42.0

# Utilities
pandas
python-dotenv==1.0.0
pynvml==11.5.0  # GPU monitoring
psutil==5.9.8   # System monitoring
```

**Why These Changes:**
- `pypdf` + `python-docx`: Load legal documents
- `sentence-transformers`: Fast embeddings (not Ollama - saves GPU)
- `bitsandbytes`: 8-bit quantization (save 50% memory)
- `pynvml`: Monitor GPU health

---

### STEP 3: Create config.py (for RTX 3050 optimization)

This file centralizes all settings so you can easily tune for your hardware.

---

### STEP 4: Create document_processor.py (core new file)

Handles: PDF → DOCX → TXT loading, chunking, embedding generation

---

### STEP 5: Update vector.py (modify existing)

Replace CSV loading with document loading, keep Chroma DB

---

### STEP 6: Update main.py (modify existing)

Replace restaurant prompt with legal template, add document upload UI

---

### STEP 7: Create utils.py (monitoring tools)

GPU monitoring, document caching, preprocessing

---

### STEP 8: Create .env (settings file)

API keys, model paths, GPU limits for RTX 3050

---

## 🔄 WORKFLOW FOR YOUR CONVERSION

### Phase 1: Update Configuration (TODAY)
1. ✅ Create folder structure
2. ✅ Update requirements.txt
3. ✅ Install new packages
4. Create config.py
5. Create .env

### Phase 2: Add Document Processing (TODAY)
6. Create document_processor.py
7. Create utils.py
8. Update vector.py
9. Create test_setup.py
10. Test PDF/DOCX loading

### Phase 3: Update Main Application (TODAY)
11. Update main.py
12. Test full pipeline
13. Add GPU monitoring

### Phase 4: Test & Deploy (TODAY)
14. Add sample legal documents
15. Run end-to-end test
16. Verify GPU usage stays <85%

---

## ⚠️ IMPORTANT MODEL SELECTION FOR RTX 3050

Your RTX 3050 has **4GB VRAM**. Here's what fits:

### ✅ RECOMMENDED SETUP:
| Component | Model | Size | GPU Memory |
|-----------|-------|------|------------|
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 | 80MB | 0.2GB |
| **LLM** | OllamaLLM (llama3.2) | Runs on CPU/GPU | ~1-2GB if on GPU |
| **Vector DB** | Chroma | Varies | 0.5-1GB |
| **Total** | - | - | ~2-3GB safe |

### Why This Works:
- **Embeddings on GPU**: Fast (0.2GB) ← all-MiniLM-L6-v2
- **LLM on CPU or partial GPU**: llama3.2 is lightweight
- **Ollama Server**: Manages everything, prevents OOM
- **Leaves 1GB buffer**: For system stability

### ⚠️ IF YOU GET GPU OUT OF MEMORY:
```python
# In config.py, change to:
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Stays same (already minimal)
LLM_DEVICE = "cpu"  # Force LLM to CPU only, embeddings on GPU
CHUNK_SIZE = 256    # Reduce from 512
BATCH_SIZE = 2      # Reduce from 4
```

---

## 📋 YOUR DOCUMENTS FOLDER STRUCTURE

Create this in `legal_documents/`:

```
legal_documents/
├── pdfs/
│   ├── contract_sample.pdf
│   ├── terms_and_conditions.pdf
│   └── README.txt  ← "DROP PDFs HERE"
├── docx/
│   ├── legal_brief.docx
│   └── README.txt  ← "DROP DOCX FILES HERE"
├── txt/
│   ├── legal_text.txt
│   └── README.txt  ← "DROP TEXT FILES HERE"
└── processed/      ← Auto-created by system
    └── (internal use)
```

**To Add Documents:**
1. Copy your PDFs to `legal_documents/pdfs/`
2. Copy your DOCX to `legal_documents/docx/`
3. Copy TXT files to `legal_documents/txt/`
4. Run app: `streamlit run main.py` (if using Streamlit)
   OR: `python main.py` (if using CLI)
5. System auto-processes all documents

---

## 🔧 HOW YOUR SYSTEM WILL WORK

### Current Flow (Restaurant Reviews):
```
CSV File (reviews.csv)
    ↓
vector.py: Load CSV with pandas
    ↓
Create embeddings (Ollama) 
    ↓
Store in Chroma DB
    ↓
main.py: User asks question
    ↓
Retrieve similar reviews (k=5)
    ↓
Ollama LLM generates answer
```

### New Flow (Legal Documents):
```
Legal Documents (PDFs/DOCX/TXT)
    ↓
document_processor.py: Extract text
    ↓
Split into chunks (512 tokens, 50 overlap)
    ↓
Create embeddings (sentence-transformers)
    ↓
Store in Chroma DB (same location, new collection)
    ↓
main.py: User asks question
    ↓
Retrieve relevant chunks (k=4)
    ↓
Ollama LLM generates legal answer
    ↓
Show sources + citations
```

**KEY DIFFERENCE:** Instead of whole CSV rows, you work with document chunks + metadata.

---

## 📝 WHICH FILES TO KEEP / MODIFY / DELETE

### KEEP (Don't delete):
- ✅ `main.py` - Modify for legal prompt
- ✅ `vector.py` - Modify for document loading
- ✅ `requirements.txt` - Update versions
- ✅ `venv/` - Your virtual environment

### DELETE (Optional - but recommended):
- ❌ `realistic_restaurant_reviews.csv` - No longer needed
- ❌ `chrome_langchain_db/` - Old Chroma DB (new one created)

### CREATE (New files):
- ✅ `config.py` - Settings for RTX 3050
- ✅ `document_processor.py` - PDF/DOCX processing
- ✅ `utils.py` - Monitoring tools
- ✅ `.env` - Environment variables
- ✅ `test_setup.py` - Verify setup
- ✅ `.gitignore` - Hide large files

### FOLDERS (Create):
- ✅ `legal_documents/{pdfs, docx, txt, processed}`
- ✅ `vectorstore/` - New Chroma DB
- ✅ `logs/` - Processing logs

---

## 🚀 QUICK SETUP COMMANDS

```powershell
# 1. Navigate to your project
cd D:\ragbot\venv\LocalAIAgentWithRAG-main

# 2. Create folders
mkdir legal_documents\pdfs, legal_documents\docx, legal_documents\txt, vectorstore, logs

# 3. Backup old database (optional)
if (Test-Path chrome_langchain_db) { 
    Rename-Item chrome_langchain_db chrome_langchain_db_backup 
}

# 4. Create .gitignore
@"
__pycache__/
*.pyc
.env
vectorstore/
*.db
chrome_langchain_db*
legal_documents/pdfs/*
legal_documents/docx/*
legal_documents/txt/*
logs/
"@ | Out-File .gitignore

# 5. Update Python packages (in virtual env)
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## ✅ NEXT: CREATE THE NEW FILES

Ready to proceed? I will create these files in order:

1. **config.py** - All settings for RTX 3050
2. **document_processor.py** - Load PDFs/DOCX, chunk text
3. **utils.py** - GPU monitoring, helpers
4. **vector.py** - Updated to load documents
5. **main.py** - Updated for legal RAG
6. **.env** - Environment settings
7. **test_setup.py** - Verify everything works

Each file will have:
- ✅ Detailed comments
- ✅ Type hints
- ✅ Error handling
- ✅ GPU optimization for RTX 3050
- ✅ Logging

---

## 📊 PROCESSING SPECS

### Document Processing:
- **Chunk Size**: 512 tokens (optimal for legal docs)
- **Chunk Overlap**: 50 tokens (maintains context)
- **Batch Size**: 4 embeddings at a time
- **GPU Memory**: ~3GB used, 1GB buffer

### Query Processing:
- **Retrieval**: Top 4 chunks (k=4)
- **LLM**: Ollama llama3.2 (on CPU or GPU as available)
- **Response Time**: 2-5 seconds per question
- **GPU Temp**: Keep <75°C (monitor via utils.py)

---

## 🎯 YOUR GOALS

After conversion, you'll have:
✅ Legal document RAG (PDF + DOCX + TXT support)
✅ GPU-optimized for RTX 3050 (no crashes/overheating)
✅ Same Chroma vector DB (familiar tech)
✅ Same Ollama LLM (familiar setup)
✅ Document source tracking + citations
✅ Auto-processing for new documents
✅ GPU health monitoring

---

## 🚨 CRITICAL: BEFORE WE START

**Check Ollama is Running:**
```powershell
# Test Ollama service
curl http://localhost:11434/api/tags

# If error, start Ollama:
ollama serve
```

This should show available models. Your existing `llama3.2` should be there.

---

## ⏱️ ESTIMATED TIME

- Setup folders + requirements: **5 mins**
- Create config.py + .env: **10 mins**
- Create document_processor.py: **15 mins**
- Create utils.py: **10 mins**
- Update vector.py: **10 mins**
- Update main.py: **15 mins**
- Test everything: **20 mins**
- **TOTAL: ~1.5 hours**

---

## ✅ Ready to Proceed?

Say "YES" and I will:
1. Create all 7 new files with GPU optimization
2. Adapt them to your existing code
3. Update your current files
4. Create test script
5. Give you step-by-step instructions

Your existing Ollama setup will continue to work - we're just replacing the data source from CSV → Legal Documents.
