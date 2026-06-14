# ChaiDocs RAG Agent 🤖

A production-ready **Retrieval-Augmented Generation (RAG)** system that answers questions about [ChaiCode](https://chaicode.com) documentation in natural language.

**Live Demo:** https://chaidocsragagent-fjfmcht7kylcm3ti3qpvln.streamlit.app/
---

## What Problem Does This Solve?

ChaiCode's documentation spans 44 pages across HTML, Git, C, Django, SQL, and DevOps courses. Finding specific information requires manual searching through multiple pages.

This system lets users ask questions in plain English and get direct, cited answers — powered by Google Gemini and grounded in ChaiCode's actual documentation.

---

## Architecture

```
Ingestion Pipeline (runs once, offline)
────────────────────────────────────────
ChaiCode docs (44 pages)
        ↓
Web scraping with BeautifulSoup (local machine)
        ↓
Saved to docs_data.json (committed to GitHub)
        ↓
RecursiveCharacterTextSplitter (chunk=1000, overlap=200)
        ↓
HuggingFace all-MiniLM-L6-v2 embeddings (local, no API quota)
        ↓
ChromaDB vectorstore (pre-built, committed to GitHub)

Serving Pipeline (runs on every user query)
────────────────────────────────────────────
User question
        ↓
HuggingFace embeds query locally (zero API cost)
        ↓
ChromaDB cosine similarity search → top 6 chunks
        ↓
format_docs converts Document objects → labelled string
        ↓
RunnablePassthrough sends original question unchanged
        ↓
Both fed into ChatPromptTemplate simultaneously
        ↓
Google Gemini 2.0 Flash generates grounded answer
        ↓
StrOutputParser returns clean text to user
```

**Key architectural decision:** Ingestion and serving pipelines are completely separated. The serving pipeline has zero dependency on external websites — making it fast, reliable, and immune to ChaiCode site downtime.

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| RAG Pipeline | LangChain | Abstracts retriever → prompt → LLM chaining |
| Embeddings | HuggingFace all-MiniLM-L6-v2 | Runs locally, no API quota, no cost |
| Vector DB | ChromaDB | Lightweight, perfect for prototype scale |
| LLM | Google Gemini 2.0 Flash | Free tier, fast, grounded responses |
| Frontend | Streamlit | Rapid AI app UI with minimal code |
| Deployment | Streamlit Cloud | Generous memory for ML apps, free tier |

---

## Why HuggingFace Instead of Gemini Embeddings?

Gemini's embedding API has a 100 requests/minute free tier limit. Embedding 249 document chunks simultaneously caused `RESOURCE_EXHAUSTED` errors on every deployment.

`all-MiniLM-L6-v2` runs directly on the server — zero API calls, zero quota issues, production-quality semantic search. The pre-built ChromaDB vectorstore is committed to GitHub so Streamlit Cloud loads it instantly on startup without any embedding at runtime.

---

## Known Limitations

- **Stateless:** No conversation memory — each question is independent
- **Static data:** Documentation scraped once; won't reflect ChaiCode updates automatically
- **Single-topic retrieval:** Multi-part comparative questions (e.g. "compare Git vs Django") may not retrieve balanced context
- **Text only:** Tables and images in documentation are not processed

**Planned improvements:**
- Database-backed conversation memory (Supabase PostgreSQL)
- Periodic re-scraping pipeline for fresh data
- Multi-query retrieval for comparative questions
- Authentication to prevent API quota abuse

---

## Project Structure

```
chaidocsRAGAgent/
├── streamlit_app.py      # Streamlit UI and session management
├── chaidocs_rag.py       # Core RAG pipeline (ingestion + serving)
├── debug.py              # Debugging script for API and DB checks
├── docs_data.json        # Pre-scraped ChaiCode documentation
├── chroma_db/            # Pre-built vectorstore (committed to GitHub)
├── requirements.txt      # Pinned dependencies
├── runtime.txt           # Python 3.11.9 specification
└── README.md
```

---

## Local Setup

```bash
# Clone repository
git clone https://github.com/deepanjali6111/chaidocsRAGAgent.git
cd chaidocsRAGAgent

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
# Create .env file with:
GEMINI_API_KEY=your_key_here

# Run locally
streamlit run streamlit_app.py
```

---

## Debugging

```bash
python debug.py
```

Checks: API key validity, URL accessibility, ChromaDB status, Gemini connection.

---

## What I Learned Building This

- **Ingestion vs serving pipeline separation** — pre-computing embeddings offline eliminates runtime API dependencies
- **Quota management** — switching from API-based to local embeddings solved RESOURCE_EXHAUSTED errors permanently  
- **Dependency pinning** — `protobuf==3.20.3` was critical; newer versions broke the Streamlit + Gemini dependency chain
- **Deployment constraints** — Render's 512MB free tier couldn't load sentence-transformers + Streamlit + ChromaDB simultaneously; Streamlit Cloud handles ML workloads better
- **RAG limitations** — single vector search struggles with multi-topic comparative queries; multi-query retrieval would solve this

---

## Built With

- [LangChain](https://langchain.com)
- [Google Gemini](https://ai.google.dev)
- [HuggingFace](https://huggingface.co)
- [ChromaDB](https://trychroma.com)
- [Streamlit](https://streamlit.io)
- [ChaiCode](https://chaicode.com) — documentation source
