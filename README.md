# ChaiDocs RAG Agent 🤖

A production-ready **Retrieval-Augmented Generation (RAG)** system that answers questions about [ChaiCode](https://chaicode.com) documentation in natural language — with evaluated answer quality.

**🚀 Live Demo:** [chaidocsragagent-fjfmcht7kylcm3ti3qpvln.streamlit.app](https://chaidocsragagent-fjfmcht7kylcm3ti3qpvln.streamlit.app)  
**📂 GitHub:** [github.com/deepanjali6111/chaidocsRAGAgent](https://github.com/deepanjali6111/chaidocsRAGAgent)

---

## What Problem Does This Solve?

ChaiCode's documentation spans 44 pages across HTML, Git, C, Django, SQL, and DevOps courses. Finding specific information requires manually searching through multiple pages.

This system lets users ask questions in plain English and get direct, cited answers — powered by Google Gemini and grounded in ChaiCode's actual documentation.

---

## RAGAS Evaluation Results

Evaluated using the **RAGAS framework** across 5 test queries covering Git, Django, Nginx, and DevOps topics:

| Metric | Score | What It Means |
|--------|-------|---------------|
| **Faithfulness** | **0.939** | 93.9% of answer claims are grounded in retrieved documentation |
| **Answer Relevancy** | **0.904** | 90.4% of answers directly address the question asked |
| **Context Precision** | **0.769** | 76.9% of retrieved chunks were relevant to the query |

These scores validate that the system produces grounded, accurate responses with minimal hallucination.

---

## Architecture

```
Ingestion Pipeline (runs once, offline on local machine)
─────────────────────────────────────────────────────────
ChaiCode docs (44 pages)
        ↓
scrape_docs.py — BeautifulSoup scraping (local machine only)
        ↓
docs_data.json — pre-scraped content committed to GitHub
        ↓
RecursiveCharacterTextSplitter (chunk=1000, overlap=200)
→ 249 semantic chunks created
        ↓
HuggingFace all-MiniLM-L6-v2 embeddings (local, zero API cost)
        ↓
ChromaDB vectorstore (pre-built, committed to GitHub)

Serving Pipeline (runs on every user query)
────────────────────────────────────────────
User question
        ↓
HuggingFace embeds query locally (zero API cost, zero quota)
        ↓
ChromaDB cosine similarity search → top 6 chunks retrieved
        ↓
format_docs converts Document objects → labelled string with sources
        ↓
RunnablePassthrough passes original question unchanged
        ↓
Both fed into ChatPromptTemplate simultaneously
        ↓
Google Gemini 2.5 Flash generates grounded answer (1 API call)
        ↓
StrOutputParser returns clean text → displayed to user
```

**Key architectural decision:** Ingestion and serving pipelines are completely separated. The serving pipeline has zero dependency on external websites — fast, reliable, immune to ChaiCode site downtime.

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| RAG Pipeline | LangChain | Abstracts retriever → prompt → LLM chaining |
| Embeddings | HuggingFace all-MiniLM-L6-v2 | Runs locally — zero API quota, zero cost |
| Vector DB | ChromaDB | Lightweight, no setup, perfect for prototype scale |
| LLM | Google Gemini 2.5 Flash | Fast, accurate, grounded responses |
| Frontend | Streamlit | Rapid AI app UI, minimal code |
| Deployment | Streamlit Cloud | Generous memory for ML workloads, free tier |
| Evaluation | RAGAS | Industry-standard RAG evaluation framework |

---

## Why HuggingFace Instead of Gemini Embeddings?

Gemini's embedding API has a 100 requests/minute free tier limit. Embedding 249 document chunks simultaneously caused `RESOURCE_EXHAUSTED` errors on every deployment.

`all-MiniLM-L6-v2` runs directly on the Streamlit Cloud server — zero API calls, zero quota issues, production-quality semantic search. The pre-built ChromaDB vectorstore is committed to GitHub so deployment loads it instantly with zero embedding overhead.

---

## Embedding Cost Reduction

| Approach | API Calls per Deployment | Cost |
|----------|--------------------------|------|
| Gemini embedding API | 249 calls (quota exhausted) | Hits rate limit |
| HuggingFace local | 0 calls | Free forever |

**100% reduction in embedding API cost** by switching to local embeddings.

---

## Known Limitations & Planned Improvements

**Current limitations:**
- Stateless — no conversation memory between sessions
- Static data — documentation scraped once, won't auto-update
- Single-topic retrieval — comparative multi-topic questions get unbalanced context
- Text only — tables and images not processed

**Planned improvements:**
- Database-backed conversation memory (Supabase PostgreSQL)
- Periodic re-scraping pipeline triggered by cron job
- Multi-query retrieval (LangChain MultiQueryRetriever) for comparative questions
- Supabase Auth for user isolation and quota protection
- RAGAS continuous evaluation on expanded test set

---

## Project Structure

```
chaidocsRAGAgent/
├── streamlit_app.py      # Streamlit UI and session management
├── chaidocs_rag.py       # Core RAG pipeline (ingestion + serving)
├── scrape_docs.py        # Offline ingestion script (BeautifulSoup)
├── evaluate_rag.py       # RAGAS evaluation script
├── debug.py              # Debugging script for API and DB checks
├── docs_data.json        # Pre-scraped ChaiCode documentation (44 pages)
├── chroma_db/            # Pre-built vectorstore (committed to GitHub)
├── requirements.txt      # Pinned dependencies
├── runtime.txt           # Python 3.11.9 specification
└── README.md
```

---

## Local Setup

```bash
git clone https://github.com/deepanjali6111/chaidocsRAGAgent.git
cd chaidocsRAGAgent

python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt

# Create .env:
GEMINI_API_KEY=your_key_here

streamlit run streamlit_app.py
```

---

## Run Evaluation

```bash
python evaluate_rag.py
```

Runs RAGAS evaluation across 5 test queries and prints faithfulness, answer relevancy, and context precision scores.

---

## What I Learned Building This

- **Pipeline separation** — offline ingestion vs online serving eliminates runtime API dependencies entirely
- **Quota management** — switching from Gemini embedding API to local HuggingFace resolved RESOURCE_EXHAUSTED errors permanently; 100% cost reduction
- **Dependency pinning** — `protobuf==3.20.3` was critical; newer versions broke the Streamlit and Gemini dependency chain
- **Deployment debugging** — newly created API keys may not immediately have full model access; reverting to established key resolved 401 errors
- **RAG evaluation** — RAGAS framework measures faithfulness (hallucination detection), answer relevancy, and context precision independently; useful for diagnosing retrieval vs generation quality issues
- **Memory constraints** — Render's 512MB free tier couldn't load sentence-transformers alongside Streamlit and ChromaDB; Streamlit Cloud handles ML workloads better

---

## Built With

- [LangChain](https://langchain.com) · [Google Gemini](https://ai.google.dev) · [HuggingFace](https://huggingface.co) · [ChromaDB](https://trychroma.com) · [Streamlit](https://streamlit.io) · [RAGAS](https://docs.ragas.io) · [ChaiCode](https://chaicode.com)
