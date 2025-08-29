

# 🤖 ChaiDocs RAG Agent

A **Retrieval-Augmented Generation (RAG) assistant** for [ChaiCode documentation](https://docs.chaicode.com), built using **Google Gemini**, **LangChain**, and **Streamlit**.

This project allows you to ask questions about ChaiCode docs in natural language. The app retrieves relevant sections of the docs, passes them into Gemini, and returns concise, context-aware answers.

---

## 🚀 Features

* **Streamlit Chat UI** with memory
* **Google Gemini API** for LLM responses
* **RAG Pipeline** powered by LangChain + ChromaDB
* **Automatic Web Scraping** of ChaiCode documentation
* **Fallback Docs Mode** (if URLs are unreachable)
* **Debugging Toolkit** (`debug.py`) to check API keys, URLs, DB, and AI connection

---

## 🛠️ Tech Stack

* [Streamlit](https://streamlit.io/) → Chat UI
* [LangChain](https://www.langchain.com/) → RAG pipeline
* [Google Generative AI (Gemini)](https://ai.google.dev/) → LLM & embeddings
* [Chroma](https://www.trychroma.com/) → Vector DB
* \[BeautifulSoup4 + Playwright] → Web scraping docs
* \[dotenv] → Environment variables

---

## 📂 Project Structure

```
chaidocsRAGAgent/
│── streamlit_app.py      # Main Streamlit UI
│── chaidocs_rag.py       # RAG pipeline (docs loader, embeddings, retriever, chain)
│── debug.py              # Debugging script
│── requirements.txt      # Dependencies
│── .env.example          # Example environment variables
│── chroma_db/            # Local vector DB (auto-created)
```

---

## ⚙️ Setup & Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/deepanjali6111/chaidocsRAGAgent.git
   cd chaidocsRAGAgent
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate   # (Linux/Mac)
   venv\Scripts\activate      # (Windows)
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up `.env` file**

   ```env
   GEMINI_API_KEY=your_google_gemini_api_key
   ```

   > 🔑 Get your API key from [Google AI Studio](https://aistudio.google.com/).

---

## ▶️ Run the App

```bash
streamlit run streamlit_app.py
```


* Ask questions like:

  * *"What is ChaiCode?"*
  * *"How do I get started?"*
  * *"Tell me about authentication"*

---

## 🧪 Debugging

If the app doesn’t work, run the debug script:

```bash
python debug.py
```

It checks:

* ✅ `GEMINI_API_KEY` is set
* ✅ Docs URLs are accessible
* ✅ Chroma DB exists
* ✅ Gemini API connection works

---

## 🔮 Roadmap / Future Work

* [ ] Deploy on Streamlit Cloud / Vercel / Hugging Face Spaces
* [ ] Add **more ChaiCode course docs** automatically
* [ ] Improve **UI/UX** with persistent chat history
* [ ] Add **API endpoints** for external use

---

## 🙌 Acknowledgements

* [LangChain](https://www.langchain.com/)
* [Google Generative AI](https://ai.google.dev/)
* [ChaiCode](https://chaicode.com/) for learning resources

---


