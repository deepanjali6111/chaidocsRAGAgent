import os
import shutil
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "DISABLED"
os.environ["GOOGLE_CLOUD_PROJECT"] = ""
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import logging
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DOCS_URLS = [
    "https://docs.chaicode.com/youtube/getting-started/",
    "https://docs.chaicode.com/youtube/chai-aur-html/welcome/",
    "https://docs.chaicode.com/youtube/chai-aur-html/introduction/",
    "https://docs.chaicode.com/youtube/chai-aur-html/emmit-crash-course/",
    "https://docs.chaicode.com/youtube/chai-aur-html/html-tags/",
    "https://docs.chaicode.com/youtube/chai-aur-git/welcome/",
    "https://docs.chaicode.com/youtube/chai-aur-git/introduction/",
    "https://docs.chaicode.com/youtube/chai-aur-git/terminology/",
    "https://docs.chaicode.com/youtube/chai-aur-git/behind-the-scenes/",
    "https://docs.chaicode.com/youtube/chai-aur-git/branches/",
    "https://docs.chaicode.com/youtube/chai-aur-git/diff-stash-tags/",
    "https://docs.chaicode.com/youtube/chai-aur-git/managing-history/",
    "https://docs.chaicode.com/youtube/chai-aur-git/github/",
    "https://docs.chaicode.com/youtube/chai-aur-c/welcome/",
    "https://docs.chaicode.com/youtube/chai-aur-c/introduction/",
    "https://docs.chaicode.com/youtube/chai-aur-c/hello-world/",
    "https://docs.chaicode.com/youtube/chai-aur-c/variables-and-constants/",
    "https://docs.chaicode.com/youtube/chai-aur-c/data-types/",
    "https://docs.chaicode.com/youtube/chai-aur-c/operators/",
    "https://docs.chaicode.com/youtube/chai-aur-c/control-flow/",
    "https://docs.chaicode.com/youtube/chai-aur-c/loops/",
    "https://docs.chaicode.com/youtube/chai-aur-c/functions/",
    "https://docs.chaicode.com/youtube/chai-aur-django/welcome/",
    "https://docs.chaicode.com/youtube/chai-aur-django/getting-started/",
    "https://docs.chaicode.com/youtube/chai-aur-django/jinja-templates/",
    "https://docs.chaicode.com/youtube/chai-aur-django/tailwind/",
    "https://docs.chaicode.com/youtube/chai-aur-django/models/",
    "https://docs.chaicode.com/youtube/chai-aur-django/relationships-and-forms/",
    "https://docs.chaicode.com/youtube/chai-aur-sql/welcome/",
    "https://docs.chaicode.com/youtube/chai-aur-sql/introduction/",
    "https://docs.chaicode.com/youtube/chai-aur-sql/postgres/",
    "https://docs.chaicode.com/youtube/chai-aur-sql/normalization/",
    "https://docs.chaicode.com/youtube/chai-aur-sql/database-design-exercise/",
    "https://docs.chaicode.com/youtube/chai-aur-sql/joins-and-keys/",
    "https://docs.chaicode.com/youtube/chai-aur-sql/joins-exercise/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/welcome/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/setup-vpc/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/setup-nginx/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/nginx-rate-limiting/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/nginx-ssl-setup/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/node-nginx-vps/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/postgresql-docker/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/postgresql-vps/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/node-logger/",
]


class ChaiDocsRAG:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY or GOOGLE_API_KEY")
        os.environ["GOOGLE_API_KEY"] = api_key
        self.vectorstore = None
        self.retriever = None
        self.chain = None
        self.docs_loaded = False

    def create_fallback_docs(self):
        logger.info("Creating fallback documents...")
        return [
            Document(
                page_content="ChaiCode is a platform for learning programming. It offers YouTube tutorials and documentation for developers.",
                metadata={"source_url": "fallback", "title": "ChaiCode Overview"}
            ),
            Document(
                page_content="To get started with ChaiCode, visit the documentation website. You can find tutorials on HTML, Git, C, Django, SQL, and DevOps.",
                metadata={"source_url": "fallback", "title": "Getting Started"}
            ),
            Document(
                page_content="ChaiCode covers HTML, Git, C programming, Django, SQL, and DevOps topics through structured YouTube course documentation.",
                metadata={"source_url": "fallback", "title": "Courses Overview"}
            )
        ]

    def load_docs(self) -> List[Document]:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            # Filter only accessible URLs
            accessible = []
            for url in DOCS_URLS:
                try:
                    r = requests.get(url, headers=headers, timeout=10)
                    if r.status_code == 200:
                        accessible.append(url)
                except Exception:
                    pass

            if not accessible:
                logger.warning("No accessible URLs, using fallback")
                return self.create_fallback_docs()

            logger.info(f"Found {len(accessible)} accessible URLs")
            loader = WebBaseLoader(
                web_paths=accessible,
                requests_per_second=2,
                header_template=headers
            )
            docs = loader.load()
            if docs:
                logger.info(f"Loaded {len(docs)} documents")
                for doc in docs:
                    doc.metadata["source_url"] = doc.metadata.get("source", "N/A")
                return docs
        except Exception as e:
            logger.error(f"Failed to load docs: {str(e)}")

        return self.create_fallback_docs()

    def process_docs(self):
        docs = self.load_docs()
        if not docs:
            logger.error("No documents to process!")
            return
        logger.info(f"Processing {len(docs)} documents...")
        splits = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        ).split_documents(docs)
        logger.info(f"Created {len(splits)} chunks")

        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
            logger.info("Cleared old chroma_db")

        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001"
        )
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6}
        )
        self.docs_loaded = True
        logger.info("Documents processed and vectorstore created successfully")

    def setup_chain(self):
        if not self.docs_loaded:
            logger.error("Documents not loaded! Call process_docs() first.")
            return
        template = """You are a helpful assistant for ChaiCode documentation.

Based on the following context, answer the user's question. If the context does not contain
relevant information, say so clearly and share whatever related info might help.

Context:
{context}

Question: {question}

Provide a helpful answer in markdown format with sources when available."""
        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1
        )
        self.chain = (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        logger.info("Chain setup completed")

    def format_docs(self, docs: List[Document]) -> str:
        if not docs:
            return "No relevant documents found."
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source_url", "Unknown")
            formatted.append(f"Document {i+1}:\n{doc.page_content.strip()}\nSource: {source}")
        logger.info(f"Formatted {len(docs)} documents for context")
        return "\n\n---\n\n".join(formatted)

    def query(self, question: str) -> str:
        if not self.chain:
            logger.info("Initializing RAG system...")
            try:
                self.process_docs()
                self.setup_chain()
            except Exception as e:
                logger.error(f"Failed to initialize RAG: {str(e)}")
                return f"Failed to initialize the system: {str(e)}"
        try:
            logger.info(f"Processing query: {question}")
            return self.chain.invoke(question.strip())
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            return f"Error processing your request: {str(e)}"


if __name__ == "__main__":
    rag = ChaiDocsRAG()
    for q in ["What is ChaiCode?", "How do I get started?", "What is Git?", "Tell me about Django"]:
        print(f"\n{'='*50}\nQuery: {q}\n{'='*50}")
        print(rag.query(q))
