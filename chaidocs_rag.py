
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "DISABLED"
os.environ["GOOGLE_CLOUD_PROJECT"] = ""
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import SitemapLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChaiDocsRAG:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing GEMINI_API_KEY in .env")

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
        loader = SitemapLoader(
            "https://docs.chaicode.com/sitemap.xml"
        )
        loader.requests_per_second = 1
        docs = loader.load()
        if docs:
            logger.info(f"Loaded {len(docs)} documents from sitemap")
            for doc in docs:
                doc.metadata["source_url"] = doc.metadata.get("source", "N/A")
            return docs
        logger.warning("Sitemap returned 0 documents, using fallback")
    except Exception as e:
        logger.error(f"Sitemap load failed: {str(e)}")

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

        embeddings = GoogleGenerativeAIEmbeddings(
            model="text-embedding-004",
            google_api_key=self.api_key,

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

Based on the following context, answer the user's question. If the context doesn't contain
relevant information, say so clearly and share whatever related info might help.

Context:
{context}

Question: {question}

Provide a helpful answer in markdown format with sources when available."""

        prompt = ChatPromptTemplate.from_template(template)

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.1,
            google_api_key=self.api_key
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
