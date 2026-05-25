
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "DISABLED"
os.environ["GOOGLE_CLOUD_PROJECT"] = ""
os.environ["USER_AGENT"] = "ChaiDocsBot/1.0"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import SitemapLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from google.generativeai import configure
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

        configure(api_key=self.api_key)

        self.vectorstore = None
        self.retriever = None
        self.chain = None
        self.docs_loaded = False

    def create_fallback_docs(self):
        """Create some test documents if sitemap is not accessible"""
        logger.info("Creating fallback test documents...")
        fallback_docs = [
            Document(
                page_content="ChaiCode is a platform for learning programming. It offers YouTube tutorials and documentation for developers.",
                metadata={"source_url": "fallback", "title": "ChaiCode Overview"}
            ),
            Document(
                page_content="To get started with ChaiCode, visit the documentation website. You can find tutorials on Python, JavaScript, and web development.",
                metadata={"source_url": "fallback", "title": "Getting Started"}
            ),
            Document(
                page_content="ChaiCode covers HTML, Git, C programming, Django, SQL, and DevOps topics through structured YouTube course documentation.",
                metadata={"source_url": "fallback", "title": "Courses Overview"}
            )
        ]
        return fallback_docs

    def load_docs(self) -> List[Document]:
        try:
            loader = SitemapLoader(
                "https://docs.chaicode.com/sitemap.xml",
                filter_urls=["https://docs.chaicode.com/youtube/"]
            )
            loader.requests_per_second = 1
            docs = loader.load()
            if docs:
                logger.info(f"Successfully loaded {len(docs)} documents from sitemap")
                for doc in docs:
                    if not hasattr(doc, 'metadata'):
                        doc.metadata = {}
                    doc.metadata["source_url"] = doc.metadata.get("source", "N/A")
                return docs
        except Exception as e:
            logger.error(f"Sitemap load failed: {str(e)}")

        return self.create_fallback_docs()

    def process_docs(self):
        docs = self.load_docs()

        if not docs:
            logger.error("No documents to process!")
            return

        logger.info(f"Processing {len(docs)} documents...")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)

        logger.info(f"Created {len(splits)} document chunks")

        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=self.api_key,
                task_type="retrieval_document"
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
            logger.info("✅ Documents processed and vectorstore created successfully")

        except Exception as e:
            logger.error(f"Failed to create vectorstore: {str(e)}")
            raise

    def setup_chain(self):
        if not self.docs_loaded:
            logger.error("Documents not loaded! Call process_docs() first.")
            return

        template = """You are a helpful assistant for ChaiCode documentation.

        Based on the following context, answer the user's question. If the context doesn't contain relevant information, acknowledge what you don't know and provide any related information that might be helpful.

        Context:
        {context}

        Question: {question}

        Please provide a helpful answer based on the available information. If you're unsure about something, say so clearly.

        Format your response in markdown with sources when available."""

        prompt = ChatPromptTemplate.from_template(template)

        try:
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

            logger.info("✅ Chain setup completed")

        except Exception as e:
            logger.error(f"Failed to setup chain: {str(e)}")
            raise

    def format_docs(self, docs: List[Document]) -> str:
        if not docs:
            return "No relevant documents found in the knowledge base."

        formatted_docs = []
        for i, doc in enumerate(docs):
            if isinstance(doc, Document) and hasattr(doc, 'page_content'):
                source = doc.metadata.get('source_url', 'Unknown source')
                content = doc.page_content.strip()
                formatted_docs.append(f"Document {i+1}:\n{content}\nSource: {source}")

        result = "\n\n---\n\n".join(formatted_docs)
        logger.info(f"Formatted {len(docs)} documents for context")
        return result

    def test_retrieval(self, query: str):
        """Test method to see what documents are being retrieved"""
        if not self.retriever:
            return "Retriever not initialized"

        try:
            docs = self.retriever.invoke(query)
            logger.info(f"Retrieved {len(docs)} documents for query: '{query}'")
            for i, doc in enumerate(docs):
                logger.info(f"Doc {i+1}: {doc.page_content[:100]}...")
            return docs
        except Exception as e:
            logger.error(f"Retrieval test failed: {str(e)}")
            return []

    def query(self, question: str) -> str:
        if not self.chain:
            logger.info("Initializing RAG system...")
            try:
                self.process_docs()
                self.setup_chain()
            except Exception as e:
                logger.error(f"Failed to initialize RAG: {str(e)}")
                return f"Failed to initialize the system: {str(e)}"

        retrieved_docs = self.test_retrieval(question)
        if not retrieved_docs:
            logger.warning("No documents retrieved for the query")

        try:
            logger.info(f"Processing query: {question}")
            response = self.chain.invoke(question.strip())
            logger.info("Query processed successfully")
            return response
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            return f"Error processing your request: {str(e)}"


if __name__ == "__main__":
    rag = ChaiDocsRAG()

    test_queries = [
        "What is ChaiCode?",
        "How do I get started?",
        "What is Git?",
        "Tell me about Django"
    ]

    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        response = rag.query(query)
        print(response)
