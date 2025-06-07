
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "DISABLED"
os.environ["GOOGLE_CLOUD_PROJECT"] = ""
os.environ["USER_AGENT"] = "ChaiDocsBot/1.0"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from google.generativeai import configure
from dotenv import load_dotenv
import logging
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChaiDocsRAG:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing GEMINI_API_KEY in .env")
        
        configure(api_key=self.api_key)
        
        # Test these URLs first - they might not exist
        self.docs_urls = [
            "https://docs.chaicode.com/youtube/getting-started/",
             "https://docs.chaicode.com/youtube/chai-aur-html/welcome/",
        "https://docs.chaicode.com/youtube/chai-aur-html/introduction/",
        "https://docs.chaicode.com/youtube/chai-aur-html/emmit-crash-course/",
        "https://docs.chaicode.com/youtube/chai-aur-html/html-tags/",
        "http://docs.chaicode.com/youtube/chai-aur-git/welcome/",
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
         "https://docs.chaicode.com/youtube/chai-aur-devops/node-logger/"

        ]
        
        self.vectorstore = None
        self.retriever = None
        self.chain = None
        self.docs_loaded = False

    def check_urls(self):
        """Check if URLs are accessible"""
        accessible_urls = []
        for url in self.docs_urls:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    accessible_urls.append(url)
                    logger.info(f"✅ URL accessible: {url}")
                else:
                    logger.warning(f"❌ URL returned {response.status_code}: {url}")
            except Exception as e:
                logger.error(f"❌ URL failed: {url} - {str(e)}")
        
        if not accessible_urls:
            logger.error("No accessible URLs found!")
            # Fallback to some test content
            return self.create_fallback_docs()
        
        self.docs_urls = accessible_urls
        return None

    def create_fallback_docs(self):
        """Create some test documents if URLs are not accessible"""
        logger.info("Creating fallback test documents...")
        fallback_docs = [
            Document(
                page_content="ChaiCode is a platform for learning programming. It offers YouTube tutorials and API documentation for developers.",
                metadata={"source_url": "fallback", "title": "ChaiCode Overview"}
            ),
            Document(
                page_content="To get started with ChaiCode, visit the documentation website. You can find tutorials on Python, JavaScript, and web development.",
                metadata={"source_url": "fallback", "title": "Getting Started"}
            ),
            Document(
                page_content="ChaiCode API provides endpoints for user authentication, video management, and content creation. Authentication is required for most operations.",
                metadata={"source_url": "fallback", "title": "API Reference"}
            )
        ]
        return fallback_docs

    def load_docs(self) -> List[Document]:
        # First check if URLs are accessible
        fallback_docs = self.check_urls()
        if fallback_docs:
            return fallback_docs
        
        try:
            loader = WebBaseLoader(
                web_paths=self.docs_urls,
                requests_per_second=1,
                header_template={"User-Agent": os.getenv("USER_AGENT", "ChaiDocsBot/1.0")}
            )
            docs = loader.load()
            
            if not docs:
                logger.warning("No documents loaded from URLs, using fallback")
                return self.create_fallback_docs()
            
            logger.info(f"Successfully loaded {len(docs)} documents")
            
            for doc in docs:
                if not hasattr(doc, 'metadata'):
                    doc.metadata = {}
                doc.metadata["source_url"] = doc.metadata.get("source", "N/A")
                
            return docs
            
        except Exception as e:
            logger.error(f"Failed to load documents: {str(e)}")
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
                model="models/embedding-001",
                google_api_key=self.api_key
            )
            
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory="./chroma_db"
            )
            
            # Lower the score threshold to be less restrictive
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",  # Remove score threshold initially
                search_kwargs={"k": 6}  # Get more results
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
                model="gemini-2.0-flash-exp",  # Use the experimental model
                temperature=0.1,  # Slightly higher for more natural responses
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
        
        # Test retrieval first
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
    # Test the system
    rag = ChaiDocsRAG()
    
    # Test queries
    test_queries = [
        "What is ChaiCode?",
        "How do I get started?",
        "What is Python?",
        "Tell me about authentication"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        response = rag.query(query)
        print(response)