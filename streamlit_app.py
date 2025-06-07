
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "DISABLED"  # Block ADC
os.environ["GOOGLE_CLOUD_PROJECT"] = ""  # Clear project ID

import streamlit as st
from chaidocs_rag import ChaiDocsRAG
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Setup page
st.set_page_config(
    page_title="Chaicode Docs Assistant (Gemini)",
    page_icon="🤖",
    layout="centered"
)

# Initialize RAG
@st.cache_resource
def get_rag():
    try:
        rag = ChaiDocsRAG()
        return rag
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {str(e)}")
        return None

rag = get_rag()

# Chat UI
st.title("🤖 Chaicode Docs Assistant")
st.caption("Powered by Google Gemini")

# Add debug info in sidebar
with st.sidebar:
    st.header("Debug Info")
    if rag:
        st.success("✅ RAG System Initialized")
    else:
        st.error("❌ RAG System Failed")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("Reset RAG System"):
        st.cache_resource.clear()
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle user input
if prompt := st.chat_input("Ask about Chaicode docs..."):
    if not rag:
        st.error("RAG system not initialized. Please check your configuration.")
        st.stop()
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching docs..."):
            try:
                response = rag.query(prompt)
                if "Error" in response or "Failed" in response:
                    st.error("There was an issue processing your query. Check the logs for details.")
            except Exception as e:
                logger.error(f"Query error: {str(e)}")
                response = f"I encountered an error while processing your question: {str(e)}"
                st.error("Something went wrong. Please try again.")
        
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

# Add some example questions
st.markdown("---")
st.markdown("**Try asking:**")
col1, col2 = st.columns(2)
with col1:
    if st.button("What is ChaiCode?"):
        st.session_state.messages.append({"role": "user", "content": "What is ChaiCode?"})
        st.rerun()
    
    if st.button("How do I get started?"):
        st.session_state.messages.append({"role": "user", "content": "How do I get started?"})
        st.rerun()

with col2:
    if st.button("Tell me about authentication"):
        st.session_state.messages.append({"role": "user", "content": "Tell me about authentication"})
        st.rerun()
    
    if st.button("API documentation"):
        st.session_state.messages.append({"role": "user", "content": "Show me API documentation"})
        st.rerun()