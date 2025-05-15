import os
import asyncio
import nest_asyncio
import streamlit as st
from dotenv import load_dotenv
from main_rag_bot import setup_retrieval_chain
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

# 🧠 Fix event loop for Streamlit + asyncio
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
nest_asyncio.apply()

# 🔐 Load environment variables
load_dotenv()

# 🔗 Qdrant + Groq Config from .env
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
collection_name = os.getenv("QDRANT_COLLECTION_NAME")

# 🎨 Streamlit UI setup
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("🧠 RAG-Powered Chatbot (Preprocessed Docs)")

# ⚡ Load existing Qdrant vectorstore
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create Qdrant client
    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
    )

    # Return LangChain vector store
    return Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings,
    )

# 🔁 Load QA chain once
if "qa_chain" not in st.session_state:
    vectorstore = load_vectorstore()
    st.session_state.qa_chain = setup_retrieval_chain(vectorstore, groq_api_key)

# 💬 Show chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📚 View Sources"):
                for doc in message["sources"]:
                    st.markdown(f"- {doc.page_content[:300]}...")

# ✍️ User prompt
if prompt := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response = st.session_state.qa_chain.invoke(prompt)
            st.markdown(response['result'])

            if response.get("source_documents"):
                with st.expander("📚 View Sources"):
                    for doc in response['source_documents']:
                        st.markdown(f"- {doc.page_content[:300]}...")

            st.session_state.messages.append({
                "role": "assistant",
                "content": response['result'],
                "sources": response.get("source_documents", [])
            })
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Sorry, I encountered an error while answering your query."
            })
