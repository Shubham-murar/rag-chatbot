import streamlit as st
from chatbot import get_answer, log_to_arize
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant as LCQdrant
from langchain_community.embeddings import HuggingFaceEmbeddings

# 🔹 Initialize Streamlit App
st.set_page_config(page_title="AI Clone Chatbot", layout="wide")
st.title("🤖 AI Clone Chatbot")

# 🔹 Initialize Qdrant Client (Persistent Storage)
collection_name = "ai_clone"
client = QdrantClient(path="qdrant_db", force_disable_lock=True)  # ✅ Opens Qdrant in read-only mode

# 🔹 Initialize Retriever
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
retriever = LCQdrant(client, collection_name, hf_embeddings.embed_query).as_retriever(search_kwargs={"k": 3})

# 🔹 User Input
user_input = st.text_input("Ask me anything about AI:")

if user_input:
    response = get_answer(user_input)

    # 🔹 Show chatbot response
    st.subheader("🤖 AI Response:")
    st.write(response)

    # 🔹 Show retrieved knowledge sources
    with st.sidebar:
        st.subheader("🔍 Retrieved Sources")
        docs = retriever.invoke(user_input)
        for doc in docs:
            st.write(f"- {doc.page_content[:200]}...")  # Show snippet

    # 🔹 User Feedback for Logging
    feedback = st.radio("Was this answer helpful?", ("👍 Yes", "👎 No"))

    # Logging Feedback to Arize
    actual_label = response if feedback == "👍 Yes" else "INCORRECT"
    log_to_arize(user_input, response, actual_label)

    st.success("✅ Your feedback has been recorded!")

